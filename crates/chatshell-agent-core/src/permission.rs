//! Tool-permission decision layer for the approval gate.
//!
//! Pure logic: no I/O, no UI, no clock. The host supplies the permission
//! mode and the session grants and receives a three-valued decision
//! (allow / deny / prompt). Precedence: hard deny (classifier blocklist)
//! -> `yolo` allow -> forced prompt (classifier warning, unparsed command,
//! network egress, private-network fetch) -> session-grant lookup ->
//! mode/tier matrix -> prompt.
//!
//! Forced prompts ignore session grants and never offer one: the signal
//! that forced them is about this specific call, so an earlier approval
//! of the same base command must not silence it.
//!
//! The authoritative blocklist enforcement stays inside the tool bodies
//! (`bash_security`, `path_policy`) so it holds in every mode, including
//! [`PermissionMode::Yolo`]; the deny produced here is a pre-check that
//! avoids showing an approval prompt for commands the tool would refuse
//! anyway.

use crate::bash_security::{self, SecurityVerdict};
use crate::grant_keys::{self, FileTier};
use crate::net_policy;
use crate::path_policy::{self, ReadTier};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Conversation-level permission mode for the approval gate.
///
/// There is deliberately no `Default`: hosts source the mode from an
/// explicit setting, never from a compiled-in fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    /// Prompt before write- and exec-tier tool calls; session grants
    /// cache approvals.
    Ask,
    /// Bash runs inside the OS sandbox without prompting, and file writes
    /// inside the workspace run silently. Prompts remain for escaping the
    /// sandbox, out-of-workspace writes, MCP, `kill_shell`, unknown tools,
    /// classifier warnings, egress, and private-network fetches. Without a
    /// usable sandbox, bash falls back to per-command prompts.
    Auto,
    /// Run tool calls without prompts. Catastrophic operations are still
    /// hard-denied by the in-tool blocklist and path policy in every mode.
    Yolo,
}

/// Per-turn facts the resolver needs beyond the mode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GateContext {
    /// The conversation working directory (writes inside it are silent in
    /// `auto`).
    pub workspace: Option<PathBuf>,
    /// The bash session runs inside a working OS sandbox.
    pub sandbox_available: bool,
    /// "Allow network this session" was granted.
    pub network_allowed: bool,
}

/// Risk tier of a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionTier {
    /// Local read-only retrieval (`read`, `grep`, `glob`, `web_search`,
    /// `web_fetch`, `skill`, `mcp_schema`).
    Read,
    /// Local mutations (`edit`, `write`) and remote MCP execution
    /// (`mcp_tool_use`).
    Write,
    /// Shell execution (`bash`, `kill_shell`) and unknown tools
    /// (fail-safe: strictest tier).
    Exec,
}

impl PermissionTier {
    /// Tier for a registered tool name. Unknown names classify as
    /// [`PermissionTier::Exec`] — the fail-safe choice.
    pub fn for_tool(tool_name: &str) -> Self {
        match tool_name {
            "read" | "grep" | "glob" | "web_search" | "web_fetch" | "skill" | "mcp_schema" => {
                PermissionTier::Read
            }
            "edit" | "write" | "mcp_tool_use" => PermissionTier::Write,
            _ => PermissionTier::Exec,
        }
    }
}

/// Why a prompt is forced past session grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcedPromptKind {
    /// The bash classifier returned a `Warn` verdict (injection patterns,
    /// `eval`, substitutions, malformed syntax, ...).
    Security,
    /// The AST layer could not analyze the command (too long or parser
    /// failure), so its validators did not run.
    Unparsed,
    /// The command contacts a non-local host through a network client.
    Network,
    /// A fetch targets a private or local network address.
    PrivateNetwork,
    /// A file-tool read hits an ask-tier `path_policy` group (shell
    /// configuration files, which often hold exported API keys).
    SensitiveRead,
}

/// A prompt that session grants cannot satisfy, with the detail shown
/// on the approval card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForcedPrompt {
    pub kind: ForcedPromptKind,
    pub detail: String,
}

/// Gate verdict for one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Execute without asking.
    Allow,
    /// Refuse; `reason` is fed back to the model as the tool result.
    Deny { reason: String },
    /// Ask the user; `subject` is the human-readable target of the call.
    /// `forced` is set when a classifier signal requires this prompt
    /// regardless of session grants; such prompts offer no grant.
    Prompt {
        subject: String,
        forced: Option<ForcedPrompt>,
    },
}

impl GateDecision {
    fn prompt(subject: impl Into<String>) -> Self {
        GateDecision::Prompt {
            subject: subject.into(),
            forced: None,
        }
    }

    fn forced(subject: impl Into<String>, kind: ForcedPromptKind, detail: String) -> Self {
        GateDecision::Prompt {
            subject: subject.into(),
            forced: Some(ForcedPrompt { kind, detail }),
        }
    }
}

/// What an approval grants: the session scope ("allow for this
/// session") or the remember scope ("remember on this device").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantScope {
    /// Every segment key of the approved bash call (see
    /// [`grant_keys::bash_grant_keys`]): `git push`, `ls`, ... All future
    /// commands whose segments are all covered run without a prompt.
    BashCommands(Vec<String>),
    /// The whole tool (`edit`/`write` for the session, and `kill_shell`).
    Tool(String),
    /// One tool on one MCP server. MCP grants stay this narrow on
    /// purpose: approving a benign tool must not unlock other tools or
    /// other servers.
    ServerTool { server: String, tool: String },
    /// Writes to the direct entries of one directory (remembered file
    /// grants; a whole-tool grant is too broad to keep across sessions).
    WriteDirectory { tool: String, dir: PathBuf },
}

impl GrantScope {
    /// Grant keys this scope adds to a grant set.
    pub fn keys(&self) -> Vec<String> {
        match self {
            GrantScope::BashCommands(keys) => keys.iter().map(|k| format!("bash:{k}")).collect(),
            GrantScope::Tool(tool) => vec![format!("tool:{tool}")],
            GrantScope::ServerTool { server, tool } => vec![format!("mcp:{server}/{tool}")],
            GrantScope::WriteDirectory { dir, .. } => {
                vec![grant_keys::dir_key(FileTier::Write, dir)]
            }
        }
    }
}

/// A set of grant keys consulted by [`resolve`].
///
/// The session store keeps one per conversation in RAM; the gate merges
/// the device's remembered keys into a snapshot before resolving. Never
/// synced: a grant made on one device must not take effect elsewhere.
#[derive(Debug, Default, Clone)]
pub struct SessionGrants {
    keys: HashSet<String>,
}

impl SessionGrants {
    /// Grant bash segment keys (`git push`, `ls`, ...).
    pub fn grant_bash_commands(&mut self, keys: &[&str]) {
        for key in keys {
            self.keys.insert(format!("bash:{key}"));
        }
    }

    /// Add every key of `scope`.
    pub fn grant_scope(&mut self, scope: &GrantScope) {
        self.keys.extend(scope.keys());
    }

    /// Add raw grant keys (remembered grants loaded from the device store).
    pub fn extend_keys(&mut self, keys: impl IntoIterator<Item = String>) {
        self.keys.extend(keys);
    }

    /// Grant a whole tool for the rest of the session (write-tier tools).
    pub fn grant_tool(&mut self, tool: &str) {
        self.keys.insert(format!("tool:{tool}"));
    }

    /// Grant one tool on one MCP server for the rest of the session.
    pub fn grant_server_tool(&mut self, server: &str, tool: &str) {
        self.keys.insert(format!("mcp:{server}/{tool}"));
    }

    /// True when every segment key of `command` is granted. A partial
    /// grant never authorizes a compound command; an unkeyable command is
    /// never covered.
    pub fn bash_command_granted(&self, command: &str) -> bool {
        grant_keys::bash_grant_keys(command).is_some_and(|keys| {
            keys.iter()
                .all(|key| self.keys.contains(&format!("bash:{key}")))
        })
    }

    /// True when a file grant covers a `tier` call on `path` (already
    /// resolved with [`grant_keys::grant_path`]).
    pub fn file_granted(&self, tier: FileTier, path: &Path) -> bool {
        grant_keys::covering_file_keys(tier, path)
            .iter()
            .any(|key| self.keys.contains(key))
    }

    /// True when the tool was granted for the session.
    pub fn tool_granted(&self, tool: &str) -> bool {
        self.keys.contains(&format!("tool:{tool}"))
    }

    /// True when this exact server/tool pair was granted.
    pub fn server_tool_granted(&self, server: &str, tool: &str) -> bool {
        self.keys.contains(&format!("mcp:{server}/{tool}"))
    }

    /// True when no grants are held.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Drop all grants.
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

/// Human-readable target of a tool call, shown in approval prompts.
pub fn subject_for(tool_name: &str, args: &Value) -> String {
    let arg = |key: &str| args.get(key).and_then(Value::as_str);
    let subject = match tool_name {
        "bash" | "kill_shell" => arg("command").map(str::to_string),
        "edit" | "write" | "read" => arg("path").map(str::to_string),
        "mcp_tool_use" => match (arg("server"), arg("tool")) {
            (Some(server), Some(tool)) => Some(format!("{server}/{tool}")),
            _ => None,
        },
        // Read tier never prompts in v1; this arm feeds the possible
        // middle tier (risk-taxonomy prompts) where fetch URLs would be
        // shown as subjects.
        "web_fetch" => arg("url").map(str::to_string),
        _ => None,
    };
    subject.unwrap_or_else(|| tool_name.to_string())
}

/// What an "allow for this session" answer memoizes for this call.
///
/// `None` when no session grant is offered (exec tier other than bash and
/// `kill_shell`, or an MCP call missing its server/tool args): the UI
/// hides the session button for those. Forced prompts never offer a
/// grant either; the gate drops the scope for them.
pub fn grant_scope(tool_name: &str, args: &Value) -> Option<GrantScope> {
    if tool_name == "bash" {
        return args
            .get("command")
            .and_then(Value::as_str)
            .and_then(grant_keys::bash_grant_keys)
            .map(GrantScope::BashCommands);
    }
    if tool_name == "mcp_tool_use" {
        let server = args.get("server").and_then(Value::as_str);
        let tool = args.get("tool").and_then(Value::as_str);
        if let (Some(server), Some(tool)) = (server, tool) {
            return Some(GrantScope::ServerTool {
                server: server.to_string(),
                tool: tool.to_string(),
            });
        }
        return None;
    }
    if tool_name == "kill_shell"
        || matches!(PermissionTier::for_tool(tool_name), PermissionTier::Write)
    {
        return Some(GrantScope::Tool(tool_name.to_string()));
    }
    None
}

/// What "remember on this device" persists for this call (mode-source
/// prompts only; the gate never offers it on forced prompts).
///
/// Narrower than the session scope where the session scope is too broad
/// to keep: file tools remember writes to the target's directory (direct
/// entries) instead of the whole tool, and only for absolute paths.
pub fn remember_scope(tool_name: &str, args: &Value) -> Option<GrantScope> {
    match tool_name {
        "edit" | "write" => {
            let path = args.get("path").and_then(Value::as_str)?;
            let resolved = grant_keys::grant_path(path)?;
            let dir = resolved.parent()?.to_path_buf();
            Some(GrantScope::WriteDirectory {
                tool: tool_name.to_string(),
                dir,
            })
        }
        _ => grant_scope(tool_name, args),
    }
}

/// Classifier signal that forces a bash prompt in prompting modes, most
/// specific first: network egress, classifier warning, then an AST layer
/// that could not run (fail closed: its validators were skipped).
/// Classifier signal that forces a bash prompt in prompting modes: a
/// classifier warning first (the most serious), then an AST layer that
/// could not run (fail closed: its validators were skipped), then network
/// egress (skipped when the session already holds a network grant).
fn bash_forced_prompt(
    command: &str,
    classification: &bash_security::CommandClassification,
    network_allowed: bool,
) -> Option<(ForcedPromptKind, String)> {
    if let SecurityVerdict::Warn(message) = &classification.verdict {
        return Some((ForcedPromptKind::Security, message.clone()));
    }
    if !classification.ast_available {
        return Some((
            ForcedPromptKind::Unparsed,
            "Command could not be fully analyzed (too long or unparseable)".to_string(),
        ));
    }
    if !network_allowed && let Some(egress) = net_policy::detect_egress(command) {
        return Some((ForcedPromptKind::Network, egress));
    }
    None
}

/// True when `path` (a file-tool argument) resolves inside `workspace`.
fn inside_workspace(path: &str, workspace: &Path) -> bool {
    let Some(target) = grant_keys::grant_path(path) else {
        return false;
    };
    let root = workspace
        .to_str()
        .and_then(grant_keys::grant_path)
        .unwrap_or_else(|| workspace.to_path_buf());
    target.starts_with(&root) && target != root
}

/// The ask-tier read block for a file-tool path, if any. Relative paths
/// are also classified against the workspace. Deny-tier hits return
/// `None`: the tool refuses them itself, so prompting would be pointless.
fn ask_tier_read(path: &str, workspace: Option<&Path>) -> Option<path_policy::ReadBlock> {
    let path = Path::new(path);
    let mut candidates = vec![path.to_path_buf()];
    if path.is_relative()
        && let Some(workspace) = workspace
    {
        candidates.push(workspace.join(path));
    }
    let blocks: Vec<_> = candidates
        .iter()
        .filter_map(|candidate| path_policy::classify_read(candidate))
        .collect();
    if blocks.iter().any(|block| block.tier == ReadTier::Deny) {
        return None;
    }
    blocks.into_iter().find(|block| block.tier == ReadTier::Ask)
}

/// Resolve one tool call against the mode and session grants, with no
/// workspace, no sandbox, and no network grant.
pub fn resolve(
    mode: PermissionMode,
    tool_name: &str,
    args: &Value,
    grants: &SessionGrants,
) -> GateDecision {
    resolve_in(mode, &GateContext::default(), tool_name, args, grants)
}

/// Resolve one tool call against the mode, the turn context, and the
/// session grants.
pub fn resolve_in(
    mode: PermissionMode,
    ctx: &GateContext,
    tool_name: &str,
    args: &Value,
    grants: &SessionGrants,
) -> GateDecision {
    let sandboxed_auto = mode == PermissionMode::Auto && ctx.sandbox_available;
    // Bash: hard-deny pre-check, then the mode, forced prompts, grants.
    // The authoritative blocklist lives inside the bash tool; this
    // pre-check denies before an approval prompt would be shown for a
    // command the tool would refuse anyway. The classification is
    // memoized, so the tool reuses it instead of re-running the pipeline.
    if tool_name == "bash" {
        // Bash without a command argument: let the tool itself surface
        // the "No command provided" error instead of prompting.
        let Some(command) = args.get("command").and_then(Value::as_str) else {
            return GateDecision::Allow;
        };
        let classification = bash_security::classify_command(command);
        if let SecurityVerdict::Block(reason) = &classification.verdict {
            return GateDecision::Deny {
                reason: reason.clone(),
            };
        }
        if mode == PermissionMode::Yolo {
            return GateDecision::Allow;
        }
        // The network grant only means something when a sandbox enforces
        // the alternative; without one, egress always prompts.
        let network_allowed = sandboxed_auto && ctx.network_allowed;
        if let Some((kind, detail)) = bash_forced_prompt(command, &classification, network_allowed)
        {
            return GateDecision::forced(command, kind, detail);
        }
        // In sandboxed auto the OS sandbox absorbs everything else;
        // escapes surface as sandbox escalations after execution.
        if sandboxed_auto || grants.bash_command_granted(command) {
            return GateDecision::Allow;
        }
        return GateDecision::prompt(command);
    }

    if mode == PermissionMode::Yolo {
        return GateDecision::Allow;
    }

    // Read tier runs silently, except fetches into private networks.
    let tier = PermissionTier::for_tool(tool_name);
    if matches!(tier, PermissionTier::Read) {
        if tool_name == "web_fetch"
            && let Some(url) = args.get("url").and_then(Value::as_str)
            && net_policy::is_private_url(url)
        {
            return GateDecision::forced(
                url,
                ForcedPromptKind::PrivateNetwork,
                format!("Fetch targets a private or local network address: {url}"),
            );
        }
        if matches!(tool_name, "read" | "grep" | "glob")
            && let Some(path) = args.get("path").and_then(Value::as_str)
            && let Some(block) = ask_tier_read(path, ctx.workspace.as_deref())
        {
            return GateDecision::forced(
                path,
                ForcedPromptKind::SensitiveRead,
                format!(
                    "Reads a {} file, which may contain secrets: {}",
                    block.label, block.matched
                ),
            );
        }
        return GateDecision::Allow;
    }

    // Auto: file writes inside the workspace run silently. The tools'
    // path policy still refuses protected paths in every mode.
    if mode == PermissionMode::Auto
        && matches!(tool_name, "edit" | "write")
        && let (Some(path), Some(workspace)) =
            (args.get("path").and_then(Value::as_str), &ctx.workspace)
        && inside_workspace(path, workspace)
    {
        return GateDecision::Allow;
    }

    if tool_name == "mcp_tool_use" {
        // MCP grants are per server/tool pair; a call missing either arg
        // matches no grant and falls through to a prompt (fail-safe).
        if let (Some(server), Some(tool)) = (
            args.get("server").and_then(Value::as_str),
            args.get("tool").and_then(Value::as_str),
        ) && grants.server_tool_granted(server, tool)
        {
            return GateDecision::Allow;
        }
    } else if grant_scope(tool_name, args).is_some() && grants.tool_granted(tool_name) {
        return GateDecision::Allow;
    } else if matches!(tool_name, "edit" | "write")
        && let Some(path) = args.get("path").and_then(Value::as_str)
        && let Some(resolved) = grant_keys::grant_path(path)
        && grants.file_granted(FileTier::Write, &resolved)
    {
        return GateDecision::Allow;
    }
    GateDecision::prompt(subject_for(tool_name, args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resolve_simple(mode: PermissionMode, tool: &str, args: Value) -> GateDecision {
        resolve(mode, tool, &args, &SessionGrants::default())
    }

    #[test]
    fn wire_values_are_lowercase() {
        assert_eq!(
            serde_json::to_value(PermissionMode::Ask).unwrap(),
            json!("ask")
        );
        assert_eq!(
            serde_json::to_value(PermissionMode::Yolo).unwrap(),
            json!("yolo")
        );
        assert_eq!(
            serde_json::from_value::<PermissionMode>(json!("yolo")).unwrap(),
            PermissionMode::Yolo
        );
        assert_eq!(
            serde_json::to_value(PermissionMode::Auto).unwrap(),
            json!("auto")
        );
    }

    #[test]
    fn tier_mapping_covers_registered_tools_and_fails_safe() {
        assert_eq!(PermissionTier::for_tool("read"), PermissionTier::Read);
        assert_eq!(PermissionTier::for_tool("web_fetch"), PermissionTier::Read);
        assert_eq!(PermissionTier::for_tool("mcp_schema"), PermissionTier::Read);
        assert_eq!(PermissionTier::for_tool("edit"), PermissionTier::Write);
        assert_eq!(
            PermissionTier::for_tool("mcp_tool_use"),
            PermissionTier::Write
        );
        assert_eq!(PermissionTier::for_tool("bash"), PermissionTier::Exec);
        assert_eq!(PermissionTier::for_tool("kill_shell"), PermissionTier::Exec);
        assert_eq!(
            PermissionTier::for_tool("some_custom_mcp_thing"),
            PermissionTier::Exec
        );
    }

    #[test]
    fn yolo_allows_write_and_exec() {
        let write = json!({"path": "/tmp/a.txt", "content": "x"});
        assert_eq!(
            resolve_simple(PermissionMode::Yolo, "write", write),
            GateDecision::Allow
        );
        let bash = json!({"command": "cargo test"});
        assert_eq!(
            resolve_simple(PermissionMode::Yolo, "bash", bash),
            GateDecision::Allow
        );
    }

    #[test]
    fn ask_prompts_write_and_exec_but_allows_read() {
        let write = json!({"path": "/tmp/a.txt", "content": "x"});
        assert_eq!(
            resolve_simple(PermissionMode::Ask, "write", write.clone()),
            GateDecision::prompt("/tmp/a.txt")
        );
        let bash = json!({"command": "cargo test"});
        assert_eq!(
            resolve_simple(PermissionMode::Ask, "bash", bash),
            GateDecision::prompt("cargo test")
        );
        let read = json!({"path": "/tmp/a.txt"});
        assert_eq!(
            resolve_simple(PermissionMode::Ask, "read", read),
            GateDecision::Allow
        );
    }

    #[test]
    fn blocklist_denies_in_every_mode_including_yolo() {
        let command = json!({"command": "rm -rf /"});
        for mode in [PermissionMode::Ask, PermissionMode::Yolo] {
            match resolve_simple(mode, "bash", command.clone()) {
                GateDecision::Deny { .. } => {}
                other => panic!("{mode:?} should deny `rm -rf /`, got {other:?}"),
            }
        }
    }

    #[test]
    fn blocklist_pre_check_takes_precedence_over_grants() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["rm"]);
        let decision = resolve(
            PermissionMode::Ask,
            "bash",
            &json!({"command": "rm -rf /"}),
            &grants,
        );
        assert!(matches!(decision, GateDecision::Deny { .. }));
    }

    #[test]
    fn bash_grant_covers_single_command_but_not_compounds() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["cargo build"]);
        assert_eq!(
            resolve(
                PermissionMode::Ask,
                "bash",
                &json!({"command": "cargo build"}),
                &grants
            ),
            GateDecision::Allow
        );
        assert_eq!(
            resolve(
                PermissionMode::Ask,
                "bash",
                &json!({"command": "cargo build && npm install"}),
                &grants
            ),
            GateDecision::prompt("cargo build && npm install")
        );
    }

    #[test]
    fn tool_grant_scopes_to_one_tool() {
        let mut grants = SessionGrants::default();
        grants.grant_tool("edit");
        assert_eq!(
            resolve(
                PermissionMode::Ask,
                "edit",
                &json!({"path": "/tmp/a.txt"}),
                &grants
            ),
            GateDecision::Allow
        );
        assert_eq!(
            resolve(
                PermissionMode::Ask,
                "write",
                &json!({"path": "/tmp/a.txt"}),
                &grants
            ),
            GateDecision::prompt("/tmp/a.txt")
        );
    }

    #[test]
    fn missing_bash_command_defers_to_tool_error() {
        assert_eq!(
            resolve_simple(PermissionMode::Ask, "bash", json!({})),
            GateDecision::Allow
        );
    }

    #[test]
    fn subjects_cover_each_tool_family() {
        assert_eq!(
            subject_for("bash", &json!({"command": "ls"})),
            "ls".to_string()
        );
        assert_eq!(
            subject_for("write", &json!({"path": "/tmp/a"})),
            "/tmp/a".to_string()
        );
        assert_eq!(
            subject_for("mcp_tool_use", &json!({"server": "s", "tool": "t"})),
            "s/t".to_string()
        );
        assert_eq!(
            subject_for("web_fetch", &json!({"url": "https://x.example"})),
            "https://x.example".to_string()
        );
        assert_eq!(
            subject_for("kill_shell", &json!({})),
            "kill_shell".to_string()
        );
    }

    #[test]
    fn grant_scope_for_exec_tools_covers_bash_and_kill_shell_only() {
        assert_eq!(
            grant_scope("kill_shell", &json!({})),
            Some(GrantScope::Tool("kill_shell".to_string()))
        );
        assert_eq!(grant_scope("some_custom_mcp_thing", &json!({})), None);
        assert_eq!(
            grant_scope("bash", &json!({"command": "git status && git push"})),
            Some(GrantScope::BashCommands(vec![
                "git push".to_string(),
                "git status".to_string()
            ]))
        );
        assert_eq!(
            grant_scope("edit", &json!({"path": "/tmp/a"})),
            Some(GrantScope::Tool("edit".to_string()))
        );
    }

    #[test]
    fn mcp_grant_scopes_to_one_server_tool_pair() {
        let mut grants = SessionGrants::default();
        grants.grant_server_tool("server-a", "fetch");
        let granted = json!({"server": "server-a", "tool": "fetch"});
        assert_eq!(
            resolve(PermissionMode::Ask, "mcp_tool_use", &granted, &grants),
            GateDecision::Allow
        );
        // Same server, different tool: not covered.
        let other_tool = json!({"server": "server-a", "tool": "delete_file"});
        assert!(matches!(
            resolve(PermissionMode::Ask, "mcp_tool_use", &other_tool, &grants),
            GateDecision::Prompt { .. }
        ));
        // Same tool name on another server: not covered.
        let other_server = json!({"server": "server-b", "tool": "fetch"});
        assert!(matches!(
            resolve(PermissionMode::Ask, "mcp_tool_use", &other_server, &grants),
            GateDecision::Prompt { .. }
        ));
    }

    #[test]
    fn mcp_call_without_server_or_tool_args_has_no_session_grant() {
        let mut grants = SessionGrants::default();
        grants.grant_server_tool("server-a", "fetch");
        // A blanket tool-level grant must never match an MCP call.
        grants.grant_tool("mcp_tool_use");
        assert!(matches!(
            resolve(PermissionMode::Ask, "mcp_tool_use", &json!({}), &grants),
            GateDecision::Prompt { .. }
        ));
        assert_eq!(grant_scope("mcp_tool_use", &json!({})), None);
        assert_eq!(
            grant_scope("mcp_tool_use", &json!({"server": "s", "tool": "t"})),
            Some(GrantScope::ServerTool {
                server: "s".to_string(),
                tool: "t".to_string()
            })
        );
    }

    fn forced_kind(decision: &GateDecision) -> Option<ForcedPromptKind> {
        match decision {
            GateDecision::Prompt {
                forced: Some(forced),
                ..
            } => Some(forced.kind),
            _ => None,
        }
    }

    #[test]
    fn warn_verdict_forces_prompt_past_session_grant() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["echo"]);
        let args = json!({"command": "echo $(whoami)"});
        let decision = resolve(PermissionMode::Ask, "bash", &args, &grants);
        assert_eq!(forced_kind(&decision), Some(ForcedPromptKind::Security));
        // The same base without the flagged construct stays covered.
        assert_eq!(
            resolve(
                PermissionMode::Ask,
                "bash",
                &json!({"command": "echo hi"}),
                &grants
            ),
            GateDecision::Allow
        );
    }

    #[test]
    fn yolo_keeps_blocklist_floor_only() {
        for command in [
            "echo $(whoami)",
            "curl https://example.com",
            &format!("echo {}", "a".repeat(12_000)),
        ] {
            assert_eq!(
                resolve_simple(PermissionMode::Yolo, "bash", json!({ "command": command })),
                GateDecision::Allow,
                "{command:.40}"
            );
        }
    }

    #[test]
    fn unanalyzable_command_fails_closed_in_ask() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["echo"]);
        let long = format!("echo {}", "a".repeat(12_000));
        let decision = resolve(
            PermissionMode::Ask,
            "bash",
            &json!({ "command": long }),
            &grants,
        );
        assert_eq!(forced_kind(&decision), Some(ForcedPromptKind::Unparsed));
    }

    #[test]
    fn malformed_syntax_forces_prompt_in_ask() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["echo"]);
        let decision = resolve(
            PermissionMode::Ask,
            "bash",
            &json!({"command": "echo 'unterminated"}),
            &grants,
        );
        assert!(forced_kind(&decision).is_some(), "got {decision:?}");
    }

    #[test]
    fn shell_config_reads_force_a_prompt_except_in_yolo() {
        let home = dirs::home_dir().unwrap();
        let zshrc = home.join(".zshrc").display().to_string();
        let mut grants = SessionGrants::default();
        grants.grant_tool("read");
        for mode in [PermissionMode::Ask, PermissionMode::Auto] {
            for tool in ["read", "grep", "glob"] {
                let decision = resolve(mode, tool, &json!({ "path": zshrc }), &grants);
                assert_eq!(
                    forced_kind(&decision),
                    Some(ForcedPromptKind::SensitiveRead),
                    "{mode:?} {tool}"
                );
            }
        }
        assert_eq!(
            resolve(
                PermissionMode::Yolo,
                "read",
                &json!({ "path": zshrc }),
                &grants
            ),
            GateDecision::Allow
        );
        // Relative paths are classified against the workspace.
        let ctx = GateContext {
            workspace: Some(home.clone()),
            ..GateContext::default()
        };
        let decision = resolve_in(
            PermissionMode::Ask,
            &ctx,
            "read",
            &json!({ "path": ".bashrc" }),
            &SessionGrants::default(),
        );
        assert_eq!(
            forced_kind(&decision),
            Some(ForcedPromptKind::SensitiveRead)
        );
    }

    #[test]
    fn deny_tier_and_ordinary_reads_do_not_prompt() {
        let home = dirs::home_dir().unwrap();
        // Deny tier: the tool refuses it; prompting would be pointless.
        let key = home.join(".ssh/id_ed25519").display().to_string();
        assert_eq!(
            resolve_simple(PermissionMode::Ask, "read", json!({ "path": key })),
            GateDecision::Allow
        );
        assert_eq!(
            resolve_simple(
                PermissionMode::Ask,
                "read",
                json!({ "path": "/tmp/notes.txt" })
            ),
            GateDecision::Allow
        );
    }

    #[test]
    fn egress_forces_network_prompt_in_ask() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["curl"]);
        let decision = resolve(
            PermissionMode::Ask,
            "bash",
            &json!({"command": "curl https://example.com"}),
            &grants,
        );
        assert_eq!(forced_kind(&decision), Some(ForcedPromptKind::Network));
        // Loopback is not egress: the grant applies.
        assert_eq!(
            resolve(
                PermissionMode::Ask,
                "bash",
                &json!({"command": "curl http://localhost:8080/health"}),
                &grants
            ),
            GateDecision::Allow
        );
    }

    #[test]
    fn private_url_fetch_prompts_in_ask_only() {
        let private = json!({"url": "http://192.168.1.1/"});
        assert_eq!(
            forced_kind(&resolve_simple(
                PermissionMode::Ask,
                "web_fetch",
                private.clone()
            )),
            Some(ForcedPromptKind::PrivateNetwork)
        );
        assert_eq!(
            resolve_simple(PermissionMode::Yolo, "web_fetch", private),
            GateDecision::Allow
        );
        assert_eq!(
            resolve_simple(
                PermissionMode::Ask,
                "web_fetch",
                json!({"url": "https://example.com/"})
            ),
            GateDecision::Allow
        );
    }

    #[test]
    fn kill_shell_session_grant_silences_later_calls() {
        let mut grants = SessionGrants::default();
        assert!(matches!(
            resolve(PermissionMode::Ask, "kill_shell", &json!({}), &grants),
            GateDecision::Prompt { forced: None, .. }
        ));
        grants.grant_tool("kill_shell");
        assert_eq!(
            resolve(PermissionMode::Ask, "kill_shell", &json!({}), &grants),
            GateDecision::Allow
        );
        // A tool grant never covers an unknown exec-tier tool.
        grants.grant_tool("custom_exec");
        assert!(matches!(
            resolve(PermissionMode::Ask, "custom_exec", &json!({}), &grants),
            GateDecision::Prompt { .. }
        ));
    }

    #[test]
    fn subcommand_grant_does_not_cover_other_subcommands() {
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["git status"]);
        let ask = |command: &str| {
            resolve(
                PermissionMode::Ask,
                "bash",
                &json!({ "command": command }),
                &grants,
            )
        };
        assert_eq!(ask("git status --short"), GateDecision::Allow);
        assert!(matches!(
            ask("git push --force"),
            GateDecision::Prompt { .. }
        ));
        assert!(matches!(
            ask("git -C repo push --force"),
            GateDecision::Prompt { .. }
        ));
    }

    #[test]
    fn unkeyable_bash_offers_no_grant_and_is_never_covered() {
        assert_eq!(
            grant_scope("bash", &json!({"command": "git --weird push"})),
            None
        );
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["git push"]);
        assert!(matches!(
            resolve(
                PermissionMode::Ask,
                "bash",
                &json!({"command": "git --weird push"}),
                &grants
            ),
            GateDecision::Prompt { .. }
        ));
    }

    #[test]
    fn remembered_directory_grant_covers_direct_entries_only() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("perm-dir-grant-{}", std::process::id()));
        let file = root.join("a.txt");
        let scope = remember_scope("write", &json!({"path": file.to_str().unwrap()})).unwrap();
        assert_eq!(
            scope,
            GrantScope::WriteDirectory {
                tool: "write".to_string(),
                dir: root.clone()
            }
        );
        let mut grants = SessionGrants::default();
        grants.grant_scope(&scope);
        let call = |tool: &str, path: &std::path::Path| {
            resolve(
                PermissionMode::Ask,
                tool,
                &json!({ "path": path.to_str().unwrap() }),
                &grants,
            )
        };
        // Sibling file, either file tool: covered.
        assert_eq!(call("edit", &root.join("b.txt")), GateDecision::Allow);
        // Nested directory and parent directory: not covered.
        assert!(matches!(
            call("write", &root.join("nested").join("c.txt")),
            GateDecision::Prompt { .. }
        ));
        assert!(matches!(
            call("write", &root.with_file_name("outside.txt")),
            GateDecision::Prompt { .. }
        ));
    }

    #[test]
    fn read_file_grant_never_covers_write() {
        let path = std::env::temp_dir().canonicalize().unwrap().join("r.txt");
        let mut grants = SessionGrants::default();
        grants.extend_keys([grant_keys::file_key(FileTier::Read, &path)]);
        assert!(matches!(
            resolve(
                PermissionMode::Ask,
                "write",
                &json!({ "path": path.to_str().unwrap() }),
                &grants
            ),
            GateDecision::Prompt { .. }
        ));
        assert!(grants.file_granted(FileTier::Read, &path));
    }

    #[test]
    fn remember_scope_matches_session_scope_except_file_tools() {
        assert_eq!(
            remember_scope("bash", &json!({"command": "cargo test"})),
            Some(GrantScope::BashCommands(vec!["cargo test".to_string()]))
        );
        assert_eq!(
            remember_scope("kill_shell", &json!({})),
            Some(GrantScope::Tool("kill_shell".to_string()))
        );
        // Relative paths are never remembered.
        assert_eq!(remember_scope("edit", &json!({"path": "src/a.rs"})), None);
    }

    fn auto_ctx(workspace: &std::path::Path, sandbox: bool, network: bool) -> GateContext {
        GateContext {
            workspace: Some(workspace.to_path_buf()),
            sandbox_available: sandbox,
            network_allowed: network,
        }
    }

    fn auto(ctx: &GateContext, tool: &str, args: Value) -> GateDecision {
        resolve_in(
            PermissionMode::Auto,
            ctx,
            tool,
            &args,
            &SessionGrants::default(),
        )
    }

    fn temp_workspace() -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("auto-ws-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn auto_runs_typical_work_without_prompts() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        for command in ["ls -la", "cargo build", "npm test", "git commit -m wip"] {
            assert_eq!(
                auto(&ctx, "bash", json!({ "command": command })),
                GateDecision::Allow,
                "{command}"
            );
        }
        let inside = ws.join("src").join("main.rs");
        for tool in ["edit", "write"] {
            assert_eq!(
                auto(&ctx, tool, json!({ "path": inside.to_str().unwrap() })),
                GateDecision::Allow
            );
        }
        assert_eq!(
            auto(&ctx, "read", json!({ "path": "/etc/hosts" })),
            GateDecision::Allow
        );
    }

    #[test]
    fn auto_prompts_out_of_workspace_writes_mcp_kill_shell_and_unknown_tools() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        let outside = ws.with_file_name("outside.txt");
        let traversal = format!("{}/../escape.txt", ws.display());
        for (tool, args) in [
            ("write", json!({ "path": outside.to_str().unwrap() })),
            ("edit", json!({ "path": traversal })),
            ("write", json!({ "path": "relative.txt" })),
            ("mcp_tool_use", json!({ "server": "s", "tool": "t" })),
            ("kill_shell", json!({})),
            ("custom_exec", json!({})),
        ] {
            assert!(
                matches!(auto(&ctx, tool, args.clone()), GateDecision::Prompt { .. }),
                "{tool} {args}"
            );
        }
    }

    #[test]
    fn auto_keeps_forced_prompts() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        let kind = |decision: GateDecision| forced_kind(&decision);
        assert_eq!(
            kind(auto(&ctx, "bash", json!({ "command": "echo $(whoami)" }))),
            Some(ForcedPromptKind::Security)
        );
        assert_eq!(
            kind(auto(
                &ctx,
                "bash",
                json!({ "command": format!("echo {}", "a".repeat(12_000)) })
            )),
            Some(ForcedPromptKind::Unparsed)
        );
        assert_eq!(
            kind(auto(
                &ctx,
                "bash",
                json!({ "command": "curl https://example.com" })
            )),
            Some(ForcedPromptKind::Network)
        );
        assert_eq!(
            kind(auto(
                &ctx,
                "web_fetch",
                json!({ "url": "http://10.0.0.1/" })
            )),
            Some(ForcedPromptKind::PrivateNetwork)
        );
        assert!(matches!(
            auto(&ctx, "bash", json!({ "command": "rm -rf /" })),
            GateDecision::Deny { .. }
        ));
    }

    #[test]
    fn auto_network_grant_silences_egress_only_with_a_sandbox() {
        let ws = temp_workspace();
        let curl = json!({ "command": "curl https://example.com" });
        assert_eq!(
            auto(&auto_ctx(&ws, true, true), "bash", curl.clone()),
            GateDecision::Allow
        );
        assert!(matches!(
            auto(&auto_ctx(&ws, false, true), "bash", curl),
            GateDecision::Prompt { .. }
        ));
    }

    #[test]
    fn auto_without_sandbox_prompts_bash_per_command() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, false, false);
        assert_eq!(
            auto(&ctx, "bash", json!({ "command": "ls" })),
            GateDecision::prompt("ls")
        );
        // Session grants still apply, exactly as in ask.
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["ls"]);
        assert_eq!(
            resolve_in(
                PermissionMode::Auto,
                &ctx,
                "bash",
                &json!({ "command": "ls" }),
                &grants
            ),
            GateDecision::Allow
        );
    }

    #[test]
    fn ask_ignores_network_grant_and_workspace() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, true);
        assert!(matches!(
            resolve_in(
                PermissionMode::Ask,
                &ctx,
                "bash",
                &json!({ "command": "curl https://example.com" }),
                &SessionGrants::default()
            ),
            GateDecision::Prompt { .. }
        ));
        assert!(matches!(
            resolve_in(
                PermissionMode::Ask,
                &ctx,
                "write",
                &json!({ "path": ws.join("a.txt").to_str().unwrap() }),
                &SessionGrants::default()
            ),
            GateDecision::Prompt { .. }
        ));
    }
}
