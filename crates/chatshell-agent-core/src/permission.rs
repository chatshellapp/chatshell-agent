//! Tool-permission decision layer for the approval gate.
//!
//! Pure logic: no I/O, no UI, no clock. The host supplies the permission
//! mode and the session grants and receives a decision (allow / deny /
//! prompt / classify). Precedence: hard deny (bash blocklist, write
//! blocklist) -> sensitive writes (denied in `yolo`) -> `yolo` allow ->
//! deletion of existing paths outside the workspace ->
//! forced prompt (classifier warning, unparsed command, network egress,
//! private-network fetch, sensitive read) -> session-grant lookup ->
//! mode/tier matrix -> prompt.
//!
//! Bash is jailed in `ask` and `auto` wherever a sandbox backend exists
//! (ADR 08): in-profile commands run silently in both, and the residual —
//! forced sources, escapes, MCP, `kill_shell`, unknown tools, file-tool
//! writes in `ask` — prompts in `ask` and is judged in `auto`.
//!
//! In [`PermissionMode::Auto`] every prompt becomes
//! [`GateDecision::Classify`]: the host routes it to the auto-mode judge
//! with deterministic evidence (see [`crate::judgment`]). Human-minted
//! grants still short-circuit where they apply in `ask`, before any
//! judgment; forced sources ignore them in every mode.
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
use crate::judgment::{Evidence, JudgmentRequest, JudgmentSource};
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
    /// The human decides the residual. Where a sandbox backend exists,
    /// bash runs jailed and in-profile commands run silently; escapes,
    /// forced sources, file-tool writes, MCP calls, `kill_shell`, and
    /// unknown tools prompt. Without a backend, every bash command
    /// prompts. Session grants cache approvals.
    Ask,
    /// Bash runs inside the OS sandbox without prompting, and file writes
    /// inside the workspace run silently. Every call `ask` would prompt
    /// for (escapes, out-of-workspace writes, MCP, `kill_shell`, unknown
    /// tools, classifier warnings, egress, private-network fetches,
    /// sensitive reads and writes, and bash without a sandbox) resolves to
    /// [`GateDecision::Classify`] for the auto-mode judge.
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
    /// The conversation scratch directory (a jail writable root, and the
    /// bash working directory when no workspace is selected).
    pub scratch_dir: Option<PathBuf>,
    /// The bash shell has left its launch directory (reported by the
    /// shell after each command): relative write targets are unprovable.
    pub cwd_drifted: bool,
}

impl GateContext {
    /// The directory bash write and deletion analysis treats as the
    /// workspace: the workspace, else the scratch directory (bash runs
    /// there when no workspace is selected).
    pub fn analysis_root(&self) -> Option<&Path> {
        self.workspace.as_deref().or(self.scratch_dir.as_deref())
    }

    /// The working directory of a bash call with `workdir`, and whether it
    /// is provable (ADR 09 §1 condition 6).
    pub fn bash_cwd(&self, workdir: Option<&str>) -> bash_security::BashCwd {
        bash_security::BashCwd::new(workdir, self.analysis_root(), self.cwd_drifted)
    }

    /// The jail's writable roots as the analysis sees them: the analysis
    /// root first, then the scratch directory.
    pub fn jail_roots(&self) -> Vec<PathBuf> {
        self.analysis_root()
            .iter()
            .map(|root| root.to_path_buf())
            .chain(self.scratch_dir.iter().cloned())
            .collect()
    }
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
    /// A write hits a write ask-tier `path_policy` group (credential
    /// store, persistence location, agent settings, host data). `yolo`,
    /// which cannot ask, denies these instead.
    SensitiveWrite,
    /// A bash command deletes a path that already exists outside the
    /// workspace. `yolo` (full access) runs these without asking.
    OutsideDeletion,
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
    /// `auto` only: route the call to the auto-mode judge. Replaces every
    /// `Prompt` the same call would get in `ask` (one call, at most one
    /// judgment).
    Classify(Box<JudgmentRequest>),
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

/// Classifier signal that forces a bash prompt in prompting modes: a
/// classifier warning first (the most serious), then an AST layer that
/// could not run (fail closed: its validators were skipped), then network
/// egress (skipped when the session already holds a network grant).
///
/// In a jail, contained redirect and heredoc warnings are absorbed (ADR 09
/// §1): the arm keeps its position, so an absorbed command still passes
/// the unparsed and egress checks, and the out-of-roots write check after
/// this function catches a resolvable target outside the roots.
fn bash_forced_prompt(
    command: &str,
    classification: &bash_security::CommandClassification,
    network_allowed: bool,
    scope: &bash_security::AbsorbScope,
) -> Option<(ForcedPromptKind, String)> {
    if let Some(message) = bash_security::effective_warning(command, classification, scope) {
        return Some((ForcedPromptKind::Security, message));
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
///
/// Relative paths are joined onto the workspace first, as the file tools
/// resolve them. The verdict depends on where the path lands, not on how
/// it is spelled: `..` components are normalized and symlinks resolved
/// through the deepest existing ancestor, so `../x` and a link pointing
/// out of the workspace land outside while `src/../x` stays inside.
pub fn inside_workspace(path: &str, workspace: &Path) -> bool {
    let joined = if Path::new(path).is_relative() {
        workspace.join(path)
    } else {
        PathBuf::from(path)
    };
    let Some(target) = joined.to_str().and_then(grant_keys::grant_path) else {
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

/// Write targets of a file-tool path: the path itself, and for a
/// relative path also its workspace-joined form (the tools resolve
/// relative paths against the working directory).
fn write_candidates(path: &str, workspace: Option<&Path>) -> Vec<PathBuf> {
    let path = Path::new(path);
    let mut candidates = vec![path.to_path_buf()];
    if path.is_relative()
        && let Some(workspace) = workspace
    {
        candidates.push(workspace.join(path));
    }
    candidates
}

/// Message for a sensitive write: shown on prompts and in `yolo` denials.
fn sensitive_write_detail(block: &path_policy::WriteBlock) -> String {
    format!("Writes to a {} ({})", block.label, block.matched)
}

/// Reason fed back to the model when `yolo` denies a sensitive write.
fn sensitive_write_denial(block: &path_policy::WriteBlock) -> String {
    format!(
        "Write blocked: {} ({}). This location is protected in yolo mode; \
         ask the user to make this change or switch to a mode that can ask.",
        block.label, block.matched
    )
}

/// Builds the non-allow outcomes for one call: a prompt in `ask`, a
/// judgment request in `auto`.
struct Escalation<'a> {
    mode: PermissionMode,
    ctx: &'a GateContext,
    tool_name: &'a str,
    args: &'a Value,
}

impl Escalation<'_> {
    fn judge(
        &self,
        subject: String,
        source: JudgmentSource,
        forced: Option<ForcedPrompt>,
    ) -> GateDecision {
        GateDecision::Classify(Box::new(JudgmentRequest {
            subject,
            source,
            forced,
            evidence: Evidence::collect(self.tool_name, self.args, self.ctx),
        }))
    }

    /// A forced source: past grants in every mode.
    fn forced(&self, subject: &str, kind: ForcedPromptKind, detail: String) -> GateDecision {
        if self.mode == PermissionMode::Auto {
            self.judge(
                subject.to_string(),
                kind.into(),
                Some(ForcedPrompt { kind, detail }),
            )
        } else {
            GateDecision::forced(subject, kind, detail)
        }
    }

    /// A mode/tier matrix arm (grants were checked already).
    fn matrix(&self, subject: String, source: JudgmentSource) -> GateDecision {
        if self.mode == PermissionMode::Auto {
            self.judge(subject, source, None)
        } else {
            GateDecision::prompt(subject)
        }
    }
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
    // Bash is jailed in both prompting modes wherever a backend exists.
    let sandboxed = mode != PermissionMode::Yolo && ctx.sandbox_available;
    let escalation = Escalation {
        mode,
        ctx,
        tool_name,
        args,
    };
    let workspace = ctx.workspace.as_deref();
    // Bash: hard-deny pre-checks, the sensitive-write floor, then the
    // mode, forced sources, grants. The authoritative checks live inside
    // the bash tool; these pre-checks deny before a prompt or judgment
    // for a command the tool would refuse anyway. The classification is
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
        // The call's cwd: `workdir` (joined onto the launch directory when
        // relative) -> workspace -> scratch. Without a workspace the scratch
        // directory is the analysis root, so the write and deletion guards
        // still have a boundary (ADR 09 §2).
        let workdir = args.get("workdir").and_then(Value::as_str);
        let bash_cwd = ctx.bash_cwd(workdir);
        let cwd = bash_cwd.as_str();
        let analysis_root = ctx.analysis_root();
        // Deleting existing paths outside the workspace escalates in the
        // prompting modes; `yolo` is full access, so only the catastrophic
        // blocklist above stops a deletion there.
        let scratch: Vec<PathBuf> = ctx.scratch_dir.iter().cloned().collect();
        if mode != PermissionMode::Yolo
            && let Some(target) =
                bash_security::outside_deletion_target(command, analysis_root, cwd, &scratch)
        {
            return escalation.forced(
                command,
                ForcedPromptKind::OutsideDeletion,
                format!(
                    "Permanently deletes an existing path outside the workspace: {}",
                    target.display()
                ),
            );
        }
        if let Some(block) = bash_security::write_ask_target(command, cwd) {
            if mode == PermissionMode::Yolo {
                return GateDecision::Deny {
                    reason: sensitive_write_denial(&block),
                };
            }
            return escalation.forced(
                command,
                ForcedPromptKind::SensitiveWrite,
                sensitive_write_detail(&block),
            );
        }
        if mode == PermissionMode::Yolo {
            return GateDecision::Allow;
        }
        // The network grant only means something when a sandbox enforces
        // the alternative; without one, egress always escalates.
        let network_allowed = sandboxed && ctx.network_allowed;
        let scope = bash_security::AbsorbScope {
            sandboxed,
            roots: ctx.jail_roots(),
            cwd: bash_cwd.clone(),
        };
        if let Some((kind, detail)) =
            bash_forced_prompt(command, &classification, network_allowed, &scope)
        {
            return escalation.forced(command, kind, detail);
        }
        // Jailed bash: a write the jail would refuse (outside the
        // workspace, scratch, and temp channel) escalates before execution;
        // an approval widens the jail for that one execution.
        if sandboxed
            && !bash_security::outside_write_targets(command, analysis_root, cwd, &scratch)
                .is_empty()
        {
            return escalation.matrix(command.to_string(), JudgmentSource::OutOfWorkspaceWrite);
        }
        // In a jail the OS sandbox absorbs everything else; escapes the
        // static pass cannot see surface as sandbox denials after execution.
        if sandboxed || grants.bash_command_granted(command) {
            return GateDecision::Allow;
        }
        return escalation.matrix(command.to_string(), JudgmentSource::UnsandboxedBash);
    }

    // File writes: the write blocklist denies in every mode (the tool
    // would refuse), and the sensitive-write floor escalates (denies in
    // `yolo`).
    if matches!(tool_name, "edit" | "write")
        && let Some(path) = args.get("path").and_then(Value::as_str)
    {
        let candidates = write_candidates(path, workspace);
        for candidate in &candidates {
            if let Err(reason) = path_policy::check_write(candidate, None) {
                return GateDecision::Deny { reason };
            }
        }
        if let Some(block) = candidates
            .iter()
            .find_map(|candidate| path_policy::classify_write(candidate))
        {
            if mode == PermissionMode::Yolo {
                return GateDecision::Deny {
                    reason: sensitive_write_denial(&block),
                };
            }
            return escalation.forced(
                path,
                ForcedPromptKind::SensitiveWrite,
                sensitive_write_detail(&block),
            );
        }
    }

    if mode == PermissionMode::Yolo {
        return GateDecision::Allow;
    }

    // Read tier runs silently, except fetches into private networks and
    // ask-tier reads.
    let tier = PermissionTier::for_tool(tool_name);
    if matches!(tier, PermissionTier::Read) {
        if tool_name == "web_fetch"
            && let Some(url) = args.get("url").and_then(Value::as_str)
            && net_policy::is_private_url(url)
        {
            return escalation.forced(
                url,
                ForcedPromptKind::PrivateNetwork,
                format!("Fetch targets a private or local network address: {url}"),
            );
        }
        if matches!(tool_name, "read" | "grep" | "glob")
            && let Some(path) = args.get("path").and_then(Value::as_str)
            && let Some(block) = ask_tier_read(path, workspace)
        {
            return escalation.forced(
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
        && let (Some(path), Some(workspace)) = (args.get("path").and_then(Value::as_str), workspace)
        && inside_workspace(path, workspace)
    {
        return GateDecision::Allow;
    }

    // Human-minted grants: the same fast paths in `ask` and `auto`.
    let subject = subject_for(tool_name, args);
    if tool_name == "mcp_tool_use" {
        // MCP grants are per server/tool pair; a call missing either arg
        // matches no grant (fail-safe).
        if let (Some(server), Some(tool)) = (
            args.get("server").and_then(Value::as_str),
            args.get("tool").and_then(Value::as_str),
        ) && grants.server_tool_granted(server, tool)
        {
            return GateDecision::Allow;
        }
        return escalation.matrix(subject, JudgmentSource::McpCall);
    }
    if grant_scope(tool_name, args).is_some() && grants.tool_granted(tool_name) {
        return GateDecision::Allow;
    }
    if matches!(tool_name, "edit" | "write")
        && let Some(path) = args.get("path").and_then(Value::as_str)
        && let Some(resolved) = grant_keys::grant_path(path)
        && grants.file_granted(FileTier::Write, &resolved)
    {
        return GateDecision::Allow;
    }
    let source = match tool_name {
        "edit" | "write" => JudgmentSource::OutOfWorkspaceWrite,
        "kill_shell" => JudgmentSource::KillShell,
        _ => JudgmentSource::UnknownTool,
    };
    escalation.matrix(subject, source)
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

    /// The forced kind of a prompt, or of the v2 prompt a judgment
    /// request carries (the interim renderer shows exactly that prompt).
    fn forced_kind(decision: &GateDecision) -> Option<ForcedPromptKind> {
        match decision {
            GateDecision::Prompt {
                forced: Some(forced),
                ..
            } => Some(forced.kind),
            GateDecision::Classify(request) => request.forced.as_ref().map(|f| f.kind),
            _ => None,
        }
    }

    /// The judgment source of a `Classify` decision.
    fn judged(decision: &GateDecision) -> Option<JudgmentSource> {
        match decision {
            GateDecision::Classify(request) => Some(request.source),
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
            scratch_dir: None,
            cwd_drifted: false,
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
        // Relative paths land where the tools resolve them: inside the
        // workspace, however they are spelled.
        for path in ["notes.md", "./notes.md", "src/main.rs", "src/../notes.md"] {
            for tool in ["edit", "write"] {
                assert_eq!(
                    auto(&ctx, tool, json!({ "path": path })),
                    GateDecision::Allow,
                    "{tool} {path}"
                );
            }
        }
        assert_eq!(
            auto(&ctx, "read", json!({ "path": "/etc/hosts" })),
            GateDecision::Allow
        );
    }

    #[cfg(unix)]
    #[test]
    fn auto_judges_relative_writes_through_a_symlink_out_of_the_workspace() {
        let base = tempfile::tempdir().unwrap();
        let ws = base.path().join("ws");
        let outside = base.path().join("outside");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, ws.join("link")).unwrap();
        let ctx = auto_ctx(&ws, true, false);
        assert_eq!(
            judged(&auto(&ctx, "write", json!({ "path": "link/x.txt" }))),
            Some(JudgmentSource::OutOfWorkspaceWrite)
        );
    }

    #[test]
    fn relative_writes_without_a_workspace_are_judged() {
        let ctx = GateContext {
            workspace: None,
            sandbox_available: true,
            network_allowed: false,
            scratch_dir: None,
            cwd_drifted: false,
        };
        assert_eq!(
            judged(&auto(&ctx, "write", json!({ "path": "notes.md" }))),
            Some(JudgmentSource::OutOfWorkspaceWrite)
        );
    }

    #[test]
    fn auto_judges_out_of_workspace_writes_mcp_kill_shell_and_unknown_tools() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        let outside = ws.with_file_name("outside.txt");
        let traversal = format!("{}/../escape.txt", ws.display());
        for (tool, args, source) in [
            (
                "write",
                json!({ "path": outside.to_str().unwrap() }),
                JudgmentSource::OutOfWorkspaceWrite,
            ),
            (
                "edit",
                json!({ "path": traversal }),
                JudgmentSource::OutOfWorkspaceWrite,
            ),
            (
                "write",
                json!({ "path": "../escape.txt" }),
                JudgmentSource::OutOfWorkspaceWrite,
            ),
            (
                "edit",
                json!({ "path": "src/../../escape.txt" }),
                JudgmentSource::OutOfWorkspaceWrite,
            ),
            (
                "mcp_tool_use",
                json!({ "server": "s", "tool": "t" }),
                JudgmentSource::McpCall,
            ),
            ("kill_shell", json!({}), JudgmentSource::KillShell),
            ("custom_exec", json!({}), JudgmentSource::UnknownTool),
        ] {
            let decision = auto(&ctx, tool, args.clone());
            assert_eq!(judged(&decision), Some(source), "{tool} {args}");
            // Matrix arms carry no forced prompt: their v2 prompt offered
            // grants, and the interim renderer keeps offering them.
            assert_eq!(forced_kind(&decision), None, "{tool} {args}");
        }
    }

    #[test]
    fn auto_judges_forced_sources() {
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
        assert_eq!(
            judged(&auto(&auto_ctx(&ws, false, true), "bash", curl)),
            Some(JudgmentSource::Network)
        );
    }

    #[test]
    fn auto_without_sandbox_judges_bash_per_command() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, false, false);
        let decision = auto(&ctx, "bash", json!({ "command": "ls" }));
        assert_eq!(judged(&decision), Some(JudgmentSource::UnsandboxedBash));
        assert_eq!(forced_kind(&decision), None);
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
    fn jailed_ask_honors_the_network_flag_but_still_prompts_file_writes() {
        let ws = temp_workspace();
        // ADR 08 SA-2: in jailed `ask` the session flag silences egress...
        let jailed = auto_ctx(&ws, true, true);
        assert_eq!(
            resolve_in(
                PermissionMode::Ask,
                &jailed,
                "bash",
                &json!({ "command": "curl https://example.com" }),
                &SessionGrants::default()
            ),
            GateDecision::Allow
        );
        // ...without a backend the flag is meaningless and egress prompts.
        let bare = auto_ctx(&ws, false, true);
        assert!(matches!(
            resolve_in(
                PermissionMode::Ask,
                &bare,
                "bash",
                &json!({ "command": "curl https://example.com" }),
                &SessionGrants::default()
            ),
            GateDecision::Prompt { .. }
        ));
        // File tools are not jailed and keep prompting in `ask` (SA-7).
        assert!(matches!(
            resolve_in(
                PermissionMode::Ask,
                &jailed,
                "write",
                &json!({ "path": ws.join("a.txt").to_str().unwrap() }),
                &SessionGrants::default()
            ),
            GateDecision::Prompt { .. }
        ));
    }

    // ---- permission v3, Phase A -------------------------------------------

    fn every_mode() -> [PermissionMode; 3] {
        [
            PermissionMode::Ask,
            PermissionMode::Auto,
            PermissionMode::Yolo,
        ]
    }

    #[test]
    fn auto_routes_all_twelve_sources_to_a_judgment_with_evidence() {
        let ws = temp_workspace();
        let keep = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let sandboxed = auto_ctx(&ws, true, false);
        let unsandboxed = auto_ctx(&ws, false, false);
        let home = dirs::home_dir().unwrap();
        let zshrc = home.join(".zshrc").display().to_string();
        let outside = ws.with_file_name("outside-v3.txt");
        let cases: Vec<(&GateContext, &str, Value, JudgmentSource)> = vec![
            (
                &sandboxed,
                "bash",
                json!({ "command": "echo $(whoami)" }),
                JudgmentSource::Security,
            ),
            (
                &sandboxed,
                "bash",
                json!({ "command": format!("echo {}", "a".repeat(12_000)) }),
                JudgmentSource::Unparsed,
            ),
            (
                &sandboxed,
                "bash",
                json!({ "command": "curl https://example.com" }),
                JudgmentSource::Network,
            ),
            (
                &sandboxed,
                "web_fetch",
                json!({ "url": "http://10.0.0.1/" }),
                JudgmentSource::PrivateNetwork,
            ),
            (
                &sandboxed,
                "read",
                json!({ "path": zshrc }),
                JudgmentSource::SensitiveRead,
            ),
            (
                &sandboxed,
                "write",
                json!({ "path": outside.to_str().unwrap() }),
                JudgmentSource::OutOfWorkspaceWrite,
            ),
            (
                &sandboxed,
                "mcp_tool_use",
                json!({ "server": "s", "tool": "t" }),
                JudgmentSource::McpCall,
            ),
            (
                &sandboxed,
                "kill_shell",
                json!({}),
                JudgmentSource::KillShell,
            ),
            (
                &sandboxed,
                "custom_exec",
                json!({}),
                JudgmentSource::UnknownTool,
            ),
            (
                &unsandboxed,
                "bash",
                json!({ "command": "ls -la" }),
                JudgmentSource::UnsandboxedBash,
            ),
            (
                &sandboxed,
                "write",
                json!({ "path": home.join(".claude/settings.json").to_str().unwrap() }),
                JudgmentSource::SensitiveWrite,
            ),
            (
                &sandboxed,
                "bash",
                json!({ "command": format!("rm -rf {}", keep.path().display()) }),
                JudgmentSource::OutsideDeletion,
            ),
        ];
        assert_eq!(cases.len(), 12);
        for (ctx, tool, args, source) in cases {
            let GateDecision::Classify(request) = auto(ctx, tool, args.clone()) else {
                panic!("{tool} {args} should be judged");
            };
            assert_eq!(request.source, source, "{tool}");
            assert_eq!(request.evidence.tool, tool);
            assert_eq!(request.evidence.jail_applies, tool == "bash");
            // Forced sources keep the v2 prompt for the interim renderer.
            let forced = matches!(
                source,
                JudgmentSource::Security
                    | JudgmentSource::Unparsed
                    | JudgmentSource::Network
                    | JudgmentSource::PrivateNetwork
                    | JudgmentSource::SensitiveRead
                    | JudgmentSource::SensitiveWrite
                    | JudgmentSource::OutsideDeletion
            );
            assert_eq!(request.forced.is_some(), forced, "{tool}");
            if let Some(forced) = &request.forced {
                assert_eq!(JudgmentSource::from(forced.kind), source);
            }
        }
    }

    #[test]
    fn ask_and_yolo_never_classify() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        for mode in [PermissionMode::Ask, PermissionMode::Yolo] {
            for (tool, args) in [
                ("bash", json!({ "command": "curl https://example.com" })),
                ("bash", json!({ "command": "ls" })),
                ("mcp_tool_use", json!({ "server": "s", "tool": "t" })),
                ("kill_shell", json!({})),
                ("web_fetch", json!({ "url": "http://10.0.0.1/" })),
            ] {
                let decision = resolve_in(mode, &ctx, tool, &args, &SessionGrants::default());
                assert_eq!(judged(&decision), None, "{mode:?} {tool}");
            }
        }
    }

    #[test]
    fn one_egress_command_yields_one_network_judgment() {
        let ws = temp_workspace();
        // Without a sandbox the command is also plain unsandboxed bash;
        // the forced source wins and is the only judgment.
        for sandbox in [true, false] {
            let decision = auto(
                &auto_ctx(&ws, sandbox, false),
                "bash",
                json!({ "command": "curl https://example.com" }),
            );
            assert_eq!(judged(&decision), Some(JudgmentSource::Network));
        }
    }

    #[test]
    fn grants_short_circuit_matrix_arms_but_not_forced_sources_in_auto() {
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, false, false);
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["echo", "ls"]);
        grants.grant_server_tool("s", "t");
        grants.grant_tool("kill_shell");
        let run =
            |tool: &str, args: Value| resolve_in(PermissionMode::Auto, &ctx, tool, &args, &grants);
        assert_eq!(run("bash", json!({ "command": "ls" })), GateDecision::Allow);
        assert_eq!(
            run("mcp_tool_use", json!({ "server": "s", "tool": "t" })),
            GateDecision::Allow
        );
        assert_eq!(run("kill_shell", json!({})), GateDecision::Allow);
        // A Warn verdict is judged even though `echo` is granted.
        assert_eq!(
            judged(&run("bash", json!({ "command": "echo $(whoami)" }))),
            Some(JudgmentSource::Security)
        );
    }

    #[test]
    fn write_blocklist_denies_before_any_prompt_or_judgment() {
        let home = dirs::home_dir().unwrap();
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        let mut grants = SessionGrants::default();
        grants.grant_tool("write");
        for path in [
            home.join(".zshrc").display().to_string(),
            "/etc/hosts".to_string(),
            ws.join(".git/hooks/pre-commit").display().to_string(),
            ".git/config".to_string(),
        ] {
            for mode in every_mode() {
                for tool in ["write", "edit"] {
                    assert!(
                        matches!(
                            resolve_in(mode, &ctx, tool, &json!({ "path": path }), &grants),
                            GateDecision::Deny { .. }
                        ),
                        "{mode:?} {tool} {path}"
                    );
                }
            }
        }
    }

    #[test]
    fn settings_and_grant_store_writes_escalate_and_yolo_denies() {
        let data = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("perm-v3-app-data-{}", std::process::id()));
        path_policy::register_private_dir(&data);
        let home = dirs::home_dir().unwrap();
        let ws = temp_workspace();
        let ctx = auto_ctx(&ws, true, false);
        // Even a whole-tool grant cannot silence the floor.
        let mut grants = SessionGrants::default();
        grants.grant_tool("write");
        for path in [
            data.join(crate::permission_store::STORE_FILE_NAME),
            data.join("settings.json"),
            home.join(".claude/settings.json"),
            home.join(".aws/credentials"),
        ] {
            let args = json!({ "path": path.to_str().unwrap() });
            assert_eq!(
                forced_kind(&resolve_in(
                    PermissionMode::Ask,
                    &ctx,
                    "write",
                    &args,
                    &grants
                )),
                Some(ForcedPromptKind::SensitiveWrite),
                "{}",
                path.display()
            );
            assert_eq!(
                judged(&resolve_in(
                    PermissionMode::Auto,
                    &ctx,
                    "write",
                    &args,
                    &grants
                )),
                Some(JudgmentSource::SensitiveWrite)
            );
            assert!(matches!(
                resolve_in(PermissionMode::Yolo, &ctx, "write", &args, &grants),
                GateDecision::Deny { .. }
            ));
        }
    }

    #[test]
    fn bash_writes_into_sensitive_locations_escalate_and_yolo_denies() {
        let home = dirs::home_dir().unwrap();
        let command = format!(
            "cp hook.json {}",
            home.join(".claude/settings.json").display()
        );
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["cp"]);
        let args = json!({ "command": command });
        assert_eq!(
            forced_kind(&resolve(PermissionMode::Ask, "bash", &args, &grants)),
            Some(ForcedPromptKind::SensitiveWrite)
        );
        assert!(matches!(
            resolve(PermissionMode::Yolo, "bash", &args, &grants),
            GateDecision::Deny { .. }
        ));
    }

    #[test]
    fn outside_deletions_escalate_past_grants_and_yolo_allows() {
        let root = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let ws = root.path().join("ws");
        let keep = root.path().join("keep");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&keep).unwrap();
        let ctx = auto_ctx(&ws, true, false);
        let args = json!({ "command": format!("rm -rf {}", keep.display()) });
        // A granted `rm` cannot silence it.
        let mut grants = SessionGrants::default();
        grants.grant_bash_commands(&["rm"]);
        let run = |mode| resolve_in(mode, &ctx, "bash", &args, &grants);
        assert_eq!(
            forced_kind(&run(PermissionMode::Ask)),
            Some(ForcedPromptKind::OutsideDeletion)
        );
        assert_eq!(
            judged(&run(PermissionMode::Auto)),
            Some(JudgmentSource::OutsideDeletion)
        );
        // Yolo is full access: the deletion runs.
        assert_eq!(run(PermissionMode::Yolo), GateDecision::Allow);
        // The catastrophic blocklist still holds there.
        assert!(matches!(
            resolve_in(
                PermissionMode::Yolo,
                &ctx,
                "bash",
                &json!({ "command": "rm -rf /" }),
                &grants
            ),
            GateDecision::Deny { .. }
        ));
        // Inside the workspace: ordinary auto bash.
        assert_eq!(
            auto(&ctx, "bash", json!({ "command": "rm -rf build" })),
            GateDecision::Allow
        );
    }

    #[test]
    fn sandboxed_auto_judges_bash_writes_outside_the_jail() {
        let root = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let ws = root.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let ctx = auto_ctx(&ws, true, false);
        let outside = root.path().join("notes.txt");
        let decision = auto(
            &ctx,
            "bash",
            json!({ "command": format!("touch {}", outside.display()) }),
        );
        assert_eq!(judged(&decision), Some(JudgmentSource::OutOfWorkspaceWrite));
        let GateDecision::Classify(request) = decision else {
            unreachable!()
        };
        assert_eq!(request.evidence.bash.unwrap().writes_outside.len(), 1);
        // Inside the workspace: the jail absorbs it.
        assert_eq!(
            auto(&ctx, "bash", json!({ "command": "touch out.txt" })),
            GateDecision::Allow
        );
    }

    #[test]
    fn evidence_jail_reflects_the_context() {
        let ws = temp_workspace();
        let ctx = GateContext {
            scratch_dir: Some(ws.join("scratch")),
            ..auto_ctx(&ws, true, false)
        };
        let GateDecision::Classify(request) = auto(
            &ctx,
            "bash",
            json!({ "command": "curl https://example.com" }),
        ) else {
            panic!("egress should be judged");
        };
        assert_eq!(
            request.evidence.jail,
            crate::judgment::JailProfile::Present {
                network_enabled: false,
                writable_roots: vec![
                    ws.display().to_string(),
                    ws.join("scratch").display().to_string()
                ],
                temp_channel: crate::sandbox::describe_temp_channel(
                    &crate::sandbox::SandboxPolicy::for_bash(Some(&ws), &ws.join("scratch"), false)
                ),
            }
        );
    }

    // ---- ADR 08 (jailed ask) and ADR 09 (fast-path alignment) ----------

    /// The resolver outcome, reduced to what the matrix asserts.
    fn outcome(decision: &GateDecision) -> String {
        match decision {
            GateDecision::Allow => "allow".into(),
            GateDecision::Deny { .. } => "deny".into(),
            GateDecision::Prompt {
                forced: Some(f), ..
            } => format!("{:?}", f.kind),
            GateDecision::Prompt { forced: None, .. } => "prompt".into(),
            GateDecision::Classify(request) => format!("{:?}", request.source),
        }
    }

    struct Jail {
        _root: tempfile::TempDir,
        ws: PathBuf,
        outside: PathBuf,
        ctx: GateContext,
    }

    fn jail() -> Jail {
        let root = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let base = root.path().canonicalize().unwrap();
        let ws = base.join("ws");
        let outside = base.join("outside");
        std::fs::create_dir_all(ws.join("src")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::create_dir_all(base.join("scratch")).unwrap();
        let ctx = GateContext {
            workspace: Some(ws.clone()),
            sandbox_available: true,
            network_allowed: false,
            scratch_dir: Some(base.join("scratch")),
            cwd_drifted: false,
        };
        Jail {
            _root: root,
            ws,
            outside,
            ctx,
        }
    }

    fn resolve_args(mode: PermissionMode, ctx: &GateContext, args: Value) -> String {
        outcome(&resolve_in(
            mode,
            ctx,
            "bash",
            &args,
            &SessionGrants::default(),
        ))
    }

    #[test]
    fn contained_redirects_absorb_in_both_jailed_modes() {
        let j = jail();
        let out = j.outside.display();
        let cases: Vec<(String, &str)> = vec![
            ("echo hi > src/x.ts".into(), "allow"),
            ("cat > src/f <<'EOF'\nhello\nEOF".into(), "allow"),
            ("cat > src/f <<'EOF'\nlog > ~/log\nEOF".into(), "allow"),
            ("cat > src/f <<-'EOF'\n\tx\n\tEOF".into(), "allow"),
            ("cat <<'EOF'\necho hi > out.txt\nEOF".into(), "Security"),
            ("cat > src/f <<'EOF'\nx\nEOF\nls".into(), "Security"),
            (
                "cat > src/f <<'EOF'\nit's\nEOF\necho x > ~/f".into(),
                "Security",
            ),
            ("echo hi > out.txt; diff < secret".into(), "Security"),
            ("echo hi > out.txt; cat <<< x".into(), "Security"),
            (
                "echo hi > out.txt; bash <<'EOF'\nls\nEOF".into(),
                "Security",
            ),
            ("bash > /dev/null <<'EOF'\nls\nEOF".into(), "Security"),
            ("ls 2>&1 | grep x".into(), "allow"),
            ("ls 2>/dev/null".into(), "allow"),
            ("ls &> src/log".into(), "allow"),
            ("echo hi >& src/f".into(), "allow"),
            ("ls >&2".into(), "allow"),
            (format!("ls &> {out}/log"), "OutOfWorkspaceWrite"),
            (format!("echo hi >& {out}/f"), "OutOfWorkspaceWrite"),
            (format!("echo hi >| {out}/f"), "OutOfWorkspaceWrite"),
            (format!("echo hi > {out}/x"), "OutOfWorkspaceWrite"),
            ("eval \"$cmd\" > src/f".into(), "Security"),
            ("echo hi > $out".into(), "Security"),
            ("echo hi > *.log".into(), "Security"),
            ("echo $x > f".into(), "Security"),
            ("curl http://x.example > $out".into(), "Security"),
            ("echo x > .git/hooks/pre-commit".into(), "Security"),
            ("echo hi > /dev/sdb1".into(), "Security"),
            ("cd ~ && echo hi > out.txt".into(), "Security"),
            (
                "while true; do cd ~; break; done; echo hi > out.txt".into(),
                "Security",
            ),
            ("[[ a > b ]] && echo hi > out.txt".into(), "Security"),
            (format!("echo hi > {}/x", j.ws.display()), "allow"),
        ];
        for mode in [PermissionMode::Ask, PermissionMode::Auto] {
            for (command, expected) in &cases {
                // `ask` renders the out-of-roots arm as a matrix prompt.
                let expected = match (mode, *expected) {
                    (PermissionMode::Ask, "OutOfWorkspaceWrite") => "prompt",
                    (_, expected) => expected,
                };
                assert_eq!(
                    resolve_args(mode, &j.ctx, json!({ "command": command })),
                    expected,
                    "{mode:?}: {command:?}"
                );
            }
        }
    }

    #[test]
    fn redirect_absorption_needs_a_provable_cwd() {
        let j = jail();
        let drifted = GateContext {
            cwd_drifted: true,
            ..j.ctx.clone()
        };
        let ask = |ctx: &GateContext, args: Value| resolve_args(PermissionMode::Ask, ctx, args);
        assert_eq!(
            ask(&drifted, json!({ "command": "echo hi > out.txt" })),
            "Security"
        );
        // Absolute targets are cwd-independent.
        let abs = format!("echo hi > {}/out.txt", j.ws.display());
        assert_eq!(ask(&drifted, json!({ "command": abs })), "allow");
        // An absolute workdir proves its own call; a relative one does not
        // after drift.
        let ws = j.ws.display().to_string();
        assert_eq!(
            ask(
                &drifted,
                json!({ "command": "echo hi > out.txt", "workdir": ws })
            ),
            "allow"
        );
        assert_eq!(
            ask(
                &drifted,
                json!({ "command": "echo hi > out.txt", "workdir": "src" })
            ),
            "Security"
        );
        assert_eq!(
            ask(
                &j.ctx,
                json!({ "command": "echo hi > out.txt", "workdir": "src" })
            ),
            "allow"
        );
        // An out-of-roots workdir places the target outside.
        let outside = j.outside.display().to_string();
        assert_eq!(
            ask(
                &j.ctx,
                json!({ "command": "echo hi > x", "workdir": outside })
            ),
            "prompt"
        );
    }

    #[test]
    fn redirects_still_warn_without_a_jail() {
        let j = jail();
        let bare = GateContext {
            sandbox_available: false,
            ..j.ctx.clone()
        };
        assert_eq!(
            resolve_args(
                PermissionMode::Ask,
                &bare,
                json!({ "command": "echo hi > x" })
            ),
            "Security"
        );
        assert_eq!(
            resolve_args(
                PermissionMode::Auto,
                &bare,
                json!({ "command": "echo hi > x" })
            ),
            "Security"
        );
    }

    #[test]
    fn jailed_ask_resolver_matrix() {
        let j = jail();
        let ask = |ctx: &GateContext, cmd: &str| {
            resolve_args(PermissionMode::Ask, ctx, json!({ "command": cmd }))
        };
        // SA-2: in-profile silence, forced kinds still prompt.
        assert_eq!(ask(&j.ctx, "ls -la"), "allow");
        assert_eq!(ask(&j.ctx, "echo $(whoami)"), "Security");
        assert_eq!(ask(&j.ctx, "curl https://example.com"), "Network");
        let flagged = GateContext {
            network_allowed: true,
            ..j.ctx.clone()
        };
        assert_eq!(ask(&flagged, "curl https://example.com"), "allow");
        // SA-3: escapes prompt before execution.
        let escape = format!("cp a {}/f", j.outside.display());
        assert_eq!(ask(&j.ctx, &escape), "prompt");
        // No backend: per-command prompts, flag ignored.
        let bare = GateContext {
            sandbox_available: false,
            network_allowed: true,
            ..j.ctx.clone()
        };
        assert_eq!(ask(&bare, "ls -la"), "prompt");
        assert_eq!(ask(&bare, "curl https://example.com"), "Network");
        // `yolo` is never jailed and never asks.
        assert_eq!(
            resolve_args(
                PermissionMode::Yolo,
                &j.ctx,
                json!({ "command": "echo hi > ~/x" })
            ),
            "allow"
        );
    }

    #[test]
    fn scratch_is_the_analysis_root_without_a_workspace() {
        let j = jail();
        let scratch_only = GateContext {
            workspace: None,
            ..j.ctx.clone()
        };
        let ask = |cmd: &str| {
            resolve_args(
                PermissionMode::Ask,
                &scratch_only,
                json!({ "command": cmd }),
            )
        };
        // Relative writes land in the scratch: in profile.
        assert_eq!(ask("echo hi > out.txt"), "allow");
        assert_eq!(ask("cat attachments/data.csv"), "allow");
        // Out-of-scratch targets surface before execution, never silently.
        assert_ne!(ask("echo hi > /etc/x"), "allow");
        assert_eq!(ask(&format!("cp x {}/f", j.outside.display())), "prompt");
        assert_eq!(ask("cp x ~/f"), "prompt");
    }
}
