//! Tool-permission decision layer for the approval gate.
//!
//! Pure logic: no I/O, no UI, no clock. The host supplies the permission
//! mode and the session grants and receives a three-valued decision
//! (allow / deny / prompt). Precedence: hard deny (classifier blocklist)
//! -> session-grant lookup -> mode/tier matrix -> prompt.
//!
//! The authoritative blocklist enforcement stays inside the tool bodies
//! (`bash_security`, `path_policy`) so it holds in every mode, including
//! [`PermissionMode::Yolo`]; the deny produced here is a pre-check that
//! avoids showing an approval prompt for commands the tool would refuse
//! anyway.

use crate::bash_security::{self, SecurityVerdict};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

/// Conversation-level permission mode for the approval gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    /// Prompt before write- and exec-tier tool calls; session grants
    /// cache approvals.
    Ask,
    /// Run tool calls without prompts. Catastrophic operations are still
    /// hard-denied by the in-tool blocklist and path policy in every mode.
    #[default]
    Yolo,
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

/// Gate verdict for one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Execute without asking.
    Allow,
    /// Refuse; `reason` is fed back to the model as the tool result.
    Deny { reason: String },
    /// Ask the user; `subject` is the human-readable target of the call.
    Prompt { subject: String },
}

/// What an "allow for this session" answer grants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantScope {
    /// Every base command of the approved bash call (all future commands
    /// whose segments are covered by these base commands).
    BashCommands(Vec<String>),
    /// The whole tool (write-tier tools only).
    Tool(String),
}

/// In-memory session grants ("allow for this session" answers).
///
/// Never persisted and never synced: a grant made in one session or on
/// one device must not take effect elsewhere.
#[derive(Debug, Default, Clone)]
pub struct SessionGrants {
    keys: HashSet<String>,
}

impl SessionGrants {
    /// Grant every base command of an approved bash call.
    pub fn grant_bash_commands(&mut self, bases: &[&str]) {
        for base in bases {
            self.keys.insert(format!("bash:{base}"));
        }
    }

    /// Grant a whole tool for the rest of the session (write-tier tools).
    pub fn grant_tool(&mut self, tool: &str) {
        self.keys.insert(format!("tool:{tool}"));
    }

    /// True when every segment base command of `command` is granted. A
    /// partial grant never authorizes a compound command.
    pub fn bash_command_granted(&self, command: &str) -> bool {
        let bases = base_commands(command);
        !bases.is_empty()
            && bases
                .iter()
                .all(|base| self.keys.contains(&format!("bash:{base}")))
    }

    /// True when the tool was granted for the session.
    pub fn tool_granted(&self, tool: &str) -> bool {
        self.keys.contains(&format!("tool:{tool}"))
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

/// Base commands of a (possibly compound) bash invocation: one entry per
/// `&&`/`||`/`;`/`|` segment, wrappers stripped, first word only. Sorted
/// and deduplicated so grant keys are stable.
pub fn base_commands(command: &str) -> Vec<String> {
    let mut bases: Vec<String> = bash_security::split_compound_command(command)
        .into_iter()
        .filter_map(|segment| {
            let stripped = bash_security::strip_safe_wrappers(segment);
            stripped
                .split_whitespace()
                .find(|word| !matches!(*word, "command" | "builtin" | "noglob" | "nocorrect"))
        })
        .map(str::to_string)
        .collect();
    bases.sort_unstable();
    bases.dedup();
    bases
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
        "web_fetch" => arg("url").map(str::to_string),
        _ => None,
    };
    subject.unwrap_or_else(|| tool_name.to_string())
}

/// What an "allow for this session" answer memoizes for this call.
///
/// `None` for tools without session-grant semantics (exec tier other
/// than bash): the UI hides the session button for those.
pub fn grant_scope(tool_name: &str, args: &Value) -> Option<GrantScope> {
    if tool_name == "bash" {
        return args
            .get("command")
            .and_then(Value::as_str)
            .map(|command| GrantScope::BashCommands(base_commands(command)));
    }
    if matches!(PermissionTier::for_tool(tool_name), PermissionTier::Write) {
        return Some(GrantScope::Tool(tool_name.to_string()));
    }
    None
}

/// Resolve one tool call against the mode and session grants.
pub fn resolve(
    mode: PermissionMode,
    tool_name: &str,
    args: &Value,
    grants: &SessionGrants,
) -> GateDecision {
    // Bash: hard-deny pre-check, then grants, then the mode matrix. The
    // authoritative blocklist lives inside the bash tool; this pre-check
    // denies before an approval prompt would be shown for a command the
    // tool would refuse anyway.
    if tool_name == "bash"
        && let Some(command) = args.get("command").and_then(Value::as_str)
    {
        if let SecurityVerdict::Block(reason) =
            bash_security::check_command(command, bash_security::check_dangerous)
        {
            return GateDecision::Deny { reason };
        }
        if grants.bash_command_granted(command) {
            return GateDecision::Allow;
        }
        return match mode {
            PermissionMode::Yolo => GateDecision::Allow,
            PermissionMode::Ask => GateDecision::Prompt {
                subject: command.to_string(),
            },
        };
    }
    // Bash without a command argument: let the tool itself surface the
    // "No command provided" error instead of prompting.
    if tool_name == "bash" {
        return GateDecision::Allow;
    }

    let tier = PermissionTier::for_tool(tool_name);
    let needs_prompt = match mode {
        PermissionMode::Yolo => false,
        PermissionMode::Ask => !matches!(tier, PermissionTier::Read),
    };
    if !needs_prompt {
        return GateDecision::Allow;
    }
    if matches!(tier, PermissionTier::Write) && grants.tool_granted(tool_name) {
        return GateDecision::Allow;
    }
    GateDecision::Prompt {
        subject: subject_for(tool_name, args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resolve_simple(mode: PermissionMode, tool: &str, args: Value) -> GateDecision {
        resolve(mode, tool, &args, &SessionGrants::default())
    }

    #[test]
    fn default_mode_is_yolo() {
        assert_eq!(PermissionMode::default(), PermissionMode::Yolo);
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
            GateDecision::Prompt {
                subject: "/tmp/a.txt".to_string()
            }
        );
        let bash = json!({"command": "cargo test"});
        assert_eq!(
            resolve_simple(PermissionMode::Ask, "bash", bash),
            GateDecision::Prompt {
                subject: "cargo test".to_string()
            }
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
        grants.grant_bash_commands(&["cargo"]);
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
            GateDecision::Prompt {
                subject: "cargo build && npm install".to_string()
            }
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
            GateDecision::Prompt {
                subject: "/tmp/a.txt".to_string()
            }
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
    fn base_commands_strips_wrappers_and_splits_compounds() {
        assert_eq!(base_commands("cargo build"), vec!["cargo".to_string()]);
        assert_eq!(
            base_commands("nice -n 10 cargo build"),
            vec!["cargo".to_string()]
        );
        assert_eq!(base_commands("GOOS=linux go build"), vec!["go".to_string()]);
        assert_eq!(
            base_commands("git status && git push | tee log"),
            vec!["git".to_string(), "tee".to_string()]
        );
        assert_eq!(base_commands(""), Vec::<String>::new());
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
    fn grant_scope_none_for_exec_tools_without_bash() {
        assert_eq!(grant_scope("kill_shell", &json!({})), None);
        assert_eq!(
            grant_scope("bash", &json!({"command": "git status && git push"})),
            Some(GrantScope::BashCommands(vec!["git".to_string()]))
        );
        assert_eq!(
            grant_scope("edit", &json!({"path": "/tmp/a"})),
            Some(GrantScope::Tool("edit".to_string()))
        );
    }
}
