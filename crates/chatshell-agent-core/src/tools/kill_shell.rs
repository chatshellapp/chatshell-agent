//! Kill shell tool for LLM agents
//!
//! Terminates the current persistent bash session. A new session will be
//! created automatically on the next `bash` tool call.

use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::bash::SharedBashSession;

#[derive(Debug, Clone, Deserialize)]
pub struct KillShellArgs {}

#[derive(Debug, thiserror::Error)]
#[error("KillShell error: {0}")]
pub struct KillShellError(String);

#[derive(Clone, Serialize, Deserialize)]
pub struct KillShellTool {
    #[serde(skip)]
    session: SharedBashSession,
}

impl std::fmt::Debug for KillShellTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KillShellTool").finish()
    }
}

impl KillShellTool {
    pub fn new(session: SharedBashSession) -> Self {
        Self { session }
    }
}

impl Tool for KillShellTool {
    const NAME: &'static str = "kill_shell";

    type Error = KillShellError;
    type Args = KillShellArgs;
    type Output = String;

    fn description(&self) -> String {
        "Terminate the current bash session. All state (environment variables, \
                working directory, background processes) will be lost. A new session will be \
                created automatically on the next bash command."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        tracing::info!("🔧 [tool-call] kill_shell");

        let mut guard = self.session.lock().await;

        if let Some(mut session) = guard.take() {
            session.kill().await;
            tracing::info!("🖥️ [kill_shell] Session terminated");
            Ok("Bash session terminated.".to_string())
        } else {
            Ok("No active bash session.".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::bash::{BashArgs, BashTool};
    use rig::agent::tool::ToolContext;
    use std::sync::Arc;

    /// Spawn a real bash session through the public path: run one command via
    /// BashTool so it lazily creates the session inside the shared handle.
    async fn live_session() -> SharedBashSession {
        let cwd = std::env::temp_dir().to_string_lossy().to_string();
        let tool: BashTool =
            BashTool::with_session(Arc::new(tokio::sync::Mutex::new(None)), Some(cwd));
        let handle = tool.session_handle();
        let mut context = ToolContext::new();
        tool.call(
            &mut context,
            BashArgs {
                command: Some("true".to_string()),
                timeout: None,
                description: None,
                workdir: None,
            },
        )
        .await
        .unwrap();
        handle
    }

    #[test]
    fn schema_advertises_no_arguments() {
        let tool = KillShellTool::new(Arc::new(tokio::sync::Mutex::new(None)));
        assert_eq!(<KillShellTool as Tool>::NAME, "kill_shell");
        let params = tool.parameters();
        assert!(
            params["properties"].as_object().unwrap().is_empty(),
            "kill_shell takes no arguments: {params}"
        );
    }

    #[tokio::test]
    async fn terminates_an_active_session_and_empties_the_slot() {
        let session = live_session().await;
        let tool = KillShellTool::new(session.clone());
        let mut context = ToolContext::new();

        let output = tool.call(&mut context, KillShellArgs {}).await.unwrap();

        assert_eq!(output, "Bash session terminated.");
        assert!(
            session.lock().await.is_none(),
            "the shared slot must be emptied so the next bash call spawns fresh"
        );
    }

    #[tokio::test]
    async fn without_an_active_session_it_reports_none() {
        let tool = KillShellTool::new(Arc::new(tokio::sync::Mutex::new(None)));
        let mut context = ToolContext::new();

        let output = tool.call(&mut context, KillShellArgs {}).await.unwrap();

        assert_eq!(output, "No active bash session.");
    }
}
