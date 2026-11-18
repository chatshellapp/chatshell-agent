//! The client-side handler installed on every MCP connection.
//!
//! Handles server notifications: `tools/list_changed` (debounced, forwarded
//! to the platform), `progress` (routed to the in-flight call that owns the
//! token), and `message` (logged via `tracing`). It never sends
//! `logging/setLevel`; servers decide what to emit.
//!
//! Logging is deprecated by SEP-2577 but still sent by servers on the
//! revisions we negotiate, so the logging types are used under
//! `#[allow(deprecated)]`.

#![allow(deprecated)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rmcp::handler::client::progress::ProgressDispatcher;
use rmcp::model::{
    ClientCapabilities, ClientInfo, Implementation, LoggingLevel, LoggingMessageNotificationParam,
    ProgressNotificationParam,
};
use rmcp::service::NotificationContext;
use rmcp::{ClientHandler, RoleClient};

/// `tools/list_changed` notifications arriving within this window collapse
/// into one refresh (a server cannot trigger a re-list storm).
pub const LIST_CHANGED_DEBOUNCE: Duration = Duration::from_millis(500);

/// Platform hooks for server notifications.
pub trait McpNotificationSink: Send + Sync + 'static {
    /// The server's tool list changed (already debounced).
    fn tools_list_changed(&self);
}

/// The identity and capabilities every ChatShell MCP client sends.
pub fn chatshell_client_info() -> ClientInfo {
    ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("chatshell", env!("CARGO_PKG_VERSION")),
    )
}

#[derive(Clone)]
pub struct ChatShellClientHandler {
    info: ClientInfo,
    server: Arc<str>,
    progress: ProgressDispatcher,
    sink: Option<Arc<dyn McpNotificationSink>>,
    list_changed_pending: Arc<AtomicBool>,
}

impl ChatShellClientHandler {
    /// `server` labels log lines (the user-facing server name).
    pub fn new(server: &str, sink: Option<Arc<dyn McpNotificationSink>>) -> Self {
        Self {
            info: chatshell_client_info(),
            server: Arc::from(server),
            progress: ProgressDispatcher::new(),
            sink,
            list_changed_pending: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Report `version` (the app's version) in `clientInfo` instead of the
    /// core crate's.
    pub fn with_client_version(mut self, version: &str) -> Self {
        self.info.client_info = Implementation::new("chatshell", version);
        self
    }

    /// Routes progress notifications to in-flight calls
    /// (see [`super::call_tool_with_options`]).
    pub fn progress_dispatcher(&self) -> &ProgressDispatcher {
        &self.progress
    }

    pub fn server_name(&self) -> &str {
        &self.server
    }

    fn schedule_list_changed(&self) {
        let Some(sink) = self.sink.clone() else {
            return;
        };
        if self.list_changed_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        let pending = self.list_changed_pending.clone();
        let server = self.server.clone();
        tokio::spawn(async move {
            tokio::time::sleep(LIST_CHANGED_DEBOUNCE).await;
            // Clear before notifying: a change during the re-list schedules
            // another refresh instead of being lost.
            pending.store(false, Ordering::Release);
            tracing::info!("MCP server '{}' reported a tool list change", server);
            sink.tools_list_changed();
        });
    }
}

fn log_server_message(server: &str, params: &LoggingMessageNotificationParam) {
    let logger = params.logger.as_deref().unwrap_or("-");
    let data = match &params.data {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    // Bound what a server can push into our logs per message.
    const MAX_LOG_CHARS: usize = 4_000;
    let data = if data.len() > MAX_LOG_CHARS {
        let mut end = MAX_LOG_CHARS;
        while !data.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &data[..end])
    } else {
        data
    };
    match params.level {
        LoggingLevel::Debug => tracing::debug!(mcp_server = server, logger, "{data}"),
        LoggingLevel::Info | LoggingLevel::Notice => {
            tracing::info!(mcp_server = server, logger, "{data}")
        }
        LoggingLevel::Warning => tracing::warn!(mcp_server = server, logger, "{data}"),
        LoggingLevel::Error
        | LoggingLevel::Critical
        | LoggingLevel::Alert
        | LoggingLevel::Emergency => tracing::error!(mcp_server = server, logger, "{data}"),
    }
}

impl ClientHandler for ChatShellClientHandler {
    fn get_info(&self) -> ClientInfo {
        self.info.clone()
    }

    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        self.schedule_list_changed();
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        self.progress.handle_notification(params).await;
    }

    async fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        log_server_message(&self.server, &params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Counter(AtomicUsize);
    impl McpNotificationSink for Counter {
        fn tools_list_changed(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn list_changed_storm_is_coalesced() {
        let counter = Arc::new(Counter(AtomicUsize::new(0)));
        let handler = ChatShellClientHandler::new("s", Some(counter.clone()));
        for _ in 0..50 {
            handler.schedule_list_changed();
        }
        tokio::time::sleep(LIST_CHANGED_DEBOUNCE * 2).await;
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);

        // A later change schedules a fresh refresh.
        handler.schedule_list_changed();
        tokio::time::sleep(LIST_CHANGED_DEBOUNCE * 2).await;
        assert_eq!(counter.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn client_info_identity() {
        let info = chatshell_client_info();
        assert_eq!(info.client_info.name, "chatshell");
        let handler = ChatShellClientHandler::new("s", None).with_client_version("9.9.9");
        assert_eq!(handler.get_info().client_info.version, "9.9.9");
        assert_eq!(handler.get_info().client_info.name, "chatshell");
    }
}
