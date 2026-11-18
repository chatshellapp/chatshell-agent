//! `tools/call` with a deadline, progress forwarding, and cancellation.
//!
//! - Timeout (fail-closed): a call that outlives its deadline is an error,
//!   never an empty success. rmcp sends `notifications/cancelled` for it.
//! - Progress: rmcp attaches a `progressToken` to every request; progress
//!   notifications for this call are forwarded to `on_progress` and, when
//!   enabled, reset the deadline (bounded by `max_total_timeout`).
//! - Cancellation: if the caller drops the call future before a response
//!   (the user stopped the generation), `notifications/cancelled` is sent
//!   best-effort on whatever transport the peer uses.

use std::time::Duration;

use futures::{FutureExt, StreamExt};
use rmcp::RoleClient;
use rmcp::handler::client::progress::ProgressDispatcher;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResult, CancelledNotificationParam,
    ClientRequest, ProgressNotificationParam, RequestId, ServerResult,
};
use rmcp::service::{Peer, PeerRequestOptions, ServiceError};

/// Default per-call deadline.
pub const DEFAULT_CALL_TIMEOUT_SECS: u64 = 30;
/// Default ceiling on a call whose deadline keeps being reset by progress.
pub const DEFAULT_MAX_TOTAL_TIMEOUT_SECS: u64 = 600;

/// Deadline settings for one `tools/call`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpCallOptions {
    /// `None` disables the deadline.
    pub timeout: Option<Duration>,
    pub reset_timeout_on_progress: bool,
    /// Hard ceiling regardless of progress; only meaningful with a timeout.
    pub max_total_timeout: Option<Duration>,
}

impl Default for McpCallOptions {
    fn default() -> Self {
        Self::from_settings(None, None, None)
    }
}

impl McpCallOptions {
    /// Build options from per-server settings. `timeout_secs == Some(0)`
    /// disables the deadline (and with it the total ceiling); `None` means
    /// the default. Same rule for `max_total_secs`.
    pub fn from_settings(
        timeout_secs: Option<u64>,
        reset_timeout_on_progress: Option<bool>,
        max_total_secs: Option<u64>,
    ) -> Self {
        let timeout = match timeout_secs.unwrap_or(DEFAULT_CALL_TIMEOUT_SECS) {
            0 => None,
            secs => Some(Duration::from_secs(secs)),
        };
        let max_total_timeout =
            timeout.and_then(
                |_| match max_total_secs.unwrap_or(DEFAULT_MAX_TOTAL_TIMEOUT_SECS) {
                    0 => None,
                    secs => Some(Duration::from_secs(secs)),
                },
            );
        Self {
            timeout,
            reset_timeout_on_progress: reset_timeout_on_progress.unwrap_or(true),
            max_total_timeout,
        }
    }

    fn to_peer_options(self) -> PeerRequestOptions {
        let mut options = PeerRequestOptions::no_options();
        options.timeout = self.timeout;
        if self.timeout.is_some() {
            options.reset_timeout_on_progress = self.reset_timeout_on_progress;
            options.max_total_timeout = self.max_total_timeout;
        }
        options
    }
}

/// Sends `notifications/cancelled` for a request when dropped while armed.
struct CancelOnDrop {
    peer: Peer<RoleClient>,
    id: RequestId,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let peer = self.peer.clone();
        let id = self.id.clone();
        runtime.spawn(async move {
            let params =
                CancelledNotificationParam::new(Some(id), Some("cancelled by user".to_string()));
            if let Err(e) = peer.notify_cancelled(params).await {
                tracing::debug!("Could not send MCP cancellation: {e}");
            }
        });
    }
}

/// Call a tool with the given deadline settings. Progress notifications for
/// this call reach `on_progress` when a `dispatcher` is supplied (the one
/// owned by the connection's [`super::ChatShellClientHandler`]).
pub async fn call_tool_with_options(
    peer: &Peer<RoleClient>,
    params: CallToolRequestParams,
    options: McpCallOptions,
    dispatcher: Option<&ProgressDispatcher>,
    mut on_progress: impl FnMut(ProgressNotificationParam) + Send,
) -> Result<CallToolResult, ServiceError> {
    let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
    let handle = peer
        .send_request_with_option(request, options.to_peer_options())
        .await?;

    let mut guard = CancelOnDrop {
        peer: peer.clone(),
        id: handle.id.clone(),
        armed: true,
    };

    let mut progress = match dispatcher {
        Some(d) => Some(d.subscribe(handle.progress_token.clone()).await),
        None => None,
    };

    let response = handle.await_response();
    tokio::pin!(response);
    let result = loop {
        let next_progress = async {
            match progress.as_mut() {
                Some(p) => p.next().await,
                None => std::future::pending().await,
            }
        };
        // Fair (unbiased) select: a server flooding progress cannot starve
        // the response future, which also carries the deadline.
        tokio::select! {
            result = &mut response => break result,
            Some(update) = next_progress => on_progress(update),
        }
    };
    // Deliver progress that was already queued when the response won the
    // race (bounded by the dispatcher's channel capacity).
    if let Some(p) = progress.as_mut() {
        while let Some(Some(update)) = p.next().now_or_never() {
            on_progress(update);
        }
    }
    // Answered, timed out (rmcp already sent the cancellation), or the
    // transport failed: nothing left to cancel.
    guard.armed = false;

    match result? {
        ServerResult::CallToolResult(r) => Ok(r),
        _ => Err(ServiceError::UnexpectedResponse),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let o = McpCallOptions::default();
        assert_eq!(o.timeout, Some(Duration::from_secs(30)));
        assert!(o.reset_timeout_on_progress);
        assert_eq!(o.max_total_timeout, Some(Duration::from_secs(600)));
    }

    #[test]
    fn zero_disables_deadline_and_ceiling() {
        let o = McpCallOptions::from_settings(Some(0), None, Some(120));
        assert_eq!(o.timeout, None);
        assert_eq!(o.max_total_timeout, None);
        let p = o.to_peer_options();
        assert_eq!(p.timeout, None);
        assert!(!p.reset_timeout_on_progress);
    }

    #[test]
    fn overrides_apply() {
        let o = McpCallOptions::from_settings(Some(5), Some(false), Some(0));
        assert_eq!(o.timeout, Some(Duration::from_secs(5)));
        assert!(!o.reset_timeout_on_progress);
        assert_eq!(o.max_total_timeout, None);
    }
}
