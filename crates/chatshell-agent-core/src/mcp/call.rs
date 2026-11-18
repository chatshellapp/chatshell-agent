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
    ClientRequest, InputResponses, ProgressNotificationParam, ReadResourceRequest,
    ReadResourceRequestParams, ReadResourceResult, RequestId, ServerResult,
};

use super::handler::ChatShellClientHandler;
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

/// Most `input_required` rounds one call may take (FR-17; the SDK default).
pub const MAX_INPUT_ROUNDS: usize = rmcp::model::DEFAULT_MRTR_MAX_ROUNDS;

/// Call a tool with the given deadline settings. Progress notifications for
/// this call reach `on_progress` when a `dispatcher` is supplied (the one
/// owned by the connection's [`super::ChatShellClientHandler`]). A
/// 2026-07-28 server may answer `input_required` (MRTR, FR-17): `input`
/// (the connection's handler) fulfils those rounds; without it, or past
/// [`MAX_INPUT_ROUNDS`], the call fails.
pub async fn call_tool_with_options(
    peer: &Peer<RoleClient>,
    params: CallToolRequestParams,
    options: McpCallOptions,
    dispatcher: Option<&ProgressDispatcher>,
    input: Option<&ChatShellClientHandler>,
    mut on_progress: impl FnMut(ProgressNotificationParam) + Send,
) -> Result<CallToolResult, ServiceError> {
    let mut params = params;
    let mut rounds = InputRounds::default();
    loop {
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params.clone()));
        match request_with_options(peer, request, options, dispatcher, &mut on_progress).await? {
            ServerResult::CallToolResult(r) => return Ok(r),
            ServerResult::InputRequiredResult(required) => {
                (params.input_responses, params.request_state) =
                    rounds.next(input, required).await?;
            }
            _ => return Err(ServiceError::UnexpectedResponse),
        }
    }
}

/// Read a resource with the same deadline, cancel-on-drop, and MRTR
/// handling as a tool call.
pub async fn read_resource_with_options(
    peer: &Peer<RoleClient>,
    uri: &str,
    options: McpCallOptions,
    input: Option<&ChatShellClientHandler>,
) -> Result<ReadResourceResult, ServiceError> {
    let mut params = ReadResourceRequestParams::new(uri);
    let mut rounds = InputRounds::default();
    loop {
        let request = ClientRequest::ReadResourceRequest(ReadResourceRequest::new(params.clone()));
        match request_with_options(peer, request, options, None, &mut |_| {}).await? {
            ServerResult::ReadResourceResult(r) => return Ok(r),
            ServerResult::InputRequiredResult(required) => {
                (params.input_responses, params.request_state) =
                    rounds.next(input, required).await?;
            }
            _ => return Err(ServiceError::UnexpectedResponse),
        }
    }
}

/// Get a prompt with the same deadline, cancel-on-drop, and MRTR handling
/// as a tool call.
pub async fn get_prompt_with_options(
    peer: &Peer<RoleClient>,
    mut params: rmcp::model::GetPromptRequestParams,
    options: McpCallOptions,
    input: Option<&ChatShellClientHandler>,
) -> Result<rmcp::model::GetPromptResult, ServiceError> {
    let mut rounds = InputRounds::default();
    loop {
        let request =
            ClientRequest::GetPromptRequest(rmcp::model::GetPromptRequest::new(params.clone()));
        match request_with_options(peer, request, options, None, &mut |_| {}).await? {
            ServerResult::GetPromptResult(r) => return Ok(r),
            ServerResult::InputRequiredResult(required) => {
                (params.input_responses, params.request_state) =
                    rounds.next(input, required).await?;
            }
            _ => return Err(ServiceError::UnexpectedResponse),
        }
    }
}

/// Bookkeeping for the `input_required` rounds of one request.
#[derive(Default)]
pub(crate) struct InputRounds {
    taken: usize,
    state_only: u32,
}

impl InputRounds {
    /// Fulfil one round through `input` and return what the retry carries
    /// (`inputResponses`, `requestState`). Fails past [`MAX_INPUT_ROUNDS`],
    /// without a handler, or on an empty round. Rounds that only carry
    /// server state back off like the SDK (50 ms doubling, capped at 250 ms).
    pub(crate) async fn next(
        &mut self,
        input: Option<&ChatShellClientHandler>,
        required: rmcp::model::InputRequiredResult,
    ) -> Result<(Option<InputResponses>, Option<String>), ServiceError> {
        self.taken += 1;
        if self.taken >= MAX_INPUT_ROUNDS {
            return Err(ServiceError::InputRequiredRoundsExceeded {
                max_rounds: MAX_INPUT_ROUNDS,
            });
        }
        let Some(handler) = input else {
            return Err(ServiceError::McpError(rmcp::ErrorData::invalid_request(
                "the server asked for input this client cannot provide here",
                None,
            )));
        };
        let requests = required.input_requests.unwrap_or_default();
        if requests.is_empty() {
            if required.request_state.is_none() {
                return Err(ServiceError::UnexpectedResponse);
            }
            let millis = (50u64 << self.state_only.min(3)).min(250);
            tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
            self.state_only += 1;
            return Ok((None, required.request_state));
        }
        self.state_only = 0;
        let responses = handler.fulfill_input_requests(requests).await?;
        Ok((
            (!responses.is_empty()).then_some(responses),
            required.request_state,
        ))
    }
}

async fn request_with_options(
    peer: &Peer<RoleClient>,
    request: ClientRequest,
    options: McpCallOptions,
    dispatcher: Option<&ProgressDispatcher>,
    on_progress: &mut (impl FnMut(ProgressNotificationParam) + Send),
) -> Result<ServerResult, ServiceError> {
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
    result
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
