//! OpenAI Chat Completions-compatible CompletionModel.
//!
//! Built on rig's generic OpenAI-compatible provider machinery: the shared
//! wire adapter already handles streaming, `reasoning_content` deltas
//! (DeepSeek-style), fragmented tool calls, and error envelopes.
//!
//! One behavior is specific to ChatShell's provider fleet:
//! - custom base URLs for arbitrary OpenAI-compatible endpoints.

use rig::client::{BearerAuth, Client, Provider, ProviderBuilder};
use rig::completion::{CompletionError, CompletionRequest};
use rig::providers::openai::completion::{
    CompletionResponse, GenericCompletionModel, OpenAICompatibleProvider,
};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Provider extension describing a generic OpenAI-compatible endpoint.
#[derive(Clone, Debug, Default)]
pub struct CompatExt {}

/// Builder satisfying rig's `ProviderBuilder` contract for [`CompatExt`].
#[derive(Clone, Debug, Default)]
pub struct CompatExtBuilder;

impl ProviderBuilder for CompatExtBuilder {
    type Extension<H>
        = CompatExt
    where
        H: rig::http_client::HttpClientExt;
    type ApiKey = BearerAuth;
    const BASE_URL: &'static str = DEFAULT_BASE_URL;

    fn build<H>(
        _builder: &rig::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> rig::http_client::Result<Self::Extension<H>>
    where
        H: rig::http_client::HttpClientExt,
    {
        Ok(CompatExt::default())
    }
}

impl rig::client::DebugExt for CompatExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
        std::iter::empty()
    }
}

impl Provider for CompatExt {
    type Builder = CompatExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl OpenAICompatibleProvider for CompatExt {
    const PROVIDER_NAME: &'static str = "openai-compat";

    // Lenient on purpose: MiniMax (and other compat endpoints) emit
    // abnormal terminal chunks whose `usage` object is missing fields —
    // observed live as `missing field 'prompt_tokens'` killing a whole
    // turn one second in. rig's stock `Usage` requires the field; parsing
    // to zeros keeps the turn alive and the provider's error envelope (if
    // any) is what should decide the outcome, not usage bookkeeping.
    type StreamingUsage = CompatStreamingUsage;
    type Response = CompletionResponse;
}

/// A client for an arbitrary OpenAI-compatible endpoint.
pub type CompatClient = Client<CompatExt>;

/// Construct an OpenAI-compatible client for `base_url` authenticated with
/// `api_key` (sent as a bearer token).
pub fn client(
    base_url: &str,
    api_key: &str,
    http_client: reqwest::Client,
) -> Result<CompatClient, CompletionError> {
    CompatClient::builder()
        .api_key(api_key)
        .base_url(base_url)
        .http_client(http_client)
        .build()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))
}

/// Chat Completions model for an OpenAI-compatible endpoint.
///
/// A thin wrapper over rig's [`GenericCompletionModel`].
#[derive(Clone)]
pub struct CompletionModel {
    client: CompatClient,
    model: String,
}

impl CompletionModel {
    pub fn new(client: CompatClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    fn generic(&self) -> GenericCompletionModel<CompatExt> {
        GenericCompletionModel::new(self.client.clone(), self.model.clone())
    }
}

impl rig::completion::CompletionModel for CompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse, CompletionError> {
        let generic = self.generic();
        generic.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<rig::streaming::StreamingCompletionResponse, CompletionError> {
        let generic = self.generic();
        generic.stream(request).await
    }
}

/// Streaming `usage` payload for the compat fleet: every field defaults to
/// zero so an abnormal terminal chunk (MiniMax omits `prompt_tokens` on
/// some error/terminal chunks) cannot fail deserialization and kill the
/// turn. [`From<CompatStreamingUsage> for rig::completion::Usage`] derives
/// `total_tokens` when the provider omits it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompatStreamingUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    /// `prompt_tokens_details`, when the provider reports it.
    #[serde(default)]
    pub prompt_tokens_details: Option<CompatPromptTokensDetails>,
}

/// `prompt_tokens_details` sub-object of the usage payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompatPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

impl From<CompatStreamingUsage> for rig::completion::Usage {
    fn from(u: CompatStreamingUsage) -> Self {
        rig::completion::Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: if u.total_tokens > 0 {
                u.total_tokens
            } else {
                u.prompt_tokens + u.completion_tokens
            },
            cached_input_tokens: u
                .prompt_tokens_details
                .map(|d| d.cached_tokens)
                .unwrap_or(0),
            ..rig::completion::Usage::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact live failure: MiniMax's abnormal terminal chunk carries a
    /// `usage` object without `prompt_tokens`. Stock rig `Usage` hard-fails
    /// ("missing field `prompt_tokens`"); the compat type must parse.
    #[test]
    fn streaming_usage_parses_without_prompt_tokens() {
        let u: CompatStreamingUsage =
            serde_json::from_str(r#"{"completion_tokens": 12, "total_tokens": 34}"#).unwrap();
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.completion_tokens, 12);

        let normalized = rig::completion::Usage::from(u);
        assert_eq!(normalized.input_tokens, 0);
        assert_eq!(normalized.output_tokens, 12);
        assert_eq!(normalized.total_tokens, 34);
    }

    #[test]
    fn streaming_usage_maps_full_payload_and_derives_missing_total() {
        let u: CompatStreamingUsage = serde_json::from_str(
            r#"{"prompt_tokens": 40, "completion_tokens": 60,
                "prompt_tokens_details": {"cached_tokens": 30}}"#,
        )
        .unwrap();
        let normalized = rig::completion::Usage::from(u);
        assert_eq!(normalized.input_tokens, 40);
        assert_eq!(normalized.output_tokens, 60);
        // total_tokens absent on the wire: derived, not zero.
        assert_eq!(normalized.total_tokens, 100);
        assert_eq!(normalized.cached_input_tokens, 30);
    }

    #[test]
    fn streaming_usage_parses_empty_object() {
        let u: CompatStreamingUsage = serde_json::from_str("{}").unwrap();
        assert_eq!(u, CompatStreamingUsage::default());
    }
}
