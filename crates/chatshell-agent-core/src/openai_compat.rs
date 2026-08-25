//! OpenAI Chat Completions-compatible CompletionModel.
//!
//! Built on rig's generic OpenAI-compatible provider machinery: the shared
//! wire adapter already handles streaming, `reasoning_content` deltas
//! (DeepSeek-style), fragmented tool calls, and error envelopes.
//!
//! Two behaviors are specific to ChatShell's provider fleet:
//! - custom base URLs for arbitrary OpenAI-compatible endpoints;
//! - `with_string_content_only()` for providers that only accept plain
//!   string content (e.g. MiniMax, Xirang): content-part arrays are
//!   flattened to the concatenation of their text parts and non-text parts
//!   (images) are dropped.

use rig::client::{BearerAuth, Client, Provider, ProviderBuilder};
use rig::completion::{CompletionError, CompletionRequest};
use rig::providers::openai::completion::{
    CompletionModelOptions, CompletionResponse, GenericCompletionModel, OpenAICompatibleProvider,
    Usage,
};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Provider extension describing a generic OpenAI-compatible endpoint.
#[derive(Clone, Debug, Default)]
pub struct CompatExt {
    string_content_only: bool,
}

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

    type StreamingUsage = Usage;
    type Response = CompletionResponse;

    fn finalize_request_body_with_options(
        &self,
        body: &mut serde_json::Value,
        _options: CompletionModelOptions,
    ) -> Result<(), CompletionError> {
        if self.string_content_only {
            flatten_message_content_arrays(body);
        }
        Ok(())
    }
}

/// Flatten every message `content` array to the concatenation of its text
/// parts, dropping non-text parts. Messages whose content is already a
/// plain string are left untouched.
fn flatten_message_content_arrays(body: &mut serde_json::Value) {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    for message in messages {
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        if let Some(parts) = content.as_array() {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("");
            *content = serde_json::Value::String(text);
        }
    }
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
/// A thin wrapper over rig's [`GenericCompletionModel`] that keeps the
/// string-content-only flag at the model level (the request-finalization
/// hook lives on the client extension).
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

    /// Restrict requests to plain-string message content for providers that
    /// reject content-part arrays. Images are dropped and user content is
    /// flattened to a single string.
    pub fn with_string_content_only(self) -> Self {
        Self {
            client: self.client.with_ext(CompatExt {
                string_content_only: true,
            }),
            model: self.model,
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
