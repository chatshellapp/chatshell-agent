//! Provider routing tables shared across desktop and iOS.
//!
//! This module is the single source of truth for provider-type-derived routing
//! data that both clients need:
//!
//! - [`default_base_url`] - the default API endpoint for OpenAI-compatible
//!   providers that lack a dedicated rig provider client (the "catch-all"
//!   providers routed through `openai_compat::CompletionModel`).
//! - [`is_string_content_only`] - whether a provider requires user message
//!   content serialized as a plain string instead of a structured content
//!   array. Providers that reject array content (e.g. DeepSeek, Baichuan,
//!   MiniMax, Xirang) must opt into `CompletionModel::with_string_content_only`.
//!
//! Named providers (OpenAI, Anthropic, OpenRouter, Ollama, DeepSeek, Groq, ...)
//! carry their default URLs inside their dedicated rig provider crates
//! (`*_provider::DEFAULT_BASE_URL`), so they are intentionally absent from
//! [`default_base_url`]. Only the catch-all OpenAI-compatible providers - those
//! with no dedicated rig client - belong here.
//!
//! Both tables previously lived inline in
//! `chatshell-desktop/src-tauri/src/llm/agent_builder.rs` as
//! `openai_compat_default_url` and `STRING_CONTENT_ONLY_PROVIDERS`. They were
//! lifted here unchanged so the values are identical and desktop simply calls
//! into core.

/// Default base URLs for OpenAI-compatible providers that have no dedicated rig
/// provider client. Returns `Some(url)` if the provider is a known catch-all
/// OpenAI-compatible endpoint, `None` otherwise.
///
/// Named providers with their own rig client (`openai`, `anthropic`,
/// `openrouter`, `ollama`, `deepseek`, `groq`, ...) are intentionally not
/// listed here - their defaults live in the rig provider crate constants.
pub fn default_base_url(provider_type: &str) -> Option<&'static str> {
    match provider_type {
        // International
        "github_models" => Some("https://models.inference.ai.azure.com"),
        "fireworks" => Some("https://api.fireworks.ai/inference/v1"),
        "nvidia" => Some("https://integrate.api.nvidia.com/v1"),
        "huggingface" => Some("https://api-inference.huggingface.co/v1"),
        "cerebras" => Some("https://api.cerebras.ai/v1"),
        "lmstudio" => Some("http://localhost:1234/v1"),
        "gpustack" => Some("http://localhost:80/v1"),
        "ovms" => Some("http://localhost:8000/v1"),
        // Chinese AI
        "zhipu" => Some("https://open.bigmodel.cn/api/paas/v4"),
        "yi" => Some("https://api.lingyiwanwu.com/v1"),
        "baichuan" => Some("https://api.baichuan-ai.com/v1"),
        "dashscope" => Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "stepfun" => Some("https://api.stepfun.com/v1"),
        "doubao" => Some("https://ark.cn-beijing.volces.com/api/v3"),
        "hunyuan" => Some("https://api.hunyuan.cloud.tencent.com/v1"),
        "tencent_cloud_ti" => Some("https://api.lkeap.cloud.tencent.com/v1"),
        "baidu_cloud" => Some("https://qianfan.baidubce.com/v2"),
        "siliconflow" => Some("https://api.siliconflow.cn/v1"),
        "modelscope" => Some("https://api-inference.modelscope.cn/v1"),
        "xirang" => Some("https://wishub-x1.ctyun.cn/v1"),
        "mimo" => Some("https://api.xiaomimimo.com/v1"),
        _ => None,
    }
}

/// Providers that reject structured content arrays and require user message
/// content serialized as a plain string. Callers building an
/// `openai_compat::CompletionModel` for these providers must apply
/// [`with_string_content_only`](crate::openai_compat::CompletionModel::with_string_content_only).
pub fn is_string_content_only(provider_type: &str) -> bool {
    matches!(
        provider_type,
        "deepseek" | "baichuan" | "minimax" | "minimax_cn" | "xirang"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_all_providers_resolve_urls() {
        assert_eq!(
            default_base_url("github_models"),
            Some("https://models.inference.ai.azure.com")
        );
        assert_eq!(
            default_base_url("zhipu"),
            Some("https://open.bigmodel.cn/api/paas/v4")
        );
    }

    #[test]
    fn named_providers_have_no_catch_all_url() {
        // Named providers keep their defaults in rig provider crate constants.
        assert_eq!(default_base_url("openai"), None);
        assert_eq!(default_base_url("anthropic"), None);
        assert_eq!(default_base_url("deepseek"), None);
    }

    #[test]
    fn unknown_providers_return_none() {
        assert_eq!(default_base_url("not_a_real_provider"), None);
    }

    #[test]
    fn string_content_only_providers_flagged() {
        assert!(is_string_content_only("deepseek"));
        assert!(is_string_content_only("baichuan"));
        assert!(is_string_content_only("minimax"));
        assert!(is_string_content_only("minimax_cn"));
        assert!(is_string_content_only("xirang"));
    }

    #[test]
    fn array_capable_providers_not_flagged() {
        assert!(!is_string_content_only("openai"));
        assert!(!is_string_content_only("groq"));
        assert!(!is_string_content_only("github_models"));
        assert!(!is_string_content_only("zhipu"));
    }
}
