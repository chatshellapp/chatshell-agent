//! Built-in provider registry - the single source of truth for provider
//! templates and capability flags shared across desktop and iOS.
//!
//! Both clients previously maintained separate hardcoded lists:
//! - Desktop: `src/components/provider-settings-dialog/constants.ts`
//!   (`BUILTIN_PROVIDERS`, `FETCH_SUPPORTED_PROVIDERS`)
//! - iOS: `Sources/ChatShellCore/Models/ProviderCatalog.swift`
//!   (`ProviderCatalog.builtin`, `fetchSupportedBuiltins`)
//!
//! This module replaces both. New providers are added here once and appear
//! on every platform automatically.

use serde::{Deserialize, Serialize};

/// A built-in provider template shown in the provider picker UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuiltinProvider {
    /// Stable identifier stored in `providers.provider_type`
    /// (e.g. `"openai"`, `"anthropic"`, `"ollama"`).
    pub id: String,
    /// Human-readable name shown in the UI.
    pub name: String,
    /// Default API base URL. Empty string for providers that require a
    /// user-supplied endpoint (e.g. Azure OpenAI).
    pub default_base_url: String,
    /// Whether this provider requires an API key. Local servers
    /// (Ollama, LM Studio, GPUStack, OVMS) do not.
    pub requires_api_key: bool,
}

/// The sentinel row for creating a custom provider endpoint.
pub fn custom_provider() -> BuiltinProvider {
    BuiltinProvider {
        id: "custom".to_string(),
        name: "Custom Provider".to_string(),
        default_base_url: String::new(),
        requires_api_key: true,
    }
}

/// Whether `provider_type` names a built-in provider template. Built-in
/// provider rows are singletons by type: every device that adds "OpenAI"
/// must produce the SAME row id, or cross-device merges duplicate the
/// provider and the type-based singleton lookups silently pick the wrong
/// twin (real-device finding: two MiniMax CN rows made model switching
/// appear broken).
pub fn is_builtin_provider_type(provider_type: &str) -> bool {
    provider_type != "custom" && builtin_providers().iter().any(|p| p.id == provider_type)
}
/// All built-in provider templates, ordered as they should appear in the UI.
pub fn builtin_providers() -> Vec<BuiltinProvider> {
    use BuiltinProvider as P;
    vec![
        P {
            id: "openai".into(),
            name: "OpenAI".into(),
            default_base_url: "https://api.openai.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "anthropic".into(),
            name: "Anthropic".into(),
            default_base_url: "https://api.anthropic.com".into(),
            requires_api_key: true,
        },
        P {
            id: "gemini".into(),
            name: "Google Gemini".into(),
            default_base_url: "https://generativelanguage.googleapis.com".into(),
            requires_api_key: true,
        },
        P {
            id: "openrouter".into(),
            name: "OpenRouter".into(),
            default_base_url: "https://openrouter.ai/api/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "azure".into(),
            name: "Azure OpenAI".into(),
            default_base_url: String::new(),
            requires_api_key: true,
        },
        P {
            id: "deepseek".into(),
            name: "DeepSeek".into(),
            default_base_url: "https://api.deepseek.com".into(),
            requires_api_key: true,
        },
        P {
            id: "groq".into(),
            name: "Groq".into(),
            default_base_url: "https://api.groq.com/openai/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "mistral".into(),
            name: "Mistral".into(),
            default_base_url: "https://api.mistral.ai".into(),
            requires_api_key: true,
        },
        P {
            id: "perplexity".into(),
            name: "Perplexity".into(),
            default_base_url: "https://api.perplexity.ai".into(),
            requires_api_key: true,
        },
        P {
            id: "together".into(),
            name: "Together AI".into(),
            default_base_url: "https://api.together.xyz".into(),
            requires_api_key: true,
        },
        P {
            id: "xai".into(),
            name: "xAI".into(),
            default_base_url: "https://api.x.ai".into(),
            requires_api_key: true,
        },
        P {
            id: "cohere".into(),
            name: "Cohere".into(),
            default_base_url: "https://api.cohere.ai".into(),
            requires_api_key: true,
        },
        P {
            id: "moonshot".into(),
            name: "Moonshot".into(),
            default_base_url: "https://api.moonshot.cn/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "hyperbolic".into(),
            name: "Hyperbolic".into(),
            default_base_url: "https://api.hyperbolic.xyz".into(),
            requires_api_key: true,
        },
        P {
            id: "galadriel".into(),
            name: "Galadriel".into(),
            default_base_url: "https://api.galadriel.com/v1/verified".into(),
            requires_api_key: true,
        },
        P {
            id: "minimax".into(),
            name: "MiniMax".into(),
            default_base_url: "https://api.minimax.io/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "minimax_cn".into(),
            name: "MiniMax CN".into(),
            default_base_url: "https://api.minimaxi.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "mira".into(),
            name: "Mira".into(),
            default_base_url: "https://api.mira.network".into(),
            requires_api_key: true,
        },
        // International
        P {
            id: "github_models".into(),
            name: "GitHub Models".into(),
            default_base_url: "https://models.inference.ai.azure.com".into(),
            requires_api_key: true,
        },
        P {
            id: "fireworks".into(),
            name: "Fireworks AI".into(),
            default_base_url: "https://api.fireworks.ai/inference/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "nvidia".into(),
            name: "NVIDIA NIM".into(),
            default_base_url: "https://integrate.api.nvidia.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "huggingface".into(),
            name: "Hugging Face".into(),
            default_base_url: "https://api-inference.huggingface.co/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "cerebras".into(),
            name: "Cerebras".into(),
            default_base_url: "https://api.cerebras.ai/v1".into(),
            requires_api_key: true,
        },
        // Chinese AI
        P {
            id: "zhipu".into(),
            name: "Zhipu AI".into(),
            default_base_url: "https://open.bigmodel.cn/api/paas/v4".into(),
            requires_api_key: true,
        },
        P {
            id: "yi".into(),
            name: "01.AI".into(),
            default_base_url: "https://api.lingyiwanwu.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "baichuan".into(),
            name: "Baichuan".into(),
            default_base_url: "https://api.baichuan-ai.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "dashscope".into(),
            name: "Alibaba Qwen".into(),
            default_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "stepfun".into(),
            name: "StepFun".into(),
            default_base_url: "https://api.stepfun.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "doubao".into(),
            name: "Doubao".into(),
            default_base_url: "https://ark.cn-beijing.volces.com/api/v3".into(),
            requires_api_key: true,
        },
        P {
            id: "hunyuan".into(),
            name: "Tencent Hunyuan".into(),
            default_base_url: "https://api.hunyuan.cloud.tencent.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "tencent_cloud_ti".into(),
            name: "Tencent Cloud TI".into(),
            default_base_url: "https://api.lkeap.cloud.tencent.com/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "baidu_cloud".into(),
            name: "Baidu Cloud".into(),
            default_base_url: "https://qianfan.baidubce.com/v2".into(),
            requires_api_key: true,
        },
        P {
            id: "siliconflow".into(),
            name: "SiliconFlow".into(),
            default_base_url: "https://api.siliconflow.cn/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "modelscope".into(),
            name: "ModelScope".into(),
            default_base_url: "https://api-inference.modelscope.cn/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "xirang".into(),
            name: "Xirang".into(),
            default_base_url: "https://wishub-x1.ctyun.cn/v1".into(),
            requires_api_key: true,
        },
        P {
            id: "mimo".into(),
            name: "Xiaomi MiMo".into(),
            default_base_url: "https://api.xiaomimimo.com/v1".into(),
            requires_api_key: true,
        },
        // Local
        P {
            id: "ollama".into(),
            name: "Ollama".into(),
            default_base_url: "http://127.0.0.1:11434".into(),
            requires_api_key: false,
        },
        P {
            id: "lmstudio".into(),
            name: "LM Studio".into(),
            default_base_url: "http://127.0.0.1:1234/v1".into(),
            requires_api_key: false,
        },
        P {
            id: "gpustack".into(),
            name: "GPUStack".into(),
            default_base_url: "http://127.0.0.1:80/v1".into(),
            requires_api_key: false,
        },
        P {
            id: "ovms".into(),
            name: "OVMS".into(),
            default_base_url: "http://127.0.0.1:8000/v1".into(),
            requires_api_key: false,
        },
    ]
}

/// Provider types that support auto-fetching models via API.
///
/// Custom OpenAI-compatible providers (`custom`, `custom_openai`) are
/// included; `custom_anthropic` is not (Anthropic has no list-models API).
/// Azure and Gemini are excluded (different auth flow / SDK).
pub fn is_supported_fetch_provider(provider_type: &str) -> bool {
    matches!(
        provider_type,
        "openai"
            | "openrouter"
            | "ollama"
            | "deepseek"
            | "groq"
            | "together"
            | "xai"
            | "moonshot"
            | "perplexity"
            | "hyperbolic"
            | "mistral"
            | "minimax"
            | "minimax_cn"
            | "mira"
            | "galadriel"
            | "cohere"
            | "custom"
            | "custom_openai"
            | "github_models"
            | "fireworks"
            | "nvidia"
            | "huggingface"
            | "cerebras"
            | "zhipu"
            | "yi"
            | "baichuan"
            | "dashscope"
            | "stepfun"
            | "doubao"
            | "hunyuan"
            | "tencent_cloud_ti"
            | "baidu_cloud"
            | "siliconflow"
            | "modelscope"
            | "xirang"
            | "mimo"
            | "lmstudio"
            | "gpustack"
            | "ovms"
    )
}

/// Whether fetching models for this provider requires an API key.
/// Local servers (Ollama, LM Studio, GPUStack, OVMS) do not.
pub fn requires_api_key_for_model_fetch(provider_type: &str) -> bool {
    !matches!(provider_type, "ollama" | "lmstudio" | "gpustack" | "ovms")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_list_includes_all_providers() {
        let providers = builtin_providers();
        assert!(providers.len() >= 38);
        assert!(providers.iter().any(|p| p.id == "openai"));
        assert!(providers.iter().any(|p| p.id == "ollama"));
        assert!(providers.iter().any(|p| p.id == "ovms"));
    }

    #[test]
    fn local_providers_dont_require_api_key() {
        let providers = builtin_providers();
        for p in &providers {
            let is_local = matches!(p.id.as_str(), "ollama" | "lmstudio" | "gpustack" | "ovms");
            assert_eq!(p.requires_api_key, !is_local, "mismatch for {}", p.id);
        }
    }

    #[test]
    fn fetch_support_excludes_azure_and_gemini() {
        assert!(!is_supported_fetch_provider("azure"));
        assert!(!is_supported_fetch_provider("gemini"));
        assert!(!is_supported_fetch_provider("custom_anthropic"));
    }

    #[test]
    fn fetch_support_includes_custom_openai() {
        assert!(is_supported_fetch_provider("custom"));
        assert!(is_supported_fetch_provider("custom_openai"));
    }

    #[test]
    fn api_key_not_required_for_local() {
        assert!(!requires_api_key_for_model_fetch("ollama"));
        assert!(!requires_api_key_for_model_fetch("lmstudio"));
        assert!(requires_api_key_for_model_fetch("openai"));
    }

    #[test]
    fn custom_provider_sentinel() {
        let c = custom_provider();
        assert_eq!(c.id, "custom");
        assert!(c.default_base_url.is_empty());
        assert!(c.requires_api_key);
    }
}
