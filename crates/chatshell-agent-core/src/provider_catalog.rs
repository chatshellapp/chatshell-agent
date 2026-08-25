//! Provider routing tables shared across desktop and iOS.
//!
//! This module is the single source of truth for provider-type-derived routing
//! data. [`resolve`] answers "how do I talk to this provider" with a
//! [`ProviderRoute`] covering every provider type both clients know:
//!
//! - which constructor family ([`Wire`]) builds the client,
//! - which core [`ProviderKind`] the provider maps to,
//! - the default base URL (where one exists),
//! - whether an API key is required and its display name for errors,
//! - whether the provider requires string-only message content,
//! - the default OpenAI API surface (`responses` vs `chat_completions`).
//!
//! Desktop consumes this table from `llm/agent_builder.rs` to dispatch
//! construction; iOS consumes it through the `resolve_provider_route` FFI
//! lookup exported by `chatshell-agent-ffi`.
//!
//! The legacy helpers [`default_base_url`] and [`is_string_content_only`]
//! remain as thin queries over the table.

use crate::types::ProviderKind;

/// Which client-construction family a provider routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    /// rig native OpenAI client - Responses API.
    Responses,
    /// `openai_compat::CompletionModel` - Chat Completions API.
    OpenAiCompat,
    /// rig native Anthropic Messages client.
    Anthropic,
    /// rig native OpenRouter client.
    OpenRouter,
    /// rig native Ollama client (no API key).
    Ollama,
    /// Azure OpenAI (endpoint + deployment auth). Constructor is desktop-side.
    Azure,
    /// Google Gemini native API. Constructor is desktop-side.
    Gemini,
    /// rig native per-provider clients (cohere/deepseek/groq/...).
    /// Constructors are desktop-side; URLs stay in the rig provider crates.
    RigNative,
}

/// Routing data for one provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderRoute {
    /// Constructor family this provider belongs to.
    pub wire: Wire,
    /// The core [`ProviderKind`] this provider maps onto. `Some` only when
    /// core itself can construct the client (`Responses` / `OpenAiCompat` /
    /// `Anthropic` / `OpenRouter` / `Ollama` wires). Desktop-only wires
    /// (`Azure` / `Gemini` / `RigNative`) carry `None`: core cannot route
    /// them, so clients construct locally or apply their own fallback.
    pub kind: Option<ProviderKind>,
    /// Default API endpoint. `None` when the caller must supply one
    /// (custom providers) or when the default lives in the rig provider
    /// crate constant (named `RigNative` / `OpenAiCompat` providers).
    pub default_base_url: Option<&'static str>,
    /// Whether an API key is required.
    pub key_required: bool,
    /// Display name for the "X API key required" error. `None` falls back
    /// to the raw provider type string.
    pub key_display: Option<&'static str>,
    /// Local endpoints accept a missing key (placeholder `no-key`).
    pub local_no_key: bool,
    /// Whether user message content must be serialized as a plain string
    /// (see [`crate::openai_compat::CompletionModel::with_string_content_only`]).
    pub string_content_only: bool,
    /// Default OpenAI API surface. `Some("responses")` or
    /// `Some("chat_completions")` for OpenAI-wire providers, `None` otherwise.
    pub default_api_style: Option<&'static str>,
}

const fn route(wire: Wire) -> ProviderRoute {
    ProviderRoute {
        wire,
        kind: None,
        default_base_url: None,
        key_required: true,
        key_display: None,
        local_no_key: false,
        string_content_only: false,
        default_api_style: None,
    }
}

/// Named-provider rows. Default URLs here were relocated from the literals
/// previously hardcoded in `agent.rs::create_agent`; `RigNative` and named
/// `OpenAiCompat` providers keep their URLs in the rig provider crates and
/// therefore carry `None`.
const NAMED_ROUTES: &[(&str, ProviderRoute)] = &[
    (
        "openai",
        ProviderRoute {
            kind: Some(ProviderKind::OpenAi),
            default_base_url: Some("https://api.openai.com/v1"),
            key_display: Some("OpenAI"),
            default_api_style: Some("responses"),
            ..route(Wire::Responses)
        },
    ),
    (
        "custom_openai",
        ProviderRoute {
            kind: Some(ProviderKind::CustomOpenAi),
            key_display: Some("Custom OpenAI-compatible"),
            default_api_style: Some("responses"),
            ..route(Wire::Responses)
        },
    ),
    (
        "anthropic",
        ProviderRoute {
            kind: Some(ProviderKind::Anthropic),
            default_base_url: Some("https://api.anthropic.com/v1"),
            key_display: Some("Anthropic"),
            ..route(Wire::Anthropic)
        },
    ),
    (
        "custom_anthropic",
        ProviderRoute {
            kind: Some(ProviderKind::Anthropic),
            key_display: Some("Custom Anthropic-compatible"),
            ..route(Wire::Anthropic)
        },
    ),
    (
        "openrouter",
        ProviderRoute {
            kind: Some(ProviderKind::OpenRouter),
            default_base_url: Some("https://openrouter.ai/api/v1"),
            key_display: Some("OpenRouter"),
            ..route(Wire::OpenRouter)
        },
    ),
    (
        "ollama",
        ProviderRoute {
            kind: Some(ProviderKind::Ollama),
            default_base_url: Some("http://127.0.0.1:11434"),
            key_required: false,
            ..route(Wire::Ollama)
        },
    ),
    (
        "azure",
        ProviderRoute {
            key_display: Some("Azure OpenAI"),
            ..route(Wire::Azure)
        },
    ),
    (
        "gemini",
        ProviderRoute {
            key_display: Some("Google Gemini"),
            ..route(Wire::Gemini)
        },
    ),
    // openai_compat-family named providers (chat completions wire).
    (
        "galadriel",
        ProviderRoute {
            kind: Some(ProviderKind::CustomOpenAi),
            key_display: Some("Galadriel"),
            default_api_style: Some("chat_completions"),
            ..route(Wire::OpenAiCompat)
        },
    ),
    (
        "minimax",
        ProviderRoute {
            kind: Some(ProviderKind::CustomOpenAi),
            key_display: Some("MiniMax"),
            string_content_only: true,
            default_api_style: Some("chat_completions"),
            ..route(Wire::OpenAiCompat)
        },
    ),
    (
        "minimax_cn",
        ProviderRoute {
            kind: Some(ProviderKind::CustomOpenAi),
            key_display: Some("MiniMax CN"),
            string_content_only: true,
            default_api_style: Some("chat_completions"),
            ..route(Wire::OpenAiCompat)
        },
    ),
    // rig native per-provider clients.
    (
        "cohere",
        ProviderRoute {
            key_display: Some("Cohere"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "deepseek",
        ProviderRoute {
            key_display: Some("DeepSeek"),
            string_content_only: true,
            ..route(Wire::RigNative)
        },
    ),
    (
        "groq",
        ProviderRoute {
            key_display: Some("Groq"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "hyperbolic",
        ProviderRoute {
            key_display: Some("Hyperbolic"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "mira",
        ProviderRoute {
            key_display: Some("Mira"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "mistral",
        ProviderRoute {
            key_display: Some("Mistral"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "moonshot",
        ProviderRoute {
            key_display: Some("Moonshot"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "perplexity",
        ProviderRoute {
            key_display: Some("Perplexity"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "together",
        ProviderRoute {
            key_display: Some("Together AI"),
            ..route(Wire::RigNative)
        },
    ),
    (
        "xai",
        ProviderRoute {
            key_display: Some("xAI"),
            ..route(Wire::RigNative)
        },
    ),
];

/// Catch-all OpenAI-compatible providers routed through
/// `openai_compat::CompletionModel`. Values are unchanged from the previous
/// inline desktop table (`openai_compat_default_url`).
const CATCH_ALL_ROUTES: &[(&str, &str)] = &[
    // International
    ("github_models", "https://models.inference.ai.azure.com"),
    ("fireworks", "https://api.fireworks.ai/inference/v1"),
    ("nvidia", "https://integrate.api.nvidia.com/v1"),
    ("huggingface", "https://api-inference.huggingface.co/v1"),
    ("cerebras", "https://api.cerebras.ai/v1"),
    ("lmstudio", "http://localhost:1234/v1"),
    ("gpustack", "http://localhost:80/v1"),
    ("ovms", "http://localhost:8000/v1"),
    // Chinese AI
    ("zhipu", "https://open.bigmodel.cn/api/paas/v4"),
    ("yi", "https://api.lingyiwanwu.com/v1"),
    ("baichuan", "https://api.baichuan-ai.com/v1"),
    (
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    ("stepfun", "https://api.stepfun.com/v1"),
    ("doubao", "https://ark.cn-beijing.volces.com/api/v3"),
    ("hunyuan", "https://api.hunyuan.cloud.tencent.com/v1"),
    ("tencent_cloud_ti", "https://api.lkeap.cloud.tencent.com/v1"),
    ("baidu_cloud", "https://qianfan.baidubce.com/v2"),
    ("siliconflow", "https://api.siliconflow.cn/v1"),
    ("modelscope", "https://api-inference.modelscope.cn/v1"),
    ("xirang", "https://wishub-x1.ctyun.cn/v1"),
    ("mimo", "https://api.xiaomimimo.com/v1"),
];

/// Local catch-all endpoints that run without an API key.
const LOCAL_NO_KEY: &[&str] = &["lmstudio", "gpustack", "ovms"];

/// Resolve routing data for a provider type. Returns `None` for unknown
/// provider types.
pub fn resolve(provider_type: &str) -> Option<ProviderRoute> {
    if let Some((_, route)) = NAMED_ROUTES.iter().find(|(name, _)| *name == provider_type) {
        return Some(*route);
    }
    let (url, is_local) = CATCH_ALL_ROUTES
        .iter()
        .find(|(name, _)| *name == provider_type)
        .map(|(_, url)| (*url, LOCAL_NO_KEY.contains(&provider_type)))?;
    Some(ProviderRoute {
        kind: Some(ProviderKind::CustomOpenAi),
        default_base_url: Some(url),
        key_required: !is_local,
        local_no_key: is_local,
        string_content_only: is_string_content_only(provider_type),
        default_api_style: Some("chat_completions"),
        ..route(Wire::OpenAiCompat)
    })
}

/// Default base URL for a provider type, if one is known here. Named
/// providers whose defaults live in rig provider crate constants
/// (`cohere`, `deepseek`, `groq`, ...) return `None`.
pub fn default_base_url(provider_type: &str) -> Option<&'static str> {
    resolve(provider_type)?.default_base_url
}

/// Whether a provider requires user message content serialized as a plain
/// string (rejects structured content arrays).
pub fn is_string_content_only(provider_type: &str) -> bool {
    matches!(
        provider_type,
        "deepseek" | "baichuan" | "minimax" | "minimax_cn" | "xirang"
    )
}

/// Default base URL for a core [`ProviderKind`], sourcing the named rows.
/// Used by `agent.rs::create_agent` so URL knowledge lives only in this table.
pub fn default_base_url_for_kind(kind: ProviderKind) -> Option<&'static str> {
    let name = match kind {
        ProviderKind::OpenAi => "openai",
        ProviderKind::Anthropic => "anthropic",
        ProviderKind::OpenRouter => "openrouter",
        ProviderKind::Ollama => "ollama",
        ProviderKind::CustomOpenAi => return None,
    };
    default_base_url(name)
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
        assert_eq!(
            default_base_url("lmstudio"),
            Some("http://localhost:1234/v1")
        );
    }

    #[test]
    fn named_providers_have_no_catch_all_url() {
        // Named providers whose defaults live in rig crates.
        assert_eq!(default_base_url("cohere"), None);
        assert_eq!(default_base_url("deepseek"), None);
        assert_eq!(default_base_url("galadriel"), None);
    }

    #[test]
    fn unknown_providers_return_none() {
        assert_eq!(default_base_url("not_a_real_provider"), None);
        assert!(resolve("not_a_real_provider").is_none());
    }

    #[test]
    fn string_content_only_providers_flagged() {
        for provider in ["deepseek", "baichuan", "minimax", "minimax_cn", "xirang"] {
            assert!(is_string_content_only(provider), "{provider}");
            assert!(resolve(provider).unwrap().string_content_only, "{provider}");
        }
    }

    #[test]
    fn array_capable_providers_not_flagged() {
        assert!(!is_string_content_only("openai"));
        assert!(!is_string_content_only("groq"));
    }

    #[test]
    fn named_routes_golden() {
        let openai = resolve("openai").unwrap();
        assert_eq!(openai.wire, Wire::Responses);
        assert_eq!(openai.kind, Some(ProviderKind::OpenAi));
        assert_eq!(openai.default_base_url, Some("https://api.openai.com/v1"));
        assert_eq!(openai.key_display, Some("OpenAI"));
        assert_eq!(openai.default_api_style, Some("responses"));
        assert!(!openai.string_content_only);

        let anthropic = resolve("anthropic").unwrap();
        assert_eq!(anthropic.wire, Wire::Anthropic);
        assert_eq!(
            anthropic.default_base_url,
            Some("https://api.anthropic.com/v1")
        );
        assert_eq!(anthropic.default_api_style, None);

        let openrouter = resolve("openrouter").unwrap();
        assert_eq!(openrouter.wire, Wire::OpenRouter);
        assert_eq!(
            openrouter.default_base_url,
            Some("https://openrouter.ai/api/v1")
        );

        let ollama = resolve("ollama").unwrap();
        assert_eq!(ollama.wire, Wire::Ollama);
        assert!(!ollama.key_required);
        assert_eq!(ollama.default_base_url, Some("http://127.0.0.1:11434"));

        let custom = resolve("custom_openai").unwrap();
        assert_eq!(custom.wire, Wire::Responses);
        assert_eq!(custom.kind, Some(ProviderKind::CustomOpenAi));
        assert_eq!(custom.default_base_url, None);
        assert_eq!(custom.default_api_style, Some("responses"));

        let azure = resolve("azure").unwrap();
        assert_eq!(azure.wire, Wire::Azure);
        assert_eq!(azure.default_base_url, None);
        assert_eq!(azure.key_display, Some("Azure OpenAI"));
        assert_eq!(azure.kind, None);

        let gemini = resolve("gemini").unwrap();
        assert_eq!(gemini.wire, Wire::Gemini);
        assert_eq!(gemini.kind, None);
        let groq = resolve("groq").unwrap();
        assert_eq!(groq.wire, Wire::RigNative);
        assert_eq!(groq.default_base_url, None);
        assert_eq!(groq.key_display, Some("Groq"));
        assert_eq!(groq.kind, None);

        let minimax = resolve("minimax").unwrap();
        assert_eq!(minimax.wire, Wire::OpenAiCompat);
        assert_eq!(minimax.default_api_style, Some("chat_completions"));
        assert!(minimax.string_content_only);
    }

    #[test]
    fn catch_all_routes_golden() {
        let zhipu = resolve("zhipu").unwrap();
        assert_eq!(zhipu.wire, Wire::OpenAiCompat);
        assert_eq!(zhipu.kind, Some(ProviderKind::CustomOpenAi));
        assert!(zhipu.key_required);
        assert_eq!(zhipu.key_display, None);
        assert_eq!(zhipu.default_api_style, Some("chat_completions"));

        let lmstudio = resolve("lmstudio").unwrap();
        assert!(!lmstudio.key_required);
        assert!(lmstudio.local_no_key);
        assert_eq!(lmstudio.default_base_url, Some("http://localhost:1234/v1"));

        let baichuan = resolve("baichuan").unwrap();
        assert!(baichuan.string_content_only);
    }

    #[test]
    fn kind_default_urls_match_named_rows() {
        assert_eq!(
            default_base_url_for_kind(ProviderKind::OpenAi),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            default_base_url_for_kind(ProviderKind::Ollama),
            Some("http://127.0.0.1:11434")
        );
        assert_eq!(default_base_url_for_kind(ProviderKind::CustomOpenAi), None);
    }

    #[test]
    fn every_table_entry_resolves_consistently() {
        for (name, route) in NAMED_ROUTES {
            assert_eq!(&resolve(name).unwrap(), route, "{name}");
        }
        for (name, url) in CATCH_ALL_ROUTES {
            assert_eq!(
                resolve(name).unwrap().default_base_url,
                Some(*url),
                "{name}"
            );
        }
    }
}
