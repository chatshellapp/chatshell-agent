//! MCP prompts (FR-13): templates a server offers for the user to insert
//! into the composer. Never injected into the system prompt; the user
//! reviews the rendered text before sending.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use rmcp::RoleClient;
use rmcp::model::{
    ContentBlock, GetPromptRequestParams, GetPromptResult, PaginatedRequestParams, Prompt,
    ResourceContents, Role,
};
use rmcp::service::Peer;
use serde::{Deserialize, Serialize};

use super::listing::paginate;
use crate::path_policy;

/// Maximum characters of a rendered prompt inserted into the composer.
pub const MAX_PROMPT_CHARS: usize = 64_000;

/// One argument a prompt takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub required: bool,
}

/// A prompt a server offers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptInfo {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub arguments: Vec<McpPromptArgument>,
}

impl From<Prompt> for McpPromptInfo {
    fn from(p: Prompt) -> Self {
        Self {
            name: p.name,
            title: p.title,
            description: p.description,
            arguments: p
                .arguments
                .unwrap_or_default()
                .into_iter()
                .map(|a| McpPromptArgument {
                    name: a.name,
                    title: a.title,
                    description: a.description,
                    required: a.required.unwrap_or(false),
                })
                .collect(),
        }
    }
}

/// Whether the server advertised the `prompts` capability.
pub fn supports_prompts(peer: &Peer<RoleClient>) -> bool {
    peer.peer_info()
        .is_some_and(|info| info.capabilities.prompts.is_some())
}

/// List a server's prompts with the shared listing guards (INV-3). Empty
/// when the server does not offer prompts.
pub async fn list_all_prompts_bounded(
    peer: &Peer<RoleClient>,
    server: &str,
) -> Result<Vec<McpPromptInfo>> {
    if !supports_prompts(peer) {
        return Ok(Vec::new());
    }
    let result = paginate(|cursor| async move {
        let page = peer
            .list_prompts(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .context("Failed to list prompts from MCP server")?;
        Ok((page.prompts, page.next_cursor))
    })
    .await?;
    if let Some(reason) = result.truncated {
        tracing::warn!("MCP server '{server}' prompt listing stopped early ({reason:?})");
    }
    Ok(result.items.into_iter().map(Into::into).collect())
}

/// INV-1: scan user-entered prompt arguments like MCP tool arguments.
pub fn scan_prompt_arguments(arguments: &BTreeMap<String, String>) -> Result<(), String> {
    for value in arguments.values() {
        path_policy::check_blocked_path(value)?;
    }
    Ok(())
}

fn block_text(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(t) => Some(t.text.clone()),
        ContentBlock::Resource(r) => match &r.resource {
            ResourceContents::TextResourceContents { text, .. } => Some(text.clone()),
            ResourceContents::BlobResourceContents { uri, .. } => Some(format!("[{uri}]")),
            _ => None,
        },
        ContentBlock::ResourceLink(link) => Some(link.uri.clone()),
        ContentBlock::Image(_) => Some("[image]".to_string()),
        ContentBlock::Audio(_) => Some("[audio]".to_string()),
        _ => None,
    }
}

/// Render a prompt for the composer: each message's text, in order. A
/// conversation-shaped prompt (with assistant turns) keeps role labels so
/// the user sees what the server wrote for which side.
pub fn render_prompt(result: &GetPromptResult) -> String {
    let labelled = result
        .messages
        .iter()
        .any(|m| matches!(m.role, Role::Assistant));
    let parts: Vec<String> = result
        .messages
        .iter()
        .filter_map(|m| {
            let text = block_text(&m.content)?;
            Some(if labelled {
                let role = match m.role {
                    Role::Assistant => "Assistant",
                    _ => "User",
                };
                format!("{role}: {text}")
            } else {
                text
            })
        })
        .collect();
    let mut text = parts.join("\n\n");
    if text.len() > MAX_PROMPT_CHARS {
        let mut end = MAX_PROMPT_CHARS;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// Fetch and render one prompt. Arguments are scanned first.
pub async fn get_prompt_text(
    peer: &Peer<RoleClient>,
    name: &str,
    arguments: BTreeMap<String, String>,
) -> Result<String> {
    scan_prompt_arguments(&arguments).map_err(|reason| anyhow::anyhow!("Blocked: {reason}"))?;
    let mut params = GetPromptRequestParams::new(name);
    if !arguments.is_empty() {
        params = params.with_arguments(
            arguments
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect(),
        );
    }
    let result = peer
        .get_prompt(params)
        .await
        .context("Failed to get prompt from MCP server")?;
    Ok(render_prompt(&result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn result(value: serde_json::Value) -> GetPromptResult {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn user_only_prompts_render_plain() {
        let r = result(json!({"messages": [
            {"role": "user", "content": {"type": "text", "text": "Review this diff"}},
            {"role": "user", "content": {"type": "resource", "resource": {
                "uri": "file:///a.diff", "text": "+x"}}}
        ]}));
        assert_eq!(render_prompt(&r), "Review this diff\n\n+x");
    }

    #[test]
    fn conversations_keep_role_labels() {
        let r = result(json!({"messages": [
            {"role": "user", "content": {"type": "text", "text": "Hi"}},
            {"role": "assistant", "content": {"type": "text", "text": "Hello"}}
        ]}));
        assert_eq!(render_prompt(&r), "User: Hi\n\nAssistant: Hello");
    }

    #[test]
    fn long_prompts_are_capped() {
        let r = result(json!({"messages": [
            {"role": "user", "content": {"type": "text", "text": "x".repeat(MAX_PROMPT_CHARS + 5)}}
        ]}));
        assert_eq!(render_prompt(&r).len(), MAX_PROMPT_CHARS);
    }

    #[test]
    fn blocked_paths_in_arguments_are_refused() {
        let mut args = BTreeMap::new();
        args.insert("file".to_string(), "~/.ssh/id_rsa".to_string());
        assert!(scan_prompt_arguments(&args).is_err());
        args.insert("file".to_string(), "README.md".to_string());
        assert!(scan_prompt_arguments(&args).is_ok());
    }

    #[test]
    fn prompt_info_maps_arguments() {
        let p: Prompt = serde_json::from_value(json!({
            "name": "review", "title": "Code review",
            "arguments": [{"name": "pr", "required": true}, {"name": "tone"}]
        }))
        .unwrap();
        let info = McpPromptInfo::from(p);
        assert_eq!(info.title.as_deref(), Some("Code review"));
        assert!(info.arguments[0].required);
        assert!(!info.arguments[1].required);
    }
}
