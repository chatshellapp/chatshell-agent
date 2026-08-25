//! Built-in tool definitions shared across desktop and iOS.
//!
//! The `web_search` and `web_fetch` tool schemas were previously duplicated:
//! - Desktop: `src-tauri/src/llm/tools/web_search.rs` / `web_fetch.rs`
//!   (rig `ToolDefinition` with inline JSON schema)
//! - iOS: `Sources/ChatShellCore/Services/Tools/ToolRegistry.swift`
//!   (`BuiltinTools.webSearch` / `webFetch` with hand-written JSON strings)
//!
//! This module is the single source of truth for the schemas sent to the LLM.
//! The schemas use the richer desktop variant (includes `max_results`,
//! `provider`, `max_chars` parameters) - iOS executors already ignore
//! parameters they don't handle.

use crate::types::ClientToolDefinition;

/// Tool definitions for the built-in `web_search` and `web_fetch` tools.
/// Pass these to `AgentRequest.tools` to enable client-side tool calling.
pub fn builtin_tool_definitions() -> Vec<ClientToolDefinition> {
    vec![web_search_definition(), web_fetch_definition()]
}

pub fn web_search_definition() -> ClientToolDefinition {
    ClientToolDefinition {
        name: "web_search".to_string(),
        description: "Search the web for information using a search engine. \
            Returns a list of relevant search results with titles, URLs, and snippets. \
            Use this when you need to find current information, facts, or references from the web."
            .to_string(),
        parameters_json: serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute"
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results to return (default: 5)"
                },
                "provider": {
                    "type": "string",
                    "enum": ["duckduckgo", "yahoo", "baidu"],
                    "description": "Search provider to use (default: duckduckgo)"
                }
            },
            "required": ["query"]
        })
        .to_string(),
    }
}

pub fn web_fetch_definition() -> ClientToolDefinition {
    ClientToolDefinition {
        name: "web_fetch".to_string(),
        description: "Fetch and extract the main content from a web page. \
            Returns the page title and cleaned text content suitable for reading. \
            Use this to read articles, documentation, blog posts, or any web page content. \
            The content is extracted using readability algorithms to remove ads and navigation."
            .to_string(),
        parameters_json: serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to fetch"
                },
                "max_chars": {
                    "type": "number",
                    "description": "Maximum characters to return (default: 50000). Use smaller values for quick summaries."
                }
            },
            "required": ["url"]
        })
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_definitions_contain_web_search_and_fetch() {
        let defs = builtin_tool_definitions();
        assert_eq!(defs.len(), 2);
        assert!(defs.iter().any(|d| d.name == "web_search"));
        assert!(defs.iter().any(|d| d.name == "web_fetch"));
    }

    #[test]
    fn web_search_schema_requires_query() {
        let def = web_search_definition();
        let schema: serde_json::Value = serde_json::from_str(&def.parameters_json).unwrap();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("query")));
    }

    #[test]
    fn web_fetch_schema_requires_url() {
        let def = web_fetch_definition();
        let schema: serde_json::Value = serde_json::from_str(&def.parameters_json).unwrap();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("url")));
    }
}
