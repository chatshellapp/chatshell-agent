//! MCP HTTP connection - connect, list tools, call tools.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use rmcp::model::{CallToolRequestParams, ClientCapabilities, ClientInfo, ContentBlock};
use rmcp::service::{Peer, RunningService};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::{RoleClient, ServiceExt};

use super::types::{McpToolInfo, McpToolResult};

/// Configuration for connecting to an MCP server via HTTP.
pub struct McpHttpConfig {
    pub endpoint: String,
    /// Bearer token for the `Authorization` header. The caller is responsible
    /// for obtaining this (plaintext, keychain, OAuth, etc.).
    pub auth_token: Option<String>,
    /// Additional HTTP headers to send with every request.
    pub custom_headers: HashMap<String, String>,
}

/// An active connection to an MCP server. Clone is cheap (Arc-based) -
/// all clones share the same underlying transport and background task.
/// Dropping the last clone disconnects.
#[derive(Clone)]
pub struct McpConnection {
    client: Peer<RoleClient>,
    _running_service: Arc<RunningService<RoleClient, ClientInfo>>,
}

impl McpConnection {
    /// Connect to an MCP server via Streamable HTTP transport.
    pub async fn connect(config: McpHttpConfig) -> Result<Self> {
        tracing::info!("Connecting to MCP server via HTTP: {}", config.endpoint);

        let running_service = connect_http_service(&config).await?;
        let client = running_service.peer().clone();

        tracing::info!("Connected to MCP server");

        Ok(Self {
            client,
            _running_service: Arc::new(running_service),
        })
    }

    /// List all tools exposed by the connected MCP server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let tools = self
            .client
            .list_all_tools()
            .await
            .context("Failed to list tools from MCP server")?;

        let tools = tools
            .into_iter()
            .map(|t| McpToolInfo {
                name: t.name.to_string(),
                description: t.description.map(|d| d.to_string()),
                input_schema_json: serde_json::to_string(&t.input_schema)
                    .unwrap_or_else(|_| "{}".to_string()),
            })
            .collect();

        Ok(tools)
    }

    /// Call a tool on the connected MCP server.
    ///
    /// `arguments_json` is a JSON object string (e.g. `{"query":"hello"}`).
    /// Non-text content in the result is skipped; only text is returned.
    pub async fn call_tool(&self, name: &str, arguments_json: &str) -> Result<McpToolResult> {
        let arguments: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(arguments_json).unwrap_or_default();

        tracing::info!("Calling MCP tool: {}", name);

        let result = self
            .client
            .call_tool({
                let mut params = CallToolRequestParams::new(name.to_string());
                params.arguments = Some(arguments);
                params
            })
            .await
            .context("MCP tool call failed")?;

        let content: String = result
            .content
            .into_iter()
            .filter_map(|c| match c {
                ContentBlock::Text(raw) => Some(raw.text.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(McpToolResult {
            content,
            is_error: result.is_error.unwrap_or(false),
        })
    }
}

/// Establish a Streamable HTTP MCP session and return the raw running
/// service. This is the shared transport-construction layer: the
/// higher-level [`McpConnection`] wraps it for iOS, and desktop's
/// `mcp/manager.rs` uses it directly so both platforms build identical
/// clients, headers, auth, and client identity.
pub async fn connect_http_service(
    config: &McpHttpConfig,
) -> Result<RunningService<RoleClient, ClientInfo>> {
    let http_client = if config.custom_headers.is_empty() {
        reqwest::Client::new()
    } else {
        let mut header_map = reqwest::header::HeaderMap::new();
        for (k, v) in &config.custom_headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                header_map.insert(name, val);
            }
        }
        reqwest::Client::builder()
            .default_headers(header_map)
            .build()
            .context("Failed to build HTTP client with custom headers")?
    };

    let mut transport_config =
        StreamableHttpClientTransportConfig::with_uri(config.endpoint.clone());
    if let Some(token) = &config.auth_token {
        transport_config = transport_config.auth_header(token.clone());
    }

    let transport = StreamableHttpClientTransport::with_client(http_client, transport_config);

    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        rmcp::model::Implementation::new("chatshell", env!("CARGO_PKG_VERSION")),
    );

    let running_service = client_info
        .serve(transport)
        .await
        .context("Failed to connect to MCP server via HTTP")?;

    Ok(running_service)
}
