//! MCP HTTP connection - connect, list tools, call tools.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RunningService};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use rmcp::{RoleClient, ServiceExt};

use super::call::{McpCallOptions, call_tool_with_options};
use super::handler::ChatShellClientHandler;
use super::listing::list_all_tools_bounded;
use super::result::render_call_tool_result;
use super::types::{McpToolInfo, McpToolResult};

/// A running MCP client session with ChatShell's notification handler.
pub type McpClientService = RunningService<RoleClient, ChatShellClientHandler>;

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
    running_service: Arc<McpClientService>,
    label: Arc<str>,
}

impl McpConnection {
    /// Connect to an MCP server via Streamable HTTP transport.
    pub async fn connect(config: McpHttpConfig) -> Result<Self> {
        tracing::info!("Connecting to MCP server via HTTP: {}", config.endpoint);

        let handler = ChatShellClientHandler::new(&config.endpoint, None);
        let running_service = connect_http_service(&config, handler).await?;
        let client = running_service.peer().clone();

        tracing::info!("Connected to MCP server");

        Ok(Self {
            client,
            label: Arc::from(config.endpoint.as_str()),
            running_service: Arc::new(running_service),
        })
    }

    /// List all tools exposed by the connected MCP server, following
    /// pagination (bounded, see [`super::listing`]).
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let tools = list_all_tools_bounded(&self.client, &self.label).await?;
        Ok(tools.iter().map(McpToolInfo::from_tool).collect())
    }

    /// Call a tool on the connected MCP server with the default deadline.
    ///
    /// `arguments_json` is a JSON object string (e.g. `{"query":"hello"}`).
    /// The result is rendered by [`render_call_tool_result`] (content blocks
    /// plus `structuredContent`).
    pub async fn call_tool(&self, name: &str, arguments_json: &str) -> Result<McpToolResult> {
        let arguments: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(arguments_json).unwrap_or_default();

        tracing::info!("Calling MCP tool: {}", name);

        let mut params = CallToolRequestParams::new(name.to_string());
        params.arguments = Some(arguments);
        let result = call_tool_with_options(
            &self.client,
            params,
            McpCallOptions::default(),
            Some(self.running_service.service().progress_dispatcher()),
            |_| {},
        )
        .await
        .context("MCP tool call failed")?;

        let rendered = render_call_tool_result(&result);
        Ok(McpToolResult {
            content: rendered.text,
            is_error: rendered.is_error,
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
    handler: ChatShellClientHandler,
) -> Result<McpClientService> {
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

    let running_service = handler
        .serve(transport)
        .await
        .context("Failed to connect to MCP server via HTTP")?;

    Ok(running_service)
}
