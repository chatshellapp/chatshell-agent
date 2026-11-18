//! 2026-07-28 revision against a scripted server over an in-memory duplex:
//! `server/discover` negotiation and legacy fallback (FR-16), MRTR
//! `input_required` rounds (FR-17), and `subscriptions/listen` (FR-18).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chatshell_agent_core::mcp::{
    ChatShellClientHandler, MAX_INPUT_ROUNDS, McpCallOptions, McpElicitationAnswer,
    McpElicitationPrompter, McpElicitationRequest, McpNotificationSink, McpProtocolMode,
    call_tool_with_options, prepare_client_peer, render_call_tool_result, serve_client,
    start_subscriptions,
};
use rmcp::model::CallToolRequestParams;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

type Log = Arc<Mutex<Vec<Value>>>;

#[derive(Clone, Copy, PartialEq)]
enum Era {
    Modern,
    Legacy,
}

async fn run_server(stream: tokio::io::DuplexStream, era: Era, log: Log) {
    let (read, write) = tokio::io::split(stream);
    let write = Arc::new(Mutex::new(write));
    let mut lines = BufReader::new(read).lines();
    let listen_id: Arc<Mutex<Option<Value>>> = Arc::default();
    while let Ok(Some(line)) = lines.next_line().await {
        let msg: Value = serde_json::from_str(&line).unwrap();
        log.lock().await.push(msg.clone());
        let Some(id) = msg.get("id").cloned() else {
            continue;
        };
        if msg.get("method").is_none() {
            continue;
        }
        let send = {
            let write = write.clone();
            move |v: Value| {
                let write = write.clone();
                async move {
                    let mut w = write.lock().await;
                    w.write_all(format!("{v}\n").as_bytes()).await.unwrap();
                    w.flush().await.unwrap();
                }
            }
        };
        let method = msg["method"].as_str().unwrap_or_default().to_string();
        let complete = |mut result: Value| {
            if era == Era::Modern {
                result["resultType"] = json!("complete");
            }
            result
        };
        match (era, method.as_str()) {
            (Era::Modern, "server/discover") => {
                send(json!({"jsonrpc": "2.0", "id": id, "result": {
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": {"tools": {"listChanged": true}},
                    "ttlMs": 0,
                    "cacheScope": "public",
                    "_meta": {"io.modelcontextprotocol/serverInfo":
                        {"name": "modern", "version": "1"}}
                }}))
                .await
            }
            (Era::Legacy, "server/discover") => {
                send(json!({"jsonrpc": "2.0", "id": id, "error": {
                    "code": -32601, "message": "Method not found"}}))
                .await
            }
            (_, "initialize") => {
                send(json!({"jsonrpc": "2.0", "id": id, "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {"listChanged": true}},
                    "serverInfo": {"name": "legacy", "version": "1"}
                }}))
                .await
            }
            (_, "tools/list") => {
                send(
                    json!({"jsonrpc": "2.0", "id": id, "result": complete(json!({
                        "tools": [{"name": "t", "inputSchema": {"type": "object"}}]
                    }))}),
                )
                .await
            }
            (_, "subscriptions/listen") => {
                *listen_id.lock().await = Some(id.clone());
                send(json!({"jsonrpc": "2.0",
                "method": "notifications/subscriptions/acknowledged",
                "params": {
                    "notifications": msg["params"]["notifications"].clone(),
                    "_meta": {"io.modelcontextprotocol/subscriptionId": id}
                }}))
                .await
            }
            (_, "tools/call") => {
                let name = msg["params"]["name"].as_str().unwrap_or_default();
                let params = &msg["params"];
                match name {
                    // One MRTR round asking for a name, then the answer.
                    "greet" => {
                        if let Some(responses) = params.get("inputResponses") {
                            assert_eq!(params["requestState"], "state-1");
                            let who = responses["who"]["content"]["name"].clone();
                            send(json!({"jsonrpc": "2.0", "id": id, "result": complete(json!({
                                "content": [{"type": "text", "text": format!("Hello {}", who.as_str().unwrap_or("?"))}]
                            }))}))
                            .await
                        } else {
                            send(json!({"jsonrpc": "2.0", "id": id, "result": {
                                "resultType": "input_required",
                                "inputRequests": {"who": {
                                    "method": "elicitation/create",
                                    "params": {"mode": "form", "message": "Your name?",
                                        "requestedSchema": {"type": "object",
                                            "properties": {"name": {"type": "string"}},
                                            "required": ["name"]}}
                                }},
                                "requestState": "state-1"
                            }}))
                            .await
                        }
                    }
                    // Never finishes asking.
                    "endless" => {
                        send(json!({"jsonrpc": "2.0", "id": id, "result": {
                            "resultType": "input_required",
                            "inputRequests": {"r": {"method": "roots/list"}},
                            "requestState": "again"
                        }}))
                        .await
                    }
                    // Changes the tool list (announced on the subscription).
                    "announce" => {
                        send(
                            json!({"jsonrpc": "2.0", "id": id, "result": complete(json!({
                                "content": [{"type": "text", "text": "ok"}]
                            }))}),
                        )
                        .await;
                        if let Some(sub) = listen_id.lock().await.clone() {
                            send(json!({"jsonrpc": "2.0",
                                "method": "notifications/tools/list_changed",
                                "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": sub}}}))
                            .await;
                        }
                    }
                    _ => {
                        send(json!({"jsonrpc": "2.0", "id": id, "error": {
                            "code": -32602, "message": "unknown tool"}}))
                        .await
                    }
                }
            }
            _ => {
                send(json!({"jsonrpc": "2.0", "id": id, "error": {
                    "code": -32601, "message": "method not found"}}))
                .await
            }
        }
    }
}

struct Counter(AtomicUsize);
impl McpNotificationSink for Counter {
    fn tools_list_changed(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct AnswerAda;
impl McpElicitationPrompter for AnswerAda {
    fn prompt(
        &self,
        _server: String,
        request: McpElicitationRequest,
        _dismissed: tokio_util::sync::CancellationToken,
    ) -> futures::future::BoxFuture<'static, McpElicitationAnswer> {
        assert!(matches!(request, McpElicitationRequest::Form { .. }));
        Box::pin(async { McpElicitationAnswer::Accept(json!({"name": "Ada"})) })
    }
}

async fn connect(
    era: Era,
    mode: McpProtocolMode,
    handler: ChatShellClientHandler,
) -> (
    rmcp::service::RunningService<rmcp::RoleClient, ChatShellClientHandler>,
    Log,
) {
    let (client_io, server_io) = tokio::io::duplex(1 << 16);
    let log: Log = Arc::default();
    tokio::spawn(run_server(server_io, era, log.clone()));
    let (r, w) = tokio::io::split(client_io);
    let service = serve_client(handler, (r, w), mode).await.unwrap();
    prepare_client_peer(service.peer()).await;
    start_subscriptions(&service).await;
    (service, log)
}

async fn methods(log: &Log) -> Vec<String> {
    log.lock()
        .await
        .iter()
        .filter_map(|m| m["method"].as_str().map(str::to_string))
        .collect()
}

fn call(name: &str) -> CallToolRequestParams {
    CallToolRequestParams::new(name.to_string())
}

#[tokio::test]
async fn opted_in_server_negotiates_2026_via_discover() {
    let (service, log) = connect(
        Era::Modern,
        McpProtocolMode::PreferModern,
        ChatShellClientHandler::new("modern", None),
    )
    .await;
    let info = service.peer().peer_info().unwrap();
    assert_eq!(info.protocol_version.to_string(), "2026-07-28");
    let seen = methods(&log).await;
    assert!(seen.contains(&"server/discover".to_string()), "{seen:?}");
    assert!(!seen.contains(&"initialize".to_string()), "{seen:?}");
}

#[tokio::test]
async fn legacy_server_falls_back_to_initialize() {
    let (service, log) = connect(
        Era::Legacy,
        McpProtocolMode::PreferModern,
        ChatShellClientHandler::new("legacy", None),
    )
    .await;
    let info = service.peer().peer_info().unwrap();
    assert_eq!(info.protocol_version.to_string(), "2025-11-25");
    let seen = methods(&log).await;
    assert!(seen.contains(&"initialize".to_string()), "{seen:?}");
}

#[tokio::test]
async fn default_mode_never_probes_discover() {
    let (_service, log) = connect(
        Era::Legacy,
        McpProtocolMode::Legacy,
        ChatShellClientHandler::new("legacy", None),
    )
    .await;
    let seen = methods(&log).await;
    assert!(!seen.contains(&"server/discover".to_string()), "{seen:?}");
}

#[tokio::test]
async fn input_required_round_is_answered_through_the_prompter() {
    let handler = ChatShellClientHandler::new("modern", None).with_elicitation(Arc::new(AnswerAda));
    let (service, log) = connect(Era::Modern, McpProtocolMode::PreferModern, handler).await;
    let result = call_tool_with_options(
        service.peer(),
        call("greet"),
        McpCallOptions::default(),
        None,
        Some(service.service()),
        |_| {},
    )
    .await
    .unwrap();
    assert_eq!(render_call_tool_result(&result).text, "Hello Ada");
    let calls = log
        .lock()
        .await
        .iter()
        .filter(|m| m["method"] == "tools/call")
        .count();
    assert_eq!(calls, 2, "one MRTR retry");
}

#[tokio::test]
async fn input_rounds_are_bounded() {
    let (service, _log) = connect(
        Era::Modern,
        McpProtocolMode::PreferModern,
        ChatShellClientHandler::new("modern", None),
    )
    .await;
    let err = call_tool_with_options(
        service.peer(),
        call("endless"),
        McpCallOptions::default(),
        None,
        Some(service.service()),
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, rmcp::service::ServiceError::InputRequiredRoundsExceeded { max_rounds } if max_rounds == MAX_INPUT_ROUNDS),
        "{err:?}"
    );
}

#[tokio::test]
async fn input_rounds_without_a_handler_fail_cleanly() {
    let (service, _log) = connect(
        Era::Modern,
        McpProtocolMode::PreferModern,
        ChatShellClientHandler::new("modern", None),
    )
    .await;
    let err = call_tool_with_options(
        service.peer(),
        call("greet"),
        McpCallOptions::default(),
        None,
        None,
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("cannot provide"), "{err}");
}

#[tokio::test]
async fn tool_list_changes_arrive_through_the_subscription() {
    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    let (service, log) = connect(
        Era::Modern,
        McpProtocolMode::PreferModern,
        ChatShellClientHandler::new("modern", Some(counter.clone())),
    )
    .await;
    let listen = log
        .lock()
        .await
        .iter()
        .find(|m| m["method"] == "subscriptions/listen")
        .cloned()
        .expect("a 2026 session subscribes");
    assert_eq!(listen["params"]["notifications"]["toolsListChanged"], true);

    call_tool_with_options(
        service.peer(),
        call("announce"),
        McpCallOptions::default(),
        None,
        Some(service.service()),
        |_| {},
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert_eq!(counter.0.load(Ordering::SeqCst), 1, "handled exactly once");
}

#[tokio::test]
async fn legacy_sessions_do_not_subscribe() {
    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    let (_service, log) = connect(
        Era::Legacy,
        McpProtocolMode::PreferModern,
        ChatShellClientHandler::new("legacy", Some(counter)),
    )
    .await;
    let seen = methods(&log).await;
    assert!(
        !seen.contains(&"subscriptions/listen".to_string()),
        "{seen:?}"
    );
}
