//! Wire-level tests for the shared MCP client layer against a scripted
//! JSON-RPC server over an in-memory duplex (no network).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chatshell_agent_core::mcp::{
    ChatShellClientHandler, McpCallOptions, McpElicitationAnswer, McpElicitationPrompter,
    McpElicitationRequest, McpNotificationSink, call_tool_with_options, get_prompt_text,
    list_all_prompts_bounded, list_all_resource_templates_bounded, list_all_resources_bounded,
    list_all_tools_bounded, prepare_client_peer, read_resource_with_options,
    render_call_tool_result, render_read_resource_result, supports_resources,
};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, ReadResourceRequestParams};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

type Log = Arc<Mutex<Vec<Value>>>;

fn tool(name: &str) -> Value {
    json!({"name": name, "inputSchema": {"type": "object"}})
}

/// Scripted server. Tools listing: 3 pages normally, or a looping cursor
/// when `loop_cursor`. Tools: `slow` never answers, `progress` emits a
/// progress notification every 300 ms for 1.2 s then answers, `structured`
/// answers with structuredContent only, `announce` answers and then sends
/// `notifications/tools/list_changed`.
async fn run_server(
    stream: tokio::io::DuplexStream,
    log: Log,
    loop_cursor: bool,
    extra_tool: Arc<AtomicUsize>,
) {
    let (read, write) = tokio::io::split(stream);
    let write = Arc::new(Mutex::new(write));
    let mut lines = BufReader::new(read).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let msg: Value = serde_json::from_str(&line).unwrap();
        log.lock().await.push(msg.clone());
        let Some(id) = msg.get("id").cloned() else {
            continue;
        };
        // A response from the client (to our elicitation request).
        if msg.get("method").is_none() {
            continue;
        }
        let method = msg["method"].as_str().unwrap_or_default().to_string();
        let write = write.clone();
        let send = move |v: Value| {
            let write = write.clone();
            async move {
                let mut w = write.lock().await;
                w.write_all(format!("{v}\n").as_bytes()).await.unwrap();
                w.flush().await.unwrap();
            }
        };
        match method.as_str() {
            "initialize" => send(json!({"jsonrpc": "2.0", "id": id, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": true}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "mock", "version": "1"}
            }}))
            .await,
            "tools/list" if extra_tool.load(Ordering::SeqCst) >= 100 => {
                // Credentials revoked: every list fails from now on.
                send(json!({"jsonrpc": "2.0", "id": id, "error": {
                    "code": -32001, "message": "Unauthorized"}}))
                .await
            }
            "tools/list" => {
                let cursor = msg["params"]["cursor"].as_str().map(str::to_string);
                let (tools, next) = if loop_cursor {
                    (vec![tool("loop")], Some("again"))
                } else {
                    match cursor.as_deref() {
                        None => (vec![tool("a"), tool("b")], Some("p2")),
                        Some("p2") => (vec![tool("c")], Some("p3")),
                        _ => {
                            let mut t = vec![tool("d")];
                            if extra_tool.load(Ordering::SeqCst) > 0 {
                                t.push(tool("added"));
                            }
                            (t, None)
                        }
                    }
                };
                // Cache hints, as a server that supports response caching
                // sends them (they must never mask a later failure).
                let mut result = json!({"tools": tools, "ttlMs": 60000, "cacheScope": "public"});
                if let Some(n) = next {
                    result["nextCursor"] = json!(n);
                }
                send(json!({"jsonrpc": "2.0", "id": id, "result": result})).await
            }
            "resources/list" => {
                let result = match msg["params"]["cursor"].as_str() {
                    None => json!({
                        "resources": [{"uri": "mem://a", "name": "a"}],
                        "nextCursor": "r2"
                    }),
                    _ => json!({"resources": [{"uri": "mem://b", "name": "b"}]}),
                };
                send(json!({"jsonrpc": "2.0", "id": id, "result": result})).await
            }
            "prompts/list" => {
                send(json!({"jsonrpc": "2.0", "id": id, "result": {"prompts": [
                    {"name": "review", "title": "Code review",
                     "arguments": [{"name": "pr", "required": true}]}
                ]}}))
                .await
            }
            "prompts/get" => {
                let pr = msg["params"]["arguments"]["pr"]
                    .as_str()
                    .unwrap_or("?")
                    .to_string();
                send(json!({"jsonrpc": "2.0", "id": id, "result": {"messages": [
                    {"role": "user", "content": {"type": "text", "text": format!("Review PR {pr}")}}
                ]}}))
                .await
            }
            "resources/templates/list" => {
                send(
                    json!({"jsonrpc": "2.0", "id": id, "result": {"resourceTemplates": [
                        {"uriTemplate": "mem://{key}", "name": "by key"}
                    ]}}),
                )
                .await
            }
            "resources/read" => {
                let uri = msg["params"]["uri"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                if uri == "mem://a" {
                    send(json!({"jsonrpc": "2.0", "id": id, "result": {"contents": [
                        {"uri": uri, "mimeType": "text/plain", "text": "alpha"}
                    ]}}))
                    .await
                } else {
                    send(json!({"jsonrpc": "2.0", "id": id, "error": {
                        "code": -32002, "message": "Resource not found"}}))
                    .await
                }
            }
            "tools/call" => {
                let name = msg["params"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let token = msg["params"]["_meta"]["progressToken"].clone();
                match name.as_str() {
                    "slow" => {}
                    "progress" => {
                        tokio::spawn(async move {
                            for i in 1..=4 {
                                tokio::time::sleep(Duration::from_millis(300)).await;
                                send(json!({"jsonrpc": "2.0", "method": "notifications/progress",
                                    "params": {"progressToken": token, "progress": i, "total": 4}}))
                                .await;
                            }
                            send(json!({"jsonrpc": "2.0", "id": id, "result": {
                                "content": [{"type": "text", "text": "done"}]}}))
                            .await;
                        });
                    }
                    "flood" => {
                        tokio::spawn(async move {
                            for i in 0u64.. {
                                send(json!({"jsonrpc": "2.0", "method": "notifications/progress",
                                    "params": {"progressToken": token, "progress": i}}))
                                .await;
                                tokio::task::yield_now().await;
                            }
                        });
                    }
                    "structured" => {
                        send(json!({"jsonrpc": "2.0", "id": id, "result": {
                            "structuredContent": {"temperature": 22.5}}}))
                        .await
                    }
                    "ask" => {
                        let log = log.clone();
                        tokio::spawn(async move {
                            send(json!({"jsonrpc": "2.0", "id": "elicit-1",
                                "method": "elicitation/create", "params": {
                                    "mode": "form", "message": "Your name?",
                                    "requestedSchema": {"type": "object",
                                        "properties": {"name": {"type": "string"}},
                                        "required": ["name"]}}}))
                            .await;
                            let answer =
                                loop {
                                    if let Some(m) = log.lock().await.iter().find(|m| {
                                        m["id"] == "elicit-1" && m.get("method").is_none()
                                    }) {
                                        break m.clone();
                                    }
                                    tokio::time::sleep(Duration::from_millis(10)).await;
                                };
                            send(json!({"jsonrpc": "2.0", "id": id, "result": {
                                "content": [{"type": "text", "text": answer["result"].to_string()}]}}))
                            .await;
                        });
                    }
                    "roots" => {
                        let log = log.clone();
                        let request = format!("roots-{id}");
                        tokio::spawn(async move {
                            send(json!({"jsonrpc": "2.0", "id": request.clone(),
                                "method": "roots/list"}))
                            .await;
                            let answer = loop {
                                if let Some(m) = log.lock().await.iter().find(|m| {
                                    m["id"] == request.as_str() && m.get("method").is_none()
                                }) {
                                    break m.clone();
                                }
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            };
                            send(json!({"jsonrpc": "2.0", "id": id, "result": {
                                "content": [{"type": "text", "text": answer["result"].to_string()}]}}))
                            .await;
                        });
                    }
                    "revoke" => {
                        extra_tool.store(100, Ordering::SeqCst);
                        send(json!({"jsonrpc": "2.0", "id": id, "result": {
                            "content": [{"type": "text", "text": "revoked"}]}}))
                        .await;
                    }
                    "announce" => {
                        extra_tool.fetch_add(1, Ordering::SeqCst);
                        send(json!({"jsonrpc": "2.0", "id": id, "result": {
                            "content": [{"type": "text", "text": "ok"}]}}))
                        .await;
                        for _ in 0..5 {
                            send(json!({"jsonrpc": "2.0",
                                "method": "notifications/tools/list_changed"}))
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

async fn connect(
    loop_cursor: bool,
    sink: Option<Arc<dyn McpNotificationSink>>,
) -> (
    rmcp::service::RunningService<rmcp::RoleClient, ChatShellClientHandler>,
    Log,
) {
    connect_with(loop_cursor, ChatShellClientHandler::new("mock", sink)).await
}

async fn connect_with(
    loop_cursor: bool,
    handler: ChatShellClientHandler,
) -> (
    rmcp::service::RunningService<rmcp::RoleClient, ChatShellClientHandler>,
    Log,
) {
    let (client_io, server_io) = tokio::io::duplex(1 << 16);
    let log: Log = Arc::default();
    tokio::spawn(run_server(
        server_io,
        log.clone(),
        loop_cursor,
        Arc::new(AtomicUsize::new(0)),
    ));
    let (r, w) = tokio::io::split(client_io);
    let service = handler.serve((r, w)).await.unwrap();
    prepare_client_peer(service.peer()).await;
    (service, log)
}

fn call(name: &str) -> CallToolRequestParams {
    CallToolRequestParams::new(name.to_string())
}

async fn cancelled_ids(log: &Log) -> Vec<Value> {
    log.lock()
        .await
        .iter()
        .filter(|m| m["method"] == "notifications/cancelled")
        .map(|m| m["params"]["requestId"].clone())
        .collect()
}

async fn request_id_of(log: &Log, tool: &str) -> Value {
    log.lock()
        .await
        .iter()
        .find(|m| m["method"] == "tools/call" && m["params"]["name"] == tool)
        .map(|m| m["id"].clone())
        .unwrap()
}

#[tokio::test]
async fn pagination_assembles_all_pages() {
    let (service, _log) = connect(false, None).await;
    let tools = list_all_tools_bounded(service.peer(), "mock")
        .await
        .unwrap();
    let names: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
    assert_eq!(names, ["a", "b", "c", "d"]);
}

#[tokio::test]
async fn looping_cursor_terminates() {
    let (service, log) = connect(true, None).await;
    let tools = list_all_tools_bounded(service.peer(), "mock")
        .await
        .unwrap();
    assert_eq!(tools.len(), 2);
    let lists = log
        .lock()
        .await
        .iter()
        .filter(|m| m["method"] == "tools/list")
        .count();
    assert_eq!(lists, 2);
}

#[tokio::test]
async fn timeout_fails_and_sends_cancellation() {
    let (service, log) = connect(false, None).await;
    let options = McpCallOptions::from_settings(Some(1), Some(false), None);
    let started = std::time::Instant::now();
    let err = call_tool_with_options(service.peer(), call("slow"), options, None, |_| {})
        .await
        .unwrap_err();
    assert!(matches!(err, rmcp::service::ServiceError::Timeout { .. }));
    assert!(started.elapsed() < Duration::from_secs(3));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let id = request_id_of(&log, "slow").await;
    assert_eq!(cancelled_ids(&log).await, vec![id]);

    // The connection survives the timeout.
    let ok = call_tool_with_options(
        service.peer(),
        call("structured"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    assert!(render_call_tool_result(&ok).text.contains("22.5"));
}

#[tokio::test]
async fn progress_resets_deadline_and_is_forwarded() {
    let (service, _log) = connect(false, None).await;
    // 1 s idle deadline, server answers after ~1.2 s with progress every 300 ms.
    let options = McpCallOptions::from_settings(Some(1), Some(true), Some(10));
    let seen = Arc::new(AtomicUsize::new(0));
    let seen2 = seen.clone();
    let result = call_tool_with_options(
        service.peer(),
        call("progress"),
        options,
        Some(service.service().progress_dispatcher()),
        move |_| {
            seen2.fetch_add(1, Ordering::SeqCst);
        },
    )
    .await
    .unwrap();
    assert_eq!(render_call_tool_result(&result).text, "done");
    assert_eq!(seen.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn progress_without_reset_still_times_out() {
    let (service, _log) = connect(false, None).await;
    let options = McpCallOptions::from_settings(Some(1), Some(false), None);
    let err = call_tool_with_options(service.peer(), call("progress"), options, None, |_| {})
        .await
        .unwrap_err();
    assert!(matches!(err, rmcp::service::ServiceError::Timeout { .. }));
}

#[tokio::test]
async fn dropping_the_call_sends_cancellation() {
    let (service, log) = connect(false, None).await;
    let peer = service.peer().clone();
    let task = tokio::spawn(async move {
        call_tool_with_options(&peer, call("slow"), McpCallOptions::default(), None, |_| {}).await
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    task.abort();
    let _ = task.await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let id = request_id_of(&log, "slow").await;
    assert_eq!(cancelled_ids(&log).await, vec![id]);
}

#[tokio::test]
async fn completed_call_sends_no_cancellation() {
    let (service, log) = connect(false, None).await;
    call_tool_with_options(
        service.peer(),
        call("structured"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(cancelled_ids(&log).await.is_empty());
}

#[tokio::test]
async fn list_changed_is_debounced_and_relist_sees_new_tool() {
    let counter = Arc::new(Counter(AtomicUsize::new(0)));
    let (service, _log) = connect(false, Some(counter.clone())).await;
    call_tool_with_options(
        service.peer(),
        call("announce"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    let tools = list_all_tools_bounded(service.peer(), "mock")
        .await
        .unwrap();
    assert!(tools.iter().any(|t| t.name == "added"));
}

#[tokio::test]
async fn progress_flood_cannot_outlive_the_ceiling() {
    let (service, _log) = connect(false, None).await;
    // Progress resets the 1 s idle deadline forever; the 2 s ceiling must win.
    let options = McpCallOptions::from_settings(Some(1), Some(true), Some(2));
    let started = std::time::Instant::now();
    let err = call_tool_with_options(
        service.peer(),
        call("flood"),
        options,
        Some(service.service().progress_dispatcher()),
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(matches!(err, rmcp::service::ServiceError::Timeout { .. }));
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn resources_are_listed_across_pages_and_read() {
    let (service, _log) = connect(false, None).await;
    assert!(supports_resources(service.peer()));
    let resources = list_all_resources_bounded(service.peer(), "mock")
        .await
        .unwrap();
    let uris: Vec<_> = resources.iter().map(|r| r.uri.as_str()).collect();
    assert_eq!(uris, ["mem://a", "mem://b"]);
    let templates = list_all_resource_templates_bounded(service.peer(), "mock")
        .await
        .unwrap();
    assert_eq!(templates[0].uri_template, "mem://{key}");

    let read = read_resource_with_options(service.peer(), "mem://a", McpCallOptions::default())
        .await
        .unwrap();
    assert!(render_read_resource_result(&read, None).contains("alpha"));

    let missing = service
        .peer()
        .read_resource(ReadResourceRequestParams::new("mem://missing"))
        .await;
    assert!(missing.is_err());
}

#[tokio::test]
async fn initialize_requests_the_2025_11_25_revision() {
    let (_service, log) = connect(false, None).await;
    let init = log
        .lock()
        .await
        .iter()
        .find(|m| m["method"] == "initialize")
        .cloned()
        .unwrap();
    assert_eq!(init["params"]["protocolVersion"], "2025-11-25");
    assert_eq!(init["params"]["clientInfo"]["name"], "chatshell");
    let methods: Vec<String> = log
        .lock()
        .await
        .iter()
        .filter_map(|m| m["method"].as_str().map(str::to_string))
        .collect();
    assert!(
        !methods.iter().any(|m| m == "server/discover"),
        "{methods:?}"
    );
}

#[tokio::test]
async fn a_failing_relist_is_an_error_not_a_stale_list() {
    let (service, _log) = connect(false, None).await;
    assert_eq!(
        list_all_tools_bounded(service.peer(), "mock")
            .await
            .unwrap()
            .len(),
        4
    );
    call_tool_with_options(
        service.peer(),
        call("revoke"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    assert!(
        list_all_tools_bounded(service.peer(), "mock")
            .await
            .is_err(),
        "revalidation must see the failure"
    );
}

struct AnswerAda;
impl McpElicitationPrompter for AnswerAda {
    fn prompt(
        &self,
        server: String,
        request: McpElicitationRequest,
        _dismissed: tokio_util::sync::CancellationToken,
    ) -> futures::future::BoxFuture<'static, McpElicitationAnswer> {
        assert_eq!(server, "mock");
        assert!(
            matches!(request, McpElicitationRequest::Form { ref message, .. } if message == "Your name?")
        );
        Box::pin(async { McpElicitationAnswer::Accept(json!({"name": "Ada"})) })
    }
}

#[tokio::test]
async fn elicitation_mid_call_reaches_the_prompter_and_returns_content() {
    let handler = ChatShellClientHandler::new("mock", None).with_elicitation(Arc::new(AnswerAda));
    let (service, log) = connect_with(false, handler).await;
    let init = log
        .lock()
        .await
        .iter()
        .find(|m| m["method"] == "initialize")
        .cloned()
        .unwrap();
    assert!(init["params"]["capabilities"]["elicitation"]["form"].is_object());
    assert!(init["params"]["capabilities"]["elicitation"]["url"].is_object());

    let result = call_tool_with_options(
        service.peer(),
        call("ask"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    let text = render_call_tool_result(&result).text;
    assert!(text.contains(r#""action":"accept""#), "{text}");
    assert!(text.contains(r#""name":"Ada""#), "{text}");
}

#[tokio::test]
async fn without_a_prompter_elicitation_is_not_advertised_and_is_declined() {
    let (service, log) = connect(false, None).await;
    let init = log
        .lock()
        .await
        .iter()
        .find(|m| m["method"] == "initialize")
        .cloned()
        .unwrap();
    assert!(init["params"]["capabilities"].get("elicitation").is_none());
    let result = call_tool_with_options(
        service.peer(),
        call("ask"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    assert!(
        render_call_tool_result(&result)
            .text
            .contains(r#""action":"decline""#)
    );
}

#[tokio::test]
async fn prompts_are_listed_and_rendered_with_arguments() {
    let (service, _log) = connect(false, None).await;
    let prompts = list_all_prompts_bounded(service.peer(), "mock")
        .await
        .unwrap();
    assert_eq!(prompts[0].name, "review");
    assert!(prompts[0].arguments[0].required);
    let mut args = std::collections::BTreeMap::new();
    args.insert("pr".to_string(), "42".to_string());
    let text = get_prompt_text(service.peer(), "review", args)
        .await
        .unwrap();
    assert_eq!(text, "Review PR 42");
}

#[tokio::test]
async fn roots_list_answers_the_current_workspace_without_hanging() {
    let root = Arc::new(std::sync::Mutex::new(None::<std::path::PathBuf>));
    let provider = {
        let root = root.clone();
        Arc::new(move || root.lock().unwrap().clone())
            as chatshell_agent_core::mcp::McpRootsProvider
    };
    let handler = ChatShellClientHandler::new("mock", None).with_roots(provider);
    let (service, log) = connect_with(false, handler).await;
    let init = log
        .lock()
        .await
        .iter()
        .find(|m| m["method"] == "initialize")
        .cloned()
        .unwrap();
    assert!(init["params"]["capabilities"]["roots"].is_object());
    assert!(
        init["params"]["capabilities"]["roots"]
            .get("listChanged")
            .is_none()
    );

    // Before any run: an empty list, not a hang.
    let result = call_tool_with_options(
        service.peer(),
        call("roots"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    assert!(
        render_call_tool_result(&result)
            .text
            .contains(r#""roots":[]"#)
    );

    *root.lock().unwrap() = Some(std::path::PathBuf::from("/tmp/project"));
    let result = call_tool_with_options(
        service.peer(),
        call("roots"),
        McpCallOptions::default(),
        None,
        |_| {},
    )
    .await
    .unwrap();
    let text = render_call_tool_result(&result).text;
    assert!(text.contains("file:///tmp/project/"), "{text}");
    assert!(text.contains(r#""name":"project""#), "{text}");
}
