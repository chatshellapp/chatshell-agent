//! Wire-level tests for the shared MCP client layer against a scripted
//! JSON-RPC server over an in-memory duplex (no network).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chatshell_agent_core::mcp::{
    ChatShellClientHandler, McpCallOptions, McpNotificationSink, call_tool_with_options,
    list_all_tools_bounded, render_call_tool_result,
};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
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
            "initialize" => {
                send(json!({"jsonrpc": "2.0", "id": id, "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": true}},
                    "serverInfo": {"name": "mock", "version": "1"}
                }}))
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
                let mut result = json!({"tools": tools});
                if let Some(n) = next {
                    result["nextCursor"] = json!(n);
                }
                send(json!({"jsonrpc": "2.0", "id": id, "result": result})).await
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
    let (client_io, server_io) = tokio::io::duplex(1 << 16);
    let log: Log = Arc::default();
    tokio::spawn(run_server(
        server_io,
        log.clone(),
        loop_cursor,
        Arc::new(AtomicUsize::new(0)),
    ));
    let (r, w) = tokio::io::split(client_io);
    let service = ChatShellClientHandler::new("mock", sink)
        .serve((r, w))
        .await
        .unwrap();
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
