//! A reply that stops at `max_tokens` is reported as `output_limited`, on
//! both the multi-turn path (no client tools) and the client-tool path the
//! mobile clients use.

use chatshell_agent_core::run_generation;
use chatshell_agent_core::types::{
    AgentEvent, AgentRequest, ChatMessage, ClientToolDefinition, MessageRole, ProviderConfig,
    ProviderKind, SamplingParameters,
};
use tokio_util::sync::CancellationToken;

/// Serve one OpenAI-compatible SSE reply ending with `finish_reason`.
fn serve_once(finish_reason: &'static str) -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        let mut request = [0u8; 16 * 1024];
        let _ = stream.read(&mut request);
        let chunk = |delta: &str, finish: &str| {
            format!(
                "data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{finish}}}]}}\n\n"
            )
        };
        let mut body = chunk("{\"content\":\"Partial answ\"}", "null");
        body.push_str(&chunk("{}", &format!("\"{finish_reason}\"")));
        body.push_str("data: [DONE]\n\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
        std::thread::sleep(std::time::Duration::from_millis(300));
    });
    port
}

async fn completed_flag(finish_reason: &'static str, with_tools: bool) -> bool {
    let port = serve_once(finish_reason);
    let tools = if with_tools {
        vec![ClientToolDefinition {
            name: "web_search".into(),
            description: "Search the web".into(),
            parameters_json: r#"{"type":"object","properties":{"query":{"type":"string"}}}"#.into(),
        }]
    } else {
        vec![]
    };
    let request = AgentRequest {
        provider: ProviderConfig {
            kind: ProviderKind::CustomOpenAi,
            api_key: Some("k".into()),
            base_url: Some(format!("http://127.0.0.1:{port}/v1")),
            api_style: Some("chat_completions".into()),
        },
        model: "m".into(),
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: "hi".into(),
            reasoning_content: None,
            images: vec![],
            tool_calls: vec![],
            tool_call_id: None,
            tool_result_call_id: None,
        }],
        parameters: SamplingParameters {
            temperature: None,
            max_tokens: Some(4),
            additional_params_json: None,
        },
        tools,
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(run_generation(request, tx, CancellationToken::new()));
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while let Some(item) = rx.recv().await {
            if let Ok(AgentEvent::Completed {
                content,
                output_limited,
                cancelled,
                ..
            }) = item
            {
                assert_eq!(content, "Partial answ");
                assert!(!cancelled);
                return output_limited;
            }
        }
        panic!("stream ended without Completed");
    })
    .await
    .expect("stream timed out")
}

#[tokio::test]
async fn length_stop_is_output_limited_on_both_paths() {
    assert!(completed_flag("length", false).await);
    assert!(completed_flag("length", true).await);
}

#[tokio::test]
async fn normal_stop_is_not_output_limited() {
    assert!(!completed_flag("stop", false).await);
    assert!(!completed_flag("stop", true).await);
}
