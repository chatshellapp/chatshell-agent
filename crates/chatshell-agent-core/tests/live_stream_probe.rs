//! Probe: full run_generation against a raw TCP SSE server.

use chatshell_agent_core::run_generation;
use chatshell_agent_core::types::{
    AgentEvent, AgentRequest, ChatMessage, MessageRole, ProviderConfig, ProviderKind,
    SamplingParameters,
};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn streams_against_raw_sse_server() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            let mut stream = stream;
            use std::io::Write;
            let mut body = String::new();
            for word in ["Hello", " world"] {
                body.push_str(&format!("data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{word}\"}},\"finish_reason\":null}}]}}\n\n"));
            }
            body.push_str("data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
            body.push_str("data: [DONE]\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
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
            max_tokens: None,
            additional_params_json: None,
        },
        tools: vec![],
    };
    let cancellation = CancellationToken::new();
    tokio::spawn(run_generation(request, tx.clone(), cancellation.clone()));
    drop(tx);

    let mut text = String::new();
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while let Some(item) = rx.recv().await {
            match item {
                Ok(AgentEvent::TextDelta { text: delta }) => text.push_str(&delta),
                Ok(other) => println!("EVENT: {:?}", other),
                Err(e) => println!("ERROR EVENT: {:?}", e),
            }
        }
    })
    .await;
    match result {
        Ok(_) => {
            println!("FINAL TEXT: {:?}", text);
            assert!(
                text.contains("Hello"),
                "expected streamed text, got {:?}",
                text
            );
        }
        Err(_) => panic!("PROBE TIMED OUT"),
    }
}
