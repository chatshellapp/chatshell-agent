//! Regression: the Anthropic wire must authenticate with `x-api-key`.
//!
//! rig's anthropic client authenticates via `x-api-key` (its `AnthropicKey`
//! scheme), which api.anthropic.com and Anthropic-compatible gateways both
//! accept while rejecting plain `Authorization: Bearer`. This test pins the
//! header scheme on the
//! non-streaming path so a rig upgrade that flips it fails loudly. The
//! desktop crate carries the streaming-path companion
//! (`llm/anthropic_wire_tests.rs`).

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use chatshell_agent_core::create_anthropic_completion_model;
use rig::completion::{CompletionModel, Message};

#[tokio::test]
async fn anthropic_wire_carries_x_api_key_header() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let captured = Arc::new(Mutex::new(String::new()));
    let sink = captured.clone();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut request = [0u8; 8192];
            let mut headers_end = 0;
            let mut raw = Vec::new();
            // Read until end of headers; the body is irrelevant here.
            loop {
                let n = stream.read(&mut request).unwrap_or(0);
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&request[..n]);
                if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    headers_end = pos + 4;
                    break;
                }
            }
            *sink.lock().unwrap() = String::from_utf8_lossy(&raw[..headers_end]).to_string();

            let body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-probe","content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    let model = create_anthropic_completion_model(
        "probe-key",
        Some(&format!("http://127.0.0.1:{port}/probe")),
        "claude-probe",
    )
    .unwrap();

    let response = model
        .completion(rig::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![Message::user("hi")],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
        .await
        .expect("completion against the local server must succeed");

    let text = response
        .choice
        .first()
        .and_then(|block| match block {
            rig::completion::AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .unwrap_or_default();
    assert_eq!(text, "Hello");

    let headers = captured.lock().unwrap().clone();
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("x-api-key: probe-key"),
        "x-api-key header must ride the request (gateways that only read x-api-key): {headers}"
    );
    assert!(
        !headers.to_ascii_lowercase().contains("authorization"),
        "the anthropic wire is x-api-key only; a Bearer header would mask gateway quirks: {headers}"
    );
    assert!(
        headers.contains("anthropic-version"),
        "rig's anthropic-version header must survive: {headers}"
    );
}
