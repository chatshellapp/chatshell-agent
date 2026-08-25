//! Regression test: web_fetch must transparently decompress gzip responses.
//!
//! Before the fix, the client advertised `Accept-Encoding: gzip` but never
//! decoded the compressed body, so fetched content was garbled binary.

use chatshell_agent_core::web_fetch::fetch_http;
use std::io::{Read, Write};

fn gzip_bytes(data: &[u8]) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

#[tokio::test]
async fn fetch_http_decodes_gzip_body() {
    let html = r#"<html><head><title>Weather Report</title></head><body><article><h1>Todays Forecast</h1><p>Café naïve façade — déjà vu, 28°C and sunny.</p></article></body></html>"#;
    let compressed = gzip_bytes(html.as_bytes());

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf); // consume the request
        let mut response = Vec::new();
        response.extend_from_slice(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                compressed.len()
            )
            .as_bytes(),
        );
        response.extend_from_slice(&compressed);
        let _ = stream.write_all(&response);
        let _ = stream.flush();
    });

    let resource = fetch_http(&format!("http://127.0.0.1:{port}/"), Some(20_000)).await;

    assert!(
        !resource.content.starts_with("Error"),
        "unexpected fetch error: {}",
        resource.content
    );
    assert!(
        resource
            .title
            .as_deref()
            .unwrap_or("")
            .contains("Weather Report"),
        "decoded title: {:?}",
        resource.title
    );
    assert!(
        resource.content.contains("déjà vu"),
        "decoded content: {}",
        resource.content
    );
    assert!(
        !resource.content.contains('\u{0}'),
        "binary residue in content"
    );
}
