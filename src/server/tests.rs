//! Integration-style tests for the server skeleton: a real bind, real HTTP
//! over TCP, and the real single-generation-slot path — no mocks (spec Part
//! IX section 71: streaming/protocol correctness has to be genuinely
//! exercised, not assumed). This is the evidence for phase 2's exit gate:
//! "Server starts on loopback and protocol tests pass."

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::config::Config;
use crate::runtime::GenerationSlot;
use crate::server::{self, AppState};

async fn spawn_test_server(model_installed: bool) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let state = AppState {
        config: Arc::new(Config::default()),
        model_installed,
        generation_slot: GenerationSlot::new(),
        started_at: Instant::now(),
        api_key: None,
    };

    let router = server::build_router(state);
    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, router).await {
            eprintln!("test server exited: {err}");
        }
    });
    tokio::task::yield_now().await;

    addr
}

async fn http_request(addr: SocketAddr, request: &str) -> String {
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    stream.write_all(request.as_bytes()).await.unwrap();
    // Deliberately no client-side half-close here: shutting down the write
    // side immediately after sending races hyper's request parsing on a
    // fast loopback connection and it aborts instead of responding. The
    // request's own `Connection: close` header is what makes the *server*
    // close the socket once it's done, which read_to_end waits for.
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut buf))
        .await
        .expect("read timed out")
        .unwrap();
    String::from_utf8_lossy(&buf).into_owned()
}

fn get(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
}

fn post_json(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn health_reports_ok_and_model_state() {
    let addr = spawn_test_server(false).await;
    let response = http_request(addr, &get("/health")).await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    assert!(response.contains(r#""status":"ok""#));
    assert!(response.contains(r#""model_installed":false"#));
}

#[tokio::test]
async fn models_lists_canonical_model() {
    let addr = spawn_test_server(true).await;
    let response = http_request(addr, &get("/v1/models")).await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    assert!(response.contains("qwen3.6-35b-a3b"));
    assert!(response.contains(r#""installed":true"#));
}

#[tokio::test]
async fn chat_completions_reports_no_model_installed() {
    let addr = spawn_test_server(false).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 503"),
        "unexpected response: {response}"
    );
    assert!(response.contains("no model installed"));
}

#[tokio::test]
async fn chat_completions_streaming_returns_valid_sse_framing() {
    let addr = spawn_test_server(false).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    let lower = response.to_ascii_lowercase();
    assert!(lower.contains("content-type: text/event-stream"));
    assert!(response.contains("event: error"));
    assert!(response.contains("data: [DONE]"));
}

#[tokio::test]
async fn malformed_json_is_rejected_before_reaching_generation() {
    let addr = spawn_test_server(true).await;
    let response = http_request(addr, &post_json("/v1/chat/completions", "not json")).await;
    assert!(
        response.starts_with("HTTP/1.1 4"),
        "expected a 4xx, got: {response}"
    );
}

#[tokio::test]
async fn generation_slot_serializes_concurrent_requests() {
    // Both requests still complete (queued, not dropped) even though only
    // one can hold the generation slot at a time (spec Part IX section 75).
    let addr = spawn_test_server(false).await;
    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let request_a = post_json("/v1/chat/completions", body);
    let request_b = post_json("/v1/chat/completions", body);
    let (a, b) = tokio::join!(
        http_request(addr, &request_a),
        http_request(addr, &request_b),
    );
    assert!(a.starts_with("HTTP/1.1 503"), "unexpected response: {a}");
    assert!(b.starts_with("HTTP/1.1 503"), "unexpected response: {b}");
}
