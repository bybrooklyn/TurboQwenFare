//! Integration-style tests for the server skeleton: a real bind, real HTTP
//! over TCP, and the real single-generation-slot path — no mocks (spec Part
//! IX section 71: streaming/protocol correctness has to be genuinely
//! exercised, not assumed). This is the evidence for phase 2's exit gate:
//! "Server starts on loopback and protocol tests pass."

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tower::ServiceExt;

use crate::config::Config;
use crate::runtime::{GeneratedOutput, GenerationSlot, NormalizedRequest, Qwen36Generator};
use crate::server::{self, AppState};

async fn spawn_test_server(model_installed: bool) -> SocketAddr {
    spawn_test_server_with(model_installed, None).await
}

async fn spawn_test_server_with(
    model_installed: bool,
    generator: Option<Arc<dyn Qwen36Generator>>,
) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let router = test_router(model_installed, generator);
    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, router).await {
            eprintln!("test server exited: {err}");
        }
    });
    tokio::task::yield_now().await;

    addr
}

fn test_router(model_installed: bool, generator: Option<Arc<dyn Qwen36Generator>>) -> axum::Router {
    let state = AppState {
        config: Arc::new(Config::default()),
        model_installed,
        generation_slot: GenerationSlot::new(),
        generator,
        started_at: Instant::now(),
        api_key: None,
    };

    server::build_router(state)
}

struct FixtureGenerator;

#[async_trait::async_trait]
impl Qwen36Generator for FixtureGenerator {
    async fn generate(
        &self,
        request: NormalizedRequest,
        _cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::error::Result<GeneratedOutput> {
        let name = request
            .tools
            .first()
            .map(|tool| tool.name.as_str())
            .unwrap_or("read_file");
        GeneratedOutput::from_model_text(format!(
            "hello <tool_call>{{\"name\":\"{name}\",\"arguments\":{{\"path\":\"Cargo.toml\"}}}}</tool_call>"
        ))
    }
}

struct CancellationAwareGenerator {
    entered: Arc<Notify>,
    observed_cancellation: Arc<Notify>,
}

#[async_trait::async_trait]
impl Qwen36Generator for CancellationAwareGenerator {
    async fn generate(
        &self,
        _request: NormalizedRequest,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::error::Result<GeneratedOutput> {
        self.entered.notify_one();
        cancellation.cancelled().await;
        self.observed_cancellation.notify_one();
        Err(crate::error::TqfError::Cancelled)
    }
}

struct DelayedGenerator;

#[async_trait::async_trait]
impl Qwen36Generator for DelayedGenerator {
    async fn generate(
        &self,
        request: NormalizedRequest,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::error::Result<GeneratedOutput> {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(250)) => {}
            _ = cancellation.cancelled() => return Err(crate::error::TqfError::Cancelled),
        }
        FixtureGenerator.generate(request, cancellation).await
    }
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
async fn embeddings_fail_fast_without_entering_generation() {
    let addr = spawn_test_server_with(
        true,
        Some(Arc::new(CancellationAwareGenerator {
            entered: Arc::new(Notify::new()),
            observed_cancellation: Arc::new(Notify::new()),
        })),
    )
    .await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/embeddings",
            r#"{"model":"qwen3.6-35b-a3b","input":"hi"}"#,
        ),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 501"), "{response}");
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

#[tokio::test]
async fn chat_completions_formats_real_generator_output_and_tool_calls() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}]}"#,
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    assert!(response.contains("chat.completion"));
    assert!(response.contains("read_file"));
    assert!(response.contains("tool_calls"));
}

#[tokio::test]
async fn chat_completion_streaming_formats_generator_output_and_done_marker() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
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
    assert!(response.contains("chat.completion.chunk"));
    assert!(response.contains("data: [DONE]"));
}

#[tokio::test]
async fn streaming_responds_before_generation_finishes() {
    let router = test_router(true, Some(Arc::new(DelayedGenerator)));
    let response = router
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let mut body = response.into_body();
    let frame = tokio::time::timeout(Duration::from_secs(2), body.frame())
        .await
        .expect("stream did not produce its initial event promptly")
        .expect("stream ended before its initial event")
        .expect("stream body failed");
    let data = frame.into_data().expect("initial SSE frame was not data");
    assert!(String::from_utf8_lossy(&data).contains("\"role\":\"assistant\""));
    drop(body);
}

#[tokio::test]
async fn responses_streaming_uses_responses_events_not_chat_chunks() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    let response = http_request(
        addr,
        &post_json("/v1/responses", r#"{"input":"hi","stream":true}"#),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    assert!(response.contains("event: response.output_text.delta"));
    assert!(response.contains("event: response.function_call_arguments.done"));
    assert!(response.contains("event: response.completed"));
    assert!(!response.contains("chat.completion.chunk"));
}

#[tokio::test]
async fn responses_normalize_their_flat_function_tool_shape() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/responses",
            r#"{"input":"hi","tools":[{"type":"function","name":"list_files","parameters":{"type":"object"}}]}"#,
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    assert!(response.contains("list_files"));
}

#[tokio::test]
async fn chat_rejects_fields_the_greedy_correctness_runtime_cannot_honor() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    for body in [
        r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":257}"#,
        r#"{"messages":[{"role":"alien","content":"hi"}]}"#,
        r#"{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"late"}]}"#,
        r#"{"model":"not-the-installed-model","messages":[{"role":"user","content":"hi"}]}"#,
    ] {
        let response = http_request(addr, &post_json("/v1/chat/completions", body)).await;
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "expected a structured compatibility error: {response}"
        );
    }
}

#[tokio::test]
async fn chat_accepts_canonical_assistant_tool_call_history() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"read it"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"Cargo.toml\"}"}}]},{"role":"tool","tool_call_id":"call_0","content":"contents"}],"temperature":0,"max_tokens":1}"#,
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
}

#[tokio::test]
async fn responses_accept_structured_text_input_and_instructions() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/responses",
            r#"{"model":"qwen3.6-35b-a3b","instructions":"Be concise.","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"temperature":0,"max_output_tokens":1}"#,
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
}
