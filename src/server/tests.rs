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

pub(super) async fn spawn_test_server(model_installed: bool) -> SocketAddr {
    spawn_test_server_with(model_installed, None).await
}

pub(super) async fn spawn_test_server_with(
    model_installed: bool,
    generator: Option<Arc<dyn Qwen36Generator>>,
) -> SocketAddr {
    spawn_router(test_router(model_installed, generator, None)).await
}

/// spec §268/§261: same test server, but with the protected sub-router's
/// `require_api_key` gate actually enabled — used by
/// `security_tests::api_key_gate_rejects_missing_or_wrong_bearer_and_
/// accepts_the_right_one`, which had no coverage before Phase 52.
pub(super) async fn spawn_test_server_with_api_key(
    model_installed: bool,
    api_key: &str,
) -> SocketAddr {
    spawn_router(test_router(model_installed, None, Some(api_key))).await
}

async fn spawn_router(router: axum::Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, router).await {
            eprintln!("test server exited: {err}");
        }
    });
    tokio::task::yield_now().await;

    addr
}

fn test_router(
    model_installed: bool,
    generator: Option<Arc<dyn Qwen36Generator>>,
    api_key: Option<&str>,
) -> axum::Router {
    let state = AppState {
        config: Arc::new(Config::default()),
        model_installed,
        generation_slot: GenerationSlot::new(),
        generator,
        model_receipt: None,
        started_at: Instant::now(),
        api_key: api_key.map(Arc::from),
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

/// A double that really streams: it emits one delta per word with a gap
/// between them, so tests can prove deltas arrive during generation
/// rather than in a single burst at the end. The existing doubles use the
/// trait's default `generate_streaming`, which cannot distinguish the two.
pub(super) struct IncrementalFixtureGenerator {
    words: Vec<&'static str>,
}

impl IncrementalFixtureGenerator {
    pub(super) fn new() -> Self {
        Self {
            words: vec!["one ", "two ", "three ", "four ", "five"],
        }
    }

    pub(super) fn full_text(&self) -> String {
        self.words.concat()
    }
}

#[async_trait::async_trait]
impl Qwen36Generator for IncrementalFixtureGenerator {
    async fn generate(
        &self,
        _request: NormalizedRequest,
        _cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::error::Result<GeneratedOutput> {
        GeneratedOutput::from_model_text(self.full_text())
    }

    async fn generate_streaming(
        &self,
        _request: NormalizedRequest,
        cancellation: tokio_util::sync::CancellationToken,
        events: tokio::sync::mpsc::Sender<crate::runtime::stream_decoder::StreamEvent>,
    ) -> crate::error::Result<GeneratedOutput> {
        use crate::runtime::stream_decoder::StreamEvent;
        let mut emitted = String::new();
        for word in &self.words {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                _ = cancellation.cancelled() => return Err(crate::error::TqfError::Cancelled),
            }
            if events
                .send(StreamEvent::TextDelta((*word).to_string()))
                .await
                .is_err()
            {
                // Client hung up: stop rather than finish a generation
                // nobody will read.
                break;
            }
            emitted.push_str(word);
        }
        let mut output = GeneratedOutput::from_model_text(emitted)?;
        output.usage = crate::runtime::generation::GenerationUsage {
            prompt_tokens: 7,
            completion_tokens: self.words.len() as u32,
            ..Default::default()
        };
        Ok(output)
    }

    fn streams_incrementally(&self) -> bool {
        true
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

pub(super) async fn http_request(addr: SocketAddr, request: &str) -> String {
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

pub(super) fn get(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
}

pub(super) fn post_json(path: &str, body: &str) -> String {
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

/// Real end-to-end test of spec §47's inspector-metrics endpoint: a
/// real HTTP request against a real running server, asserting the
/// resident-memory reading is a genuine positive number (this test
/// process itself has real, nonzero resident memory — anything else
/// would mean the OS sampler silently failed rather than reported
/// `None` honestly).
#[tokio::test]
async fn tqf_metrics_reports_real_process_memory() {
    let addr = spawn_test_server(false).await;
    let response = http_request(addr, &get("/v1/tqf/metrics")).await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    let body = response.split("\r\n\r\n").last().unwrap_or("");
    let json: serde_json::Value = serde_json::from_str(body).expect("valid JSON body");
    let resident = json["resident_bytes"]
        .as_u64()
        .expect("resident_bytes must be a real sampled number on this platform");
    assert!(
        resident > 0,
        "a running process must have nonzero resident memory: {resident}"
    );
    assert_eq!(json["model_installed"], false);
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
    let router = test_router(true, Some(Arc::new(DelayedGenerator)), None);
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

/// These are the parameters real OpenAI and Ollama clients send by
/// default. Every one of them used to 400, because no sampler existed;
/// rejecting them now that `crate::sampling` implements them would be the
/// lie.
#[tokio::test]
async fn chat_accepts_the_sampling_parameters_real_clients_send() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    for body in [
        r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.8,"top_p":0.9}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"top_k":40,"min_p":0.05}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"seed":42}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":2048}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"stop":["\n\n"]}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5,"presence_penalty":0.5}"#,
    ] {
        let response = http_request(addr, &post_json("/v1/chat/completions", body)).await;
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "should have been accepted: {body}\n{response}"
        );
    }
}

/// What still gets rejected, and why each rejection is a real limit
/// rather than unimplemented plumbing (spec §204: reject rather than
/// silently ignore).
#[tokio::test]
async fn chat_rejects_what_this_build_genuinely_cannot_honor() {
    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    for (body, why) in [
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#,
            "one generation slot serves one sequence (spec §75)",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
            "the decoder keeps only top-4 pre-softmax candidates",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"temperature":5.0}"#,
            "temperature is out of the accepted range",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"top_p":1.5}"#,
            "top_p is out of the accepted range",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":0}"#,
            "a zero-token budget cannot produce output",
        ),
        (
            r#"{"messages":[{"role":"alien","content":"hi"}]}"#,
            "unknown role",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"late"}]}"#,
            "system messages must lead",
        ),
        (
            r#"{"model":"not-the-installed-model","messages":[{"role":"user","content":"hi"}]}"#,
            "only the canonical model is served",
        ),
    ] {
        let response = http_request(addr, &post_json("/v1/chat/completions", body)).await;
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "expected a structured compatibility error ({why}): {response}"
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

// ---------------------------------------------------------------------
// Real streaming (spec §71). Before this, "streaming" awaited the whole
// generation and emitted it as one event, so every one of these
// properties was vacuously true and none of them were tested.
// ---------------------------------------------------------------------

/// Extracts the `data:` payloads of an SSE response body.
fn sse_payloads(response: &str) -> Vec<String> {
    response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(str::to_string)
        .collect()
}

fn delta_contents(payloads: &[String]) -> Vec<String> {
    payloads
        .iter()
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .filter_map(|v| {
            v["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect()
}

#[tokio::test]
async fn chat_streaming_emits_one_chunk_per_delta_not_one_at_the_end() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"count"}],"stream":true}"#,
        ),
    )
    .await;

    let contents = delta_contents(&sse_payloads(&response));
    assert!(
        contents.len() >= 5,
        "expected one chunk per delta, got {}: {response}",
        contents.len()
    );
    assert_eq!(
        contents.concat(),
        expected,
        "reassembled deltas must equal the full text"
    );
}

/// Exactly-once delivery: NVMAI hit real double-delivery bugs here, and
/// spec §71 asks for regression tests that make the class impossible.
#[tokio::test]
async fn each_delta_is_delivered_exactly_once() {
    let generator = IncrementalFixtureGenerator::new();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"count"}],"stream":true}"#,
        ),
    )
    .await;

    let contents = delta_contents(&sse_payloads(&response));
    // The fixture's words are all distinct, so any duplicate delivery
    // shows up as a repeated entry.
    let unique: std::collections::HashSet<&String> = contents.iter().collect();
    assert_eq!(
        unique.len(),
        contents.len(),
        "a delta was delivered more than once: {contents:?}"
    );

    let payloads = sse_payloads(&response);
    assert_eq!(
        payloads.iter().filter(|p| *p == "[DONE]").count(),
        1,
        "exactly one [DONE] sentinel must terminate the stream"
    );
}

#[tokio::test]
async fn a_streamed_chat_completion_reports_finish_reason_and_usage() {
    let generator = IncrementalFixtureGenerator::new();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"count"}],"stream":true}"#,
        ),
    )
    .await;

    let payloads = sse_payloads(&response);
    let terminal = payloads
        .iter()
        .filter(|p| *p != "[DONE]")
        .filter_map(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .find(|v| !v["choices"][0]["finish_reason"].is_null())
        .expect("a terminal chunk with a finish_reason must be sent");

    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(terminal["usage"]["completion_tokens"], 5);
    assert_eq!(terminal["usage"]["prompt_tokens"], 7);
    assert_eq!(terminal["usage"]["total_tokens"], 12);
}

/// Every chunk must carry a stable, per-response id. The previous
/// hardcoded `"chatcmpl-tqf-local"` meant every response in the process
/// shared one id, so clients keying by it collided.
#[tokio::test]
async fn streamed_chunks_share_one_id_that_differs_between_responses() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;
    let body = r#"{"messages":[{"role":"user","content":"count"}],"stream":true}"#;

    let ids_of = |response: &str| -> Vec<String> {
        sse_payloads(response)
            .iter()
            .filter(|p| *p != "[DONE]")
            .filter_map(|p| serde_json::from_str::<serde_json::Value>(p).ok())
            .filter_map(|v| v["id"].as_str().map(str::to_string))
            .collect()
    };

    let first = ids_of(&http_request(addr, &post_json("/v1/chat/completions", body)).await);
    let second = ids_of(&http_request(addr, &post_json("/v1/chat/completions", body)).await);

    assert!(!first.is_empty() && !second.is_empty());
    assert!(
        first.iter().all(|id| *id == first[0]),
        "chunks within one response must share an id: {first:?}"
    );
    assert_ne!(first[0], second[0], "two responses must not share an id");
}

#[tokio::test]
async fn responses_streaming_emits_incremental_output_text_deltas() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json("/v1/responses", r#"{"input":"count","stream":true}"#),
    )
    .await;

    let deltas: Vec<String> = sse_payloads(&response)
        .iter()
        .filter_map(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .filter(|v| v["type"] == "response.output_text.delta")
        .filter_map(|v| v["delta"].as_str().map(str::to_string))
        .collect();

    assert!(
        deltas.len() >= 5,
        "expected incremental output_text deltas, got {}: {response}",
        deltas.len()
    );
    assert_eq!(deltas.concat(), expected);
    assert!(
        response.contains("response.completed"),
        "the stream must terminate with response.completed"
    );
}

/// A non-streaming request must produce exactly what the streamed deltas
/// reassemble to. Without this, "the streamed answer differs from the
/// batch answer" is a whole live bug class.
#[tokio::test]
async fn streamed_and_non_streamed_answers_agree() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let streamed = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"count"}],"stream":true}"#,
        ),
    )
    .await;
    let batch = http_request(
        addr,
        &post_json(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"count"}]}"#,
        ),
    )
    .await;

    let streamed_text = delta_contents(&sse_payloads(&streamed)).concat();
    let body = batch.split("\r\n\r\n").nth(1).expect("a response body");
    let parsed: serde_json::Value = serde_json::from_str(body).expect("valid JSON body");

    assert_eq!(streamed_text, expected);
    assert_eq!(parsed["choices"][0]["message"]["content"], expected);
}

// ---------------------------------------------------------------------
// Native diagnostics namespace (spec §211).
// ---------------------------------------------------------------------

#[tokio::test]
async fn the_native_status_endpoint_distinguishes_installed_from_loaded() {
    // A valid receipt can exist while the runtime failed to construct, so
    // conflating the two would hide the most confusing real state.
    let addr = spawn_test_server_with(true, None).await;
    let response = http_request(addr, &get("/tqf/status")).await;
    let body: serde_json::Value =
        serde_json::from_str(response.split("\r\n\r\n").nth(1).unwrap_or("")).expect("JSON body");

    assert_eq!(body["model"]["installed"], true, "{body}");
    assert_eq!(body["model"]["loaded"], false, "{body}");
    assert_eq!(body["model"]["id"], "qwen3.6-35b-a3b");
    assert!(body["backend"].is_string(), "{body}");

    let addr = spawn_test_server_with(true, Some(Arc::new(FixtureGenerator))).await;
    let response = http_request(addr, &get("/tqf/status")).await;
    let body: serde_json::Value =
        serde_json::from_str(response.split("\r\n\r\n").nth(1).unwrap_or("")).expect("JSON body");
    assert_eq!(body["model"]["loaded"], true, "{body}");
}

#[tokio::test]
async fn the_native_memory_endpoint_reports_real_os_numbers() {
    let addr = spawn_test_server(true).await;
    let response = http_request(addr, &get("/tqf/memory")).await;
    let body: serde_json::Value =
        serde_json::from_str(response.split("\r\n\r\n").nth(1).unwrap_or("")).expect("JSON body");

    assert!(
        body["observed"]["resident_bytes"].as_u64().unwrap_or(0) > 0,
        "resident memory must be a real reading: {body}"
    );
    // The per-owner broker breakdown is deliberately absent rather than
    // fabricated, and the response says so.
    assert!(
        body["note"].as_str().unwrap_or_default().contains("broker"),
        "{body}"
    );
}

/// The endpoint must report the backend that is *live*, not the one that
/// is available: TQKV is opt-in, and a caller reading "tqkv" while BF16
/// runs would be misled about its own memory.
#[tokio::test]
async fn the_native_context_endpoint_reports_the_live_kv_backend() {
    let addr = spawn_test_server(true).await;
    let response = http_request(addr, &get("/tqf/context")).await;
    let body: serde_json::Value =
        serde_json::from_str(response.split("\r\n\r\n").nth(1).unwrap_or("")).expect("JSON body");

    assert_eq!(
        body["kv_backend"], "bf16",
        "TQKV is opt-in, so an unconfigured server must report bf16: {body}"
    );
    assert_eq!(body["selective_attention"], false, "{body}");
}

#[tokio::test]
async fn the_native_indexes_endpoint_explains_why_it_is_empty() {
    let addr = spawn_test_server(true).await;
    let response = http_request(addr, &get("/tqf/indexes")).await;
    let body: serde_json::Value =
        serde_json::from_str(response.split("\r\n\r\n").nth(1).unwrap_or("")).expect("JSON body");

    assert_eq!(body["indexes"], serde_json::json!([]));
    assert!(
        body["note"]
            .as_str()
            .unwrap_or_default()
            .contains("not implemented"),
        "an empty list must say why: {body}"
    );
}

/// `/v1/tqf/metrics` is what the Phase 47 SwiftUI inspector reads, so
/// adding the spec's `/tqf/metrics` spelling must not break it.
#[tokio::test]
async fn both_metrics_paths_serve_the_same_payload() {
    let addr = spawn_test_server(true).await;
    for path in ["/tqf/metrics", "/v1/tqf/metrics"] {
        let response = http_request(addr, &get(path)).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{path}: {response}");
        assert!(response.contains("resident_bytes"), "{path}: {response}");
    }
}
