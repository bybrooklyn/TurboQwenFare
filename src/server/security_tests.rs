//! Phase 52 (spec §261, "server fuzz/security tests": malformed JSON,
//! giant payloads, invalid UTF-8 at the HTTP boundary, auth parsing) and
//! spec §268 ("non-loopback API uses auth by default"). Real raw-TCP
//! requests against the real running axum server (same harness as
//! `tests.rs`), not mocked parsing — a malformed-input handler bug would
//! show up here as a hang, a panic that kills the test process, or a
//! non-4xx status, not as a mock assertion mismatch.
//!
//! `malformed_json_is_rejected_before_reaching_generation` in `tests.rs`
//! already covers one item on spec §261's list (malformed JSON); this
//! module covers the rest that are meaningfully testable against the
//! server surface this crate actually exposes today. Not attempted:
//! "path traversal in model/index native APIs" (no HTTP route accepts a
//! filesystem path yet — retrieval is MCP/stdio-only per Phase 44, and
//! the model path is a CLI flag, not a request field) and "MCP argument
//! bounds" (MCP is a separate stdio transport, out of this HTTP-server
//! module's scope — real coverage for it belongs next to `src/mcp/`).

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::tests::{get, post_json, spawn_test_server};

/// Like `tests::http_request`, but tolerant of the server closing the
/// connection *before* the client finishes writing an oversized body —
/// a `write_all` on a raw byte slice, not a `&str` (so invalid UTF-8 can
/// be sent), and write errors (`BrokenPipe`) are swallowed rather than
/// unwrapped, since a fast, correct rejection is exactly what some of
/// these tests are checking for.
async fn raw_request(addr: SocketAddr, request: &[u8]) -> String {
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.write_all(request)).await;
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut buf)).await;
    String::from_utf8_lossy(&buf).into_owned()
}

/// spec §261 "giant arrays/messages": a 32 MB single-message body. The
/// real behavior (not assumed): axum's `Json` extractor applies a 2 MB
/// default body limit unless a route opts out, so the server is
/// expected to reject this fast with `413 Payload Too Large` — often
/// before the client has even finished writing the body, which is why
/// this uses `raw_request` (tolerant of `BrokenPipe`) rather than
/// `tests::http_request`.
#[tokio::test]
async fn giant_message_body_is_rejected_not_hung_or_crashed() {
    let addr = spawn_test_server(true).await;
    let big = "x".repeat(32 * 1024 * 1024);
    let body = format!(r#"{{"messages":[{{"role":"user","content":"{big}"}}]}}"#);
    let request = post_json("/v1/chat/completions", &body);

    let started = std::time::Instant::now();
    let response = raw_request(addr, request.as_bytes()).await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(3),
        "giant body took {elapsed:?} to resolve — looks hung, not fast-rejected"
    );
    assert!(
        response.starts_with("HTTP/1.1 4") || response.is_empty(),
        "expected a 4xx (or a closed connection with nothing readable), got: {response:.200}"
    );
}

/// spec §261 "invalid UTF-8 at HTTP boundary": a request body that is
/// syntactically almost-JSON but contains a raw invalid UTF-8 byte
/// inside the string content. `serde_json` and axum's `Json` extractor
/// must reject this as a parse error, not panic the request-handling
/// task (which — since the test server is `tokio::spawn`ed — would show
/// up here as a hung/reset connection, not a process crash, so the real
/// assertion is "the server is still answering ordinary requests
/// afterward").
#[tokio::test]
async fn invalid_utf8_body_is_rejected_and_server_stays_healthy() {
    let addr = spawn_test_server(true).await;
    let mut body = br#"{"messages":[{"role":"user","content":""#.to_vec();
    body.push(0xFF); // invalid UTF-8 continuation byte, never valid standalone
    body.push(0xFE);
    body.extend_from_slice(br#""}]}"#);

    let mut request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(&body);

    let response = raw_request(addr, &request).await;
    assert!(
        response.starts_with("HTTP/1.1 4"),
        "expected a 4xx for invalid UTF-8 in the body, got: {response:.200}"
    );

    // The server (and its single-slot generation scheduler) must still be
    // answering ordinary requests — a malformed-body task must not have
    // wedged shared state.
    let health = raw_request(addr, get("/health").as_bytes()).await;
    assert!(
        health.starts_with("HTTP/1.1 200"),
        "server did not recover after an invalid-UTF-8 body: {health:.200}"
    );
}

/// spec §268 "non-loopback API uses auth by default" — `require_api_key`
/// (`src/server/auth.rs`) had zero test coverage before this phase.
/// Real requests against a real router built with `api_key: Some(...)`,
/// covering: no `Authorization` header, wrong scheme, wrong token, and
/// the correct token.
#[tokio::test]
async fn api_key_gate_rejects_missing_or_wrong_bearer_and_accepts_the_right_one() {
    let addr = super::tests::spawn_test_server_with_api_key(true, "s3cr3t-test-key").await;

    let no_header = post_json(
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
    );
    let response = raw_request(addr, no_header.as_bytes()).await;
    assert!(
        response.starts_with("HTTP/1.1 401"),
        "missing Authorization header should be rejected: {response:.200}"
    );

    let wrong_scheme = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Authorization: Basic s3cr3t-test-key\r\nContent-Type: application/json\r\n\
         Content-Length: 2\r\nConnection: close\r\n\r\n{}";
    let response = raw_request(addr, wrong_scheme.as_bytes()).await;
    assert!(
        response.starts_with("HTTP/1.1 401"),
        "non-Bearer scheme should be rejected: {response:.200}"
    );

    let wrong_token = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Authorization: Bearer not-the-real-key\r\nContent-Type: application/json\r\n\
         Content-Length: 2\r\nConnection: close\r\n\r\n{}";
    let response = raw_request(addr, wrong_token.as_bytes()).await;
    assert!(
        response.starts_with("HTTP/1.1 401"),
        "wrong bearer token should be rejected: {response:.200}"
    );

    // `/health` lives outside the protected sub-router (unauthenticated
    // health checks are intentional — spec §268 gates the *API*, not
    // liveness probing) and must stay reachable with no key at all.
    let health = raw_request(addr, get("/health").as_bytes()).await;
    assert!(
        health.starts_with("HTTP/1.1 200"),
        "unauthenticated health check should not require the API key: {health:.200}"
    );

    let correct = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
         Authorization: Bearer s3cr3t-test-key\r\nContent-Type: application/json\r\n\
         Content-Length: 2\r\nConnection: close\r\n\r\n{}";
    let response = raw_request(addr, correct.as_bytes()).await;
    // Past the auth gate now — an empty JSON object fails ordinary
    // request-shape validation (400), not auth (401). The point of this
    // assertion is specifically "not 401".
    assert!(
        !response.starts_with("HTTP/1.1 401"),
        "correct bearer token should pass the auth gate: {response:.200}"
    );
}

/// spec §261 "header abuse": an oversized single header value (a 1 MB
/// `Authorization` header). Must be rejected cleanly (or hit hyper's own
/// header-size limit, which closes the connection) — never a hang or a
/// crash that takes the test process down with it.
#[tokio::test]
async fn oversized_header_value_is_rejected_not_hung() {
    let addr = spawn_test_server(true).await;
    let huge_token = "a".repeat(1024 * 1024);
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {huge_token}\r\n\
         Connection: close\r\n\r\n"
    );

    let started = std::time::Instant::now();
    let response = raw_request(addr, request.as_bytes()).await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(3),
        "oversized header took {elapsed:?} to resolve — looks hung"
    );
    // Either hyper's own header-size cap closes the connection (empty
    // response) or the request is parsed and answered normally (this
    // route has no api_key configured, so it is unauthenticated) — both
    // are "handled," neither is a crash or a hang.
    assert!(
        response.starts_with("HTTP/1.1") || response.is_empty(),
        "unexpected response shape: {response:.200}"
    );
}
