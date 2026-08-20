//! Real-HTTP tests for the Ollama-compatible surface.
//!
//! These use the same real-bind, real-TCP style as `server::tests`,
//! because the failure modes this adapter exists to prevent are wire
//! framing bugs — a `data:` prefix, a missing terminal object — that an
//! in-process handler call would never surface.

use std::sync::Arc;

use serde_json::Value;

use crate::server::tests::{
    http_request, post_json, spawn_test_server_with, spawn_test_server_with_api_key,
    IncrementalFixtureGenerator,
};

fn get(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
}

fn body_of(response: &str) -> &str {
    response.split("\r\n\r\n").nth(1).unwrap_or("")
}

fn json_body(response: &str) -> Value {
    serde_json::from_str(body_of(response))
        .unwrap_or_else(|e| panic!("expected a JSON body, got {response:?}: {e}"))
}

/// Splits an NDJSON body into its lines, which is what a real client's
/// parser does.
fn ndjson_lines(response: &str) -> Vec<Value> {
    body_of(response)
        .lines()
        .filter(|line| !line.trim().is_empty())
        // Skip HTTP chunked-transfer size markers, which appear when the
        // response is streamed over a raw socket.
        .filter(|line| line.trim_start().starts_with('{'))
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("NDJSON line is not bare JSON: {line:?}: {e}"))
        })
        .collect()
}

// ------------------------------------------------------------- liveness

/// Clients probe these before they have anywhere to put a credential, so
/// they must answer without one — and must answer with the exact strings
/// clients match on.
#[tokio::test]
async fn root_and_version_answer_the_liveness_handshake() {
    let addr = spawn_test_server_with(false, None).await;

    let root = http_request(addr, &get("/")).await;
    assert!(root.starts_with("HTTP/1.1 200"), "{root}");
    assert!(
        root.contains("Ollama is running"),
        "clients match this exact string: {root}"
    );

    let version = http_request(addr, &get("/api/version")).await;
    assert!(version.starts_with("HTTP/1.1 200"), "{version}");
    assert!(json_body(&version)["version"].is_string(), "{version}");
}

/// The security-critical half of the router split: liveness is open,
/// everything that generates or enumerates is not. Merging the Ollama
/// routes at the top level (the easy mistake) would expose generation
/// with no key on a `0.0.0.0` bind (spec §74).
#[tokio::test]
async fn liveness_is_unauthenticated_but_generation_and_inventory_are_not() {
    let addr = spawn_test_server_with_api_key(true, "secret-key").await;

    for path in ["/", "/api/version", "/health"] {
        let response = http_request(addr, &get(path)).await;
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "{path} must not require a key: {response}"
        );
    }

    for path in ["/api/tags", "/api/ps"] {
        let response = http_request(addr, &get(path)).await;
        assert!(
            response.starts_with("HTTP/1.1 401"),
            "{path} must require a key: {response}"
        );
    }

    for path in ["/api/chat", "/api/generate", "/api/show", "/api/embed"] {
        let response = http_request(addr, &post_json(path, r#"{"model":"qwen3.6:35b"}"#)).await;
        assert!(
            response.starts_with("HTTP/1.1 401"),
            "{path} must require a key: {response}"
        );
    }

    // And the key actually works.
    let authorized = "GET /api/tags HTTP/1.1\r\nHost: localhost\r\n\
         Authorization: Bearer secret-key\r\nConnection: close\r\n\r\n";
    let response = http_request(addr, authorized).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
}

// ------------------------------------------------------------ inventory

/// An empty library is what real Ollama reports when nothing is
/// installed — not an error.
#[tokio::test]
async fn tags_and_ps_report_an_empty_library_rather_than_failing() {
    let addr = spawn_test_server_with(false, None).await;

    for path in ["/api/tags", "/api/ps"] {
        let response = http_request(addr, &get(path)).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{path}: {response}");
        assert_eq!(
            json_body(&response)["models"],
            serde_json::json!([]),
            "{path} must return an empty array"
        );
    }
}

#[tokio::test]
async fn ps_lists_the_loaded_model_with_real_resident_bytes() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;
    let response = http_request(addr, &get("/api/ps")).await;
    let models = &json_body(&response)["models"];

    assert_eq!(models.as_array().map(Vec::len), Some(1), "{response}");
    assert_eq!(models[0]["name"], "qwen3.6:35b");
    assert!(
        models[0]["size"].as_u64().unwrap_or(0) > 0,
        "resident size must be a real measurement: {response}"
    );
    // Experts stream from SSD and the GPU-resident path is opt-in, so 0
    // is the honest answer rather than an unset field.
    assert_eq!(models[0]["size_vram"], 0);
}

#[tokio::test]
async fn show_rejects_an_unknown_model_and_reports_a_missing_one_honestly() {
    let addr = spawn_test_server_with(false, None).await;

    let unknown = http_request(addr, &post_json("/api/show", r#"{"model":"llama3"}"#)).await;
    assert!(unknown.starts_with("HTTP/1.1 400"), "{unknown}");
    assert!(
        json_body(&unknown)["error"].is_string(),
        "Ollama's envelope is a flat string: {unknown}"
    );

    let missing = http_request(addr, &post_json("/api/show", r#"{"model":"qwen3.6:35b"}"#)).await;
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
}

/// Not required by spec §210, but a 501 that says why beats an anonymous
/// 404 that looks like a routing bug.
#[tokio::test]
async fn model_management_endpoints_return_501_with_a_reason() {
    let addr = spawn_test_server_with(true, None).await;

    for path in ["/api/pull", "/api/push", "/api/create", "/api/copy"] {
        let response = http_request(addr, &post_json(path, r#"{"name":"llama3"}"#)).await;
        assert!(
            response.starts_with("HTTP/1.1 501"),
            "{path} should be 501, not 404: {response}"
        );
        let body = json_body(&response);
        let error = body["error"].as_str().unwrap_or_default();
        assert!(
            error.contains("one pinned model"),
            "{path} must say why: {error}"
        );
    }
}

// ----------------------------------------------------------- generation

#[tokio::test]
async fn chat_streams_ndjson_with_a_terminal_done_object() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"count"}]}"#,
        ),
    )
    .await;

    assert!(
        response.to_lowercase().contains("application/x-ndjson"),
        "NDJSON content-type is what clients switch parsers on: {response}"
    );
    // SSE framing here would parse as garbage in every Ollama client.
    assert!(
        !response.contains("data: "),
        "SSE prefix leaked: {response}"
    );
    assert!(
        !response.contains("[DONE]"),
        "SSE sentinel leaked: {response}"
    );

    let lines = ndjson_lines(&response);
    assert!(
        lines.len() > 1,
        "expected incremental lines, got {}: {response}",
        lines.len()
    );

    let (last, partials) = lines.split_last().expect("at least one line");
    for line in partials {
        assert_eq!(line["done"], false, "partial lines must carry done:false");
        assert_eq!(line["model"], "qwen3.6:35b", "the requested tag is echoed");
    }

    // Clients stop on done:true, not on stream close.
    assert_eq!(last["done"], true, "terminal object missing: {response}");
    assert_eq!(last["done_reason"], "stop");
    assert!(last["eval_count"].as_u64().unwrap_or(0) > 0, "{last}");
    assert!(
        last["prompt_eval_count"].as_u64().unwrap_or(0) > 0,
        "{last}"
    );
    assert!(last["total_duration"].is_number(), "{last}");

    let streamed: String = partials
        .iter()
        .filter_map(|line| line["message"]["content"].as_str())
        .collect();
    assert_eq!(
        streamed, expected,
        "deltas must reassemble to the full text"
    );
}

#[tokio::test]
async fn chat_with_stream_false_returns_one_object_carrying_the_whole_message() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/api/chat",
            r#"{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;

    let body = json_body(&response);
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["message"]["content"], expected);
    assert_eq!(body["done_reason"], "stop");
}

/// `/api/generate` differs from `/api/chat` only in this key, and a
/// client parsing one shape against the other silently sees empty output.
#[tokio::test]
async fn generate_uses_response_rather_than_message() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/api/generate",
            r#"{"model":"qwen3.6:35b","stream":false,"prompt":"hi"}"#,
        ),
    )
    .await;

    let body = json_body(&response);
    assert_eq!(body["response"], expected);
    assert!(
        body["message"].is_null(),
        "generate must not nest a message"
    );
    assert_eq!(body["done"], true);
}

/// Ollama's `stream` defaults to true, the opposite of OpenAI's. A client
/// that omits it expects a stream.
#[tokio::test]
async fn stream_defaults_to_true_unlike_openai() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;
    let response = http_request(
        addr,
        &post_json(
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    assert!(
        response.to_lowercase().contains("application/x-ndjson"),
        "omitting stream must still stream: {response}"
    );
}

// ------------------------------------------------------ parameter policy

#[tokio::test]
async fn ollama_style_tags_are_accepted_and_echoed_back() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;

    for tag in [
        "qwen3.6:35b",
        "qwen3.6",
        "qwen3.6:latest",
        "qwen3.6-35b-a3b",
    ] {
        let response = http_request(
            addr,
            &post_json(
                "/api/chat",
                &format!(
                    r#"{{"model":"{tag}","stream":false,"messages":[{{"role":"user","content":"hi"}}]}}"#
                ),
            ),
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 200"), "{tag}: {response}");
        assert_eq!(
            json_body(&response)["model"],
            tag,
            "clients key session state on the tag they sent"
        );
    }
}

#[tokio::test]
async fn the_sampling_options_real_clients_send_are_accepted() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;

    for options in [
        r#"{"temperature":0.8,"top_p":0.9,"top_k":40}"#,
        r#"{"seed":42,"repeat_penalty":1.1,"repeat_last_n":64}"#,
        r#"{"num_predict":-1}"#,
        r#"{"stop":["\n\n"],"min_p":0.05}"#,
        // Ollama's own shipped defaults for strategies this build lacks.
        r#"{"mirostat":0,"tfs_z":1.0,"typical_p":1.0}"#,
        r#"{"num_gpu":0,"numa":false,"unknown_future_option":7}"#,
    ] {
        let body = format!(
            r#"{{"model":"qwen3.6:35b","stream":false,"messages":[{{"role":"user","content":"hi"}}],"options":{options}}}"#
        );
        let response = http_request(addr, &post_json("/api/chat", &body)).await;
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "options {options} should be accepted: {response}"
        );
    }
}

/// Each of these would produce silently wrong output if accepted and
/// ignored, which spec §204 forbids.
#[tokio::test]
async fn parameters_that_cannot_be_honored_are_rejected_not_ignored() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;

    for (path, body, why) in [
        (
            "/api/generate",
            r#"{"model":"qwen3.6:35b","prompt":"x","raw":true}"#,
            "raw would skip the pinned chat template",
        ),
        (
            "/api/generate",
            r#"{"model":"qwen3.6:35b","prompt":"x","context":[1,2,3]}"#,
            "context has no TQF equivalent",
        ),
        (
            "/api/generate",
            r#"{"model":"qwen3.6:35b","prompt":"x","template":"{{ .Prompt }}"}"#,
            "the template is pinned",
        ),
        (
            "/api/generate",
            r#"{"model":"qwen3.6:35b","prompt":"x","suffix":"tail"}"#,
            "fill-in-the-middle is not implemented",
        ),
        (
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"x"}],"format":"json"}"#,
            "there is no grammar enforcement",
        ),
        (
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"x"}],"options":{"mirostat":2}}"#,
            "mirostat is not implemented",
        ),
        (
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"x","images":["aGk="]}]}"#,
            "the vision path is not wired",
        ),
    ] {
        let response = http_request(addr, &post_json(path, body)).await;
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "{path} should reject ({why}): {response}"
        );
        assert!(
            json_body(&response)["error"].is_string(),
            "errors use Ollama's flat envelope: {response}"
        );
    }
}

/// Spec §212: each surface returns its own error shape. An Ollama client
/// cannot parse OpenAI's nested `{"error": {"message": ...}}`.
#[tokio::test]
async fn unavailable_state_uses_ollamas_flat_error_envelope() {
    let addr = spawn_test_server_with(false, None).await;

    let response = http_request(
        addr,
        &post_json(
            "/api/chat",
            r#"{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    let error = &json_body(&response)["error"];
    assert!(
        error.is_string(),
        "Ollama's envelope is a flat string, not a nested object: {response}"
    );
    assert!(
        error.as_str().unwrap_or_default().contains("no model"),
        "the message must say what is actually wrong: {error}"
    );
}

#[tokio::test]
async fn embeddings_report_the_missing_checkpoint_pin_honestly() {
    let addr = spawn_test_server_with(true, None).await;

    for path in ["/api/embed", "/api/embeddings"] {
        let response = http_request(addr, &post_json(path, r#"{"input":"hello"}"#)).await;
        assert!(response.starts_with("HTTP/1.1 501"), "{path}: {response}");
        let error = json_body(&response)["error"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(
            error.contains("not pinned"),
            "{path} must name the real gap, not just decline: {error}"
        );
    }
}

// -------------------------------------------------------- body decoding

/// Ollama's own README documents every endpoint as a bare
/// `curl http://localhost:11434/api/generate -d '{...}'`, and `curl -d`
/// sends `Content-Type: application/x-www-form-urlencoded`. Real Ollama
/// parses the body anyway; `axum::Json` would 415 it.
///
/// This is the exact class of bug this module exists to prevent — a
/// complete, correct endpoint list that every documented invocation
/// bounces off — so it is tested at the wire, with the header real curl
/// actually sends rather than with no header at all.
#[tokio::test]
async fn documented_curl_bodies_are_accepted_without_a_json_content_type() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;

    for (path, body) in [
        (
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        ),
        (
            "/api/generate",
            r#"{"model":"qwen3.6:35b","prompt":"hi","stream":false}"#,
        ),
    ] {
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: \
             application/x-www-form-urlencoded\r\nContent-Length: {}\r\nConnection: \
             close\r\n\r\n{body}",
            body.len()
        );
        let response = http_request(addr, &request).await;
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "{path} must accept the body real curl sends: {response}"
        );
    }

    // `/api/show` takes the same tolerant body, but this fixture server
    // has no receipt, so the *correct* answer is a 404 for an unknown
    // model. What matters here is that it got far enough to look: a 415
    // would mean the body was never parsed.
    let body = r#"{"model":"qwen3.6:35b"}"#;
    let show = http_request(
        addr,
        &format!(
            "POST /api/show HTTP/1.1\r\nHost: localhost\r\nContent-Type: \
             application/x-www-form-urlencoded\r\nContent-Length: {}\r\nConnection: \
             close\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    assert!(
        !show.starts_with("HTTP/1.1 415"),
        "the body must be parsed regardless of content type: {show}"
    );
    assert!(json_body(&show)["error"].is_string(), "{show}");
}

/// A body that genuinely is not JSON must still be refused — and refused
/// in Ollama's own flat `{"error": "..."}` envelope (spec §212), not
/// axum's plain-text rejection, because that is the shape a client's
/// error path parses.
#[tokio::test]
async fn a_malformed_body_is_refused_in_ollamas_error_envelope() {
    let addr = spawn_test_server_with(true, None).await;

    for body in ["not json at all", ""] {
        let request = format!(
            "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: \
             close\r\n\r\n{body}",
            body.len()
        );
        let response = http_request(addr, &request).await;
        assert!(
            response.starts_with("HTTP/1.1 400"),
            "{body:?} must be a 400: {response}"
        );
        let error = json_body(&response);
        assert!(
            error["error"].is_string(),
            "expected Ollama's flat error envelope, got {error}"
        );
    }
}

/// A streaming response commits to its status code with the first byte,
/// so readiness has to be checked *before* the adapter decides to stream.
/// It wasn't: `/api/chat` with no model installed answered
/// `200 OK`, then wrote `{"error": ...}` as the first NDJSON line — a
/// success status followed by an object with no `message` and no `done`.
/// A client's error path never fires, its message parser gets something
/// it cannot read, and because clients stop on `done: true` it then waits
/// on a connection that will never produce another line.
#[tokio::test]
async fn an_unready_server_fails_with_a_status_not_a_success_stream() {
    let addr = spawn_test_server_with(false, None).await;

    for (path, body) in [
        (
            "/api/chat",
            r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        ("/api/generate", r#"{"model":"qwen3.6:35b","prompt":"hi"}"#),
    ] {
        // No "stream" key: streaming is Ollama's default, which is the
        // shape that regressed.
        let response = http_request(addr, &post_json(path, body)).await;
        assert!(
            response.starts_with("HTTP/1.1 503"),
            "{path} must report unreadiness as a status: {response}"
        );

        // And the body must still terminate a client that got far enough
        // to read it.
        let line = ndjson_lines(&response)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("expected one NDJSON line: {response}"));
        assert!(line["error"].is_string(), "{line}");
        assert_eq!(
            line["done"], true,
            "clients stop on `done`; without it they hang: {line}"
        );
    }
}
