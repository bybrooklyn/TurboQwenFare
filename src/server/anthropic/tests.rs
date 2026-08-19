//! Real-HTTP tests for the Anthropic Messages surface.

use std::sync::Arc;

use serde_json::Value;

use crate::server::tests::{
    http_request, post_json, spawn_test_server_with, IncrementalFixtureGenerator,
};

fn body_of(response: &str) -> &str {
    response.split("\r\n\r\n").nth(1).unwrap_or("")
}

fn json_body(response: &str) -> Value {
    serde_json::from_str(body_of(response))
        .unwrap_or_else(|e| panic!("expected a JSON body, got {response:?}: {e}"))
}

fn sse_events(response: &str) -> Vec<(String, Value)> {
    let mut events = Vec::new();
    let mut name = String::new();
    for line in response.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            name = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data: ") {
            if let Ok(value) = serde_json::from_str(rest.trim()) {
                events.push((std::mem::take(&mut name), value));
            }
        }
    }
    events
}

#[tokio::test]
async fn messages_returns_anthropics_content_block_shape() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/v1/messages",
            r#"{"model":"qwen3.6-35b-a3b","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;

    let body = json_body(&response);
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    // Anthropic nests content in typed blocks, not a bare string.
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], expected);
    assert_eq!(body["stop_reason"], "end_turn");
    assert!(body["usage"]["output_tokens"].is_number(), "{body}");
}

/// Anthropic's API requires `max_tokens`; matching that beats inventing a
/// default they do not have, because a client relying on the real API's
/// rejection would otherwise get silently different behavior here.
#[tokio::test]
async fn max_tokens_is_required_as_it_is_in_the_real_api() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/messages",
            r#"{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert_eq!(
        json_body(&response)["error"]["type"],
        "invalid_request_error"
    );
}

/// Spec §212: each surface returns its own error shape. Anthropic's is
/// `{"type":"error","error":{"type":...,"message":...}}` — neither
/// OpenAI's nor Ollama's.
///
/// Both error paths are checked. The validation path and the
/// service-unavailable path are produced by different code (this module
/// versus `server::stub`), and a live probe found the latter still
/// emitting OpenAI's nested envelope while the former was already
/// correct.
#[tokio::test]
async fn errors_use_anthropics_own_envelope() {
    let addr = spawn_test_server_with(true, None).await;

    // Validation failure.
    let response = http_request(
        addr,
        &post_json(
            "/v1/messages",
            r#"{"model":"gpt-4","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    let body = json_body(&response);
    assert_eq!(body["type"], "error", "validation path: {body}");
    assert!(body["error"]["message"].is_string(), "{body}");

    // Service-unavailable failure, with no loaded generator.
    let response = http_request(
        addr,
        &post_json(
            "/v1/messages",
            r#"{"model":"qwen3.6-35b-a3b","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    let body = json_body(&response);
    assert_eq!(body["type"], "error", "unavailable path: {body}");
    assert_eq!(body["error"]["type"], "overloaded_error", "{body}");
    // The server was built with a receipt but no generator, so the
    // message must describe *that* state rather than a generic failure.
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("runtime is not loaded"),
        "the message must say what is actually wrong: {body}"
    );
}

#[tokio::test]
async fn the_system_prompt_is_accepted_as_a_top_level_field() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;
    for system in [r#""be brief""#, r#"[{"type":"text","text":"be brief"}]"#] {
        let body = format!(
            r#"{{"model":"qwen3.6-35b-a3b","max_tokens":16,"system":{system},"messages":[{{"role":"user","content":"hi"}}]}}"#
        );
        let response = http_request(addr, &post_json("/v1/messages", &body)).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{system}: {response}");
    }
}

/// Images must be refused rather than dropped: answering a question about
/// a picture the model never saw is worse than declining.
#[tokio::test]
async fn image_blocks_are_rejected_rather_than_silently_dropped() {
    let addr =
        spawn_test_server_with(true, Some(Arc::new(IncrementalFixtureGenerator::new()))).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/messages",
            r#"{"model":"qwen3.6-35b-a3b","max_tokens":16,"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}"#,
        ),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    let message = json_body(&response)["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(message.contains("vision"), "must say why: {message}");
}

/// Clients drive their UI off this exact event sequence, so its order
/// matters as much as the text it carries.
#[tokio::test]
async fn streaming_follows_anthropics_event_sequence() {
    let generator = IncrementalFixtureGenerator::new();
    let expected = generator.full_text();
    let addr = spawn_test_server_with(true, Some(Arc::new(generator))).await;

    let response = http_request(
        addr,
        &post_json(
            "/v1/messages",
            r#"{"model":"qwen3.6-35b-a3b","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"count"}]}"#,
        ),
    )
    .await;

    let events = sse_events(&response);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

    assert_eq!(names.first(), Some(&"message_start"), "{names:?}");
    assert_eq!(names.get(1), Some(&"content_block_start"), "{names:?}");
    assert_eq!(names.last(), Some(&"message_stop"), "{names:?}");
    assert!(names.contains(&"content_block_stop"), "{names:?}");
    assert!(names.contains(&"message_delta"), "{names:?}");

    let deltas: Vec<&str> = events
        .iter()
        .filter(|(name, _)| name == "content_block_delta")
        .filter_map(|(_, value)| value["delta"]["text"].as_str())
        .collect();
    assert!(deltas.len() >= 5, "expected incremental deltas: {names:?}");
    assert_eq!(deltas.concat(), expected);

    let (_, message_delta) = events
        .iter()
        .find(|(name, _)| name == "message_delta")
        .expect("message_delta must be sent");
    assert_eq!(message_delta["delta"]["stop_reason"], "end_turn");
}

/// Claude Code calls this before sending a long conversation, so the
/// number has to be the tokenizer's rather than an estimate.
#[tokio::test]
async fn count_tokens_reports_a_real_count_or_says_it_cannot() {
    let addr = spawn_test_server_with(true, None).await;
    let response = http_request(
        addr,
        &post_json(
            "/v1/messages/count_tokens",
            r#"{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}]}"#,
        ),
    )
    .await;
    // With no loaded generator there is no tokenizer, and saying so beats
    // returning a plausible guess.
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert_eq!(json_body(&response)["type"], "error");
}
