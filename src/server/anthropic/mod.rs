//! Anthropic Messages-compatible surface (spec Part IX section 72, §208).
//!
//! Exists for the same practical reason the Ollama surface does: Claude
//! Code and gateway-style clients redirect through `ANTHROPIC_BASE_URL`,
//! and `tqf --open claude` writes exactly that redirect. Without this
//! module that flag would point a client at endpoints that do not exist.
//!
//! Reuses the whole normalization and generation pipeline; only the wire
//! shapes and the streaming event names are Anthropic's.

use std::convert::Infallible;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;

use crate::runtime::stream_decoder::StreamEvent;
use crate::runtime::{
    GeneratedOutput, Message, NormalizedRequest, ProtocolFlavor, Role, SamplingParams,
};
use crate::server::model_id::{self, CANONICAL_MODEL_ID};
use crate::server::stream::CancelOnDrop;
use crate::server::{stub, AppState};

#[cfg(test)]
mod tests;

type SseItem = Result<Event, Infallible>;
type AnthropicStream = CancelOnDrop<ReceiverStream<SseItem>>;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
}

#[derive(Deserialize)]
struct MessagesRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    /// Anthropic requires this; so does this adapter, matching their API
    /// rather than inventing a default they do not have.
    max_tokens: Option<u32>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    stop_sequences: Vec<String>,
}

#[derive(Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Value,
}

/// Anthropic's error envelope, distinct from OpenAI's and Ollama's
/// (spec §212).
fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {"type": kind, "message": message.into()},
        })),
    )
        .into_response()
}

fn invalid(message: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

/// Anthropic content is either a plain string or an array of typed
/// blocks. Only text blocks are readable here — an image block is
/// rejected rather than silently dropped, which would answer a question
/// about a picture the model never saw.
fn content_text(content: &Value) -> std::result::Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(
                        block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                    Some("image") => {
                        return Err("image content blocks are not supported yet: the vision \
                                    encoder is not wired into the request path in this build"
                            .to_string())
                    }
                    Some(other) => return Err(format!("unsupported content block type {other:?}")),
                    None => return Err("content blocks require a type".to_string()),
                }
            }
            Ok(text)
        }
        _ => Err("content must be a string or an array of content blocks".to_string()),
    }
}

fn parse_role(role: &str) -> std::result::Result<Role, String> {
    match role {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        other => Err(format!(
            "unsupported message role {other:?}: Anthropic messages are user or assistant \
             (system is a top-level field)"
        )),
    }
}

/// Anthropic's `stop_reason` vocabulary, which differs from both
/// OpenAI's and Ollama's.
fn stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

fn usage(output: &GeneratedOutput) -> Value {
    serde_json::json!({
        "input_tokens": output.usage.prompt_tokens,
        "output_tokens": output.usage.completion_tokens,
    })
}

/// Returns a message rather than a built `Response`: every failure here
/// is a 400, so the caller wraps it, and threading a whole
/// `axum::Response` through the `Err` arm makes the `Result` large enough
/// to be worth avoiding on a per-request path.
fn normalize(req: MessagesRequest) -> std::result::Result<NormalizedRequest, String> {
    model_id::resolve(req.model.as_deref())?;
    let Some(max_tokens) = req.max_tokens else {
        return Err("max_tokens is required".to_string());
    };
    if req.messages.is_empty() {
        return Err("messages must contain at least one item".to_string());
    }

    let mut messages = Vec::new();
    // Anthropic carries the system prompt as a top-level field rather
    // than a message; internally it is a leading system message.
    if let Some(system) = &req.system {
        match content_text(system) {
            Ok(text) if !text.is_empty() => messages.push(Message {
                role: Role::System,
                content: text,
                tool_calls: Vec::new(),
            }),
            Ok(_) => {}
            Err(message) => return Err(message),
        }
    }
    for message in &req.messages {
        let role = parse_role(&message.role)?;
        let content = content_text(&message.content)?;
        messages.push(Message {
            role,
            content,
            tool_calls: Vec::new(),
        });
    }

    let mut sampling = SamplingParams {
        max_output_tokens: Some(max_tokens),
        stop_sequences: req
            .stop_sequences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect(),
        ..SamplingParams::default()
    };
    if let Some(temperature) = req.temperature {
        if !(0.0..=1.0).contains(&temperature) {
            return Err("temperature must be between 0 and 1".to_string());
        }
        sampling.temperature = temperature;
    }
    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err("top_p must be between 0 and 1".to_string());
        }
        sampling.top_p = top_p;
    }
    if let Some(top_k) = req.top_k {
        sampling.top_k = (top_k > 0).then_some(top_k);
    }

    let mut normalized = NormalizedRequest::new(ProtocolFlavor::Anthropic, messages, req.stream);
    normalized.sampling = sampling;
    Ok(normalized)
}

async fn messages(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    let normalized = match normalize(req) {
        Ok(normalized) => normalized,
        Err(message) => return invalid(message),
    };
    if normalized.stream {
        return stream_messages(state, normalized).into_response();
    }
    match stub::generate(&state, &normalized).await {
        Ok(output) => Json(message_object(&output)).into_response(),
        Err(response) => response,
    }
}

fn message_id() -> String {
    format!("msg_tqf_{:x}", crate::server::openai::unix_seconds())
}

fn message_object(output: &GeneratedOutput) -> Value {
    serde_json::json!({
        "id": message_id(),
        "type": "message",
        "role": "assistant",
        "model": CANONICAL_MODEL_ID,
        "content": [{"type": "text", "text": output.text}],
        "stop_reason": stop_reason(output.finish_reason),
        "stop_sequence": Value::Null,
        "usage": usage(output),
    })
}

fn sse(name: &str, payload: Value) -> SseItem {
    Ok(Event::default().event(name).data(payload.to_string()))
}

/// Anthropic's streaming state machine (§208): `message_start`,
/// `content_block_start`, N `content_block_delta`, `content_block_stop`,
/// `message_delta`, `message_stop`. Clients drive their UI off these
/// event names, so the sequence matters as much as the text.
fn stream_messages(state: AppState, request: NormalizedRequest) -> Sse<AnthropicStream> {
    let (sender, receiver) = tokio::sync::mpsc::channel::<SseItem>(32);
    let session = crate::runtime::Session::new();
    let cancellation = session.cancellation.clone();
    let cancellation_on_drop = cancellation.clone();
    let id = message_id();

    tokio::spawn(async move {
        let preamble = vec![
            sse(
                "message_start",
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": id,
                        "type": "message",
                        "role": "assistant",
                        "model": CANONICAL_MODEL_ID,
                        "content": [],
                        "stop_reason": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }),
            ),
            sse(
                "content_block_start",
                serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }),
            ),
        ];
        for event in preamble {
            if sender.send(event).await.is_err() {
                cancellation.cancel();
                return;
            }
        }

        let (model_events, mut model_rx) = tokio::sync::mpsc::channel(32);
        let pump_state = state.clone();
        let pump_request = request.clone();
        let generation = tokio::spawn(async move {
            stub::generate_streaming_with_session(&pump_state, &pump_request, session, model_events)
                .await
        });

        loop {
            tokio::select! {
                event = model_rx.recv() => {
                    let Some(event) = event else { break };
                    let rendered = match event {
                        StreamEvent::TextDelta(text) => sse(
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            }),
                        ),
                        // Anthropic's own name for reasoning deltas.
                        StreamEvent::Reasoning(text) => sse(
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "thinking_delta", "thinking": text},
                            }),
                        ),
                        StreamEvent::ToolCall(_) => continue,
                    };
                    if sender.send(rendered).await.is_err() {
                        cancellation.cancel();
                        return;
                    }
                }
                _ = sender.closed() => {
                    cancellation.cancel();
                    return;
                }
            }
        }

        let tail = match generation.await {
            Ok(Ok(output)) => vec![
                sse(
                    "content_block_stop",
                    serde_json::json!({"type": "content_block_stop", "index": 0}),
                ),
                sse(
                    "message_delta",
                    serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason(output.finish_reason),
                            "stop_sequence": Value::Null,
                        },
                        "usage": usage(&output),
                    }),
                ),
                sse("message_stop", serde_json::json!({"type": "message_stop"})),
            ],
            Ok(Err(_)) | Err(_) => vec![sse(
                "error",
                serde_json::json!({
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "generation unavailable"},
                }),
            )],
        };
        for event in tail {
            if sender.send(event).await.is_err() {
                cancellation.cancel();
                return;
            }
        }
    });

    Sse::new(CancelOnDrop {
        inner: ReceiverStream::new(receiver),
        cancellation: cancellation_on_drop,
    })
    .keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// Claude Code calls this before sending a long conversation. Reports the
/// real tokenizer's count, not an estimate.
async fn count_tokens(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    // `max_tokens` is irrelevant here, so a caller may legitimately omit
    // it; supply one so the shared normalizer's requirement does not
    // reject an otherwise-valid counting request.
    let req = MessagesRequest {
        max_tokens: req.max_tokens.or(Some(1)),
        ..req
    };
    let normalized = match normalize(req) {
        Ok(normalized) => normalized,
        Err(message) => return invalid(message),
    };
    let Some(generator) = &state.generator else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            "no model is loaded, so its tokenizer is unavailable for counting",
        );
    };
    match generator.count_prompt_tokens(&normalized) {
        Ok(tokens) => Json(serde_json::json!({ "input_tokens": tokens })).into_response(),
        Err(err) => error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            err.to_string(),
        ),
    }
}
