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
    /// Anthropic's tool shape matches OpenAI's function schema closely
    /// enough that the shared normalizer handles both. Without this
    /// field the definitions were silently discarded, so the model was
    /// never told the tools existed while `stop_reason` could still
    /// claim `tool_use`.
    #[serde(default)]
    tools: Vec<Value>,
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
/// Anthropic declares a tool as `{name, description, input_schema}`;
/// the shared normalizer expects OpenAI's `{type:"function", function:
/// {name, description, parameters}}`. Translating here keeps one
/// normalizer rather than two nearly-identical ones.
fn anthropic_tools_as_function_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.get("name").cloned().unwrap_or(Value::Null),
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": tool
                        .get("input_schema")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({"type": "object"})),
                },
            })
        })
        .collect()
}

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
    normalized.tools =
        crate::server::openai::normalize_tools(anthropic_tools_as_function_tools(&req.tools))?;
    Ok(normalized)
}

async fn messages(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    let normalized = match normalize(req) {
        Ok(normalized) => normalized,
        Err(message) => return invalid(message),
    };
    // Checked before the stream branch: once bytes are out the status is
    // spent, and a 200 followed by an in-band error is a success the
    // client's error path never sees (see `stub::pre_stream_not_ready`).
    if let Some(message) = stub::readiness_error(&state) {
        return stub::pre_stream_not_ready(normalized.protocol, message);
    }
    if normalized.stream {
        return stream_messages(state, normalized).into_response();
    }
    match stub::generate(&state, &normalized).await {
        Ok(output) => Json(message_object(&output)).into_response(),
        Err(response) => response,
    }
}

/// Per-response id. A bare timestamp collides for every request inside
/// the same second, which matters to clients that key state by id — the
/// same defect the OpenAI adapter's `response_id` carries a counter to
/// avoid.
/// Which kind of content block is currently open.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
}

impl BlockKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Thinking => "thinking",
            Self::ToolUse => "tool_use",
        }
    }
}

/// Tracks Anthropic's typed content blocks across a stream: opens one
/// lazily when the first delta of a kind arrives, closes it when the kind
/// changes, and advances the block index — which is what makes a stream
/// that mixes reasoning, text, and tool calls parse in a real SDK client.
#[derive(Default)]
struct BlockWriter {
    open: Option<BlockKind>,
    index: usize,
}

impl BlockWriter {
    fn start(&mut self, kind: BlockKind, block: Value) -> Vec<Result<Event, Infallible>> {
        let mut events = self.close_open();
        if events.is_empty() && self.open.is_some() {
            unreachable!("close_open always emits when a block is open");
        }
        events.push(sse(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": self.index,
                "content_block": block,
            }),
        ));
        self.open = Some(kind);
        events
    }

    fn delta(&mut self, kind: BlockKind, delta: Value) -> Vec<Result<Event, Infallible>> {
        let mut events = if self.open == Some(kind) {
            Vec::new()
        } else {
            let block = match kind {
                BlockKind::Text => serde_json::json!({"type": "text", "text": ""}),
                BlockKind::Thinking => serde_json::json!({"type": "thinking", "thinking": ""}),
                BlockKind::ToolUse => serde_json::json!({"type": "tool_use"}),
            };
            self.start(kind, block)
        };
        events.push(sse(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": self.index,
                "delta": delta,
            }),
        ));
        events
    }

    fn tool_use(
        &mut self,
        call: &crate::runtime::generation::GeneratedToolCall,
    ) -> Vec<Result<Event, Infallible>> {
        let mut events = self.start(
            BlockKind::ToolUse,
            serde_json::json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": {},
            }),
        );
        events.push(sse(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": self.index,
                "delta": {"type": "input_json_delta", "partial_json": call.arguments_json},
            }),
        ));
        events.extend(self.close_open());
        events
    }

    fn close_open(&mut self) -> Vec<Result<Event, Infallible>> {
        let Some(_) = self.open.take() else {
            return Vec::new();
        };
        let stop = sse(
            "content_block_stop",
            serde_json::json!({"type": "content_block_stop", "index": self.index}),
        );
        self.index += 1;
        vec![stop]
    }
}

fn message_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "msg_tqf_{:x}{:x}",
        crate::server::openai::unix_seconds(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Anthropic represents a tool call as a `tool_use` content block whose
/// `input` is the parsed argument object, not a JSON string.
fn tool_use_blocks(output: &GeneratedOutput) -> Vec<Value> {
    output
        .tool_calls
        .iter()
        .map(|call| {
            serde_json::json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": serde_json::from_str::<Value>(&call.arguments_json)
                    .unwrap_or_else(|_| serde_json::json!({})),
            })
        })
        .collect()
}

fn message_object(output: &GeneratedOutput) -> Value {
    // Without the tool_use blocks the body claimed `stop_reason:
    // "tool_use"` while carrying no tool call for the client to execute.
    let mut content = Vec::new();
    if !output.text.is_empty() {
        content.push(serde_json::json!({"type": "text", "text": output.text}));
    }
    content.extend(tool_use_blocks(output));
    if content.is_empty() {
        content.push(serde_json::json!({"type": "text", "text": ""}));
    }

    serde_json::json!({
        "id": message_id(),
        "type": "message",
        "role": "assistant",
        "model": CANONICAL_MODEL_ID,
        "content": content,
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
        let preamble = vec![sse(
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
        )];
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

        // Anthropic's content blocks are typed and must be opened before
        // any delta and closed before the next block opens. Emitting a
        // `thinking_delta` against a block started as `{"type":"text"}`
        // makes a real SDK client raise on the first delta — and since the
        // Qwen3.6 prompt always opens `<think>`, that is every streamed
        // response. So the block is opened lazily, by the first event that
        // needs one, and reopened whenever the kind changes.
        let mut blocks = BlockWriter::default();

        loop {
            tokio::select! {
                event = model_rx.recv() => {
                    let Some(event) = event else { break };
                    let rendered = match event {
                        StreamEvent::TextDelta(text) => blocks.delta(
                            BlockKind::Text,
                            serde_json::json!({"type": "text_delta", "text": text}),
                        ),
                        StreamEvent::Reasoning(text) => blocks.delta(
                            BlockKind::Thinking,
                            serde_json::json!({"type": "thinking_delta", "thinking": text}),
                        ),
                        // A tool call arrives already complete, so its
                        // block opens, streams its arguments as one
                        // `input_json_delta`, and closes immediately.
                        StreamEvent::ToolCall(call) => blocks.tool_use(&call),
                    };
                    for event in rendered {
                        if sender.send(event).await.is_err() {
                            cancellation.cancel();
                            return;
                        }
                    }
                }
                _ = sender.closed() => {
                    cancellation.cancel();
                    return;
                }
            }
        }

        // Whatever block is still open has to be closed before
        // `message_delta`, or the client sees an unterminated block.
        for event in blocks.close_open() {
            if sender.send(event).await.is_err() {
                cancellation.cancel();
                return;
            }
        }

        let tail = match generation.await {
            Ok(Ok(output)) => vec![
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
        // A capability the running generator lacks is a 501, not a 400:
        // telling a client its request was invalid when the request was
        // fine sends it looking for a bug it does not have.
        Err(crate::error::TqfError::Model(crate::error::ModelError::Unsupported(message))) => {
            error(StatusCode::NOT_IMPLEMENTED, "api_error", message)
        }
        Err(err) => error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            err.to_string(),
        ),
    }
}
