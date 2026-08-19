//! OpenAI-compatible surface: model listing plus Responses/Chat/Embeddings
//! stubs (spec Part IX sections 70-71). Generation itself lands in later
//! phases; these routes validate request shape and exercise the real
//! request-queue/cancellation path today.

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tokio_util::sync::CancellationToken;

use crate::runtime::{
    GeneratedOutput, Message, MessageToolCall, NormalizedRequest, ProtocolFlavor, Role,
    ToolDefinition,
};
use crate::server::{stub, AppState};

const CANONICAL_MODEL_ID: &str = "qwen3.6-35b-a3b";

type SseItem = Result<Event, Infallible>;

/// Axum drops the response stream when the client stops consuming it. Tie
/// that lifetime directly to the model session instead of relying on a later
/// channel send to discover the disconnect.
struct CancelOnDropStream {
    inner: ReceiverStream<SseItem>,
    cancellation: CancellationToken,
}

impl Stream for CancelOnDropStream {
    type Item = SseItem;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl Drop for CancelOnDropStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[derive(Serialize)]
struct ModelObject {
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
    installed: bool,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: CANONICAL_MODEL_ID,
            object: "model",
            owned_by: "turboqwenfare",
            installed: state.model_installed,
        }],
    })
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Vec<Value>,
}

#[derive(Deserialize)]
struct ChatCompletionsRequest {
    /// Model selection is a no-op until more than one model can be served;
    /// accepted so well-formed OpenAI clients don't get a 400 for sending it.
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

fn default_object_schema() -> Value {
    serde_json::json!({"type": "object"})
}

/// Normalizes both accepted OpenAI tool shapes: Chat Completions nests the
/// function definition under `function`, while Responses supplies its
/// function fields directly. The model never sees either wire shape.
fn normalize_tools(tools: Vec<Value>) -> std::result::Result<Vec<ToolDefinition>, String> {
    tools
        .into_iter()
        .map(|tool| {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                return Err("only function tools are supported".to_string());
            }
            let function = tool.get("function").unwrap_or(&tool);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())
                .ok_or_else(|| "function tool requires a non-empty name".to_string())?;
            let description = function
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(default_object_schema);
            Ok(ToolDefinition {
                name: name.to_string(),
                description: description.to_string(),
                parameters_json_schema: parameters.to_string(),
            })
        })
        .collect()
}

fn invalid_request(message: String) -> Response {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {"message": message, "type": "invalid_request_error"}
        })),
    )
        .into_response()
}

fn parse_role(role: &str) -> std::result::Result<Role, String> {
    match role {
        "system" | "developer" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(format!("unsupported message role {other:?}")),
    }
}

fn validate_message_order(messages: &[Message]) -> std::result::Result<(), String> {
    if messages
        .iter()
        .enumerate()
        .any(|(index, message)| message.role == Role::System && index != 0)
    {
        return Err("system/developer messages must be the first message".to_string());
    }
    Ok(())
}

fn chat_content_text(content: &Value) -> std::result::Result<String, String> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
                if kind != "text" {
                    return Err(format!(
                        "unsupported Chat Completions content part {kind:?}"
                    ));
                }
                text.push_str(
                    part.get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "chat text content part requires text".to_string())?,
                );
            }
            Ok(text)
        }
        _ => Err("chat message content must be text, text parts, or null".to_string()),
    }
}

fn chat_message_tool_calls(calls: &[Value]) -> std::result::Result<Vec<MessageToolCall>, String> {
    calls
        .iter()
        .map(|call| {
            if call.get("type").and_then(Value::as_str) != Some("function") {
                return Err("only function message tool calls are supported".to_string());
            }
            let function = call
                .get("function")
                .ok_or_else(|| "message tool call requires function".to_string())?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())
                .ok_or_else(|| {
                    "message tool call requires a non-empty function name".to_string()
                })?;
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .ok_or_else(|| "message tool call arguments must be a JSON string".to_string())?;
            let parsed: Value = serde_json::from_str(arguments).map_err(|error| {
                format!("message tool call arguments are invalid JSON: {error}")
            })?;
            if !parsed.is_object() {
                return Err("message tool call arguments must encode a JSON object".to_string());
            }
            Ok(MessageToolCall {
                name: name.to_string(),
                arguments_json: parsed.to_string(),
            })
        })
        .collect()
}

fn validate_model(model: Option<&str>) -> std::result::Result<(), String> {
    match model {
        None | Some(CANONICAL_MODEL_ID) | Some("Qwen3.6-35B-A3B") => Ok(()),
        Some(model) => Err(format!(
            "model {model:?} is not available; use {CANONICAL_MODEL_ID:?}"
        )),
    }
}

/// Every sampling knob the request carries, so `apply_sampling` reads as
/// one mapping rather than a growing positional argument list.
#[derive(Default)]
struct RequestedSampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    seed: Option<u64>,
    maximum: Option<u32>,
    stop: Option<Value>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
}

/// Maps OpenAI's parameter spelling onto the one internal
/// `SamplingParams` (spec §153). These used to be rejected wholesale
/// because no sampler existed; now that `crate::sampling` implements them,
/// rejecting values real clients send by default would be the lie.
///
/// Absent temperature stays `0.0` (greedy), matching the internal default
/// — a client that says nothing about sampling gets deterministic output.
fn apply_sampling(
    normalized: &mut NormalizedRequest,
    requested: RequestedSampling,
) -> std::result::Result<(), String> {
    let RequestedSampling {
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        maximum,
        stop,
        frequency_penalty,
        presence_penalty,
    } = requested;

    normalized.sampling.temperature = match temperature {
        None => 0.0,
        Some(value) => range("temperature", value, 0.0, 2.0)?,
    };
    if let Some(top_p) = top_p {
        normalized.sampling.top_p = range("top_p", top_p, 0.0, 1.0)?;
    }
    if let Some(top_k) = top_k {
        normalized.sampling.top_k = (top_k > 0).then_some(top_k);
    }
    if let Some(min_p) = min_p {
        normalized.sampling.min_p = Some(range("min_p", min_p, 0.0, 1.0)?);
    }
    normalized.sampling.seed = seed;
    if let Some(value) = frequency_penalty {
        normalized.sampling.frequency_penalty = range("frequency_penalty", value, -2.0, 2.0)?;
    }
    if let Some(value) = presence_penalty {
        normalized.sampling.presence_penalty = range("presence_penalty", value, -2.0, 2.0)?;
    }
    if let Some(maximum) = maximum {
        if maximum == 0 {
            return Err("max output tokens must be at least 1".to_string());
        }
        normalized.sampling.max_output_tokens = Some(maximum);
    }
    normalized.sampling.stop_sequences = parse_stop_sequences(stop)?;
    Ok(())
}

fn range(name: &str, value: f32, low: f32, high: f32) -> std::result::Result<f32, String> {
    if !value.is_finite() || value < low || value > high {
        return Err(format!(
            "{name} must be a finite value between {low} and {high}"
        ));
    }
    Ok(value)
}

/// OpenAI accepts `stop` as either a single string or an array of up to
/// four. Empty strings are dropped rather than accepted: a zero-length
/// stop sequence matches immediately and would end every generation at
/// the first token.
fn parse_stop_sequences(stop: Option<Value>) -> std::result::Result<Vec<String>, String> {
    const MAX_STOP_SEQUENCES: usize = 4;
    let Some(stop) = stop else {
        return Ok(Vec::new());
    };
    let sequences = match stop {
        Value::Null => Vec::new(),
        Value::String(one) => vec![one],
        Value::Array(many) => many
            .into_iter()
            .map(|entry| match entry {
                Value::String(text) => Ok(text),
                _ => Err("stop entries must be strings".to_string()),
            })
            .collect::<std::result::Result<Vec<_>, _>>()?,
        _ => return Err("stop must be a string or an array of strings".to_string()),
    };
    if sequences.len() > MAX_STOP_SEQUENCES {
        return Err(format!(
            "at most {MAX_STOP_SEQUENCES} stop sequences are supported"
        ));
    }
    Ok(sequences.into_iter().filter(|s| !s.is_empty()).collect())
}

/// What this build still cannot honor. Both rejections are real
/// limitations rather than unimplemented plumbing, and spec §204 requires
/// rejecting rather than silently ignoring them.
fn validate_unsupported_options(
    n: Option<u32>,
    logprobs: Option<bool>,
) -> std::result::Result<(), String> {
    if n.unwrap_or(1) != 1 {
        // v1 runs one active generation and queues the rest (spec §75), so
        // there is no second sequence to return.
        return Err("n must be 1: this build serves one generation per request".to_string());
    }
    if logprobs.unwrap_or(false) {
        // The decoder keeps only its top-4 pre-softmax candidates for
        // diagnostics; reporting those as logprobs would be wrong, not
        // merely partial.
        return Err("logprobs are not implemented in this build".to_string());
    }
    Ok(())
}

fn apply_tool_choice(
    tools: &mut Vec<ToolDefinition>,
    choice: Option<&Value>,
) -> std::result::Result<(), String> {
    match choice {
        None => Ok(()),
        Some(Value::String(choice)) if choice == "auto" => Ok(()),
        Some(Value::String(choice)) if choice == "none" => {
            tools.clear();
            Ok(())
        }
        Some(_) => Err("only tool_choice=auto or tool_choice=none is implemented".to_string()),
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    if let Err(message) = validate_model(req.model.as_deref()) {
        return invalid_request(message);
    }
    if req.messages.is_empty() {
        return invalid_request("messages must contain at least one item".to_string());
    }
    if let Err(message) = validate_unsupported_options(req.n, req.logprobs) {
        return invalid_request(message);
    }
    let messages = match req
        .messages
        .into_iter()
        .map(|m| {
            let role = parse_role(&m.role)?;
            if role != Role::Assistant && !m.tool_calls.is_empty() {
                return Err("tool_calls are valid only on assistant messages".to_string());
            }
            Ok(Message {
                role,
                content: chat_content_text(&m.content)?,
                tool_calls: chat_message_tool_calls(&m.tool_calls)?,
            })
        })
        .collect::<std::result::Result<Vec<_>, String>>()
    {
        Ok(messages) => messages,
        Err(message) => return invalid_request(message),
    };
    if let Err(message) = validate_message_order(&messages) {
        return invalid_request(message);
    }
    let mut tools = match normalize_tools(req.tools) {
        Ok(tools) => tools,
        Err(message) => return invalid_request(message),
    };
    if let Err(message) = apply_tool_choice(&mut tools, req.tool_choice.as_ref()) {
        return invalid_request(message);
    }
    let mut normalized =
        NormalizedRequest::new(ProtocolFlavor::OpenAiChatCompletions, messages, req.stream);
    normalized.tools = tools;
    let requested = RequestedSampling {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        seed: req.seed,
        maximum: req.max_completion_tokens.or(req.max_tokens),
        stop: req.stop,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    };
    if let Err(message) = apply_sampling(&mut normalized, requested) {
        return invalid_request(message);
    }
    if normalized.stream {
        return stream_chat_completion(state, normalized).into_response();
    }
    match stub::generate(&state, &normalized).await {
        Ok(output) => Json(chat_completion(output)).into_response(),
        Err(response) => response,
    }
}

fn response_content_text(content: &Value) -> std::result::Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let kind = part
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("input_text");
                if !matches!(kind, "input_text" | "output_text" | "text") {
                    return Err(format!("unsupported Responses content part {kind:?}"));
                }
                let value = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Responses text content part requires text".to_string())?;
                text.push_str(value);
            }
            Ok(text)
        }
        _ => Err("Responses message content must be text or text parts".to_string()),
    }
}

fn response_input_messages(input: &Value) -> std::result::Result<Vec<Message>, String> {
    match input {
        Value::String(text) => Ok(vec![Message {
            role: Role::User,
            content: text.clone(),
            tool_calls: Vec::new(),
        }]),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Responses message item requires role".to_string())?;
                let content = item
                    .get("content")
                    .ok_or_else(|| "Responses message item requires content".to_string())?;
                Ok(Message {
                    role: parse_role(role)?,
                    content: response_content_text(content)?,
                    tool_calls: Vec::new(),
                })
            })
            .collect(),
        _ => Err("Responses input must be a string or an array of message items".to_string()),
    }
}

async fn responses(State(state): State<AppState>, Json(req): Json<Value>) -> Response {
    if let Err(message) = validate_model(req.get("model").and_then(Value::as_str)) {
        return invalid_request(message);
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let input = match req.get("input") {
        Some(input) => input,
        None => return invalid_request("Responses input is required".to_string()),
    };
    let mut messages = match response_input_messages(input) {
        Ok(messages) if !messages.is_empty() => messages,
        Ok(_) => return invalid_request("Responses input must not be empty".to_string()),
        Err(message) => return invalid_request(message),
    };
    if let Some(instructions) = req.get("instructions") {
        let Some(instructions) = instructions.as_str() else {
            return invalid_request("Responses instructions must be a string".to_string());
        };
        messages.insert(
            0,
            Message {
                role: Role::System,
                content: instructions.to_string(),
                tool_calls: Vec::new(),
            },
        );
    }
    if let Err(message) = validate_message_order(&messages) {
        return invalid_request(message);
    }
    let raw_tools = match req.get("tools") {
        Some(Value::Array(tools)) => tools.clone(),
        Some(_) => return invalid_request("tools must be an array".to_string()),
        None => Vec::new(),
    };
    let mut tools = match normalize_tools(raw_tools) {
        Ok(tools) => tools,
        Err(message) => return invalid_request(message),
    };
    if let Err(message) = apply_tool_choice(&mut tools, req.get("tool_choice")) {
        return invalid_request(message);
    }
    let mut normalized = NormalizedRequest::new(ProtocolFlavor::OpenAiResponses, messages, stream);
    normalized.tools = tools;
    let maximum = match req.get("max_output_tokens") {
        Some(value) => match value.as_u64().and_then(|value| u32::try_from(value).ok()) {
            Some(value) => Some(value),
            None => {
                return invalid_request(
                    "max_output_tokens must be a nonnegative integer".to_string(),
                )
            }
        },
        None => None,
    };
    let temperature = match req.get("temperature") {
        Some(value) => match value.as_f64() {
            Some(value) => Some(value as f32),
            None => return invalid_request("temperature must be a number".to_string()),
        },
        None => None,
    };
    let top_p = match req.get("top_p") {
        Some(value) => match value.as_f64() {
            Some(value) => Some(value as f32),
            None => return invalid_request("top_p must be a number".to_string()),
        },
        None => None,
    };
    let requested = RequestedSampling {
        temperature,
        top_p,
        maximum,
        stop: req.get("stop").cloned(),
        ..RequestedSampling::default()
    };
    if let Err(message) = apply_sampling(&mut normalized, requested) {
        return invalid_request(message);
    }
    if normalized.stream {
        return stream_response(state, normalized).into_response();
    }
    match stub::generate(&state, &normalized).await {
        Ok(output) => Json(response_object(output)).into_response(),
        Err(response) => response,
    }
}

async fn embeddings(State(_state): State<AppState>, Json(_req): Json<Value>) -> Response {
    (
        axum::http::StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "embeddings are not implemented",
                "type": "invalid_request_error"
            }
        })),
    )
        .into_response()
}

#[derive(Serialize)]
struct ChatCompletion {
    id: &'static str,
    object: &'static str,
    model: &'static str,
    choices: Vec<ChatChoice>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u8,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiToolCall>,
}

#[derive(Serialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionCall,
}

#[derive(Serialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

fn tool_calls(output: &GeneratedOutput) -> Vec<OpenAiToolCall> {
    output
        .tool_calls
        .iter()
        .map(|call| OpenAiToolCall {
            id: call.id.clone(),
            kind: "function",
            function: OpenAiFunctionCall {
                name: call.name.clone(),
                arguments: call.arguments_json.clone(),
            },
        })
        .collect()
}

fn chat_completion(output: GeneratedOutput) -> ChatCompletion {
    let tool_calls = tool_calls(&output);
    ChatCompletion {
        id: "chatcmpl-tqf-local",
        object: "chat.completion",
        model: CANONICAL_MODEL_ID,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: output.text,
                tool_calls,
            },
            finish_reason: output.finish_reason,
        }],
    }
}

fn response_object(output: GeneratedOutput) -> Value {
    serde_json::json!({
        "id": "resp-tqf-local",
        "object": "response",
        "model": CANONICAL_MODEL_ID,
        "output_text": output.text,
        "finish_reason": output.finish_reason,
        "tool_calls": tool_calls(&output),
    })
}

fn chat_completion_events(output: GeneratedOutput) -> Vec<Result<Event, Infallible>> {
    let mut events = Vec::new();
    if !output.text.is_empty() {
        events.push(Ok(Event::default().data(
            serde_json::json!({
                "id": "chatcmpl-tqf-local",
                "object": "chat.completion.chunk",
                "model": CANONICAL_MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": output.text}, "finish_reason": null}],
            })
            .to_string(),
        )));
    }
    if !output.tool_calls.is_empty() {
        events.push(Ok(Event::default().data(
            serde_json::json!({
                "id": "chatcmpl-tqf-local",
                "object": "chat.completion.chunk",
                "model": CANONICAL_MODEL_ID,
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls(&output)}, "finish_reason": null}],
            })
            .to_string(),
        )));
    }
    events.push(Ok(Event::default().data(
        serde_json::json!({
            "id": "chatcmpl-tqf-local",
            "object": "chat.completion.chunk",
            "model": CANONICAL_MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": output.finish_reason}],
        })
        .to_string(),
    )));
    events.push(Ok(Event::default().data("[DONE]")));
    events
}

fn stream_chat_completion(state: AppState, request: NormalizedRequest) -> Sse<CancelOnDropStream> {
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let session = crate::runtime::Session::new();
    let cancellation = session.cancellation.clone();
    let cancellation_on_drop = cancellation.clone();
    tokio::spawn(async move {
        let role = Ok(Event::default().data(
            serde_json::json!({
                "id": "chatcmpl-tqf-local",
                "object": "chat.completion.chunk",
                "model": CANONICAL_MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
            })
            .to_string(),
        ));
        if sender.send(role).await.is_err() {
            cancellation.cancel();
            return;
        }
        tokio::select! {
            output = stub::generate_with_session(&state, &request, session) => {
                let events = match output {
                    Ok(output) => chat_completion_events(output),
                    Err(_) => vec![
                        Ok(Event::default().event("error").data("generation unavailable")),
                        Ok(Event::default().data("[DONE]")),
                    ],
                };
                for event in events {
                    if sender.send(event).await.is_err() {
                        cancellation.cancel();
                        break;
                    }
                }
            }
            _ = sender.closed() => cancellation.cancel(),
        }
    });
    Sse::new(CancelOnDropStream {
        inner: ReceiverStream::new(receiver),
        cancellation: cancellation_on_drop,
    })
    .keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// Responses API uses typed events rather than chat-completion chunks. The
/// runtime still produces the same protocol-neutral `GeneratedOutput`; only
/// this boundary chooses the externally visible event names.
fn response_events(output: GeneratedOutput) -> Vec<Result<Event, Infallible>> {
    let mut events = Vec::new();
    if !output.text.is_empty() {
        events.push(Ok(Event::default()
            .event("response.output_text.delta")
            .data(
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "delta": output.text,
                })
                .to_string(),
            )));
    }
    for call in tool_calls(&output) {
        events.push(Ok(Event::default()
            .event("response.function_call_arguments.done")
            .data(
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                })
                .to_string(),
            )));
    }
    events.push(Ok(Event::default().event("response.completed").data(
        serde_json::json!({
            "type": "response.completed",
            "response": response_object(output),
        })
        .to_string(),
    )));
    events
}

fn stream_response(state: AppState, request: NormalizedRequest) -> Sse<CancelOnDropStream> {
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let session = crate::runtime::Session::new();
    let cancellation = session.cancellation.clone();
    let cancellation_on_drop = cancellation.clone();
    tokio::spawn(async move {
        let created = Ok(Event::default().event("response.created").data(
            serde_json::json!({
                "type": "response.created",
                "response": {"id": "resp-tqf-local", "status": "in_progress"},
            })
            .to_string(),
        ));
        if sender.send(created).await.is_err() {
            cancellation.cancel();
            return;
        }
        tokio::select! {
            output = stub::generate_with_session(&state, &request, session) => {
                let events = match output {
                    Ok(output) => response_events(output),
                    Err(_) => vec![Ok(Event::default().event("error").data(
                        serde_json::json!({
                            "type": "error",
                            "error": {"message": "generation unavailable", "code": "model_not_ready"},
                        }).to_string(),
                    ))],
                };
                for event in events {
                    if sender.send(event).await.is_err() {
                        cancellation.cancel();
                        break;
                    }
                }
            }
            _ = sender.closed() => cancellation.cancel(),
        }
    });
    Sse::new(CancelOnDropStream {
        inner: ReceiverStream::new(receiver),
        cancellation: cancellation_on_drop,
    })
    .keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/embeddings", post(embeddings))
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn dropping_the_sse_stream_cancels_its_model_session() {
        let (sender, receiver) = tokio::sync::mpsc::channel::<SseItem>(1);
        let cancellation = CancellationToken::new();
        let stream = CancelOnDropStream {
            inner: ReceiverStream::new(receiver),
            cancellation: cancellation.clone(),
        };
        assert!(!cancellation.is_cancelled());
        drop(stream);
        assert!(cancellation.is_cancelled());
        drop(sender);
    }
}
