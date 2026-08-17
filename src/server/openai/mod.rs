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

fn apply_sampling(
    normalized: &mut NormalizedRequest,
    temperature: Option<f32>,
    top_p: Option<f32>,
    maximum: Option<u32>,
) -> std::result::Result<(), String> {
    if let Some(temperature) = temperature {
        if !temperature.is_finite() || temperature != 0.0 {
            return Err(
                "this correctness runtime currently supports only temperature=0".to_string(),
            );
        }
        normalized.sampling.temperature = temperature;
    } else {
        normalized.sampling.temperature = 0.0;
    }
    if let Some(top_p) = top_p {
        if !top_p.is_finite() || top_p != 1.0 {
            return Err("this correctness runtime currently supports only top_p=1".to_string());
        }
        normalized.sampling.top_p = top_p;
    }
    if let Some(maximum) = maximum {
        if maximum > 256 {
            return Err("max output tokens must be at most 256 in this build".to_string());
        }
        normalized.sampling.max_output_tokens = Some(maximum);
    }
    Ok(())
}

fn validate_unsupported_options(
    n: Option<u32>,
    stop: Option<&Value>,
    logprobs: Option<bool>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> std::result::Result<(), String> {
    if n.unwrap_or(1) != 1 {
        return Err("n must be 1".to_string());
    }
    if stop.is_some_and(|value| !value.is_null()) {
        return Err("stop sequences are not implemented in this build".to_string());
    }
    if logprobs.unwrap_or(false) {
        return Err("logprobs are not implemented in this build".to_string());
    }
    if frequency_penalty.unwrap_or(0.0) != 0.0 || presence_penalty.unwrap_or(0.0) != 0.0 {
        return Err(
            "frequency and presence penalties are not implemented in this build".to_string(),
        );
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
    if let Err(message) = validate_unsupported_options(
        req.n,
        req.stop.as_ref(),
        req.logprobs,
        req.frequency_penalty,
        req.presence_penalty,
    ) {
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
    let maximum = req.max_completion_tokens.or(req.max_tokens);
    if let Err(message) = apply_sampling(&mut normalized, req.temperature, req.top_p, maximum) {
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
    if let Err(message) = apply_sampling(&mut normalized, temperature, top_p, maximum) {
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
