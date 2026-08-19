//! Ollama-compatible surface (spec Part IX section 73, §210).
//!
//! TQF binds Ollama's port 11434 by default, so an Ollama client connects
//! successfully whether or not this module exists — which is why its
//! absence produced a server that looked up and 404'd everything.
//!
//! Two framing details matter more than the endpoint list, because
//! getting either wrong breaks every client while `curl` still looks fine:
//!
//! 1. **NDJSON, not SSE.** Ollama streams `application/x-ndjson`: one bare
//!    JSON object per line, no `data:` prefix, no `[DONE]` sentinel.
//! 2. **The terminal object is load-bearing.** Clients (ollama-js,
//!    ollama-python, LangChain, Open WebUI) stop on `"done": true`, not on
//!    stream close. Omitting it hangs them.
//!
//! Also note `stream` defaults to **true** here, the opposite of OpenAI.

#[cfg(test)]
mod tests;
mod time;

use std::convert::Infallible;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;

use crate::runtime::stream_decoder::StreamEvent;
use crate::runtime::{
    GeneratedOutput, Message, NormalizedRequest, ProtocolFlavor, Role, SamplingParams,
};
use crate::server::model_id::{self, CANONICAL_MODEL_ID, OLLAMA_MODEL_TAG};
use crate::server::stream::CancelOnDrop;
use crate::server::{stub, AppState};

use time::{rfc3339_utc, rfc3339_utc_now};

/// The exact body a real Ollama returns for `GET /`. Clients probe this
/// for liveness before doing anything else.
const LIVENESS_BODY: &str = "Ollama is running";

/// Reported by `/api/version`. Marked as TQF so a client that logs it
/// cannot mistake this for a real Ollama install, while still parsing as
/// a version string.
const REPORTED_VERSION: &str = concat!("0.0.0-tqf-", env!("CARGO_PKG_VERSION"));

// ---------------------------------------------------------------- routes

/// Everything that can generate, embed, or enumerate what is installed.
/// These are merged into the authenticated sub-router: putting them at the
/// top level would expose generation with no key on a `0.0.0.0` bind
/// (spec §74).
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .route("/api/embed", post(embed))
        .route("/api/embeddings", post(embed))
        .route("/api/tags", get(tags))
        .route("/api/show", post(show))
        .route("/api/ps", get(ps))
        // Model management. Spec §210 says these are not required; a 501
        // that says why is better client UX than an anonymous 404.
        .route("/api/pull", post(unsupported))
        .route("/api/push", post(unsupported))
        .route("/api/create", post(unsupported))
        .route("/api/copy", post(unsupported))
        .route("/api/delete", axum::routing::delete(unsupported))
}

/// Pre-credential liveness probes. These carry a fixed string and a
/// version number — no model data — and clients call them before they have
/// anywhere to put an API key.
///
/// `get()` also answers HEAD with the body stripped, which is what the
/// `HEAD /` probe some clients use expects.
pub fn unauthenticated_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(root))
        .route("/api/version", get(version))
}

async fn root() -> &'static str {
    LIVENESS_BODY
}

async fn version() -> Json<Value> {
    Json(serde_json::json!({ "version": REPORTED_VERSION }))
}

async fn unsupported() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": format!(
                "TurboQwenFare serves one pinned model ({CANONICAL_MODEL_ID}) and does not \
                 implement model management. Pull, push, create, copy, and delete are not \
                 available."
            )
        })),
    )
        .into_response()
}

// -------------------------------------------------------- request shapes

/// Ollama's `stream` defaults to **true**, unlike OpenAI's. A client that
/// omits it expects a stream.
fn stream_default() -> bool {
    true
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    #[serde(default = "stream_default")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    format: Option<Value>,
    #[serde(default)]
    think: Option<Value>,
    /// Accepted and ignored: TQF never unloads the model, so there is no
    /// idle timer for this to set. Rejecting it would break clients that
    /// always send it.
    #[serde(default, rename = "keep_alive")]
    _keep_alive: Option<Value>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    images: Vec<Value>,
}

#[derive(Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default = "stream_default")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
    #[serde(default)]
    format: Option<Value>,
    /// `raw: true` asks for the prompt to bypass templating. TQF always
    /// renders the pinned Qwen3.6 chat template, so honoring this is
    /// impossible and silently templating anyway would be a lie.
    #[serde(default)]
    raw: Option<bool>,
    #[serde(default)]
    template: Option<String>,
    #[serde(default)]
    suffix: Option<String>,
    /// Ollama's opaque conversation-state vector. TQF has no equivalent,
    /// and partially honoring it yields silently wrong continuations.
    #[serde(default)]
    context: Option<Value>,
    #[serde(default)]
    images: Vec<Value>,
    #[serde(default, rename = "keep_alive")]
    _keep_alive: Option<Value>,
}

/// Ollama nests sampling under `options`. Unknown keys are ignored rather
/// than rejected — clients send many, and most are backend hints.
#[derive(Deserialize, Default)]
struct OllamaOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    seed: Option<i64>,
    num_predict: Option<i64>,
    stop: Option<Value>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    // Sampling strategies this build does not implement. Ollama's own
    // defaults ship the no-op values (`mirostat: 0`, `tfs_z: 1.0`,
    // `typical_p: 1.0`), so accepting the no-op and rejecting anything
    // else is the difference between working with the ecosystem and
    // 400ing half of it.
    mirostat: Option<u32>,
    tfs_z: Option<f32>,
    typical_p: Option<f32>,
}

impl OllamaOptions {
    fn into_sampling(self) -> std::result::Result<SamplingParams, String> {
        if self.mirostat.unwrap_or(0) != 0 {
            return Err("mirostat sampling is not implemented in this build".to_string());
        }
        if self.tfs_z.is_some_and(|v| v != 1.0) {
            return Err("tail-free sampling (tfs_z) is not implemented in this build".to_string());
        }
        if self.typical_p.is_some_and(|v| v != 1.0) {
            return Err("typical_p sampling is not implemented in this build".to_string());
        }

        let mut sampling = SamplingParams::default();
        if let Some(temperature) = self.temperature {
            sampling.temperature = bounded("temperature", temperature, 0.0, 2.0)?;
        }
        if let Some(top_p) = self.top_p {
            sampling.top_p = bounded("top_p", top_p, 0.0, 1.0)?;
        }
        if let Some(top_k) = self.top_k {
            sampling.top_k = (top_k > 0).then_some(top_k);
        }
        if let Some(min_p) = self.min_p {
            sampling.min_p = Some(bounded("min_p", min_p, 0.0, 1.0)?);
        }
        // Ollama uses -1 for "random seed", which is the same as absent.
        sampling.seed = self.seed.filter(|s| *s >= 0).map(|s| s as u64);
        if let Some(penalty) = self.repeat_penalty {
            sampling.repeat_penalty = bounded("repeat_penalty", penalty, 0.0, 2.0)?;
        }
        if let Some(window) = self.repeat_last_n {
            // -1 means "the whole context"; the decoder bounds it by the
            // history it actually has, so a large value is safe.
            sampling.repeat_last_n = if window < 0 {
                usize::MAX
            } else {
                window as usize
            };
        }
        if let Some(penalty) = self.frequency_penalty {
            sampling.frequency_penalty = bounded("frequency_penalty", penalty, -2.0, 2.0)?;
        }
        if let Some(penalty) = self.presence_penalty {
            sampling.presence_penalty = bounded("presence_penalty", penalty, -2.0, 2.0)?;
        }
        // `num_predict` of -1 (infinite) and -2 (fill context) both mean
        // "no explicit cap", which the runtime already handles.
        sampling.max_output_tokens = self
            .num_predict
            .filter(|n| *n > 0)
            .map(|n| n.min(u32::MAX as i64) as u32);
        sampling.stop_sequences = parse_stop(self.stop)?;
        Ok(sampling)
    }
}

fn bounded(name: &str, value: f32, low: f32, high: f32) -> std::result::Result<f32, String> {
    if !value.is_finite() || value < low || value > high {
        return Err(format!(
            "{name} must be a finite value between {low} and {high}"
        ));
    }
    Ok(value)
}

fn parse_stop(stop: Option<Value>) -> std::result::Result<Vec<String>, String> {
    match stop {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(one)) => Ok(vec![one].into_iter().filter(|s| !s.is_empty()).collect()),
        Some(Value::Array(many)) => many
            .into_iter()
            .map(|entry| match entry {
                Value::String(text) => Ok(text),
                _ => Err("stop entries must be strings".to_string()),
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(|all| all.into_iter().filter(|s| !s.is_empty()).collect()),
        Some(_) => Err("stop must be a string or an array of strings".to_string()),
    }
}

fn parse_role(role: &str) -> std::result::Result<Role, String> {
    match role {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(format!("unsupported message role {other:?}")),
    }
}

fn bad_request(message: impl Into<String>) -> Response {
    // Ollama's error envelope is a flat `{"error": "..."}`, not OpenAI's
    // nested object (spec §212: each surface returns its own shape).
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": message.into() })),
    )
        .into_response()
}

// ------------------------------------------------------- response shapes

/// The nanosecond timings every Ollama client reads to display tok/s.
/// Measured, never synthesized: reporting zeros here would be a quiet lie
/// in a number users actually look at.
#[derive(Serialize, Default)]
struct Timings {
    total_duration: u64,
    /// Always 0: the model is already resident when a request arrives, so
    /// there is no per-request load. This is honest rather than unset.
    load_duration: u64,
    prompt_eval_count: u32,
    prompt_eval_duration: u64,
    eval_count: u32,
    eval_duration: u64,
}

impl From<&GeneratedOutput> for Timings {
    fn from(output: &GeneratedOutput) -> Self {
        let usage = output.usage;
        let prefill = usage.prefill.as_nanos().min(u64::MAX as u128) as u64;
        let decode = usage.decode.as_nanos().min(u64::MAX as u128) as u64;
        Self {
            total_duration: prefill.saturating_add(decode),
            load_duration: 0,
            prompt_eval_count: usage.prompt_tokens,
            prompt_eval_duration: prefill,
            eval_count: usage.completion_tokens,
            eval_duration: decode,
        }
    }
}

/// Ollama's `done_reason` has no `tool_calls` value — a tool call is
/// reported as `"stop"` with a populated `message.tool_calls`, so passing
/// the runtime's OpenAI-flavored reason straight through would emit an
/// invalid value.
fn done_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "length",
        _ => "stop",
    }
}

fn ollama_tool_calls(output: &GeneratedOutput) -> Vec<Value> {
    output
        .tool_calls
        .iter()
        .map(|call| {
            serde_json::json!({
                "function": {
                    "name": call.name,
                    "arguments": serde_json::from_str::<Value>(&call.arguments_json)
                        .unwrap_or_else(|_| Value::String(call.arguments_json.clone())),
                }
            })
        })
        .collect()
}

/// Whether this response is chat-shaped (`message`) or generate-shaped
/// (`response`) — the two differ only in that key.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Chat,
    Generate,
}

impl Shape {
    fn content(self, text: &str) -> (&'static str, Value) {
        match self {
            Self::Chat => (
                "message",
                serde_json::json!({"role": "assistant", "content": text}),
            ),
            Self::Generate => ("response", Value::String(text.to_string())),
        }
    }
}

/// One streamed, not-yet-final line.
fn partial_line(model: &str, shape: Shape, event: &StreamEvent) -> Option<Value> {
    let mut object = serde_json::json!({
        "model": model,
        "created_at": rfc3339_utc_now(),
        "done": false,
    });

    match event {
        StreamEvent::TextDelta(text) => {
            let (key, value) = shape.content(text);
            object[key] = value;
        }
        StreamEvent::Reasoning(text) => match shape {
            // Ollama's own field name for reasoning content. Surfacing it
            // beats dead air: on a reasoning model the think block can run
            // hundreds of tokens before any visible text.
            Shape::Chat => {
                object["message"] =
                    serde_json::json!({"role": "assistant", "content": "", "thinking": text});
            }
            Shape::Generate => {
                object["response"] = Value::String(String::new());
                object["thinking"] = Value::String(text.clone());
            }
        },
        // Tool calls ride on the terminal object, where Ollama puts them.
        StreamEvent::ToolCall(_) => return None,
    }
    Some(object)
}

/// The terminal object. Clients stop on `"done": true` rather than on
/// stream close, so this must always be sent.
fn final_line(model: &str, shape: Shape, output: &GeneratedOutput, streaming: bool) -> Value {
    let mut object = serde_json::json!({
        "model": model,
        "created_at": rfc3339_utc_now(),
        "done": true,
        "done_reason": done_reason(output.finish_reason),
    });

    // While streaming, the text already went out as deltas, so the
    // terminal object carries an empty content field (what real Ollama
    // does). A non-streaming response carries the whole thing.
    let text = if streaming { "" } else { output.text.as_str() };
    let (key, mut value) = shape.content(text);
    let calls = ollama_tool_calls(output);
    if !calls.is_empty() {
        match shape {
            Shape::Chat => value["tool_calls"] = Value::Array(calls),
            Shape::Generate => object["tool_calls"] = Value::Array(calls),
        }
    }
    object[key] = value;

    let timings = serde_json::to_value(Timings::from(output)).unwrap_or_default();
    if let (Some(target), Some(source)) = (object.as_object_mut(), timings.as_object()) {
        for (name, value) in source {
            target.insert(name.clone(), value.clone());
        }
    }
    object
}

fn error_line(message: &str) -> Value {
    serde_json::json!({ "error": message })
}

/// Recovers the human-readable reason from the response `stub` built, so
/// a streamed failure reports the same actionable message a
/// non-streaming request would have received.
async fn describe_failure(response: Response) -> String {
    use axum::body::to_bytes;
    const MAX_ERROR_BODY: usize = 8 * 1024;
    let fallback = "generation unavailable".to_string();
    let Ok(bytes) = to_bytes(response.into_body(), MAX_ERROR_BODY).await else {
        return fallback;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return fallback;
    };
    value
        .get("error")
        .and_then(|error| {
            error
                .as_str()
                .map(str::to_string)
                .or_else(|| error.get("message")?.as_str().map(str::to_string))
        })
        .unwrap_or(fallback)
}

// -------------------------------------------------------- NDJSON framing

/// One newline-delimited JSON line.
fn ndjson_line(value: &Value) -> Bytes {
    let mut line = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    line.push('\n');
    Bytes::from(line)
}

type NdjsonItem = std::result::Result<Bytes, Infallible>;
type NdjsonBody = CancelOnDrop<ReceiverStream<NdjsonItem>>;

/// Builds the NDJSON response.
///
/// `Sse` is deliberately not used: it prepends `data: ` and appends a
/// blank line, which Ollama clients cannot parse. This is a raw body with
/// an explicit content type — the one framing detail that breaks every
/// client while `curl` still looks fine.
fn ndjson_response(body: NdjsonBody) -> Response {
    (
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(body),
    )
        .into_response()
}

/// Runs a streamed generation and writes one NDJSON line per event,
/// terminating with the `done: true` object clients wait for.
fn stream_ndjson(
    state: AppState,
    request: NormalizedRequest,
    model: String,
    shape: Shape,
) -> Response {
    let (sender, receiver) = tokio::sync::mpsc::channel::<NdjsonItem>(32);
    let session = crate::runtime::Session::new();
    let cancellation = session.cancellation.clone();
    let cancellation_on_drop = cancellation.clone();

    tokio::spawn(async move {
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
                    if let Some(line) = partial_line(&model, shape, &event) {
                        if sender.send(Ok(ndjson_line(&line))).await.is_err() {
                            cancellation.cancel();
                            return;
                        }
                    }
                }
                // Catches a receiver dropped while this task waits for
                // send capacity; `CancelOnDrop` catches the rest.
                _ = sender.closed() => {
                    cancellation.cancel();
                    return;
                }
            }
        }

        let terminal = match generation.await {
            Ok(Ok(output)) => final_line(&model, shape, &output, true),
            // Carry the real reason rather than a generic one: "no model
            // installed yet; run `tqf`" is actionable, "generation
            // unavailable" is not.
            Ok(Err(response)) => error_line(&describe_failure(response).await),
            Err(_) => error_line("the generation worker stopped unexpectedly"),
        };
        let _ = sender.send(Ok(ndjson_line(&terminal))).await;
    });

    ndjson_response(CancelOnDrop {
        inner: ReceiverStream::new(receiver),
        cancellation: cancellation_on_drop,
    })
}

// ------------------------------------------------------------- handlers

async fn chat(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    if let Err(message) = model_id::resolve(req.model.as_deref()) {
        return bad_request(message);
    }
    if req.messages.is_empty() {
        return bad_request("messages must contain at least one item");
    }
    if let Some(rejection) = reject_unsupported(req.format.as_ref(), req.think.as_ref()) {
        return rejection;
    }
    if req.messages.iter().any(|m| !m.images.is_empty()) {
        return bad_request(
            "image inputs are not supported yet: the vision encoder is not wired into the \
             request path in this build",
        );
    }

    let messages = match req
        .messages
        .into_iter()
        .map(|m| {
            Ok(Message {
                role: parse_role(&m.role)?,
                content: m.content,
                tool_calls: Vec::new(),
            })
        })
        .collect::<std::result::Result<Vec<_>, String>>()
    {
        Ok(messages) => messages,
        Err(message) => return bad_request(message),
    };

    // Ollama's tool objects use the same OpenAI function-tool JSON, so the
    // OpenAI adapter's normalization applies verbatim.
    let tools = match crate::server::openai::normalize_tools(req.tools) {
        Ok(tools) => tools,
        Err(message) => return bad_request(message),
    };

    let sampling = match req.options.into_sampling() {
        Ok(sampling) => sampling,
        Err(message) => return bad_request(message),
    };

    let mut normalized = NormalizedRequest::new(ProtocolFlavor::Ollama, messages, req.stream);
    normalized.tools = tools;
    normalized.sampling = sampling;

    // Echo the exact string the client asked for: real Ollama does, and
    // clients key session state on it (spec §203's "unless the client
    // requires echoing its requested alias").
    let echoed = req.model.unwrap_or_else(|| OLLAMA_MODEL_TAG.to_string());
    respond(state, normalized, echoed, Shape::Chat).await
}

async fn generate(State(state): State<AppState>, Json(req): Json<GenerateRequest>) -> Response {
    if let Err(message) = model_id::resolve(req.model.as_deref()) {
        return bad_request(message);
    }
    if let Some(rejection) = reject_unsupported(req.format.as_ref(), None) {
        return rejection;
    }
    // Each of these is rejected rather than ignored because honoring it
    // partially would produce silently wrong output (spec §204).
    if req.raw == Some(true) {
        return bad_request(
            "raw: true is not supported: this server always renders the pinned Qwen3.6 chat \
             template, and templating a prompt labelled raw would misrepresent what ran",
        );
    }
    if req.template.is_some() {
        return bad_request("template overrides are not supported: the chat template is pinned");
    }
    if req.context.is_some() {
        return bad_request(
            "the context parameter is not supported: this server has no equivalent \
             conversation-state vector, and partially honoring it would silently change the \
             continuation",
        );
    }
    if req.suffix.is_some() {
        return bad_request("suffix (fill-in-the-middle) is not implemented in this build");
    }
    if !req.images.is_empty() {
        return bad_request(
            "image inputs are not supported yet: the vision encoder is not wired into the \
             request path in this build",
        );
    }

    let mut messages = Vec::new();
    if let Some(system) = req.system.filter(|s| !s.is_empty()) {
        messages.push(Message {
            role: Role::System,
            content: system,
            tool_calls: Vec::new(),
        });
    }
    messages.push(Message {
        role: Role::User,
        content: req.prompt,
        tool_calls: Vec::new(),
    });

    let sampling = match req.options.into_sampling() {
        Ok(sampling) => sampling,
        Err(message) => return bad_request(message),
    };
    let mut normalized = NormalizedRequest::new(ProtocolFlavor::Ollama, messages, req.stream);
    normalized.sampling = sampling;

    let echoed = req.model.unwrap_or_else(|| OLLAMA_MODEL_TAG.to_string());
    respond(state, normalized, echoed, Shape::Generate).await
}

/// Rejections shared by chat and generate.
fn reject_unsupported(format: Option<&Value>, think: Option<&Value>) -> Option<Response> {
    // `format` asks for grammar-constrained output. Spec §204: implement
    // it only when enforcement is real, and until then reject — accepting
    // it and returning unconstrained text is the failure mode that wastes
    // the most of a caller's time.
    if format.is_some_and(|value| !value.is_null()) {
        return Some(bad_request(
            "format (JSON mode / schema) is not supported: this build has no grammar \
             enforcement, and accepting the parameter without honoring it would return \
             unconstrained output that looks like it was constrained",
        ));
    }
    if think.is_some_and(|value| value == &Value::Bool(true)) {
        return Some(bad_request(
            "think: true is not supported: reasoning content is streamed on a separate \
             channel but is not exposed as a client-controllable mode in this build",
        ));
    }
    None
}

async fn respond(
    state: AppState,
    normalized: NormalizedRequest,
    model: String,
    shape: Shape,
) -> Response {
    if normalized.stream {
        return stream_ndjson(state, normalized, model, shape);
    }
    match stub::generate(&state, &normalized).await {
        Ok(output) => Json(final_line(&model, shape, &output, false)).into_response(),
        Err(response) => response,
    }
}

async fn embed(State(_state): State<AppState>, Json(_req): Json<Value>) -> Response {
    // Honest about *why*, not just that: the embedding runtime exists and
    // is oracle-validated (Phase 37), but `source::pinned` names no
    // pplx-embed artifact, so this build has no way to acquire the
    // checkpoint. See spec §86.
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "embeddings are not available in this build: the pplx-embed helper model \
                      is implemented and validated but is not pinned as an installable \
                      artifact, so there is no checkpoint to serve"
        })),
    )
        .into_response()
}

async fn tags(State(state): State<AppState>) -> Json<Value> {
    // An empty list is what real Ollama returns for an empty library, and
    // is what clients expect — not an error.
    let Some(receipt) = &state.model_receipt else {
        return Json(serde_json::json!({ "models": [] }));
    };

    let size = std::fs::metadata(&receipt.tqf_path)
        .map(|m| m.len())
        .unwrap_or(0);
    Json(serde_json::json!({
        "models": [{
            "name": OLLAMA_MODEL_TAG,
            "model": OLLAMA_MODEL_TAG,
            "modified_at": rfc3339_utc(receipt.installed_at_unix, 0),
            // Real values from the trusted receipt, not plausible-looking
            // invented ones.
            "size": size,
            "digest": receipt.conversion_fingerprint_blake3,
            "details": model_details(),
        }]
    }))
}

async fn show(State(state): State<AppState>, Json(req): Json<Value>) -> Response {
    let requested = req
        .get("model")
        .or_else(|| req.get("name"))
        .and_then(Value::as_str);
    if let Err(message) = model_id::resolve(requested) {
        return bad_request(message);
    }
    if state.model_receipt.is_none() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "no model is installed; run `tqf` to complete first-run setup"
            })),
        )
            .into_response();
    }

    use crate::model::qwen36::geometry::Qwen36Geometry;
    Json(serde_json::json!({
        "details": model_details(),
        "model_info": {
            "general.architecture": "qwen3moe",
            "general.parameter_count": 35_000_000_000u64,
            // The configured logical context, not a model constant: this
            // is what the running server will actually accept.
            "qwen3moe.context_length": state
                .config
                .context_limit_tokens
                .unwrap_or(128 * 1024),
            "qwen3moe.embedding_length": Qwen36Geometry::HIDDEN_SIZE,
            "qwen3moe.block_count": Qwen36Geometry::NUM_LAYERS,
            "qwen3moe.attention.head_count": Qwen36Geometry::FULL_ATTENTION_HEADS,
            "qwen3moe.attention.head_count_kv": Qwen36Geometry::FULL_KV_HEADS,
            "qwen3moe.rope.freq_base": Qwen36Geometry::ROPE_THETA,
            "qwen3moe.expert_count": Qwen36Geometry::NUM_EXPERTS,
            "qwen3moe.expert_used_count": Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN,
            "qwen3moe.vocab_size": Qwen36Geometry::VOCAB_SIZE,
        },
        // Deliberately no "vision": the encoder exists but is not wired
        // into the request path, and advertising it would make clients
        // send images this build rejects.
        "capabilities": ["completion", "tools"],
        "license": crate::source::pinned::LICENSE_ID,
        // Empty rather than a fabricated Modelfile: TQF is not built from
        // one, and inventing plausible text would be worse than nothing.
        "modelfile": "",
        "template": "",
        "parameters": "",
    }))
    .into_response()
}

async fn ps(State(state): State<AppState>) -> Json<Value> {
    if state.generator.is_none() {
        return Json(serde_json::json!({ "models": [] }));
    }

    let resident = crate::memory::os_sampler::sample_process_footprint()
        .map(|(resident, _, _)| resident.0)
        .unwrap_or(0);
    Json(serde_json::json!({
        "models": [{
            "name": OLLAMA_MODEL_TAG,
            "model": OLLAMA_MODEL_TAG,
            // Real OS-observed resident bytes, the same number
            // /v1/tqf/metrics serves.
            "size": resident,
            // Honest 0: experts stream from SSD, and the GPU-resident
            // path is opt-in and measured a negative (Phase 20).
            "size_vram": 0,
            "digest": state
                .model_receipt
                .as_ref()
                .map(|r| r.conversion_fingerprint_blake3.clone())
                .unwrap_or_default(),
            "details": model_details(),
            // TQF never unloads, so keep_alive has no meaning here.
            "expires_at": "9999-12-31T23:59:59.000000000Z",
        }]
    }))
}

fn model_details() -> Value {
    serde_json::json!({
        "parent_model": "",
        "format": "tqf",
        "family": "qwen3moe",
        "families": ["qwen3moe"],
        "parameter_size": "35B",
        "quantization_level": "Q4_K_M",
    })
}
