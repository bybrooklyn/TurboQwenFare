//! OpenAI-compatible surface: model listing plus Responses/Chat/Embeddings
//! stubs (spec Part IX sections 70-71). Generation itself lands in later
//! phases; these routes validate request shape and exercise the real
//! request-queue/cancellation path today.

use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::runtime::{Message, NormalizedRequest, ProtocolFlavor, Role};
use crate::server::{stub, AppState};

const CANONICAL_MODEL_ID: &str = "qwen3.6-35b-a3b";

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
    content: String,
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
}

fn parse_role(role: &str) -> Role {
    match role {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    let messages = req
        .messages
        .into_iter()
        .map(|m| Message {
            role: parse_role(&m.role),
            content: m.content,
        })
        .collect();
    let normalized =
        NormalizedRequest::new(ProtocolFlavor::OpenAiChatCompletions, messages, req.stream);
    stub::not_ready(&state, &normalized).await
}

/// Responses request shape is intentionally loose (any JSON object) rather
/// than a committed struct: the endpoint is not implemented yet, and typing
/// the full Responses schema now would be guessing at a contract before
/// it's needed.
async fn responses(State(state): State<AppState>, Json(req): Json<Value>) -> Response {
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let input = req
        .get("input")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let messages = vec![Message {
        role: Role::User,
        content: input,
    }];
    let normalized = NormalizedRequest::new(ProtocolFlavor::OpenAiResponses, messages, stream);
    stub::not_ready(&state, &normalized).await
}

async fn embeddings(State(state): State<AppState>, Json(_req): Json<Value>) -> Response {
    let normalized = NormalizedRequest::new(ProtocolFlavor::OpenAiEmbeddings, Vec::new(), false);
    stub::not_ready(&state, &normalized).await
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/embeddings", post(embeddings))
}
