//! Shared generation admission and unavailable-state translation for every
//! endpoint. A loaded Qwen generator runs through the real single-request
//! queue; missing installation/runtime states produce an honest response.

use std::convert::Infallible;

use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tokio_stream::Stream;

use crate::runtime::stream_decoder::StreamEvent;
use crate::runtime::{GeneratedOutput, NormalizedRequest, ProtocolFlavor, Session};
use crate::server::AppState;

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: &'static str,
}

/// Runs a normalized request through the real single-active-generation slot
/// before calling the loaded Qwen generator or returning its honest
/// unavailable state.
pub async fn generate(
    state: &AppState,
    request: &NormalizedRequest,
) -> std::result::Result<GeneratedOutput, Response> {
    let session = Session::new();
    generate_with_session(state, request, session).await
}

/// Streaming adapters create the session before returning their response so
/// dropping the body can cancel queued or in-progress reference work.
pub async fn generate_with_session(
    state: &AppState,
    request: &NormalizedRequest,
    session: Session,
) -> std::result::Result<GeneratedOutput, Response> {
    tracing::debug!(
        session_id = %session.id,
        protocol = ?request.protocol,
        stream = request.stream,
        "queued generation"
    );

    let Some(_permit) = state.generation_slot.acquire(&session.cancellation).await else {
        return Err(not_ready_body(
            "request cancelled before it reached the generation slot",
            request,
        ));
    };

    if !state.model_installed {
        return Err(not_ready_body(
            "no model installed yet; run `tqf` to complete first-run setup",
            request,
        ));
    }
    let Some(generator) = &state.generator else {
        return Err(not_ready_body(
            "model receipt exists but its Qwen runtime is not loaded",
            request,
        ));
    };
    match generator
        .generate(request.clone(), session.cancellation.clone())
        .await
    {
        Ok(output) => Ok(output),
        Err(error) => Err(not_ready_body(
            &format!("generation failed: {error}"),
            request,
        )),
    }
}

/// Streaming twin of [`generate_with_session`].
///
/// It repeats the same admission checks rather than sharing a helper on
/// purpose: the single generation slot must be held for the whole
/// streamed generation, so the permit has to live in this function's body
/// until the generator returns.
pub async fn generate_streaming_with_session(
    state: &AppState,
    request: &NormalizedRequest,
    session: Session,
    events: tokio::sync::mpsc::Sender<StreamEvent>,
) -> std::result::Result<GeneratedOutput, Response> {
    tracing::debug!(
        session_id = %session.id,
        protocol = ?request.protocol,
        "queued streaming generation"
    );

    let Some(_permit) = state.generation_slot.acquire(&session.cancellation).await else {
        return Err(not_ready_body(
            "request cancelled before it reached the generation slot",
            request,
        ));
    };

    if !state.model_installed {
        return Err(not_ready_body(
            "no model installed yet; run `tqf` to complete first-run setup",
            request,
        ));
    }
    let Some(generator) = &state.generator else {
        return Err(not_ready_body(
            "model receipt exists but its Qwen runtime is not loaded",
            request,
        ));
    };
    generator
        .generate_streaming(request.clone(), session.cancellation.clone(), events)
        .await
        .map_err(|error| not_ready_body(&format!("generation failed: {error}"), request))
}

/// Translates an unavailable state into the error envelope the *calling
/// protocol* uses (spec §212: OpenAI surfaces return OpenAI-like errors,
/// Ollama surfaces return Ollama-like ones, all mapped from one internal
/// taxonomy).
///
/// Before this was protocol-aware, an Ollama client asking a server with
/// no model installed got OpenAI's nested `{"error": {"message": ...}}`
/// on an `/api/*` route, which Ollama clients do not parse.
fn not_ready_body(message: &str, request: &NormalizedRequest) -> Response {
    match request.protocol {
        ProtocolFlavor::Ollama => ollama_not_ready(message, request.stream),
        ProtocolFlavor::Anthropic => anthropic_not_ready(message, request.stream),
        _ if request.stream => sse_error(message).into_response(),
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: ErrorDetail {
                    message: message.to_string(),
                    r#type: "service_unavailable",
                    code: "model_not_ready",
                },
            }),
        )
            .into_response(),
    }
}

/// Anthropic's `{"type": "error", "error": {...}}` envelope. A streaming
/// request receives it as an SSE `error` event, which is how Anthropic's
/// own stream reports a mid-flight failure.
fn anthropic_not_ready(message: &str, stream: bool) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {"type": "overloaded_error", "message": message},
    });
    if stream {
        return Sse::new(tokio_stream::iter(vec![Ok::<_, Infallible>(
            Event::default().event("error").data(body.to_string()),
        )]))
        .into_response();
    }
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

/// Ollama's flat `{"error": "..."}` envelope. A streaming request gets it
/// as a single NDJSON line, since the client is already parsing that
/// framing and an abrupt close would look like a network fault.
fn ollama_not_ready(message: &str, stream: bool) -> Response {
    let body = serde_json::json!({ "error": message });
    if stream {
        let mut line = body.to_string();
        line.push('\n');
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
            line,
        )
            .into_response();
    }
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

fn sse_error(message: &str) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let error_event = Event::default().event("error").data(message);
    let done_event = Event::default().data("[DONE]");
    Sse::new(tokio_stream::iter(vec![Ok(error_event), Ok(done_event)]))
}
