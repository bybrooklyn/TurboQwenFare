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

use crate::runtime::{GeneratedOutput, NormalizedRequest, Session};
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
            request.stream,
        ));
    };

    if !state.model_installed {
        return Err(not_ready_body(
            "no model installed yet; run `tqf` to complete first-run setup",
            request.stream,
        ));
    }
    let Some(generator) = &state.generator else {
        return Err(not_ready_body(
            "model receipt exists but its Qwen runtime is not loaded",
            request.stream,
        ));
    };
    match generator
        .generate(request.clone(), session.cancellation.clone())
        .await
    {
        Ok(output) => Ok(output),
        Err(error) => Err(not_ready_body(
            &format!("generation failed: {error}"),
            request.stream,
        )),
    }
}

fn not_ready_body(message: &str, stream: bool) -> Response {
    if stream {
        sse_error(message).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: ErrorDetail {
                    message: message.to_string(),
                    r#type: "service_unavailable",
                    code: "model_not_ready",
                },
            }),
        )
            .into_response()
    }
}

fn sse_error(message: &str) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let error_event = Event::default().event("error").data(message);
    let done_event = Event::default().data("[DONE]");
    Sse::new(tokio_stream::iter(vec![Ok(error_event), Ok(done_event)]))
}
