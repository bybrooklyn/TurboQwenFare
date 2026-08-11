//! Shared "not ready yet" response building for every generation endpoint.
//! Real generation arrives with the model core (phases 10-16); until then,
//! every endpoint still exercises the real request-queue/cancellation path
//! (spec Part IX section 75) before honestly reporting why it can't answer.

use std::convert::Infallible;

use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tokio_stream::Stream;

use crate::runtime::{NormalizedRequest, Session};
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
/// (so queuing and cancellation behavior is exercised end to end), then
/// returns the honest reason generation can't proceed yet.
pub async fn not_ready(state: &AppState, request: &NormalizedRequest) -> Response {
    let session = Session::new();
    tracing::debug!(
        session_id = %session.id,
        protocol = ?request.protocol,
        stream = request.stream,
        "queued stub generation"
    );

    let Some(_permit) = state.generation_slot.acquire(&session.cancellation).await else {
        return not_ready_body(
            "request cancelled before it reached the generation slot",
            request.stream,
        );
    };

    if !state.model_installed {
        return not_ready_body(
            "no model installed yet; run `tqf` to complete first-run setup",
            request.stream,
        );
    }

    not_ready_body(
        "model is installed but generation is not implemented yet (phases 10-16)",
        request.stream,
    )
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
    let error_event = Event::default().event("error").data(message.to_string());
    let done_event = Event::default().data("[DONE]");
    Sse::new(tokio_stream::iter(vec![Ok(error_event), Ok(done_event)]))
}
