//! API-key gate for non-loopback exposure (spec Part IX section 74). A
//! no-op whenever `state.api_key` is `None` (loopback, or explicit
//! `--insecure`).

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use crate::server::AppState;

#[derive(Serialize)]
struct ErrorBody {
    error: &'static str,
}

pub async fn require_api_key(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.api_key.as_deref() else {
        return next.run(request).await;
    };

    let authorized = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|token| token == expected);

    if authorized {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorBody {
                error: "missing or invalid API key (spec Part IX section 74: this \
                        server is bound to a non-loopback address)",
            }),
        )
            .into_response()
    }
}
