//! Native tqf control/status endpoints (spec Part IX section 69).

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::server::AppState;

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    model_installed: bool,
    uptime_seconds: u64,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        model_installed: state.model_installed,
        uptime_seconds: state.started_at.elapsed().as_secs(),
    })
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/health", get(health))
}
