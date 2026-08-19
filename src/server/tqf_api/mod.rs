//! Native tqf control/status endpoints (spec Part IX section 69).

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::memory::os_sampler::sample_process_footprint;
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

/// spec §47's "the inspector consumes metrics": real OS-observed
/// process memory (Phase 24's sampler, no broker dependency needed for
/// this reading) plus the same real fields `/health` exposes. This
/// endpoint is read-only by construction — it has no mutating
/// counterpart, matching spec §47's "must not change runtime policy
/// directly except through supported configuration actions" (there are
/// none to expose here yet; see the Phase 47 qualification doc).
#[derive(Serialize)]
struct MetricsResponse {
    uptime_seconds: u64,
    model_installed: bool,
    /// `None` on a platform this crate's OS sampler doesn't support.
    resident_bytes: Option<u64>,
    virtual_bytes: Option<u64>,
    resident_peak_bytes: Option<u64>,
}

async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let footprint = sample_process_footprint();
    Json(MetricsResponse {
        uptime_seconds: state.started_at.elapsed().as_secs(),
        model_installed: state.model_installed,
        resident_bytes: footprint.map(|(resident, _, _)| resident.0),
        virtual_bytes: footprint.map(|(_, virt, _)| virt.0),
        resident_peak_bytes: footprint.map(|(_, _, peak)| peak.0),
    })
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/tqf/metrics", get(metrics))
}
