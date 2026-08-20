//! Native tqf control/status endpoints (spec Part IX section 69).

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use serde_json::Value;

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

/// Spec §211's native diagnostics namespace, kept separate from the
/// compatibility surfaces.
///
/// `/v1/tqf/metrics` stays as an alias because the Phase 47 SwiftUI
/// inspector already reads it; `/tqf/metrics` is the spec's own spelling
/// and what new callers should use.
/// The only endpoint that must answer before any credential is
/// presented: `bind::probe_health` uses it to tell an existing tqf from
/// some other process squatting on the port, which happens during bind,
/// before a key could be supplied. It reports liveness and version only.
pub fn public_routes() -> Router<AppState> {
    Router::new().route("/health", get(health))
}

/// Native diagnostics (spec §211). These sit behind the API-key layer:
/// §211 requires that "sensitive project paths should not be exposed to
/// non-loopback clients without authentication", and `/tqf/status`
/// reports the model container path, source revision, and the memory and
/// context configuration. Loopback binds mint no key, so local tools —
/// including the GUI inspector — are unaffected.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/tqf/metrics", get(metrics))
        .route("/tqf/metrics", get(metrics))
        .route("/tqf/status", get(status))
        .route("/tqf/memory", get(memory))
        .route("/tqf/context", get(context))
        .route("/tqf/indexes", get(indexes))
        .route("/tqf/index/search", axum::routing::post(index_search))
}

/// What is installed, loaded, and serving. The native counterpart to
/// `tqf status`, reporting the same facts over HTTP.
async fn status(State(state): State<AppState>) -> Json<Value> {
    let receipt = state.model_receipt.as_ref();
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": state.started_at.elapsed().as_secs(),
        "model": {
            "id": crate::server::model_id::CANONICAL_MODEL_ID,
            "installed": state.model_installed,
            // Installed and loaded are different states: a valid receipt
            // can exist while the runtime failed to construct.
            "loaded": state.generator.is_some(),
            "family": receipt.map(|r| r.model_family.clone()),
            "source_revision": receipt.and_then(|r| r.source_revision.clone()),
            "container": receipt.map(|r| r.tqf_path.display().to_string()),
        },
        "config": {
            "memory_budget_bytes": state.config.memory_budget_bytes,
            "context_limit_tokens": state.config.context_limit_tokens,
            "vision_enabled": state.config.enable_vision,
        },
        "backend": crate::setup::hardware::detect().backend,
    }))
}

/// OS-observed process memory. The broker's own per-owner accounting is
/// deliberately absent: the broker belongs to the loaded runtime and is
/// not reachable from here, and inventing a breakdown would be worse
/// than omitting one. `/tqf/status` names the configured budget.
async fn memory(State(state): State<AppState>) -> Json<Value> {
    let footprint = sample_process_footprint();

    // Phase 24's rule: report both numbers or neither. A configuration is
    // not "4G certified" because steady-state decode is 3.9G if admission
    // spiked the OS-observed footprint to 4.7G, so the broker's own
    // accounting and the OS reading are only meaningful together.
    let reserved = state
        .generator
        .as_ref()
        .and_then(|g| g.broker())
        .map(|broker| {
            let snapshot = broker.snapshot();
            serde_json::json!({
                "budget_bytes": snapshot.budget.0,
                "reserved_bytes": snapshot.reserved.0,
                "peak_bytes": snapshot.peak.0,
                "by_owner": {
                    "core": snapshot.by_owner.core,
                    "gdn_state": snapshot.by_owner.gdn_state,
                    "context_hot": snapshot.by_owner.context_hot,
                    "context_cold": snapshot.by_owner.context_cold,
                    "expert_pinned": snapshot.by_owner.expert_pinned,
                    "expert_probation": snapshot.by_owner.expert_probation,
                    "io_staging": snapshot.by_owner.io_staging,
                    "scratch": snapshot.by_owner.scratch,
                    "server_reserve": snapshot.by_owner.server_reserve,
                    "helper_model": snapshot.by_owner.helper_model,
                },
            })
        });

    Json(serde_json::json!({
        "budget_bytes": state.config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024),
        "observed": {
            "resident_bytes": footprint.map(|(resident, _, _)| resident.0),
            "virtual_bytes": footprint.map(|(_, virtual_bytes, _)| virtual_bytes.0),
            "resident_peak_bytes": footprint.map(|(_, _, peak)| peak.0),
        },
        // `null` when no runtime is loaded, which is honest: there is no
        // broker yet, as opposed to a broker reporting zero.
        "reserved": reserved,
    }))
}

/// Context capability and which KV backend is actually in use.
async fn context(State(state): State<AppState>) -> Json<Value> {
    let (backend, precision) = crate::context::tqkv::configured_backend_description();
    Json(serde_json::json!({
        "limit_tokens": state.config.context_limit_tokens.unwrap_or(128 * 1024),
        "kv_backend": backend,
        "kv_precision": precision,
        "selective_attention": false,
    }))
}

/// Spec §211's index listing, reporting the indexes loaded at startup
/// from the registry (spec §218) — including the roots that failed to
/// load or went stale, since a silently missing index is
/// indistinguishable from one that was never synced.
async fn indexes(State(state): State<AppState>) -> Json<Value> {
    let loaded: Vec<Value> = state
        .indexes
        .roots
        .iter()
        .map(|root| {
            serde_json::json!({
                "root": root.root.display().to_string(),
                "index": root.index_path.display().to_string(),
                "generation": root.generation,
                "files": root.file_count,
                "lexical_terms": root.term_count,
                // Stated per index rather than once, because it is a
                // property of what this index contains: the semantic lane
                // needs the helper model, which is not installed.
                "lanes": ["lexical", "exact"],
            })
        })
        .collect();

    // Registered roots that did not load are reported, not omitted: a
    // silently missing index looks identical to one that was never
    // synced.
    let failed: Vec<Value> = state
        .indexes
        .failed
        .iter()
        .map(|(root, why)| serde_json::json!({"root": root.display().to_string(), "error": why}))
        .collect();
    let stale: Vec<String> = state
        .indexes
        .stale
        .iter()
        .map(|root| root.display().to_string())
        .collect();

    let mut body = serde_json::json!({
        "indexes": loaded,
        "failed": failed,
        "stale": stale,
    });
    // An empty list with no explanation leaves a caller guessing whether
    // indexing is unsupported or simply unused.
    if state.indexes.roots.is_empty()
        && state.indexes.failed.is_empty()
        && state.indexes.stale.is_empty()
    {
        body["note"] =
            serde_json::json!("no root is synced; run `tqf sync <path>` to build and register one");
    }
    Json(body)
}

#[derive(serde::Deserialize)]
struct IndexSearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Case-sensitive exact identifier lookup, which bypasses BM25
    /// entirely (spec §83, §182) rather than being blended into it.
    #[serde(default)]
    exact: bool,
}

fn default_top_k() -> usize {
    10
}

/// `POST /tqf/index/search` (spec §211).
async fn index_search(
    State(state): State<AppState>,
    Json(request): Json<IndexSearchRequest>,
) -> Response {
    if request.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "query must not be empty"})),
        )
            .into_response();
    }
    // Spec §44: no index is an ordinary state, not an error. Report an
    // empty result set and say why, so a client can tell "nothing matched"
    // from "nothing is indexed".
    if state.indexes.is_empty() {
        return Json(serde_json::json!({
            "hits": [],
            "note": "no root is synced; run `tqf sync <path>`",
        }))
        .into_response();
    }

    let top_k = request.top_k.clamp(1, 100);
    let hits = if request.exact {
        state.indexes.exact_lookup(request.query.trim())
    } else {
        state.indexes.search(&request.query, top_k)
    };

    Json(serde_json::json!({
        "hits": hits
            .iter()
            .take(top_k)
            .map(|hit| serde_json::json!({
                "root": hit.root,
                "path": hit.path,
                "score": hit.score,
            }))
            .collect::<Vec<_>>(),
        "lane": if request.exact { "exact" } else { "lexical" },
    }))
    .into_response()
}
