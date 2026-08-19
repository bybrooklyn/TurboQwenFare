//! Protocol servers. All flavors normalize into one internal request/event
//! representation before reaching the runtime (spec Part IV section 26;
//! Part IX).

pub mod anthropic;
pub mod auth;
pub mod bind;
#[cfg(test)]
mod conformance;
pub mod model_id;
pub mod ollama;
pub mod openai;
#[cfg(test)]
mod security_tests;
pub mod stream;
pub mod stub;
#[cfg(test)]
mod tests;
pub mod tqf_api;

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use tokio::net::TcpListener;

use crate::config::Config;
use crate::runtime::{GenerationSlot, Qwen36Generator};

/// Shared state handed to every route. Cloned per-request by axum, so its
/// contents are themselves cheap-to-clone handles (`Arc`, atomics).
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    /// Whether a valid trusted receipt was found at startup (spec Part V
    /// section 36). Real model *loading* is a much later phase; this only
    /// reflects "is there something on disk to load."
    pub model_installed: bool,
    pub generation_slot: GenerationSlot,
    /// `None` until a trusted converted Qwen3.6 model has loaded. The
    /// protocol layer never fabricates output when this is absent.
    pub generator: Option<Arc<dyn Qwen36Generator>>,
    /// The validated receipt for the installed model, when there is one.
    /// Inventory endpoints (`/api/tags`, `/api/show`, `/v1/models`) report
    /// size, digest, and source revision from this rather than
    /// synthesizing values that merely look right.
    pub model_receipt: Option<Arc<crate::setup::receipt::ModelReceipt>>,
    pub started_at: Instant,
    /// `Some` only for non-loopback binds without `--insecure` (spec Part
    /// IX section 74); enforced by the `auth` middleware on the protected
    /// sub-router.
    pub api_key: Option<Arc<str>>,
}

/// Assembles the whole HTTP surface.
///
/// The split between the two tiers is security-critical, not stylistic.
/// Everything that can generate, embed, or enumerate what is installed
/// goes inside `protected`, so a `0.0.0.0` bind requires the API key
/// tqf mints for it (spec §74). The unauthenticated tier is limited to
/// fixed-content liveness probes that clients call *before* they have
/// anywhere to put a credential:
///
/// - `/health` — also what `bind::probe_health` uses to recognize another
///   tqf on a busy port, so it must answer without a key.
/// - `/v1/tqf/metrics` — process uptime and memory, no model data.
/// - `GET`/`HEAD /` and `/api/version` — Ollama's liveness handshake.
///
/// Merging the Ollama routes at this top level is the path of least
/// resistance and would silently expose generation with no auth.
pub fn build_router(state: AppState) -> Router {
    let protected = Router::new()
        .merge(openai::routes())
        .merge(ollama::routes())
        .merge(anthropic::routes())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_api_key,
        ));

    Router::new()
        .merge(tqf_api::routes())
        .merge(ollama::unauthenticated_routes())
        .merge(protected)
        .with_state(state)
}

/// Binds and serves until a shutdown signal arrives, honoring in-flight
/// requests (spec Part IX section 75: cancellation must not just vanish
/// connections).
pub async fn serve(listener: TcpListener, state: AppState) -> std::io::Result<()> {
    let router = build_router(state);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(err) => {
                tracing::warn!(%err, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received, draining in-flight requests");
}
