//! Protocol servers. All flavors normalize into one internal request/event
//! representation before reaching the runtime (spec Part IV section 26;
//! Part IX).

pub mod anthropic;
pub mod auth;
pub mod bind;
pub mod ollama;
pub mod openai;
pub mod stub;
#[cfg(test)]
mod tests;
pub mod tqf_api;

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use tokio::net::TcpListener;

use crate::config::Config;
use crate::runtime::GenerationSlot;

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
    pub started_at: Instant,
    /// `Some` only for non-loopback binds without `--insecure` (spec Part
    /// IX section 74); enforced by the `auth` middleware on the protected
    /// sub-router.
    pub api_key: Option<Arc<str>>,
}

pub fn build_router(state: AppState) -> Router {
    let protected =
        Router::new()
            .merge(openai::routes())
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                auth::require_api_key,
            ));

    Router::new()
        .merge(tqf_api::routes())
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
