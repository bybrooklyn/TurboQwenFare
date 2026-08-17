//! Session identity and the single-active-generation contract (spec Part IV
//! section 26; Part IX section 75: v1 runs one generation at a time and
//! queues the rest, with clean cancellation).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(u64);

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

impl SessionId {
    pub fn next() -> Self {
        Self(NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sess-{}", self.0)
    }
}

/// Placeholders: token/context/prefix state belong to the tokenizer (phase
/// 9) and context subsystems (phase 27+). They live here now only so
/// `Session`'s shape matches spec section 26 and won't need to be
/// restructured when those phases land.
#[derive(Debug, Default)]
pub struct TokenStore;
#[derive(Debug, Default)]
pub struct ContextState;
#[derive(Debug, Clone)]
pub struct PrefixHandle;

#[derive(Debug)]
pub struct Session {
    pub id: SessionId,
    pub token_history: TokenStore,
    pub context: ContextState,
    pub prefix: Option<PrefixHandle>,
    pub cancellation: CancellationToken,
}

impl Session {
    pub fn new() -> Self {
        Self {
            id: SessionId::next(),
            token_history: TokenStore::default(),
            context: ContextState::default(),
            prefix: None,
            cancellation: CancellationToken::new(),
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Axum cancels a request handler future when its client disappears.
        // Turning that ordinary Rust drop into an explicit token signal also
        // stops queued or spawn-blocking model work for non-streaming calls.
        self.cancellation.cancel();
    }
}

/// v1 has exactly one active generation; every other request queues behind
/// this permit. Holding the permit *is* "generating" (spec Part IX section
/// 75, Part IV section 25).
#[derive(Debug, Clone)]
pub struct GenerationSlot {
    semaphore: Arc<Semaphore>,
}

impl GenerationSlot {
    pub fn new() -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(1)),
        }
    }

    /// Waits for the single generation slot. Returns `None` if
    /// `cancellation` fires first (client disconnect / explicit cancel)
    /// rather than the request ever reaching the front of the queue.
    pub async fn acquire(&self, cancellation: &CancellationToken) -> Option<OwnedSemaphorePermit> {
        tokio::select! {
            permit = self.semaphore.clone().acquire_owned() => permit.ok(),
            _ = cancellation.cancelled() => None,
        }
    }
}

impl Default for GenerationSlot {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropping_a_session_signals_all_cancellation_observers() {
        let session = Session::new();
        let observer = session.cancellation.clone();
        assert!(!observer.is_cancelled());
        drop(session);
        assert!(observer.is_cancelled());
    }
}
