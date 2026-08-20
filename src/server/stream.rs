//! Client-disconnect handling shared by every streaming surface.
//!
//! Hoisted out of the OpenAI adapter so the SSE and NDJSON paths cannot
//! drift: two hand-copied cancellation implementations that slowly
//! disagree is exactly the bug class spec §71 is about.

use std::pin::Pin;
use std::task::{Context, Poll};

use tokio_stream::Stream;
use tokio_util::sync::CancellationToken;

/// Ties a response body's lifetime directly to its model session.
///
/// Axum drops the body when the client stops consuming it. Without this,
/// a disconnect is only noticed at the *next* channel send — which never
/// happens if the generator is blocked mid-decode, so the generation runs
/// to completion holding the single generation slot with nobody reading.
pub struct CancelOnDrop<S> {
    pub inner: S,
    pub cancellation: CancellationToken,
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::wrappers::ReceiverStream;

    #[test]
    fn dropping_a_stream_cancels_its_session_for_any_item_type() {
        // Proven for both wire formats, since one shared implementation
        // now backs SSE and NDJSON alike.
        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(1);
        let cancellation = CancellationToken::new();
        let stream = CancelOnDrop {
            inner: ReceiverStream::new(bytes_rx),
            cancellation: cancellation.clone(),
        };
        assert!(!cancellation.is_cancelled());
        drop(stream);
        assert!(cancellation.is_cancelled(), "NDJSON body drop must cancel");
        drop(bytes_tx);

        let (text_tx, text_rx) = tokio::sync::mpsc::channel::<String>(1);
        let cancellation = CancellationToken::new();
        let stream = CancelOnDrop {
            inner: ReceiverStream::new(text_rx),
            cancellation: cancellation.clone(),
        };
        drop(stream);
        assert!(cancellation.is_cancelled(), "SSE body drop must cancel");
        drop(text_tx);
    }
}
