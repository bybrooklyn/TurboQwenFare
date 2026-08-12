//! Event/timing primitives (spec §282 phase 10 "event timing"; spec §135
//! "GPU/I/O overlap contract" names `MTLEvent`/`MTLSharedEvent`-style
//! signal/wait as the cross-command-buffer dependency mechanism a future
//! decode loop needs).
//!
//! `GpuStopwatch` measures wall-clock elapsed time bracketing a
//! `commit()` + `wait_until_completed()` round trip — sufficient for the
//! synthetic bandwidth/GEMV harness's throughput numbers. It is
//! deliberately *not* GPU-side `kernelStartTime`/`kernelEndTime`
//! sampling: the `metal` crate version pinned here does not expose those
//! accessors, and CPU-side wall-clock timing around a synchronous wait is
//! an honest (if less precise) REFERENCE BASELINE rather than a fabricated
//! GPU-only number (spec §114: "Do not measure only GPU kernel time and
//! call it decode time" cuts the other way too — a benchmark that can only
//! measure wall time should say so, not pretend otherwise).

use std::time::{Duration, Instant};

use metal_sys::SharedEvent;

use super::context::MetalContext;

pub struct GpuStopwatch {
    started_at: Instant,
}

impl GpuStopwatch {
    pub fn start() -> Self {
        Self {
            started_at: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

/// Wraps an `MTLSharedEvent` for cross-command-buffer signal/wait
/// (spec §135). CPU-side waiting here is a polling loop on
/// `signaled_value()` rather than the block-based `notify` API — a
/// correct, if coarser-grained, REFERENCE BASELINE; a production
/// implementation should switch to `notify` to avoid burning a CPU core
/// while waiting.
pub struct EventFence {
    event: SharedEvent,
}

impl EventFence {
    pub fn new(ctx: &MetalContext) -> Self {
        Self {
            event: ctx.device().new_shared_event(),
        }
    }

    pub fn signaled_value(&self) -> u64 {
        self.event.signaled_value()
    }

    pub fn set_signaled_value(&self, value: u64) {
        self.event.set_signaled_value(value);
    }

    pub fn metal_event(&self) -> &SharedEvent {
        &self.event
    }

    /// Busy-polls until `signaled_value() >= target` or `timeout` elapses.
    /// Returns `true` if the target was reached.
    pub fn wait_until_signaled(&self, target: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while self.signaled_value() < target {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwatch_reports_nonzero_elapsed_after_work() {
        let sw = GpuStopwatch::start();
        std::thread::sleep(Duration::from_millis(1));
        assert!(sw.elapsed() >= Duration::from_millis(1));
    }

    #[test]
    fn event_fence_wait_succeeds_once_already_signaled() {
        // metal-rs's Objective-C object wrappers are not `Send`/`Sync`, so
        // this exercises the same signal-then-wait contract a real
        // cross-command-buffer dependency would use without needing an
        // actual second thread to drive it.
        let Ok(ctx) = MetalContext::init() else {
            eprintln!("skipping Metal test: no device available in this environment");
            return;
        };
        let fence = EventFence::new(&ctx);
        assert_eq!(fence.signaled_value(), 0);
        fence.set_signaled_value(1);
        assert!(fence.wait_until_signaled(1, Duration::from_secs(1)));
    }

    #[test]
    fn event_fence_wait_times_out_when_never_signaled() {
        let Ok(ctx) = MetalContext::init() else {
            eprintln!("skipping Metal test: no device available in this environment");
            return;
        };
        let fence = EventFence::new(&ctx);
        assert!(!fence.wait_until_signaled(1, Duration::from_millis(20)));
    }
}
