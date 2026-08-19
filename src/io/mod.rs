//! Explicit asynchronous SSD I/O: parallel reads, read-ahead, staging
//! buffers with broker-owned leases (spec Part VI section 43; Part VII).
//!
//! Phase 19 ("parallel I/O/read-ahead", spec §112 row 19) starts with the
//! NVMAI R9 precedent: fan independently reserved reads out across a bounded
//! thread pool instead of issuing them one at a time. This module only
//! schedules fetches that the caller has already made independent (distinct
//! destination, distinct broker reservation) - it does not itself reserve
//! anything, so invariant #5 ("every async I/O op owns/borrows a destination
//! lease that outlives completion") stays the caller's responsibility, same
//! as the serial path it replaces.

use std::thread;

use crate::error::Result;

/// Read fan-out policy. `Serial` is the Phase 18 baseline; `Parallel` is the
/// Phase 19 candidate. Both remain selectable so a caller can A/B them
/// (invariant #10 - every performance optimization needs a debug control).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadFanout {
    Serial,
    Parallel { workers: usize },
}

impl ReadFanout {
    /// NVMAI's R9 finding used a small worker pool bounded by expected
    /// concurrent SSD queue depth, not by CPU core count; TQF starts with the
    /// same order of magnitude and treats it as a benchmark-selected default,
    /// not a locked constant.
    pub const DEFAULT_WORKERS: usize = 4;

    pub fn parallel_default() -> Self {
        Self::Parallel {
            workers: Self::DEFAULT_WORKERS,
        }
    }

    /// Reads `TQF_EXPERT_IO_FANOUT` (`serial`, or `parallel:<N>`/`parallel`)
    /// so the fan-out policy can be A/B'd from outside the binary without a
    /// user-facing quality-mode maze (spec invariant #10, §114).
    pub fn from_env(var: &str) -> Option<Self> {
        let value = std::env::var(var).ok()?;
        let value = value.trim();
        if value.eq_ignore_ascii_case("serial") {
            return Some(Self::Serial);
        }
        if value.eq_ignore_ascii_case("parallel") {
            return Some(Self::parallel_default());
        }
        let workers = value
            .strip_prefix("parallel:")
            .or_else(|| value.strip_prefix("parallel="))?
            .parse::<usize>()
            .ok()?;
        Some(Self::Parallel {
            workers: workers.max(1),
        })
    }
}

/// Runs `fetch` once per item, either serially or fanned out across a bounded
/// thread pool, and returns results in the original order. `fetch` is called
/// with a borrowed item so a failing fetch touches only its own slot; because
/// each fetch reserves and populates an independent destination, one item's
/// error never leaves another item's already-completed read incomplete or
/// racily written. The first error encountered while scanning results in
/// original order is returned, which keeps failure reporting deterministic
/// regardless of which worker happened to finish first.
pub fn fetch_all<T, I, F>(fanout: ReadFanout, items: &[I], fetch: F) -> Result<Vec<T>>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> Result<T> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let workers = match fanout {
        ReadFanout::Serial => 1,
        ReadFanout::Parallel { workers } => workers.max(1).min(items.len()),
    };
    if workers <= 1 {
        return items.iter().map(&fetch).collect();
    }

    let chunk_size = items.len().div_ceil(workers);
    let mut results: Vec<Option<Result<T>>> = (0..items.len()).map(|_| None).collect();
    thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk_size)
            .enumerate()
            .map(|(worker_index, chunk)| {
                let fetch = &fetch;
                let start = worker_index * chunk_size;
                (
                    start,
                    scope.spawn(move || chunk.iter().map(fetch).collect::<Vec<_>>()),
                )
            })
            .collect();
        for (start, handle) in handles {
            let chunk_results = handle.join().expect("expert I/O worker thread panicked");
            for (offset, result) in chunk_results.into_iter().enumerate() {
                results[start + offset] = Some(result);
            }
        }
    });
    results
        .into_iter()
        .map(|slot| slot.expect("fetch_all schedules every item exactly once"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[test]
    fn serial_and_parallel_fanout_produce_identical_ordered_results() {
        let items: Vec<u32> = (0..17).collect();
        let serial = fetch_all(ReadFanout::Serial, &items, |&value| {
            Ok::<_, crate::error::TqfError>(value * 2)
        })
        .unwrap();
        let parallel = fetch_all(ReadFanout::Parallel { workers: 5 }, &items, |&value| {
            Ok::<_, crate::error::TqfError>(value * 2)
        })
        .unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(
            serial,
            items.iter().map(|value| value * 2).collect::<Vec<_>>()
        );
    }

    #[test]
    fn parallel_fanout_touches_every_item_exactly_once() {
        let items: Vec<u32> = (0..64).collect();
        let calls = AtomicUsize::new(0);
        let result = fetch_all(ReadFanout::Parallel { workers: 8 }, &items, |&value| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, crate::error::TqfError>(value)
        })
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 64);
        assert_eq!(result, items);
    }

    #[test]
    fn first_error_in_original_order_is_returned_deterministically() {
        let items: Vec<u32> = (0..10).collect();
        let error = fetch_all(ReadFanout::Parallel { workers: 4 }, &items, |&value| {
            if value == 3 || value == 7 {
                Err(crate::error::ModelError::Unsupported(format!("item {value}")).into())
            } else {
                Ok::<_, crate::error::TqfError>(value)
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("item 3"));
    }

    /// Proves the fan-out really overlaps work, by observing concurrency
    /// directly rather than inferring it from a wall-clock ratio.
    ///
    /// The previous version of this test asserted
    /// `parallel < serial * 3/4`, which is not a portable claim: on a
    /// contended shared CI runner the serial arm alone overshot its own
    /// theoretical 200ms by 1.65x, and the ratio collapsed to 1.06x even
    /// though the implementation was fine. Peak observed concurrency is
    /// the property that actually distinguishes the two paths, and it
    /// holds on any machine that can run two threads.
    #[test]
    fn parallel_fanout_actually_overlaps_fetches_while_serial_does_not() {
        /// Records how many fetches were ever in flight simultaneously.
        struct Watermark {
            in_flight: AtomicUsize,
            peak: AtomicUsize,
        }

        impl Watermark {
            fn observe<T>(&self, body: impl FnOnce() -> T) -> T {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(now, Ordering::SeqCst);
                // Long enough that genuinely concurrent fetches overlap
                // even on a slow runner, short enough to stay cheap.
                thread::sleep(Duration::from_millis(20));
                let result = body();
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                result
            }
        }

        let items: Vec<u32> = (0..8).collect();

        let serial = Watermark {
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        };
        fetch_all(ReadFanout::Serial, &items, |_| {
            serial.observe(|| Ok::<_, crate::error::TqfError>(()))
        })
        .unwrap();
        assert_eq!(
            serial.peak.load(Ordering::SeqCst),
            1,
            "the serial path must never have two fetches in flight"
        );

        let parallel = Watermark {
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        };
        fetch_all(ReadFanout::Parallel { workers: 4 }, &items, |_| {
            parallel.observe(|| Ok::<_, crate::error::TqfError>(()))
        })
        .unwrap();
        assert!(
            parallel.peak.load(Ordering::SeqCst) >= 2,
            "the parallel path never overlapped two fetches (peak {}); \
             it is scheduling serially",
            parallel.peak.load(Ordering::SeqCst)
        );
    }

    /// The Phase 19 exit gate's wall-clock claim ("M4 I/O strategy
    /// selected by measured end-to-end result").
    ///
    /// `#[ignore]`d rather than deleted: the measurement is real and worth
    /// keeping, but a shared CI runner cannot hold its premise — thread
    /// spawn latency and sleep overshoot there swamp a 200ms workload. Run
    /// it on the machine whose I/O strategy you are actually choosing,
    /// where Phase 19 measured 29.5x on the real checkpoint:
    ///
    /// ```text
    /// cargo test --release -- --ignored parallel_fanout_wall_time
    /// ```
    #[test]
    #[ignore = "wall-clock ratio is not portable to a contended CI runner; run locally"]
    fn parallel_fanout_wall_time_beats_serial() {
        let items: Vec<u32> = (0..8).collect();
        let per_item = Duration::from_millis(25);

        let serial_start = Instant::now();
        fetch_all(ReadFanout::Serial, &items, |_| {
            thread::sleep(per_item);
            Ok::<_, crate::error::TqfError>(())
        })
        .unwrap();
        let serial_elapsed = serial_start.elapsed();

        let parallel_start = Instant::now();
        fetch_all(ReadFanout::Parallel { workers: 4 }, &items, |_| {
            thread::sleep(per_item);
            Ok::<_, crate::error::TqfError>(())
        })
        .unwrap();
        let parallel_elapsed = parallel_start.elapsed();

        println!(
            "serial {serial_elapsed:?} parallel {parallel_elapsed:?} \
             ({:.2}x)",
            serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64()
        );
        assert!(
            parallel_elapsed < serial_elapsed * 3 / 4,
            "expected parallel fan-out ({parallel_elapsed:?}) to clearly beat serial ({serial_elapsed:?})"
        );
    }

    #[test]
    fn from_env_parses_serial_and_parallel_forms() {
        let var = "TQF_TEST_IO_FANOUT_PARSE";
        std::env::set_var(var, "serial");
        assert_eq!(ReadFanout::from_env(var), Some(ReadFanout::Serial));
        std::env::set_var(var, "parallel");
        assert_eq!(
            ReadFanout::from_env(var),
            Some(ReadFanout::parallel_default())
        );
        std::env::set_var(var, "parallel:6");
        assert_eq!(
            ReadFanout::from_env(var),
            Some(ReadFanout::Parallel { workers: 6 })
        );
        std::env::remove_var(var);
        assert_eq!(ReadFanout::from_env(var), None);
    }
}
