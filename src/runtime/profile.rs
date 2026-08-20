//! Where a decode run's wall clock actually went.
//!
//! Phase 25 measured 78% of decode as demand expert I/O against a 15 tok/s
//! floor (spec §4) it misses by an order of magnitude. Every number needed
//! to see that already existed — `DecodeTimings` per step, and the expert
//! cache's own hit/miss/byte/stall counters — but nothing added them up,
//! so answering "where did the time go" meant re-deriving it by hand from
//! a qualification document each time.
//!
//! This is deliberately an accumulator over measurements taken elsewhere,
//! not a new measurement path. A profiler that samples differently from
//! the thing being optimized is how a 10x win on a stage that was 3% of
//! the total gets celebrated.
//!
//! Two rules it follows, both from the contributor list in spec §114:
//! it never reports GPU kernel time as decode time (the stage timings
//! are wall clock around the whole step), and it reports the model's
//! I/O stall alongside its compute, because a compute-only breakdown of
//! an out-of-core runtime describes a machine nobody is running.

use std::time::Duration;

use crate::experts::ExpertCacheStats;
use crate::ids::{Bytes, LayerId};
use crate::runtime::decode::DecodeTimings;

/// Accumulates per-step timings and the expert cache's own counters over
/// a run.
///
/// Expert statistics are recorded as a *delta* against a baseline taken
/// when profiling started, so a profile covering ten tokens of a long
/// session reports those ten tokens rather than the session's lifetime
/// totals.
#[derive(Debug, Clone, Default)]
pub struct DecodeProfile {
    steps: u64,
    embedding: Duration,
    per_layer: Vec<(LayerId, Duration)>,
    final_norm: Duration,
    lm_head: Duration,
    sampling: Duration,
    total: Duration,
    baseline: Option<ExpertCacheStats>,
    latest: Option<ExpertCacheStats>,
}

impl DecodeProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the expert-cache counters as they stood before the first
    /// profiled step. Calling it more than once keeps the earliest.
    pub fn set_baseline(&mut self, stats: ExpertCacheStats) {
        if self.baseline.is_none() {
            self.baseline = Some(stats);
        }
    }

    /// Folds in one decode step.
    ///
    /// `total` is measured around the whole step by the caller rather
    /// than summed from the stages: the difference between the two is
    /// exactly the work no stage timer covers, and hiding it would make
    /// the breakdown add to 100% while describing less than all of it.
    pub fn record_step(&mut self, timings: &DecodeTimings, total: Duration) {
        self.steps += 1;
        self.embedding += timings.embedding;
        self.final_norm += timings.final_norm;
        self.lm_head += timings.lm_head;
        self.sampling += timings.sampling;
        self.total += total;

        for (layer, elapsed) in &timings.layers {
            match self.per_layer.iter_mut().find(|(id, _)| id == layer) {
                Some((_, accumulated)) => *accumulated += *elapsed,
                None => self.per_layer.push((*layer, *elapsed)),
            }
        }
    }

    /// Records the expert-cache counters after a step. The last one wins,
    /// so the report covers baseline-to-here.
    pub fn record_expert_stats(&mut self, stats: ExpertCacheStats) {
        self.latest = Some(stats);
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn is_empty(&self) -> bool {
        self.steps == 0
    }

    pub fn report(&self) -> DecodeProfileReport {
        let layers: Duration = self.per_layer.iter().map(|(_, d)| *d).sum();
        // The stage timers nest: `layers` already contains the expert
        // work, so they are not added together. `unaccounted` is what the
        // step took beyond every stage timer, which is the honest name
        // for it.
        let staged = self.embedding + layers + self.final_norm + self.lm_head + self.sampling;
        let unaccounted = self.total.saturating_sub(staged);

        let experts = match (&self.baseline, &self.latest) {
            (Some(base), Some(now)) => Some(ExpertActivity {
                hits: now.hits.saturating_sub(base.hits),
                misses: now.misses.saturating_sub(base.misses),
                evictions: now.evictions.saturating_sub(base.evictions),
                miss_bytes: Bytes(now.raw_miss_bytes.0.saturating_sub(base.raw_miss_bytes.0)),
                demand_io: Duration::from_nanos(
                    now.demand_io_nanos.saturating_sub(base.demand_io_nanos) as u64,
                ),
                resident_bytes: now.resident_bytes,
            }),
            // One-sided is not half a delta: without a baseline the
            // numbers would silently be session totals attributed to
            // this run.
            _ => None,
        };

        DecodeProfileReport {
            steps: self.steps,
            total: self.total,
            embedding: self.embedding,
            layers,
            final_norm: self.final_norm,
            lm_head: self.lm_head,
            sampling: self.sampling,
            unaccounted,
            slowest_layers: self.slowest_layers(),
            experts,
        }
    }

    fn slowest_layers(&self) -> Vec<(LayerId, Duration)> {
        let mut sorted = self.per_layer.clone();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.truncate(5);
        sorted
    }
}

/// Expert-cache activity over the profiled window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertActivity {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub miss_bytes: Bytes,
    pub demand_io: Duration,
    pub resident_bytes: Bytes,
}

impl ExpertActivity {
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        (total > 0).then(|| self.hits as f64 / total as f64)
    }
}

#[derive(Debug, Clone)]
pub struct DecodeProfileReport {
    pub steps: u64,
    pub total: Duration,
    pub embedding: Duration,
    pub layers: Duration,
    pub final_norm: Duration,
    pub lm_head: Duration,
    pub sampling: Duration,
    /// Step time no stage timer covered.
    pub unaccounted: Duration,
    pub slowest_layers: Vec<(LayerId, Duration)>,
    /// `None` when the run had no expert cache, or when no baseline was
    /// taken — never a fabricated zero.
    pub experts: Option<ExpertActivity>,
}

impl DecodeProfileReport {
    /// Seconds per token, the denominator of the §4 floor.
    pub fn seconds_per_token(&self) -> Option<f64> {
        (self.steps > 0).then(|| self.total.as_secs_f64() / self.steps as f64)
    }

    pub fn tokens_per_second(&self) -> Option<f64> {
        self.seconds_per_token()
            .filter(|s| *s > 0.0)
            .map(|s| 1.0 / s)
    }

    /// The share of wall clock spent stalled on expert reads.
    ///
    /// This is the number Phase 25 put at 78%, and the one that decides
    /// whether the next optimization should be compute or I/O.
    pub fn demand_io_fraction(&self) -> Option<f64> {
        let experts = self.experts.as_ref()?;
        (self.total.as_secs_f64() > 0.0)
            .then(|| experts.demand_io.as_secs_f64() / self.total.as_secs_f64())
    }

    fn share(&self, part: Duration) -> f64 {
        if self.total.as_secs_f64() <= 0.0 {
            return 0.0;
        }
        100.0 * part.as_secs_f64() / self.total.as_secs_f64()
    }

    /// A human-readable breakdown.
    ///
    /// Layer time is marked as containing the expert work rather than
    /// listed beside it, because the two overlap — presenting nested
    /// timers as siblings is how a breakdown ends up summing past 100%.
    pub fn render(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        let _ = writeln!(out, "decode profile: {} tokens", self.steps);
        match (self.seconds_per_token(), self.tokens_per_second()) {
            (Some(spt), Some(tps)) => {
                let _ = writeln!(
                    out,
                    "  {:.3} s/token  ({:.2} tok/s; the spec §4 floor is 15)",
                    spt, tps
                );
            }
            _ => {
                let _ = writeln!(out, "  no completed steps");
            }
        }

        let _ = writeln!(out, "\nstage totals (layers contain the expert work):");
        for (label, part) in [
            ("embedding", self.embedding),
            ("layers", self.layers),
            ("final norm", self.final_norm),
            ("lm head", self.lm_head),
            ("sampling", self.sampling),
            ("unaccounted", self.unaccounted),
        ] {
            let _ = writeln!(
                out,
                "  {label:<12} {:>10.3} s  {:>5.1}%",
                part.as_secs_f64(),
                self.share(part)
            );
        }

        if !self.slowest_layers.is_empty() {
            let _ = writeln!(out, "\nslowest layers:");
            for (layer, elapsed) in &self.slowest_layers {
                let _ = writeln!(
                    out,
                    "  layer {:<3} {:>10.3} s  {:>5.1}%",
                    layer.0,
                    elapsed.as_secs_f64(),
                    self.share(*elapsed)
                );
            }
        }

        match &self.experts {
            Some(experts) => {
                let _ = writeln!(out, "\nexpert cache:");
                let _ = writeln!(
                    out,
                    "  {} hits, {} misses{}, {} evictions",
                    experts.hits,
                    experts.misses,
                    experts
                        .hit_rate()
                        .map(|r| format!(" ({:.1}% hit rate)", r * 100.0))
                        .unwrap_or_default(),
                    experts.evictions
                );
                let _ = writeln!(
                    out,
                    "  {:.1} MiB demanded from disk in {:.3} s ({:.1}% of wall clock)",
                    experts.miss_bytes.0 as f64 / (1024.0 * 1024.0),
                    experts.demand_io.as_secs_f64(),
                    self.demand_io_fraction().unwrap_or(0.0) * 100.0
                );
                if self.steps > 0 {
                    let _ = writeln!(
                        out,
                        "  {:.1} MiB/token",
                        experts.miss_bytes.0 as f64 / (1024.0 * 1024.0) / self.steps as f64
                    );
                }
                if let Some(fraction) = self.demand_io_fraction() {
                    let _ = writeln!(
                        out,
                        "\n{}",
                        if fraction > 0.5 {
                            "I/O bound: this run spends most of its time waiting on expert \
                             reads. Faster kernels cannot help until that shrinks — the \
                             levers are cache capacity, container placement, and prefetch."
                        } else {
                            "Compute bound: expert reads are not the limit in this run, so \
                             the stage breakdown above is where to look."
                        }
                    );
                }
            }
            None => {
                let _ = writeln!(
                    out,
                    "\nexpert cache: not reported (no cache in this runtime, or no baseline \
                     was taken before the first step)"
                );
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timings(embedding_ms: u64, layer_ms: &[u64], head_ms: u64) -> DecodeTimings {
        DecodeTimings {
            embedding: Duration::from_millis(embedding_ms),
            layers: layer_ms
                .iter()
                .enumerate()
                .map(|(i, ms)| (LayerId(i as u8), Duration::from_millis(*ms)))
                .collect(),
            final_norm: Duration::from_millis(1),
            lm_head: Duration::from_millis(head_ms),
            sampling: Duration::from_millis(1),
        }
    }

    fn stats(hits: u64, misses: u64, miss_bytes: u64, io_nanos: u128) -> ExpertCacheStats {
        ExpertCacheStats {
            hits,
            misses,
            raw_miss_bytes: Bytes(miss_bytes),
            demand_io_nanos: io_nanos,
            ..ExpertCacheStats::default()
        }
    }

    #[test]
    fn per_layer_time_accumulates_across_steps_rather_than_being_overwritten() {
        let mut profile = DecodeProfile::new();
        for _ in 0..4 {
            profile.record_step(&timings(1, &[10, 20, 30], 5), Duration::from_millis(100));
        }
        let report = profile.report();

        assert_eq!(report.steps, 4);
        assert_eq!(report.layers, Duration::from_millis(4 * 60));
        // Sorted by cost, so the first entry is the layer worth looking at.
        assert_eq!(report.slowest_layers[0].0, LayerId(2));
        assert_eq!(report.slowest_layers[0].1, Duration::from_millis(120));
    }

    /// The stage timers nest — layer time already contains the expert
    /// work — so the report must not present them as siblings that sum
    /// past 100%. What it does report is the gap between the whole step
    /// and every stage timer, which is real and otherwise invisible.
    #[test]
    fn time_no_stage_timer_covered_is_reported_rather_than_absorbed() {
        let mut profile = DecodeProfile::new();
        // Stages total 34 ms (1 embedding + 30 layers + 1 final norm
        // + 1 lm head + 1 sampling); the step really took 100 ms.
        profile.record_step(&timings(1, &[10, 20], 1), Duration::from_millis(100));
        let report = profile.report();

        assert_eq!(report.unaccounted, Duration::from_millis(66));
        let rendered = report.render();
        assert!(rendered.contains("unaccounted"), "{rendered}");
    }

    /// Expert counters are lifetime totals on the cache, so a profile
    /// covering part of a session has to subtract where it started.
    /// Reporting the raw totals would credit this run with every fetch
    /// the process ever made.
    #[test]
    fn expert_counters_are_a_delta_against_the_baseline() {
        let mut profile = DecodeProfile::new();
        profile.set_baseline(stats(100, 50, 5_000_000, 2_000_000_000));
        profile.record_step(&timings(1, &[10], 1), Duration::from_millis(100));
        profile.record_expert_stats(stats(140, 60, 9_000_000, 3_500_000_000));

        let experts = profile.report().experts.expect("a baseline was taken");
        assert_eq!(experts.hits, 40);
        assert_eq!(experts.misses, 10);
        assert_eq!(experts.miss_bytes, Bytes(4_000_000));
        assert_eq!(experts.demand_io, Duration::from_millis(1500));
        assert!((experts.hit_rate().unwrap() - 0.8).abs() < 1e-9);
    }

    /// Without a baseline the delta is unknowable, and a zero would read
    /// as "this run fetched nothing" — the opposite of the truth.
    #[test]
    fn a_missing_baseline_reports_nothing_rather_than_zero() {
        let mut profile = DecodeProfile::new();
        profile.record_step(&timings(1, &[10], 1), Duration::from_millis(100));
        profile.record_expert_stats(stats(140, 60, 9_000_000, 3_500_000_000));

        let report = profile.report();
        assert!(report.experts.is_none());
        assert!(report.demand_io_fraction().is_none());
        let rendered = report.render();
        assert!(rendered.contains("no baseline was taken"), "{rendered}");
    }

    /// The headline the whole instrument exists for: which side of the
    /// 15 tok/s problem this run is on. Phase 25 measured 78% I/O.
    #[test]
    fn the_report_names_whether_the_run_is_io_or_compute_bound() {
        let mut io_bound = DecodeProfile::new();
        io_bound.set_baseline(stats(0, 0, 0, 0));
        io_bound.record_step(&timings(1, &[900], 1), Duration::from_secs(1));
        io_bound.record_expert_stats(stats(2, 8, 200 * 1024 * 1024, 780_000_000));

        let report = io_bound.report();
        assert!((report.demand_io_fraction().unwrap() - 0.78).abs() < 1e-6);
        assert_eq!(report.tokens_per_second().unwrap(), 1.0);
        let rendered = report.render();
        assert!(rendered.contains("I/O bound"), "{rendered}");
        assert!(rendered.contains("the spec §4 floor is 15"), "{rendered}");

        let mut compute_bound = DecodeProfile::new();
        compute_bound.set_baseline(stats(0, 0, 0, 0));
        compute_bound.record_step(&timings(1, &[900], 1), Duration::from_secs(1));
        compute_bound.record_expert_stats(stats(10, 0, 0, 50_000_000));
        assert!(compute_bound.report().render().contains("Compute bound"));
    }

    #[test]
    fn an_empty_profile_renders_without_dividing_by_zero() {
        let report = DecodeProfile::new().report();
        assert!(report.seconds_per_token().is_none());
        assert!(report.tokens_per_second().is_none());
        let rendered = report.render();
        assert!(rendered.contains("no completed steps"), "{rendered}");
    }
}
