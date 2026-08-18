//! Phase 23 predictive prefetch (spec §295, §45): a statistical predictor
//! over exact route transitions and co-routing, used purely as an I/O
//! scheduling hint. Prediction never alters expert IDs or router weights
//! (spec invariant #7): the worst possible outcome is wasted SSD traffic.
//!
//! The offline replay below is the Phase 23 measurement instrument: it
//! replays the same exact-router trace format as the Phase 21/22 replays
//! through a byte-budgeted whole-expert cache with a speculative prefetch
//! queue, reporting precision, recall, timeliness, and wasted bytes -
//! the four metrics spec §295 names - at several prefetch depths, so the
//! live prefetch default is chosen from measured data rather than
//! intuition.

use serde::Serialize;

use crate::error::{ModelError, Result};
use crate::experts::policy::{ExpertRouteTrace, ReplayConfig};
use crate::ids::{ExpertId, LayerId};

/// Default decay per route event for transition scores. Scores are
/// multiplied by this on every event that does not touch the key. The
/// half-life is ~23 route events: route patterns drift with content, so
/// three stale observations must not outrank one fresh observation after
/// ~50 events of unrelated traffic.
const TRANSITION_DECAY: f64 = 0.97;

#[derive(Debug, Clone, Copy, Default)]
struct Score {
    value: f64,
    last_event: u64,
}

/// Route-transition/co-routing table: for each (layer, expert) observed in
/// a route, how strongly each expert in the *next* route event is expected
/// to appear. One table per consecutive layer pair, so layer 39's table
/// covers the token-boundary transition into layer 0.
#[derive(Debug, Clone)]
pub struct TransitionPredictor {
    /// Keyed by (layer, expert) -> (target_layer, target_expert) -> score.
    scores: std::collections::HashMap<
        (LayerId, ExpertId),
        std::collections::HashMap<(LayerId, ExpertId), Score>,
    >,
    events: u64,
}

impl Default for TransitionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionPredictor {
    pub fn new() -> Self {
        Self {
            scores: std::collections::HashMap::new(),
            events: 0,
        }
    }

    /// Feeds one observed route event and the following one (which may be
    /// the next layer of the same token or layer 0 of the next token).
    pub fn observe(
        &mut self,
        from_layer: LayerId,
        from_route: &[ExpertId],
        to_layer: LayerId,
        to_route: &[ExpertId],
    ) {
        self.events = self.events.saturating_add(1);
        for &from in from_route {
            let row = self.scores.entry((from_layer, from)).or_default();
            for &to in to_route {
                let entry = row.entry((to_layer, to)).or_default();
                let age = self.events.saturating_sub(entry.last_event);
                entry.value = entry.value * TRANSITION_DECAY.powi(age as i32) + 1.0;
                entry.last_event = self.events;
            }
        }
    }

    /// Predicts which experts the next route event (at `to_layer`) will
    /// demand, given the route just observed at `from_layer`. Returns the
    /// top candidates by transition score, excluding experts the caller
    /// says are already resident, up to `budget`.
    pub fn predict(
        &self,
        from_layer: LayerId,
        from_route: &[ExpertId],
        to_layer: LayerId,
        budget: usize,
        is_resident: impl Fn(ExpertId) -> bool,
    ) -> Vec<ExpertId> {
        let mut ranked: Vec<(ExpertId, f64)> = Vec::new();
        for &from in from_route {
            let Some(row) = self.scores.get(&(from_layer, from)) else {
                continue;
            };
            for (&(target_layer, target), score) in row {
                if target_layer != to_layer || is_resident(target) {
                    continue;
                }
                let age = self.events.saturating_sub(score.last_event);
                let decayed = score.value * TRANSITION_DECAY.powi(age as i32);
                match ranked.iter_mut().find(|(expert, _)| *expert == target) {
                    Some((_, total)) => *total += decayed,
                    None => ranked.push((target, decayed)),
                }
            }
        }
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(budget);
        ranked.into_iter().map(|(expert, _)| expert).collect()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PrefetchReplayResult {
    pub route_events: u64,
    /// Demanded experts that were already resident (demand-fill or prior
    /// prefetch) - the no-stall demands.
    pub hits: u64,
    /// Bytes fetched by demand misses (the SSD stall proxy).
    pub demand_miss_bytes: u64,
    /// Experts prefetched speculatively.
    pub prefetched: u64,
    /// Prefetched experts that arrived in time and were demanded.
    pub prefetch_hits: u64,
    /// Bytes of prefetched experts evicted before any demand (wasted).
    pub prefetch_wasted_bytes: u64,
    /// |predicted ∩ demanded| / |predicted| over all predictions.
    pub precision_numerator: u64,
    pub precision_denominator: u64,
    /// |predicted ∩ demanded| / |demanded| over all demands.
    pub recall_numerator: u64,
    pub recall_denominator: u64,
}

impl PrefetchReplayResult {
    pub fn precision(&self) -> f64 {
        if self.precision_denominator == 0 {
            1.0
        } else {
            self.precision_numerator as f64 / self.precision_denominator as f64
        }
    }

    pub fn recall(&self) -> f64 {
        if self.recall_denominator == 0 {
            0.0
        } else {
            self.recall_numerator as f64 / self.recall_denominator as f64
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    layer: LayerId,
    expert: ExpertId,
    bytes: u64,
    last_use: u64,
    probation: bool,
}

struct PrefetchReplayCache {
    config: ReplayConfig,
    clock: u64,
    resident: u64,
    entries: Vec<Entry>,
    result: PrefetchReplayResult,
}

impl PrefetchReplayCache {
    fn new(config: ReplayConfig) -> Result<Self> {
        if config.capacity.0 == 0 {
            return Err(ModelError::Unsupported(
                "prefetch replay requires a nonzero byte capacity".to_string(),
            )
            .into());
        }
        Ok(Self {
            config,
            clock: 0,
            resident: 0,
            entries: Vec::new(),
            result: PrefetchReplayResult::default(),
        })
    }

    /// Prefetched entries land here; they occupy capacity and are counted
    /// as wasted if evicted before any demand.
    fn deliver(&mut self, layer: LayerId, expert: ExpertId, bytes: u64) {
        if self
            .entries
            .iter()
            .any(|entry| entry.layer == layer && entry.expert == expert)
        {
            return;
        }
        self.evict_for(bytes, &[]);
        let entry = Entry {
            layer,
            expert,
            bytes,
            last_use: self.clock,
            probation: true,
        };
        self.resident = self.resident.saturating_add(bytes);
        self.entries.push(entry);
        self.result.prefetched = self.result.prefetched.saturating_add(1);
    }

    fn evict_for(&mut self, required: u64, pinned: &[(LayerId, ExpertId)]) {
        while self.resident.saturating_add(required) > self.config.capacity.0 {
            let index = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| !pinned.contains(&(entry.layer, entry.expert)))
                .min_by(|(_, left), (_, right)| {
                    // Probation entries evict before any demand entry,
                    // then LRU order.
                    left.probation
                        .cmp(&right.probation)
                        .then_with(|| left.last_use.cmp(&right.last_use))
                        .then_with(|| left.layer.cmp(&right.layer))
                        .then_with(|| left.expert.cmp(&right.expert))
                });
            let Some((index, _)) = index else {
                return;
            };
            let entry = self.entries.swap_remove(index);
            self.resident = self.resident.saturating_sub(entry.bytes);
            if entry.probation && entry.last_use == self.clock {
                // Prefetched this event but already being evicted - it
                // arrived too late or never fit; count the bytes wasted
                // only when it was never demanded.
                self.result.prefetch_wasted_bytes = self
                    .result
                    .prefetch_wasted_bytes
                    .saturating_add(entry.bytes);
            } else if entry.probation {
                self.result.prefetch_wasted_bytes = self
                    .result
                    .prefetch_wasted_bytes
                    .saturating_add(entry.bytes);
            }
        }
    }

    /// One demand route event. `predicted_last_event` is the prefetch set
    /// issued at the *previous* event (already delivered), so hits on
    /// those probation entries count as prefetch hits.
    ///
    /// Metric semantics: precision = |delivered predictions that served a
    /// demand| / |delivered predictions| (the useful-prefetch rate);
    /// recall = |demands that had been predicted| / |demands| (prediction
    /// coverage, independent of whether delivery survived eviction).
    fn route(
        &mut self,
        layer: LayerId,
        experts: [ExpertId; 8],
        bytes_per_expert: u64,
        predicted_last_event: &[ExpertId],
    ) {
        self.clock = self.clock.saturating_add(1);
        self.result.route_events = self.result.route_events.saturating_add(1);
        let pinned: Vec<(LayerId, ExpertId)> =
            experts.iter().map(|expert| (layer, *expert)).collect();
        self.result.precision_denominator = self
            .result
            .precision_denominator
            .saturating_add(predicted_last_event.len() as u64);
        let mut missing = 0u64;
        for &expert in &experts {
            self.result.recall_denominator = self.result.recall_denominator.saturating_add(1);
            let was_predicted = predicted_last_event.contains(&expert);
            if was_predicted {
                self.result.recall_numerator = self.result.recall_numerator.saturating_add(1);
            }
            if let Some(entry) = self
                .entries
                .iter_mut()
                .find(|entry| entry.layer == layer && entry.expert == expert)
            {
                if entry.probation {
                    if was_predicted {
                        self.result.prefetch_hits = self.result.prefetch_hits.saturating_add(1);
                        self.result.precision_numerator =
                            self.result.precision_numerator.saturating_add(1);
                    }
                    entry.probation = false;
                }
                entry.last_use = self.clock;
                self.result.hits = self.result.hits.saturating_add(1);
            } else {
                missing = missing.saturating_add(bytes_per_expert);
            }
        }
        self.evict_for(missing, &pinned);
        for &expert in &experts {
            if self
                .entries
                .iter()
                .any(|entry| entry.layer == layer && entry.expert == expert)
            {
                continue;
            }
            self.entries.push(Entry {
                layer,
                expert,
                bytes: bytes_per_expert,
                last_use: self.clock,
                probation: false,
            });
            self.resident = self.resident.saturating_add(bytes_per_expert);
            self.result.demand_miss_bytes = self
                .result
                .demand_miss_bytes
                .saturating_add(bytes_per_expert);
        }
    }
}

/// Replays the trace with a transition predictor and one-event-ahead
/// prefetch: predictions issued at event t are delivered at the start of
/// event t+1 (one route event of lead time - the compute window the live
/// loop can overlap an SSD read into), and only then can they serve a
/// demand.
pub fn replay_prefetch(
    trace: &ExpertRouteTrace,
    config: ReplayConfig,
    depth: usize,
    bytes_per_expert: u64,
) -> Result<PrefetchReplayResult> {
    if trace.schema_version != 1 {
        return Err(ModelError::Unsupported(format!(
            "unsupported expert route-trace schema {}",
            trace.schema_version
        ))
        .into());
    }
    let mut cache = PrefetchReplayCache::new(config)?;
    let mut predictor = TransitionPredictor::new();
    let mut previous: Option<(LayerId, Vec<ExpertId>)> = None;
    let mut pending: Vec<(LayerId, ExpertId)> = Vec::new();

    for step in &trace.steps {
        for (index, layer) in step.layers.iter().enumerate() {
            let experts = layer.expert_ids.map(ExpertId);
            let layer_id = LayerId(layer.layer);

            // Deliver last event's predictions (one event of lead time).
            for &(target_layer, expert) in &pending {
                cache.deliver(target_layer, expert, bytes_per_expert);
            }
            let delivered_experts: Vec<ExpertId> =
                pending.iter().map(|&(_, expert)| expert).collect();
            cache.route(layer_id, experts, bytes_per_expert, &delivered_experts);

            if let Some((from_layer, from_route)) = &previous {
                predictor.observe(*from_layer, from_route, layer_id, &experts);
            }
            previous = Some((layer_id, experts.to_vec()));

            // Predict the layer after this one: the next layer of the
            // same token, or layer 0 of the next token at a step
            // boundary.
            let to_layer = step
                .layers
                .get(index + 1)
                .map(|next| LayerId(next.layer))
                .unwrap_or(LayerId(0));
            pending = if depth == 0 {
                Vec::new()
            } else {
                let resident = |expert: ExpertId| {
                    cache
                        .entries
                        .iter()
                        .any(|entry| entry.layer == to_layer && entry.expert == expert)
                };
                predictor
                    .predict(layer_id, &experts, to_layer, depth, resident)
                    .into_iter()
                    .map(|expert| (to_layer, expert))
                    .collect()
            };
        }
    }
    Ok(cache.result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::policy::{CachePolicyKind, ExpertRouteLayer, ExpertRouteStep};
    use crate::ids::Bytes;

    fn layer(layer: u8, experts: [u16; 8]) -> ExpertRouteLayer {
        ExpertRouteLayer {
            layer,
            expert_ids: experts,
            weights: [0.125; 8],
        }
    }

    fn trace(layers: Vec<ExpertRouteLayer>) -> ExpertRouteTrace {
        ExpertRouteTrace {
            schema_version: 1,
            fixture_id: "prefetch-fixture".to_string(),
            model_source_sha256: "00".repeat(32),
            steps: layers
                .into_iter()
                .enumerate()
                .map(|(index, layer)| ExpertRouteStep {
                    decode_step: index + 1,
                    input_token: index as u32,
                    output_token: index as u32 + 1,
                    layers: vec![layer],
                })
                .collect(),
        }
    }

    fn lru(bytes: u64) -> ReplayConfig {
        ReplayConfig {
            capacity: Bytes(bytes),
            policy: CachePolicyKind::Lru,
            half_life_events: 1,
        }
    }

    const EXPERT: u64 = 1_769_472;

    #[test]
    fn predictor_learns_an_exact_transition_and_scores_confidence() {
        let mut predictor = TransitionPredictor::new();
        let first: Vec<ExpertId> = [0u16, 1, 2, 3, 4, 5, 6, 7].map(ExpertId).to_vec();
        let second: Vec<ExpertId> = [10u16, 11, 12, 13, 14, 15, 16, 17].map(ExpertId).to_vec();
        for _ in 0..4 {
            predictor.observe(LayerId(0), &first, LayerId(1), &second);
        }
        let predicted = predictor.predict(LayerId(0), &first, LayerId(1), 8, |_| false);
        // All four repeats: the second set should score highest; ties
        // within it break toward smaller IDs, so it is the top-8 exactly.
        assert_eq!(predicted, second);
    }

    #[test]
    fn predictor_excludes_resident_experts() {
        let mut predictor = TransitionPredictor::new();
        let first: Vec<ExpertId> = [0u16, 1, 2, 3, 4, 5, 6, 7].map(ExpertId).to_vec();
        let second: Vec<ExpertId> = [10u16, 11, 12, 13, 14, 15, 16, 17].map(ExpertId).to_vec();
        predictor.observe(LayerId(0), &first, LayerId(1), &second);
        // Only five of eight candidates pass the resident filter; all of
        // them must appear, in ascending-ID tie order.
        let predicted =
            predictor.predict(LayerId(0), &first, LayerId(1), 8, |expert| expert.0 < 13);
        let expected: Vec<ExpertId> = [13u16, 14, 15, 16, 17].map(ExpertId).to_vec();
        assert_eq!(predicted, expected);
    }

    #[test]
    fn prefetch_replay_hits_arriving_predictions_and_logs_waste() {
        // Five events alternating R1/R2 at a capacity that holds exactly
        // one route. The transition table needs one occurrence to learn a
        // transition (prediction excludes already-resident experts, so
        // the self-loop R2->R2 yields nothing while R2 is resident). The
        // learned R1->R2 transition fires at event 4's prediction: its
        // prefetch is delivered at event 5 (one event of lead time) and
        // converts the whole demand into prefetch hits.
        let r1 = [0, 1, 2, 3, 4, 5, 6, 7];
        let r2 = [8, 9, 10, 11, 12, 13, 14, 15];
        let trace = trace(vec![
            layer(0, r1),
            layer(0, r2),
            layer(0, r2),
            layer(0, r1),
            layer(0, r2),
        ]);
        let result = replay_prefetch(&trace, lru(EXPERT * 8), 8, EXPERT).unwrap();
        assert_eq!(result.route_events, 5);
        assert_eq!(
            result.prefetched, 8,
            "predictor issues after learning the transition"
        );
        assert_eq!(
            result.prefetch_hits, 8,
            "one-event-ahead prefetch serves the next demand"
        );
        assert_eq!(
            result.prefetch_wasted_bytes, 0,
            "exact predictions waste nothing"
        );
        assert_eq!(result.precision(), 1.0);
        assert!(result.recall() > 0.0);
        // Three cold events fetch 24 experts; the prefetched fifth event
        // costs nothing on the demand path.
        assert_eq!(result.demand_miss_bytes, EXPERT * 24);
    }

    #[test]
    fn depth_zero_disables_speculation_entirely() {
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
        ]);
        let result = replay_prefetch(&trace, lru(EXPERT * 16), 0, EXPERT).unwrap();
        assert_eq!(result.prefetched, 0);
        assert_eq!(result.prefetch_hits, 0);
        assert_eq!(result.prefetch_wasted_bytes, 0);
    }

    #[test]
    fn decay_lets_recent_transitions_outrank_stale_ones() {
        let mut predictor = TransitionPredictor::new();
        let from: Vec<ExpertId> = [0u16, 1, 2, 3, 4, 5, 6, 7].map(ExpertId).to_vec();
        let stale: Vec<ExpertId> = [20u16, 21, 22, 23, 24, 25, 26, 27].map(ExpertId).to_vec();
        let fresh: Vec<ExpertId> = [30u16, 31, 32, 33, 34, 35, 36, 37].map(ExpertId).to_vec();
        for _ in 0..3 {
            predictor.observe(LayerId(0), &from, LayerId(1), &stale);
        }
        // ~50 intervening events drown the stale scores.
        for index in 0..50 {
            let noise: Vec<ExpertId> = (0..8)
                .map(|offset| ExpertId(40 + index * 8 + offset as u16))
                .collect();
            let next: Vec<ExpertId> = (0..8)
                .map(|offset| ExpertId(40 + (index + 1) * 8 + offset as u16))
                .collect();
            predictor.observe(LayerId(5), &noise, LayerId(6), &next);
        }
        predictor.observe(LayerId(0), &from, LayerId(1), &fresh);
        let predicted = predictor.predict(LayerId(0), &from, LayerId(1), 8, |_| false);
        assert_eq!(
            predicted, fresh,
            "fresh transition must outrank stale after decay"
        );
    }

    #[test]
    #[ignore = "requires a qualification route-trace artifact"]
    fn qualification_trace_measures_prefetch_precision_recall_and_waste() {
        let path = std::env::var("TQF_QUALIFICATION_ROUTE_TRACE")
            .expect("set TQF_QUALIFICATION_ROUTE_TRACE");
        let bytes = std::fs::read(path).unwrap();
        let trace: ExpertRouteTrace = serde_json::from_slice(&bytes).unwrap();
        for capacity_mib in [512u64, 768, 1024] {
            for depth in [0usize, 4, 8] {
                let result = replay_prefetch(
                    &trace,
                    ReplayConfig {
                        capacity: Bytes(capacity_mib * 1024 * 1024),
                        policy: CachePolicyKind::Lru,
                        half_life_events: 1,
                    },
                    depth,
                    EXPERT,
                )
                .unwrap();
                println!(
                    "prefetch_replay capacity_mib={capacity_mib} depth={depth} result={}",
                    serde_json::to_string(&result).unwrap()
                );
            }
        }
    }
}
