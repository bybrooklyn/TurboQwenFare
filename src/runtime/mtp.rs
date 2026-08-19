//! MTP (multi-token prediction) accounting and controller (spec §68,
//! §172; Phase 33, spec §305). "Qwen3.6 ships MTP... TQF implements MTP as
//! another schedule candidate. The controller turns it on only when
//! accepted tokens/second increases... A draft mechanism that predicts
//! multiple tokens but causes substantially more unique expert fetches can
//! lose despite fewer target iterations."
//!
//! The real MTP sidecar checkpoint (`source::pinned::MTP_FILENAME`, a
//! separate ~1 GiB GGUF) is not installed in this environment, and a full
//! second forward-pass runtime (its own embedding/hidden norms, one-layer
//! MoE, projection) is out of scope for this phase — that is comparable in
//! size to the phases that built the *target* model's forward pass. This
//! module implements the parts that do not require the sidecar to exist:
//! the accept/reject verification semantics and accounting NVMAI's real
//! `StreamingMTPDecoder` uses (`StreamingMTP.swift`,
//! `/Volumes/flash1/tqf-research/NVMAI/sources/NVMAI/Runtime/Generation/`),
//! the adaptive hysteresis controller (spec §172), and expert-union
//! bandwidth accounting measured against the real committed router trace
//! (`docs/research/qualification/raw-a-128-route-trace.json`) rather than
//! synthetic data.

use crate::ids::ExpertId;
use crate::model::qwen36::geometry::Qwen36Geometry;

const HIDDEN: usize = Qwen36Geometry::HIDDEN_SIZE;
const EXPERT_WIDTH: usize = Qwen36Geometry::ROUTED_EXPERT_WIDTH;
const TOP_K: usize = Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN;
const Q4K_BLOCK_BYTES: usize = 144;

fn q4_bytes(rows: usize, cols: usize) -> usize {
    rows * (cols / 256) * Q4K_BLOCK_BYTES
}

/// One routed expert's stored Q4_K bytes (gate + up + down), constant
/// across every expert/layer since all routed experts share
/// `ROUTED_EXPERT_WIDTH` (spec §117) — the same formula
/// `experts::WholeExpertLfuCache` uses for real cache accounting.
pub fn expert_bytes() -> u64 {
    (q4_bytes(EXPERT_WIDTH, HIDDEN)
        + q4_bytes(EXPERT_WIDTH, HIDDEN)
        + q4_bytes(HIDDEN, EXPERT_WIDTH)) as u64
}

/// Spec §172's MTP runtime contract, matching NVMAI's real
/// `MTPStatistics` field-for-field (a verified-in-production design, not
/// reinvented here).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MtpStatistics {
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub target_backbone_passes: u64,
    pub emitted_tokens: u64,
}

impl MtpStatistics {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.drafted_tokens as f64
        }
    }

    /// >1.0 means MTP is emitting more tokens per target backbone pass than
    /// the 1-token-per-pass non-speculative baseline — the direct "net
    /// accepted tok/s" proxy spec §172 asks the controller to track.
    pub fn emitted_tokens_per_target_pass(&self) -> f64 {
        if self.target_backbone_passes == 0 {
            0.0
        } else {
            self.emitted_tokens as f64 / self.target_backbone_passes as f64
        }
    }

    pub fn record(&mut self, accepted: bool, emitted: u64, target_passes: u64) {
        self.drafted_tokens += 1;
        if accepted {
            self.accepted_tokens += 1;
        }
        self.target_backbone_passes += target_passes;
        self.emitted_tokens += emitted;
    }
}

/// Official Qwen3.6 MTP semantics (NVMAI `StreamingMTPDecoder.advance`,
/// spec §172 "accepted-token verification"): the target verifies the
/// `[boundary_token, draft_token]` pair by greedy comparison. On accept,
/// both the draft token and the target's next prediction are emitted (2
/// tokens for 1 target backbone pass); on reject, only the target's own
/// prediction is emitted (1 token for 1 pass) and the draft's KV row must
/// be rolled back by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyOutcome {
    Accepted { emitted_tokens: u64 },
    Rejected { emitted_tokens: u64 },
}

pub fn verify_pair(target_prediction_after_boundary: u32, draft_token: u32) -> VerifyOutcome {
    if target_prediction_after_boundary == draft_token {
        VerifyOutcome::Accepted { emitted_tokens: 2 }
    } else {
        VerifyOutcome::Rejected { emitted_tokens: 1 }
    }
}

/// Spec §172: "The adaptive controller disables MTP when rolling net
/// benefit is negative beyond hysteresis." Net benefit is
/// `emitted_tokens_per_target_pass - 1.0` (the non-speculative baseline);
/// a rolling window smooths single-step noise, and separate on/off
/// thresholds (rather than one threshold) give the hysteresis band spec
/// §172 names, so the controller doesn't flap at the boundary.
pub struct MtpController {
    window: Vec<f64>,
    window_capacity: usize,
    enabled: bool,
    enable_threshold: f64,
    disable_threshold: f64,
}

impl MtpController {
    pub fn new(window_capacity: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_capacity),
            window_capacity,
            enabled: false,
            enable_threshold: 0.10,
            disable_threshold: -0.05,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn rolling_net_benefit(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        }
    }

    /// Feeds one verification outcome's net benefit
    /// (`emitted_tokens - 1.0`, i.e. tokens gained/lost versus the
    /// non-speculative baseline for that one target pass) and re-decides
    /// on/off. Returns the (possibly updated) enabled state.
    pub fn record_and_decide(&mut self, emitted_tokens: u64) -> bool {
        let net_benefit = emitted_tokens as f64 - 1.0;
        if self.window.len() == self.window_capacity {
            self.window.remove(0);
        }
        self.window.push(net_benefit);

        // "Rolling" implies a sustained trend, not a single lucky/unlucky
        // sample — don't flip state until there is a full window's worth
        // of evidence behind the average.
        if self.window.len() < self.window_capacity {
            return self.enabled;
        }
        let rolling = self.rolling_net_benefit();
        if !self.enabled && rolling > self.enable_threshold {
            self.enabled = true;
        } else if self.enabled && rolling < self.disable_threshold {
            self.enabled = false;
        }
        self.enabled
    }
}

/// Spec §68/§172's "unique expert bytes touched, union of routed experts
/// across draft tokens" and "extra expert bytes" metrics, computed from
/// two real per-layer top-8 router selections. `boundary`/`draft` would
/// come from the target's verification pass and the draft's proposal in a
/// real MTP session; measured here against real consecutive-step router
/// selections from the committed trace (see `tests::real_trace_...`
/// below) as an honest proxy in the absence of a real sidecar draft.
pub struct UnionBandwidth {
    pub union_experts: usize,
    pub union_bytes: u64,
    pub separate_experts_sum: usize,
    pub separate_bytes_sum: u64,
    pub saved_bytes: u64,
}

pub fn union_bandwidth(boundary: &[ExpertId; TOP_K], draft: &[ExpertId; TOP_K]) -> UnionBandwidth {
    let mut union: Vec<ExpertId> = boundary.to_vec();
    for &e in draft {
        if !union.contains(&e) {
            union.push(e);
        }
    }
    let per_expert = expert_bytes();
    let union_bytes = union.len() as u64 * per_expert;
    let separate_experts_sum = boundary.len() + draft.len();
    let separate_bytes_sum = separate_experts_sum as u64 * per_expert;
    UnionBandwidth {
        union_experts: union.len(),
        union_bytes,
        separate_experts_sum,
        separate_bytes_sum,
        saved_bytes: separate_bytes_sum.saturating_sub(union_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::policy::ExpertRouteTrace;
    use std::path::Path;

    #[test]
    fn verify_pair_matches_nvmai_accept_reject_semantics() {
        assert_eq!(
            verify_pair(42, 42),
            VerifyOutcome::Accepted { emitted_tokens: 2 }
        );
        assert_eq!(
            verify_pair(42, 7),
            VerifyOutcome::Rejected { emitted_tokens: 1 }
        );
    }

    #[test]
    fn statistics_track_acceptance_rate_and_emitted_per_pass() {
        let mut stats = MtpStatistics::default();
        stats.record(true, 2, 1);
        stats.record(false, 1, 1);
        stats.record(true, 2, 1);
        assert_eq!(stats.drafted_tokens, 3);
        assert_eq!(stats.accepted_tokens, 2);
        assert!((stats.acceptance_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(stats.emitted_tokens, 5);
        assert_eq!(stats.target_backbone_passes, 3);
        assert!((stats.emitted_tokens_per_target_pass() - 5.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn controller_enables_only_after_a_sustained_positive_rolling_benefit() {
        let mut controller = MtpController::new(4);
        assert!(!controller.enabled());
        // A single accept isn't enough to fill a window of hysteresis.
        controller.record_and_decide(2);
        assert!(!controller.enabled());
        // Four straight accepts (net benefit +1.0 each) clears the window
        // average well above the enable threshold.
        for _ in 0..4 {
            controller.record_and_decide(2);
        }
        assert!(controller.enabled());
    }

    #[test]
    fn controller_disables_after_a_sustained_negative_rolling_benefit() {
        let mut controller = MtpController::new(4);
        for _ in 0..4 {
            controller.record_and_decide(2);
        }
        assert!(controller.enabled());
        // A run of draft failures that cost a target pass and produced no
        // usable output (net benefit -1.0 each) should push the rolling
        // average below the disable threshold.
        for _ in 0..4 {
            controller.record_and_decide(0);
        }
        assert!(!controller.enabled());
    }

    #[test]
    fn union_bandwidth_counts_overlap_correctly() {
        let boundary = [
            ExpertId(1),
            ExpertId(2),
            ExpertId(3),
            ExpertId(4),
            ExpertId(5),
            ExpertId(6),
            ExpertId(7),
            ExpertId(8),
        ];
        let draft = [
            ExpertId(1),
            ExpertId(2),
            ExpertId(3),
            ExpertId(4),
            ExpertId(9),
            ExpertId(10),
            ExpertId(11),
            ExpertId(12),
        ];
        let result = union_bandwidth(&boundary, &draft);
        assert_eq!(result.union_experts, 12); // 4 shared, 8 unique
        assert_eq!(result.separate_experts_sum, 16);
        assert!(result.saved_bytes > 0);
        assert_eq!(
            result.union_bytes + result.saved_bytes,
            result.separate_bytes_sum
        );
    }

    /// Real measured expert-union bandwidth (spec §68/§172), computed from
    /// the committed real router trace
    /// (`docs/research/qualification/raw-a-128-route-trace.json`, 128 real
    /// greedy decode steps x 40 layers on the canonical checkpoint) rather
    /// than synthetic router selections. Every consecutive step pair
    /// (step i as "boundary", step i+1 as "draft") is a defensible proxy
    /// for the *accepted* case specifically: MTP's draft token, when
    /// accepted, is by definition the model's own real next token — which
    /// is exactly what step i+1 already is in this real greedy trace.
    #[test]
    fn real_trace_measures_expert_union_savings_across_consecutive_steps() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("docs/research/qualification/raw-a-128-route-trace.json");
        let text = std::fs::read_to_string(&path).unwrap();
        let trace: ExpertRouteTrace = serde_json::from_str(&text).unwrap();

        let mut total_union_bytes = 0u64;
        let mut total_separate_bytes = 0u64;
        let mut pairs = 0u64;
        for window in trace.steps.windows(2) {
            let (boundary_step, draft_step) = (&window[0], &window[1]);
            for boundary_layer in &boundary_step.layers {
                let draft_layer = draft_step
                    .layers
                    .iter()
                    .find(|l| l.layer == boundary_layer.layer)
                    .expect("trace layers are 1:1 across steps");
                let boundary_ids: [ExpertId; TOP_K] = boundary_layer.expert_ids.map(ExpertId);
                let draft_ids: [ExpertId; TOP_K] = draft_layer.expert_ids.map(ExpertId);
                let result = union_bandwidth(&boundary_ids, &draft_ids);
                total_union_bytes += result.union_bytes;
                total_separate_bytes += result.separate_bytes_sum;
                pairs += 1;
            }
        }
        let saved = total_separate_bytes - total_union_bytes;
        let saved_pct = saved as f64 / total_separate_bytes as f64 * 100.0;
        println!(
            "phase33_mtp_bandwidth layer_pairs={pairs} separate_bytes={total_separate_bytes} union_bytes={total_union_bytes} saved_bytes={saved} saved_pct={saved_pct:.2}"
        );
        assert!(pairs > 0);
        assert!(total_union_bytes <= total_separate_bytes);
    }
}
