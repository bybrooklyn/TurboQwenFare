//! Phase 29 "128K production gate" (spec §301; table row 29: "4G, 128K,
//! <=1%, >=15 tok/s populated-context floor").
//!
//! A literal end-to-end 128K-token decode is not something this session can
//! run: Phase 25's own measured floor is 1.775-2.34 s/token even at trivial
//! context length on this machine (external-flash-drive expert I/O
//! dominates, `docs/research/qualification/phase-25-m4-assault.md`), which
//! puts a real 131,072-token populate-then-time run at many hours to days
//! of wall clock — the same kind of infeasible-at-this-hardware-scale
//! finding the project already records honestly elsewhere (Phase 15's
//! 512-token gate, Phase 25's 15 tok/s floor).
//!
//! What *is* tractable, and is a real, separate question from I/O: does the
//! attention computation itself — O(context length) per decode step,
//! independent of expert I/O — become its own bottleneck as context grows
//! toward 128K? `FullAttentionLayer::seed_synthetic_history_for_benchmark`
//! populates a real cache (real broker accounting, real page sealing for
//! TQKV) without paying the O(n) cost of computing attention at every
//! intermediate depth, so a single real attention step can be timed at
//! deep, otherwise-unreachable context lengths.

use std::time::Instant;

use crate::context::tqkv::TqkvPrecision;
use crate::error::Result;
use crate::ids::{Bytes, LayerId};
use crate::memory::MemoryBroker;
use crate::model::qwen36::attention::{BackendChoice, FullAttentionLayer};
use crate::model::qwen36::geometry::Qwen36Geometry;

const HEADS: usize = Qwen36Geometry::FULL_ATTENTION_HEADS;
const HEAD_DIM: usize = Qwen36Geometry::FULL_HEAD_DIM;
const KV_WIDTH: usize = Qwen36Geometry::FULL_KV_HEADS * HEAD_DIM;

#[derive(Debug, Clone, Copy)]
pub struct ScalingPoint {
    pub context_tokens: usize,
    pub seed_elapsed: std::time::Duration,
    pub one_step_elapsed: std::time::Duration,
}

/// Seeds `context_tokens` of synthetic history, then times exactly one real
/// `decode_projected` attention step at that depth. `max_tokens` must be
/// `>= context_tokens + 1` (the layer's declared capacity, matching every
/// other `FullAttentionLayer` construction site).
pub fn measure_one_step_at_depth(
    choice: BackendChoice,
    context_tokens: usize,
    max_tokens: usize,
) -> Result<ScalingPoint> {
    let broker = MemoryBroker::new(Bytes(8 * 1024 * 1024 * 1024));
    let mut layer = FullAttentionLayer::new_with_backend(&broker, LayerId(3), max_tokens, choice)?;

    let seed_started = Instant::now();
    layer.seed_synthetic_history_for_benchmark(context_tokens)?;
    let seed_elapsed = seed_started.elapsed();
    assert_eq!(layer.cache_len(), context_tokens);

    let mut state = 0xABCDu64;
    let mut xorshift = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 3.0
    };
    let q: Vec<f32> = (0..HEADS * HEAD_DIM).map(|_| xorshift()).collect();
    let gate: Vec<f32> = vec![4.0; HEADS * HEAD_DIM];
    let k: Vec<f32> = (0..KV_WIDTH).map(|_| xorshift()).collect();
    let v: Vec<f32> = (0..KV_WIDTH).map(|_| xorshift()).collect();
    let q_norm = vec![1.0; HEAD_DIM];
    let k_norm = vec![1.0; HEAD_DIM];

    let step_started = Instant::now();
    layer.decode_projected(q, &gate, k, &v, &q_norm, &k_norm)?;
    let one_step_elapsed = step_started.elapsed();

    Ok(ScalingPoint {
        context_tokens,
        seed_elapsed,
        one_step_elapsed,
    })
}

/// Phase 29's memory half of the "4G, 128K" gate: actually *construct* (not
/// just formula-check, see Phase 27's `bytes_for_tokens` test) all ten
/// full-attention layers at 128K token capacity under TQKV-Q4, inside one
/// broker, and confirm the reservation succeeds and leaves headroom under
/// a 4 GiB budget for everything else (resident core + expert cache).
pub fn all_ten_layers_at_128k_tqkv_q4_reserved_bytes(broker: &MemoryBroker) -> Result<Bytes> {
    all_ten_layers_reserved_bytes(broker, 131_072, BackendChoice::Tqkv(TqkvPrecision::Q4))
}

/// Same construction, parameterized by context length and backend — Phase
/// 31 (spec §303) reuses this at 256K for both BF16 and TQKV-Q4.
pub fn all_ten_layers_reserved_bytes(
    broker: &MemoryBroker,
    max_tokens: usize,
    choice: BackendChoice,
) -> Result<Bytes> {
    let mut layers = Vec::with_capacity(Qwen36Geometry::FULL_ATTENTION_LAYERS);
    for layer in 0..Qwen36Geometry::NUM_LAYERS {
        if Qwen36Geometry::layer_kind(LayerId(layer as u8)) == crate::ids::LayerKind::FullAttention
        {
            layers.push(FullAttentionLayer::new_with_backend(
                broker,
                LayerId(layer as u8),
                max_tokens,
                choice,
            )?);
        }
    }
    assert_eq!(layers.len(), Qwen36Geometry::FULL_ATTENTION_LAYERS);
    Ok(broker.snapshot().reserved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_full_attention_layers_at_128k_tqkv_q4_fit_under_a_4gib_broker() {
        let four_gib = Bytes(4 * 1024 * 1024 * 1024);
        let broker = MemoryBroker::new(four_gib);
        let reserved = all_ten_layers_at_128k_tqkv_q4_reserved_bytes(&broker).unwrap();
        assert!(
            reserved.0 < four_gib.0,
            "TQKV-Q4 128K reservation {} exceeded the 4 GiB budget",
            reserved.0
        );
        println!(
            "phase29_memory tqkv_q4_128k_all_layers_bytes={} gib={:.3} headroom_gib={:.3}",
            reserved.0,
            reserved.0 as f64 / (1024.0 * 1024.0 * 1024.0),
            (four_gib.0 - reserved.0) as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }

    /// Phase 31 memory half (spec §303, table row 31: "256K usable within
    /// 4G and <=1%"): really constructs all ten full-attention layers at
    /// 262,144-token (256K) capacity, for both TQKV-Q4 and the BF16
    /// reference, inside one 4 GiB broker.
    #[test]
    fn ten_full_attention_layers_at_256k_capacity_check() {
        let four_gib = Bytes(4 * 1024 * 1024 * 1024);

        let broker_q4 = MemoryBroker::new(four_gib);
        let q4_reserved = all_ten_layers_reserved_bytes(
            &broker_q4,
            262_144,
            BackendChoice::Tqkv(TqkvPrecision::Q4),
        )
        .unwrap();
        assert!(
            q4_reserved.0 < four_gib.0,
            "TQKV-Q4 256K reservation {} exceeded the 4 GiB budget",
            q4_reserved.0
        );

        let broker_bf16 = MemoryBroker::new(Bytes(u64::MAX / 2));
        let bf16_reserved =
            all_ten_layers_reserved_bytes(&broker_bf16, 262_144, BackendChoice::Bf16).unwrap();

        println!(
            "phase31_memory tqkv_q4_256k_bytes={} gib={:.3} bf16_256k_bytes={} gib={:.3} bf16_fits_4gib={}",
            q4_reserved.0,
            q4_reserved.0 as f64 / (1024.0 * 1024.0 * 1024.0),
            bf16_reserved.0,
            bf16_reserved.0 as f64 / (1024.0 * 1024.0 * 1024.0),
            bf16_reserved.0 < four_gib.0,
        );
    }

    #[test]
    fn seeding_reaches_the_declared_depth_and_one_step_completes() {
        let point =
            measure_one_step_at_depth(BackendChoice::Bf16, 300, 512).unwrap();
        assert_eq!(point.context_tokens, 300);
        assert!(point.one_step_elapsed.as_secs() < 5);
    }

    /// Phase 29/31's populated-context attention-cost scaling table (spec
    /// §301, §303), extended through 256K for the Phase 31 "measure at
    /// 256K, decide full-vs-TQAttn" trigger. Run with
    /// `--release --ignored --nocapture`; the measured numbers are
    /// transcribed into `docs/research/qualification/phase-29-128k-gate.md`
    /// and `phase-31-256k-tqattn-trigger.md`. Debug builds are ~10-50x
    /// slower than release for this scalar numeric loop, so this is
    /// release-only and marked ignored like the other real-hardware
    /// qualification runs (it needs no checkpoint, but does need minutes of
    /// wall clock at the top end).
    #[test]
    #[ignore = "run with --release; Phase 29/31 populated-context attention scaling"]
    fn attention_cost_scales_with_populated_context_depth_toward_128k() {
        let depths = [
            512usize, 4_096, 16_384, 65_536, 131_072, 262_144,
        ];
        for &depth in &depths {
            let max_tokens = depth + 1;
            for (name, choice) in [
                ("BF16", BackendChoice::Bf16),
                ("TQKV-Q4", BackendChoice::Tqkv(TqkvPrecision::Q4)),
            ] {
                let point = measure_one_step_at_depth(choice, depth, max_tokens).unwrap();
                println!(
                    "phase29_scaling backend={name:<8} context_tokens={depth:<7} seed_ms={:<8} one_step_ms={}",
                    point.seed_elapsed.as_millis(),
                    point.one_step_elapsed.as_millis(),
                );
            }
        }
    }
}
