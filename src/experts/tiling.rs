//! Phase 22 offline replay (spec §294, §139): simulates a byte-budgeted
//! cache whose admission unit is one expert tile instead of one whole
//! expert, over the same exact-router trace format the Phase 21 policy
//! replay consumes. This tool changes nothing about routing or computed
//! results - it only simulates residency/admission, so the Phase 22 A/B
//! (whole vs 64/128/256/mixed tilings) is reproducible without touching
//! the real checkpoint.
//!
//! Metrics match the spec's Phase 22 demands: hit ratio is reported
//! alongside syscall count (one read per missing tile) and overread. Since
//! every Qwen Q4_K tile size is 4096-aligned, padding overread is
//! structurally zero for all candidates; the load-bearing costs are the
//! fetched bytes and the read syscalls. `fetched_never_reused_bytes`
//! additionally counts bytes fetched then evicted before any hit, the
//! direct measure of wasted SSD traffic.

use serde::Serialize;

use crate::error::{ModelError, Result};
use crate::experts::policy::{CachePolicyKind, ExpertRouteTrace, ReplayConfig};
use crate::format::tqf::tiling::{
    tile_plan, NeuronWidth, DOWN_HIDDEN, EXPERT_HIDDEN, EXPERT_NEURONS, Q4K_BLOCK_BYTES,
};
use crate::ids::{Bytes, ExpertId, LayerId};
use crate::model::qwen36::geometry::Qwen36Geometry;

/// Canonical Qwen3.6 Q4_K routed-expert region sizes (spec §117 geometry):
/// gate/up are [512 neurons, 2048 hidden], down is [2048 hidden, 512
/// neurons], all row-major Q4_K.
pub fn canonical_expert_region_bytes() -> (u32, u32, u32) {
    let gate = EXPERT_NEURONS * (EXPERT_HIDDEN / 256) * Q4K_BLOCK_BYTES;
    let down = DOWN_HIDDEN * (EXPERT_NEURONS / 256) * Q4K_BLOCK_BYTES;
    (gate, gate, down)
}

/// Byte sizes of every tile of one expert under `width`, in physical order.
pub fn tile_sizes(width: NeuronWidth) -> Vec<u32> {
    let (gate, up, down) = canonical_expert_region_bytes();
    tile_plan(width, gate, up, down)
        .iter()
        .map(|tile| tile.stored_bytes)
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TileReplayResult {
    /// Total (layer, expert, tile) demands across the trace.
    pub tile_demands: u64,
    pub tile_hits: u64,
    pub tile_misses: u64,
    /// Bytes fetched from SSD on tile misses (before any alignment).
    pub raw_miss_bytes: u64,
    /// One read syscall per missing tile.
    pub read_syscalls: u64,
    /// Alignment padding fetched beyond demanded bytes. Structurally zero
    /// for Qwen Q4_K tilings (every tile size is 4096-aligned); reported
    /// anyway so a future misaligned layout cannot silently regress it.
    pub overread_bytes: u64,
    /// Bytes fetched then evicted without ever being hit - wasted SSD
    /// traffic under this admission granularity.
    pub fetched_never_reused_bytes: u64,
    pub evictions: u64,
    pub peak_resident_bytes: u64,
}

#[derive(Debug, Clone)]
struct TileEntry {
    layer: LayerId,
    expert: ExpertId,
    tile_index: usize,
    bytes: u32,
    hits: u64,
    /// Intrusive LRU list links over the `entries` slab.
    newer: Option<usize>,
    older: Option<usize>,
}

struct TileReplayCache {
    config: ReplayConfig,
    resident: u64,
    /// O(1) key lookup into the slab.
    index: std::collections::HashMap<(LayerId, ExpertId, usize), usize>,
    entries: Vec<TileEntry>,
    /// Most-recently-used end of the LRU list.
    mru: Option<usize>,
    /// Least-recently-used end of the LRU list.
    lru: Option<usize>,
    result: TileReplayResult,
}

impl TileReplayCache {
    fn new(config: ReplayConfig) -> Result<Self> {
        if config.capacity.0 == 0 {
            return Err(ModelError::Unsupported(
                "tile replay requires a nonzero byte capacity".to_string(),
            )
            .into());
        }
        if config.policy != CachePolicyKind::Lru {
            return Err(ModelError::Unsupported(
                "tile replay uses the Phase 21 benchmark-selected LRU policy only".to_string(),
            )
            .into());
        }
        Ok(Self {
            config,
            resident: 0,
            index: std::collections::HashMap::new(),
            entries: Vec::new(),
            mru: None,
            lru: None,
            result: TileReplayResult::default(),
        })
    }

    /// Unlinks a slab index from the LRU list.
    fn unlink(&mut self, index: usize) {
        let newer = self.entries[index].newer.take();
        let older = self.entries[index].older.take();
        match newer {
            Some(newer) => self.entries[newer].older = older,
            None => self.mru = older,
        }
        match older {
            Some(older) => self.entries[older].newer = newer,
            None => self.lru = newer,
        }
    }

    /// Pushes a slab index to the MRU end of the LRU list.
    fn push_mru(&mut self, index: usize) {
        self.entries[index].newer = None;
        self.entries[index].older = self.mru;
        match self.mru {
            Some(mru) => self.entries[mru].newer = Some(index),
            None => self.lru = Some(index),
        }
        self.mru = Some(index);
    }

    fn route(&mut self, layer: LayerId, experts: [ExpertId; 8], sizes: &[u32]) -> Result<()> {
        let pinned: Vec<(LayerId, ExpertId, usize)> = experts
            .iter()
            .flat_map(|expert| (0..sizes.len()).map(move |tile_index| (layer, *expert, tile_index)))
            .collect();
        let mut missing_bytes = 0u64;
        for expert in experts {
            for (tile_index, &bytes) in sizes.iter().enumerate() {
                self.result.tile_demands = self.result.tile_demands.saturating_add(1);
                let key = (layer, expert, tile_index);
                if self.index.contains_key(&key) {
                    self.result.tile_hits = self.result.tile_hits.saturating_add(1);
                } else {
                    self.result.tile_misses = self.result.tile_misses.saturating_add(1);
                    missing_bytes = missing_bytes.saturating_add(bytes as u64);
                }
            }
        }
        while self.resident.saturating_add(missing_bytes) > self.config.capacity.0 {
            // Walk the LRU list from the least-recently-used end, skipping
            // this route's pinned tiles.
            let mut cursor = self.lru;
            let candidate = loop {
                let Some(index) = cursor else {
                    break None;
                };
                let entry = &self.entries[index];
                let key = (entry.layer, entry.expert, entry.tile_index);
                if pinned.contains(&key) {
                    cursor = entry.newer;
                    continue;
                }
                break Some(index);
            };
            let index = candidate.ok_or_else(|| {
                ModelError::Unsupported(format!(
                    "tile replay cannot fit one route's pinned tiles in {} bytes at this granularity",
                    self.config.capacity.0
                ))
            })?;
            self.unlink(index);
            let key = {
                let entry = self.entries.swap_remove(index);
                self.resident = self.resident.saturating_sub(entry.bytes as u64);
                self.result.evictions = self.result.evictions.saturating_add(1);
                if entry.hits == 0 {
                    self.result.fetched_never_reused_bytes = self
                        .result
                        .fetched_never_reused_bytes
                        .saturating_add(entry.bytes as u64);
                }
                (entry.layer, entry.expert, entry.tile_index)
            };
            self.index.remove(&key);
            // Fix up links for the swapped-in last element.
            if index < self.entries.len() {
                let moved_key = {
                    let moved = &self.entries[index];
                    (moved.layer, moved.expert, moved.tile_index)
                };
                self.index.insert(moved_key, index);
                let newer = self.entries[index].newer;
                let older = self.entries[index].older;
                if let Some(newer) = newer {
                    self.entries[newer].older = Some(index);
                } else {
                    self.mru = Some(index);
                }
                if let Some(older) = older {
                    self.entries[older].newer = Some(index);
                } else {
                    self.lru = Some(index);
                }
            }
            // When the removed entry was the last slab element nothing
            // was swapped in; the unlink above already cleared its links.
        }
        for expert in experts {
            for (tile_index, &bytes) in sizes.iter().enumerate() {
                let key = (layer, expert, tile_index);
                if let Some(&index) = self.index.get(&key) {
                    let entry = &mut self.entries[index];
                    entry.hits = entry.hits.saturating_add(1);
                    self.unlink(index);
                    self.push_mru(index);
                    continue;
                }
                let index = self.entries.len();
                self.entries.push(TileEntry {
                    layer,
                    expert,
                    tile_index,
                    bytes,
                    hits: 0,
                    newer: None,
                    older: None,
                });
                self.index.insert(key, index);
                self.push_mru(index);
                self.resident = self.resident.saturating_add(bytes as u64);
                self.result.raw_miss_bytes =
                    self.result.raw_miss_bytes.saturating_add(bytes as u64);
                self.result.read_syscalls = self.result.read_syscalls.saturating_add(1);
                self.result.overread_bytes = self.result.overread_bytes.saturating_add(
                    (bytes as u64).div_ceil(4096).saturating_mul(4096) - bytes as u64,
                );
                self.result.peak_resident_bytes =
                    self.result.peak_resident_bytes.max(self.resident);
            }
        }
        Ok(())
    }
}

/// Replays an exact-router trace through a tile-granular LRU cache.
/// `width` selects the tiling; `sizes` must be `tile_sizes(width)`.
pub fn replay_trace_tiled(
    trace: &ExpertRouteTrace,
    config: ReplayConfig,
    width: NeuronWidth,
    sizes: &[u32],
) -> Result<TileReplayResult> {
    if trace.schema_version != 1 {
        return Err(ModelError::Unsupported(format!(
            "unsupported expert route-trace schema {}",
            trace.schema_version
        ))
        .into());
    }
    let expected = tile_plan(
        width,
        canonical_expert_region_bytes().0,
        canonical_expert_region_bytes().1,
        canonical_expert_region_bytes().2,
    )
    .len();
    if sizes.len() != expected {
        return Err(ModelError::Unsupported(format!(
            "tile replay got {} tile sizes but {width:?} layout has {expected}",
            sizes.len()
        ))
        .into());
    }
    let mut cache = TileReplayCache::new(config)?;
    for step in &trace.steps {
        for layer in &step.layers {
            cache.route(LayerId(layer.layer), layer.expert_ids.map(ExpertId), sizes)?;
        }
    }
    Ok(cache.result)
}

/// Convenience: the same replay with the canonical Qwen tile sizes for
/// `width` built in.
pub fn replay_trace_tiled_canonical(
    trace: &ExpertRouteTrace,
    config: ReplayConfig,
    width: NeuronWidth,
) -> Result<TileReplayResult> {
    replay_trace_tiled(trace, config, width, &tile_sizes(width))
}

/// The minimum cache capacity (bytes) at which a single exact route (eight
/// routed experts) is admissible under `width`. Every selected expert's
/// tiles must be pinnable at once, so this is eight experts' full bytes
/// regardless of tiling - what tiling changes is admission/eviction
/// granularity, not the peak pin set.
pub fn minimum_route_capacity(width: NeuronWidth) -> Bytes {
    let expert_bytes: u64 = tile_sizes(width).iter().map(|&bytes| bytes as u64).sum();
    Bytes(expert_bytes * Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::policy::{ExpertRouteLayer, ExpertRouteStep};

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
            fixture_id: "tile-fixture".to_string(),
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

    #[test]
    fn tiled_eviction_granularity_preserves_partial_residency() {
        // Three alternating routes. Capacity holds exactly one route (8
        // experts) plus five N128 gate tiles. Whole-expert admission must
        // evict an entire expert to admit one byte, so it re-fetches
        // everything every event (0 hits). Tile admission evicts only as
        // many tiles as the byte budget demands, so a few tiles survive
        // each cycle and hit on the alternation - fetched bytes drop
        // below the whole-expert total. (The surviving set is three
        // tiles, not five: down tiles are twice a gate tile's size, so
        // the byte-conditional eviction loop stops after expert 7's
        // tiles t7/t8/t9.)
        let sizes = tile_sizes(NeuronWidth::N128);
        let expert_bytes: u64 = sizes.iter().map(|&b| b as u64).sum();
        let five_tiles: u64 = sizes.iter().take(5).map(|&b| b as u64).sum();
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [8, 9, 10, 11, 12, 13, 14, 15]),
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
        ]);
        let result = replay_trace_tiled(
            &trace,
            lru(expert_bytes * 8 + five_tiles),
            NeuronWidth::N128,
            &sizes,
        )
        .unwrap();
        assert_eq!(result.tile_demands, 240);
        assert_eq!(result.tile_hits, 3);
        assert_eq!(result.tile_misses, 237);
        // Whole-expert admission at the same capacity fetches 24 whole
        // experts (0 hits); the tiled run fetched strictly fewer bytes.
        let whole = crate::experts::policy::replay_trace(&trace, lru(expert_bytes * 8), |_, _| {
            Ok(Bytes(expert_bytes))
        })
        .unwrap();
        assert_eq!(whole.hits, 0);
        assert!(result.raw_miss_bytes < whole.raw_miss_bytes);
        assert!(
            result.read_syscalls > whole.misses,
            "tiling trades reads for bytes"
        );
    }

    #[test]
    fn whole_and_tiled_agree_at_whole_capacity() {
        // At a capacity holding many experts, a warm repeat of an
        // identical route is all hits under any tiling.
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
        ]);
        let (gate, up, down) = canonical_expert_region_bytes();
        let expert = (gate + up + down) as u64;
        for width in [
            NeuronWidth::N64,
            NeuronWidth::N128,
            NeuronWidth::N256,
            NeuronWidth::Mixed128,
            NeuronWidth::Whole,
        ] {
            let result = replay_trace_tiled_canonical(&trace, lru(expert * 16), width).unwrap();
            assert_eq!(
                result.tile_misses,
                8 * tile_sizes(width).len() as u64,
                "{width:?}"
            );
            assert_eq!(
                result.tile_hits,
                8 * tile_sizes(width).len() as u64,
                "{width:?}"
            );
            assert_eq!(result.raw_miss_bytes, expert * 8, "{width:?}");
        }
    }

    #[test]
    fn waste_metric_counts_fetched_never_reused_bytes() {
        // One-shot demand then eviction: every fetched byte is wasted.
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [8, 9, 10, 11, 12, 13, 14, 15]),
        ]);
        let (gate, up, down) = canonical_expert_region_bytes();
        let expert_bytes = (gate + up + down) as u64;
        // Capacity 8 experts exactly: the second route evicts the first.
        let result =
            replay_trace_tiled_canonical(&trace, lru(expert_bytes * 8), NeuronWidth::N128).unwrap();
        assert_eq!(result.tile_hits, 0);
        assert_eq!(result.raw_miss_bytes, expert_bytes * 16);
        // The second route's entries are still resident at trace end, so
        // only the first route's fetched bytes were evicted unreused.
        assert_eq!(result.fetched_never_reused_bytes, expert_bytes * 8);
    }

    #[test]
    fn every_candidate_tile_size_is_4096_aligned() {
        for width in [
            NeuronWidth::N64,
            NeuronWidth::N128,
            NeuronWidth::N256,
            NeuronWidth::Mixed128,
            NeuronWidth::Whole,
        ] {
            for bytes in tile_sizes(width) {
                assert_eq!(bytes % 4096, 0, "{width:?} tile of {bytes} bytes");
            }
        }
    }

    #[test]
    fn minimum_route_capacity_is_eight_experts_under_any_tiling() {
        let expected = 8 * 3 * 589_824;
        for width in [
            NeuronWidth::Whole,
            NeuronWidth::N64,
            NeuronWidth::N128,
            NeuronWidth::N256,
            NeuronWidth::Mixed128,
        ] {
            assert_eq!(minimum_route_capacity(width).0, expected, "{width:?}");
        }
    }

    #[test]
    #[ignore = "requires a qualification route-trace artifact"]
    fn qualification_trace_compares_whole_and_tiled_granularity() {
        let path = std::env::var("TQF_QUALIFICATION_ROUTE_TRACE")
            .expect("set TQF_QUALIFICATION_ROUTE_TRACE");
        let bytes = std::fs::read(path).unwrap();
        let trace: ExpertRouteTrace = serde_json::from_slice(&bytes).unwrap();
        for capacity_mib in [128u64, 256, 512, 768, 1024] {
            let config = lru(capacity_mib * 1024 * 1024);
            for width in [
                NeuronWidth::Whole,
                NeuronWidth::N64,
                NeuronWidth::N128,
                NeuronWidth::N256,
                NeuronWidth::Mixed128,
            ] {
                match replay_trace_tiled_canonical(&trace, config, width) {
                    Ok(result) => println!(
                        "tiled_replay capacity_mib={capacity_mib} width={width:?} result={}",
                        serde_json::to_string(&result).unwrap()
                    ),
                    Err(error) => println!(
                        "tiled_replay capacity_mib={capacity_mib} width={width:?} infeasible: {error}"
                    ),
                }
            }
        }
    }

    #[test]
    fn replay_rejects_wrong_tile_size_table_and_unknown_schema() {
        let trace = trace(vec![layer(0, [0, 1, 2, 3, 4, 5, 6, 7])]);
        let sizes = tile_sizes(NeuronWidth::N128);
        assert!(replay_trace_tiled(&trace, lru(1024 * 1024), NeuronWidth::N64, &sizes).is_err());
        let mut bad_schema = trace.clone();
        bad_schema.schema_version = 2;
        assert!(
            replay_trace_tiled(&bad_schema, lru(1024 * 1024), NeuronWidth::N128, &sizes).is_err()
        );
    }
}
