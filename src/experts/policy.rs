//! Offline replay for Phase 21's benchmark-selected global expert-cache
//! policies. This module never predicts or changes a route: it consumes exact
//! router IDs and simulates only residency/admission under a byte budget.

use serde::{Deserialize, Serialize};

use crate::error::{ModelError, Result};
use crate::ids::{Bytes, ExpertId, LayerId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CachePolicyKind {
    Lru,
    Lfu,
    DecayedCostAware,
}

#[derive(Debug, Clone, Copy)]
pub struct ReplayConfig {
    pub capacity: Bytes,
    pub policy: CachePolicyKind,
    pub half_life_events: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertRouteTrace {
    pub schema_version: u32,
    pub fixture_id: String,
    pub model_source_sha256: String,
    pub steps: Vec<ExpertRouteStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertRouteStep {
    pub decode_step: usize,
    pub input_token: u32,
    pub output_token: u32,
    pub layers: Vec<ExpertRouteLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertRouteLayer {
    pub layer: u8,
    pub expert_ids: [u16; 8],
    pub weights: [f32; 8],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ReplayResult {
    pub route_events: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub raw_miss_bytes: u64,
    pub peak_resident_bytes: u64,
}

#[derive(Debug, Clone)]
struct Entry {
    layer: LayerId,
    expert: ExpertId,
    bytes: Bytes,
    frequency: u64,
    last_use: u64,
}

struct ReplayCache {
    config: ReplayConfig,
    clock: u64,
    resident: Bytes,
    entries: Vec<Entry>,
    result: ReplayResult,
}

impl ReplayCache {
    fn new(config: ReplayConfig) -> Result<Self> {
        if config.capacity.0 == 0 {
            return Err(ModelError::Unsupported(
                "cache-policy replay requires a nonzero byte capacity".to_string(),
            )
            .into());
        }
        if config.policy == CachePolicyKind::DecayedCostAware && config.half_life_events == 0 {
            return Err(ModelError::Unsupported(
                "decayed cache-policy replay requires a nonzero half-life".to_string(),
            )
            .into());
        }
        Ok(Self {
            config,
            clock: 0,
            resident: Bytes(0),
            entries: Vec::new(),
            result: ReplayResult::default(),
        })
    }

    fn utility(&self, entry: &Entry) -> f64 {
        match self.config.policy {
            CachePolicyKind::Lru => entry.last_use as f64,
            CachePolicyKind::Lfu => entry.frequency as f64 * 1.0e12 + entry.last_use as f64,
            CachePolicyKind::DecayedCostAware => {
                let age = self.clock.saturating_sub(entry.last_use) as f64;
                let decay = 2.0f64.powf(-(age / self.config.half_life_events as f64));
                entry.frequency as f64 * decay * entry.bytes.0 as f64
            }
        }
    }

    fn eviction_candidate(&self, selected: &[(LayerId, ExpertId)]) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !selected.contains(&(entry.layer, entry.expert)))
            .min_by(|(_, left), (_, right)| {
                self.utility(left)
                    .total_cmp(&self.utility(right))
                    .then_with(|| left.last_use.cmp(&right.last_use))
                    .then_with(|| left.layer.cmp(&right.layer))
                    .then_with(|| left.expert.cmp(&right.expert))
            })
            .map(|(index, _)| index)
    }

    fn route<F>(&mut self, layer: LayerId, experts: [ExpertId; 8], mut size_of: F) -> Result<()>
    where
        F: FnMut(LayerId, ExpertId) -> Result<Bytes>,
    {
        self.clock = self.clock.saturating_add(1);
        self.result.route_events = self.result.route_events.saturating_add(1);
        let selected = experts.map(|expert| (layer, expert));
        let mut missing = Vec::new();
        let mut missing_bytes = 0u64;
        for expert in experts {
            if !missing.iter().any(|(candidate, _)| *candidate == expert)
                && !self
                    .entries
                    .iter()
                    .any(|entry| entry.layer == layer && entry.expert == expert)
            {
                let bytes = size_of(layer, expert)?;
                if bytes.0 == 0 {
                    return Err(ModelError::Unsupported(
                        "cache-policy replay encountered a zero-byte expert".to_string(),
                    )
                    .into());
                }
                missing_bytes = missing_bytes.saturating_add(bytes.0);
                missing.push((expert, bytes));
            }
        }
        if missing_bytes > self.config.capacity.0 {
            return Err(ModelError::Unsupported(format!(
                "one exact route needs {missing_bytes} bytes, exceeding replay capacity {}",
                self.config.capacity.0
            ))
            .into());
        }
        while self.resident.0.saturating_add(missing_bytes) > self.config.capacity.0 {
            let index = self.eviction_candidate(&selected).ok_or_else(|| {
                ModelError::Unsupported(
                    "cache-policy replay could not evict without violating route pinning"
                        .to_string(),
                )
            })?;
            let entry = self.entries.swap_remove(index);
            self.resident.0 = self.resident.0.saturating_sub(entry.bytes.0);
            self.result.evictions = self.result.evictions.saturating_add(1);
        }

        for expert in experts {
            if let Some(entry) = self
                .entries
                .iter_mut()
                .find(|entry| entry.layer == layer && entry.expert == expert)
            {
                entry.frequency = entry.frequency.saturating_add(1);
                entry.last_use = self.clock;
                self.result.hits = self.result.hits.saturating_add(1);
                continue;
            }
            let index = missing
                .iter()
                .position(|(candidate, _)| *candidate == expert)
                .ok_or_else(|| crate::error::InternalError {
                    incident_id: "cache-replay-missing-size".to_string(),
                    message: "planned replay miss lost its byte size".to_string(),
                })?;
            let (_, bytes) = missing.swap_remove(index);
            self.entries.push(Entry {
                layer,
                expert,
                bytes,
                frequency: 1,
                last_use: self.clock,
            });
            self.resident.0 = self.resident.0.saturating_add(bytes.0);
            self.result.misses = self.result.misses.saturating_add(1);
            self.result.raw_miss_bytes = self.result.raw_miss_bytes.saturating_add(bytes.0);
            self.result.peak_resident_bytes = self.result.peak_resident_bytes.max(self.resident.0);
        }
        Ok(())
    }
}

pub fn replay_trace<F>(
    trace: &ExpertRouteTrace,
    config: ReplayConfig,
    mut size_of: F,
) -> Result<ReplayResult>
where
    F: FnMut(LayerId, ExpertId) -> Result<Bytes>,
{
    if trace.schema_version != 1 {
        return Err(ModelError::Unsupported(format!(
            "unsupported expert route-trace schema {}",
            trace.schema_version
        ))
        .into());
    }
    let mut cache = ReplayCache::new(config)?;
    for step in &trace.steps {
        for layer in &step.layers {
            cache.route(
                LayerId(layer.layer),
                layer.expert_ids.map(ExpertId),
                &mut size_of,
            )?;
        }
    }
    Ok(cache.result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen36::weights::LoadedQwen36Expert;

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
            fixture_id: "fixture".to_string(),
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

    #[test]
    fn exact_route_entries_are_pinned_during_atomic_admission() {
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [0, 1, 2, 3, 8, 9, 10, 11]),
        ]);
        let result = replay_trace(
            &trace,
            ReplayConfig {
                capacity: Bytes(8 * 1024),
                policy: CachePolicyKind::Lru,
                half_life_events: 1,
            },
            |_, _| Ok(Bytes(1024)),
        )
        .unwrap();
        assert_eq!(result.hits, 4);
        assert_eq!(result.misses, 12);
        assert_eq!(result.evictions, 4);
        assert_eq!(result.peak_resident_bytes, 8 * 1024);
    }

    #[test]
    fn lfu_retains_repeated_entries_that_lru_would_evict() {
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [8, 9, 10, 11, 12, 13, 14, 15]),
            layer(0, [16, 17, 18, 19, 20, 21, 22, 23]),
            layer(0, [4, 5, 6, 7, 24, 25, 26, 27]),
        ]);
        let replay = |policy| {
            replay_trace(
                &trace,
                ReplayConfig {
                    capacity: Bytes(12 * 1024),
                    policy,
                    half_life_events: 2,
                },
                |_, _| Ok(Bytes(1024)),
            )
            .unwrap()
        };
        assert!(replay(CachePolicyKind::Lfu).hits > replay(CachePolicyKind::Lru).hits);
    }

    #[test]
    fn cost_aware_policy_accounts_variable_reload_bytes() {
        let trace = trace(vec![
            layer(0, [0, 1, 2, 3, 4, 5, 6, 7]),
            layer(0, [8, 9, 10, 11, 12, 13, 14, 15]),
            layer(0, [0, 1, 2, 3, 16, 17, 18, 19]),
        ]);
        let result = replay_trace(
            &trace,
            ReplayConfig {
                capacity: Bytes(20 * 1024),
                policy: CachePolicyKind::DecayedCostAware,
                half_life_events: 4,
            },
            |_, expert| Ok(Bytes(if expert.0 < 4 { 2048 } else { 1024 })),
        )
        .unwrap();
        assert!(result.hits >= 4);
        assert!(result.raw_miss_bytes > 0);
    }

    #[test]
    #[ignore = "requires a qualification route-trace artifact"]
    fn qualification_trace_replays_all_phase21_policy_candidates() {
        let path = std::env::var("TQF_QUALIFICATION_ROUTE_TRACE")
            .expect("set TQF_QUALIFICATION_ROUTE_TRACE");
        let bytes = std::fs::read(path).unwrap();
        let trace: ExpertRouteTrace = serde_json::from_slice(&bytes).unwrap();
        let expert_bytes = LoadedQwen36Expert::canonical_stored_bytes();
        for capacity_mib in [256u64, 512, 768, 1024] {
            for policy in [
                CachePolicyKind::Lru,
                CachePolicyKind::Lfu,
                CachePolicyKind::DecayedCostAware,
            ] {
                let result = replay_trace(
                    &trace,
                    ReplayConfig {
                        capacity: Bytes(capacity_mib * 1024 * 1024),
                        policy,
                        half_life_events: 160,
                    },
                    |_, _| Ok(expert_bytes),
                )
                .unwrap();
                println!(
                    "policy={policy:?} capacity_mib={capacity_mib} result={}",
                    serde_json::to_string(&result).unwrap()
                );
            }
        }
    }
}
