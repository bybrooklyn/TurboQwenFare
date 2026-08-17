//! MoE reference path and global whole-expert cache/streaming baseline.
//! Adaptive admission and parallel I/O remain later benchmark-gated phases.
//! Phase 14 intentionally starts with every routed expert resident so the
//! router and expert math have an unambiguous correctness oracle before any
//! I/O/cache policy can complicate failures (spec §286, §149-151).

pub mod policy;

use crate::backend::reference::{q4k_gemv, sigmoid, silu};
use crate::dev::inventory::TensorRole;
use crate::error::{ModelError, Result};
use crate::ids::{Bytes, ExpertId, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};
use crate::model::qwen36::geometry::Qwen36Geometry;
use crate::model::qwen36::weights::{
    LoadedQwen36Expert, LoadedQwen36Tensor, Qwen36Activation, Qwen36WeightLoader,
};

const HIDDEN: usize = Qwen36Geometry::HIDDEN_SIZE;
const EXPERT_WIDTH: usize = Qwen36Geometry::ROUTED_EXPERT_WIDTH;
const NUM_EXPERTS: usize = Qwen36Geometry::NUM_EXPERTS;
const TOP_K: usize = Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN;
const Q4K_BLOCK_BYTES: usize = 144;

fn q4_bytes(rows: usize, cols: usize) -> usize {
    rows * (cols / 256) * Q4K_BLOCK_BYTES
}

fn require_len(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::Shape {
            tensor: name,
            expected,
            actual,
        }
        .into())
    }
}

/// Exact top-8 router result. `ids` identify storage/compute only; no
/// predictor is permitted to mutate them (spec invariant #7).
#[derive(Debug, Clone, PartialEq)]
pub struct RouterResult {
    pub ids: [ExpertId; TOP_K],
    pub weights: [f32; TOP_K],
}

impl RouterResult {
    /// FP32 softmax followed by a stable descending top-k. Equal logits break
    /// toward smaller expert IDs, which makes reference tests deterministic.
    pub fn from_logits(logits: &[f32]) -> Result<Self> {
        require_len("router logits", logits.len(), NUM_EXPERTS)?;
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(ModelError::Unsupported(
                "exact router logits must all be finite".to_string(),
            )
            .into());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let normalizer = probabilities.iter().sum::<f32>();
        let mut ranked: Vec<(usize, f32)> = probabilities
            .into_iter()
            .enumerate()
            .map(|(id, p)| (id, p / normalizer))
            .collect();
        ranked.sort_by(|(left_id, left), (right_id, right)| {
            right.total_cmp(left).then_with(|| left_id.cmp(right_id))
        });
        let selected = &ranked[..TOP_K];
        let selected_total = selected
            .iter()
            .map(|(_, probability)| probability)
            .sum::<f32>();
        let mut ids = [ExpertId(0); TOP_K];
        let mut weights = [0.0; TOP_K];
        for (slot, &(id, probability)) in selected.iter().enumerate() {
            ids[slot] = ExpertId(id as u16);
            weights[slot] = probability / selected_total;
        }
        let route = Self { ids, weights };
        route.validate()?;
        Ok(route)
    }

    fn validate(&self) -> Result<()> {
        let mut seen = [false; NUM_EXPERTS];
        let mut total = 0.0f32;
        for (&id, &weight) in self.ids.iter().zip(&self.weights) {
            let index = id.0 as usize;
            if index >= NUM_EXPERTS || seen[index] {
                return Err(ModelError::Unsupported(
                    "exact router plan must contain eight distinct canonical expert IDs"
                        .to_string(),
                )
                .into());
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(ModelError::Unsupported(
                    "exact router weights must be finite and non-negative".to_string(),
                )
                .into());
            }
            seen[index] = true;
            total += weight;
        }
        if !total.is_finite() || (total - 1.0).abs() > 1e-4 {
            return Err(ModelError::Unsupported(
                "exact router weights must sum to one".to_string(),
            )
            .into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_bytes: Bytes,
    /// Bytes demanded from SSD on cache misses, before any read-ahead or
    /// compression policy. This is the Phase-18 baseline metric.
    pub raw_miss_bytes: Bytes,
}

impl Default for ExpertCacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            resident_bytes: Bytes(0),
            raw_miss_bytes: Bytes(0),
        }
    }
}

/// Immutable binding plan derived from Qwen's real top-8 result. Cache
/// policy may decide where bytes live, but this plan deliberately carries the
/// router IDs and weights unchanged into execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertLoadPlan {
    pub plan_id: u64,
    pub layer: LayerId,
    pub route: RouterResult,
    pub hits: [bool; TOP_K],
    pub miss_bytes: Bytes,
}

struct CachedExpert {
    layer: LayerId,
    expert: ExpertId,
    frequency: u64,
    last_used: u64,
    pin_count: u16,
    value: LoadedQwen36Expert,
}

/// Global, whole-expert LFU cache. The cache plans from the *actual* router
/// result and never influences its IDs or weights. It has a dedicated byte
/// cap below the broker's global budget; entries retain their broker leases
/// and are dropped before a new expert allocation is attempted.
pub struct WholeExpertLfuCache {
    capacity: Bytes,
    resident_bytes: Bytes,
    clock: u64,
    next_plan_id: u64,
    active_plan: Option<ExpertLoadPlan>,
    stats: ExpertCacheStats,
    entries: Vec<CachedExpert>,
}

impl WholeExpertLfuCache {
    pub fn new(capacity: Bytes) -> Self {
        Self {
            capacity,
            resident_bytes: Bytes(0),
            clock: 0,
            next_plan_id: 1,
            active_plan: None,
            stats: ExpertCacheStats::default(),
            entries: Vec::new(),
        }
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let mut stats = self.stats;
        stats.resident_bytes = self.resident_bytes;
        stats
    }

    /// Plans an exact route before cache-miss allocations. This gives the
    /// later I/O queue a stable, inspectable demand list without ever letting
    /// a predictor substitute an expert ID or router weight.
    pub fn plan_exact_route(
        &self,
        loader: &Qwen36WeightLoader,
        layer: LayerId,
        route: &RouterResult,
    ) -> Result<ExpertLoadPlan> {
        route.validate()?;
        let mut hits = [false; TOP_K];
        let mut miss_bytes = 0u64;
        for (slot, expert) in route.ids.iter().copied().enumerate() {
            let hit = self
                .entries
                .iter()
                .any(|entry| entry.layer == layer && entry.expert == expert);
            hits[slot] = hit;
            if !hit {
                miss_bytes =
                    miss_bytes.saturating_add(loader.expert_stored_bytes(layer, expert)?.0);
            }
        }
        Ok(ExpertLoadPlan {
            plan_id: 0,
            layer,
            route: route.clone(),
            hits,
            miss_bytes: Bytes(miss_bytes),
        })
    }

    /// Materializes one immutable exact-router transaction before any routed
    /// computation begins. All selected hits are pinned and all selected
    /// misses are loaded before the plan becomes active, so an entry named by
    /// the plan cannot be evicted between routing and its final use.
    pub(crate) fn prepare_exact_route(
        &mut self,
        loader: &Qwen36WeightLoader,
        layer: LayerId,
        route: &RouterResult,
    ) -> Result<ExpertLoadPlan> {
        if self.active_plan.is_some() {
            return Err(crate::error::InternalError {
                incident_id: "expert-plan-overlap".to_string(),
                message: "whole-expert baseline permits only one active exact route".to_string(),
            }
            .into());
        }

        let mut plan = self.plan_exact_route(loader, layer, route)?;
        let misses = plan
            .route
            .ids
            .iter()
            .copied()
            .zip(plan.hits)
            .filter_map(|(expert, hit)| (!hit).then_some(expert))
            .collect::<Vec<_>>();
        let required = misses.iter().try_fold(0u64, |bytes, &expert| {
            Ok::<_, crate::error::TqfError>(
                bytes.saturating_add(loader.expert_stored_bytes(layer, expert)?.0),
            )
        })?;
        if required > self.capacity.0 {
            return Err(ModelError::Unsupported(format!(
                "expert cache capacity {} bytes cannot hold the {} missing experts in one exact route ({} bytes)",
                self.capacity.0,
                misses.len(),
                required
            ))
            .into());
        }

        while self.resident_bytes.0.saturating_add(required) > self.capacity.0 {
            if self
                .evict_coldest_excluding(layer, &plan.route.ids)
                .is_none()
            {
                return Err(ModelError::Unsupported(format!(
                    "expert cache capacity {} bytes cannot pin all eight experts in one exact route",
                    self.capacity.0
                ))
                .into());
            }
        }

        // Stage every miss first. A failed read drops all staged broker leases
        // and leaves no partially valid cache entry behind.
        let mut staged = Vec::with_capacity(misses.len());
        for expert in misses {
            staged.push((expert, loader.load_expert(layer, expert)?));
        }

        let plan_id = self.next_plan_id;
        self.next_plan_id = self.next_plan_id.wrapping_add(1).max(1);
        plan.plan_id = plan_id;
        for (&expert, hit) in plan.route.ids.iter().zip(plan.hits) {
            self.clock = self.clock.wrapping_add(1);
            if hit {
                let entry = self
                    .entries
                    .iter_mut()
                    .find(|entry| entry.layer == layer && entry.expert == expert)
                    .ok_or_else(|| crate::error::InternalError {
                        incident_id: format!("expert-plan-{plan_id}-lost-hit"),
                        message: "exact-route hit disappeared before plan activation".to_string(),
                    })?;
                entry.frequency = entry.frequency.saturating_add(1);
                entry.last_used = self.clock;
                entry.pin_count = entry.pin_count.saturating_add(1);
                self.stats.hits = self.stats.hits.saturating_add(1);
                continue;
            }
            let index = staged
                .iter()
                .position(|(id, _)| *id == expert)
                .ok_or_else(|| crate::error::InternalError {
                    incident_id: format!("expert-plan-{plan_id}-unstaged-miss"),
                    message: "exact-route miss was not present in the staged transaction"
                        .to_string(),
                })?;
            let (_, value) = staged.swap_remove(index);
            let stored_bytes = value.stored_bytes();
            self.resident_bytes.0 = self.resident_bytes.0.saturating_add(stored_bytes.0);
            self.stats.misses = self.stats.misses.saturating_add(1);
            self.stats.raw_miss_bytes.0 =
                self.stats.raw_miss_bytes.0.saturating_add(stored_bytes.0);
            self.entries.push(CachedExpert {
                layer,
                expert,
                frequency: 1,
                last_used: self.clock,
                pin_count: 1,
                value,
            });
        }
        debug_assert!(staged.is_empty());
        self.active_plan = Some(plan.clone());
        Ok(plan)
    }

    pub(crate) fn planned_expert(
        &self,
        plan: &ExpertLoadPlan,
        expert: ExpertId,
    ) -> Result<&LoadedQwen36Expert> {
        if self.active_plan.as_ref() != Some(plan) || !plan.route.ids.contains(&expert) {
            return Err(crate::error::InternalError {
                incident_id: format!("expert-plan-{}-binding", plan.plan_id),
                message: "routed computation requested an expert outside its active exact plan"
                    .to_string(),
            }
            .into());
        }
        self.entries
            .iter()
            .find(|entry| {
                entry.layer == plan.layer && entry.expert == expert && entry.pin_count > 0
            })
            .map(|entry| &entry.value)
            .ok_or_else(|| {
                crate::error::InternalError {
                    incident_id: format!("expert-plan-{}-missing", plan.plan_id),
                    message: "active exact plan lost a pinned expert binding".to_string(),
                }
                .into()
            })
    }

    pub(crate) fn finish_exact_route(&mut self, plan: &ExpertLoadPlan) -> Result<()> {
        if self.active_plan.as_ref() != Some(plan) {
            return Err(crate::error::InternalError {
                incident_id: format!("expert-plan-{}-finish", plan.plan_id),
                message: "attempted to finish a route that is not active".to_string(),
            }
            .into());
        }
        for expert in plan.route.ids {
            let entry = self
                .entries
                .iter_mut()
                .find(|entry| entry.layer == plan.layer && entry.expert == expert)
                .ok_or_else(|| crate::error::InternalError {
                    incident_id: format!("expert-plan-{}-lost-pin", plan.plan_id),
                    message: "active exact plan lost a selected expert before release".to_string(),
                })?;
            entry.pin_count = entry.pin_count.saturating_sub(1);
        }
        self.active_plan = None;
        Ok(())
    }

    pub fn get_or_load(
        &mut self,
        loader: &Qwen36WeightLoader,
        layer: LayerId,
        expert: ExpertId,
    ) -> Result<&LoadedQwen36Expert> {
        self.clock = self.clock.wrapping_add(1);
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.layer == layer && entry.expert == expert)
        {
            let entry = &mut self.entries[index];
            entry.frequency = entry.frequency.saturating_add(1);
            entry.last_used = self.clock;
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(&entry.value);
        }

        let required = loader.expert_stored_bytes(layer, expert)?;
        if required > self.capacity {
            return Err(ModelError::Unsupported(format!(
                "expert cache capacity {} bytes cannot hold one {} byte Qwen expert",
                self.capacity.0, required.0
            ))
            .into());
        }
        while self.resident_bytes.0.saturating_add(required.0) > self.capacity.0 {
            if self.evict_coldest().is_none() {
                return Err(ModelError::Unsupported(
                    "all whole-expert cache entries are pinned by an active exact route"
                        .to_string(),
                )
                .into());
            }
        }
        // `load_expert` acquires its broker lease before allocating the
        // destination Vec. All cache evictions above happened first.
        let value = loader.load_expert(layer, expert)?;
        self.stats.misses = self.stats.misses.saturating_add(1);
        self.stats.raw_miss_bytes.0 = self.stats.raw_miss_bytes.0.saturating_add(required.0);
        self.resident_bytes.0 = self.resident_bytes.0.saturating_add(required.0);
        self.entries.push(CachedExpert {
            layer,
            expert,
            frequency: 1,
            last_used: self.clock,
            pin_count: 0,
            value,
        });
        Ok(&self.entries.last().expect("just pushed cache entry").value)
    }

    fn evict_coldest(&mut self) -> Option<()> {
        let (index, _) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.pin_count == 0)
            .min_by_key(|(_, entry)| {
                (entry.frequency, entry.last_used, entry.layer, entry.expert)
            })?;
        self.remove_entry(index);
        Some(())
    }

    fn evict_coldest_excluding(
        &mut self,
        selected_layer: LayerId,
        selected_experts: &[ExpertId; TOP_K],
    ) -> Option<()> {
        let (index, _) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.pin_count == 0
                    && (entry.layer != selected_layer || !selected_experts.contains(&entry.expert))
            })
            .min_by_key(|(_, entry)| {
                (entry.frequency, entry.last_used, entry.layer, entry.expert)
            })?;
        self.remove_entry(index);
        Some(())
    }

    fn remove_entry(&mut self, index: usize) {
        let entry = self.entries.swap_remove(index);
        self.resident_bytes.0 = self
            .resident_bytes
            .0
            .saturating_sub(entry.value.stored_bytes().0);
        drop(entry);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
    }
}

/// Q4 tensors for one SwiGLU-style MoE expert. `gate`/`up` are [512, 2048],
/// `down` is [2048, 512], all row-major Q4_K.
pub struct ResidentExpert<'a> {
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
}

impl<'a> ResidentExpert<'a> {
    fn validate(&self, name: &'static str) -> Result<()> {
        require_len(name, self.gate.len(), q4_bytes(EXPERT_WIDTH, HIDDEN))?;
        require_len(name, self.up.len(), q4_bytes(EXPERT_WIDTH, HIDDEN))?;
        require_len(name, self.down.len(), q4_bytes(HIDDEN, EXPERT_WIDTH))?;
        Ok(())
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let gate = q4k_gemv(self.gate, input, EXPERT_WIDTH, HIDDEN);
        let up = q4k_gemv(self.up, input, EXPERT_WIDTH, HIDDEN);
        let activated: Vec<f32> = silu(&gate)
            .into_iter()
            .zip(up)
            .map(|(gate, up)| gate * up)
            .collect();
        q4k_gemv(self.down, &activated, HIDDEN, EXPERT_WIDTH)
    }
}

/// Resident correctness profile for a whole MoE block. Borrowed weight bytes
/// are owned by the model loader; that loader must reserve their backing
/// lease before constructing this view. This module reserves its own transient
/// activation space before running the reference computation.
pub struct ResidentMoe<'a> {
    pub router: &'a [u8],
    pub shared: ResidentExpert<'a>,
    pub shared_gate: &'a [f32],
    pub routed: Vec<ResidentExpert<'a>>,
}

impl<'a> ResidentMoe<'a> {
    pub fn validate(&self) -> Result<()> {
        require_len("router", self.router.len(), q4_bytes(NUM_EXPERTS, HIDDEN))?;
        require_len("shared_expert_gate", self.shared_gate.len(), HIDDEN)?;
        self.shared.validate("shared expert")?;
        require_len("resident routed experts", self.routed.len(), NUM_EXPERTS)?;
        for expert in &self.routed {
            expert.validate("routed expert")?;
        }
        Ok(())
    }

    /// Reference MoE ordering: exact router -> shared expert -> all selected
    /// routed experts -> weighted accumulation. The returned route is exposed
    /// for parity dumps and future `--router-trace` diagnostics.
    pub fn forward(
        &self,
        broker: &MemoryBroker,
        input: &[f32],
    ) -> Result<(Vec<f32>, RouterResult)> {
        require_len("MoE input", input.len(), HIDDEN)?;
        self.validate()?;
        // router + two intermediate vectors + output, with enough headroom
        // for one routed expert; selected experts execute serially in this
        // correctness profile, so they do not need eight simultaneous buffers.
        let scratch_bytes = Bytes(((NUM_EXPERTS + 2 * EXPERT_WIDTH + 3 * HIDDEN) * 4) as u64);
        let _scratch = broker.reserve(
            MemoryOwner::Scratch,
            MemoryClass::Transient,
            scratch_bytes,
            64,
        )?;
        let route = RouterResult::from_logits(&q4k_gemv(self.router, input, NUM_EXPERTS, HIDDEN))?;

        let mut output = self.shared.forward(input);
        let shared_gate_logit: f32 = self
            .shared_gate
            .iter()
            .zip(input)
            .map(|(weight, value)| weight * value)
            .sum();
        let shared_scale = sigmoid(&[shared_gate_logit])[0];
        for value in &mut output {
            *value *= shared_scale;
        }
        for (&id, &weight) in route.ids.iter().zip(route.weights.iter()) {
            let expert_output = self.routed[id.0 as usize].forward(input);
            for (output, expert) in output.iter_mut().zip(expert_output) {
                *output += weight * expert;
            }
        }
        Ok((output, route))
    }
}

/// Canonical Qwen3.6 resident-MoE binding used by the Phase 14 development
/// profile.  This retains the original GGUF storage types: the router and
/// shared gate are FP32 in the official Q4_K_M release, projections may be
/// Q8_0, and the routed expert planes are Q4_K.  It therefore deliberately
/// does not reuse the older all-Q4K micro-kernel fixture above.
///
/// Holding this value intentionally pins three complete routed-expert tensors
/// for one layer.  It is a high-memory correctness oracle only; the Phase 18
/// cache will replace its retention policy while preserving the same exact
/// router result and matvec semantics.
pub struct Qwen36ResidentMoe {
    router: LoadedQwen36Tensor,
    shared_input_gate: LoadedQwen36Tensor,
    shared_gate: LoadedQwen36Tensor,
    shared_up: LoadedQwen36Tensor,
    shared_down: LoadedQwen36Tensor,
    routed_gate: LoadedQwen36Tensor,
    routed_up: LoadedQwen36Tensor,
    routed_down: LoadedQwen36Tensor,
}

/// Streaming equivalent of the resident Phase-14 oracle. Resident shared
/// weights still have ordinary typed extents; routed weights are obtained
/// only after exact routing, then cached by whole expert.
pub struct Qwen36StreamingMoe {
    router: LoadedQwen36Tensor,
    shared_input_gate: LoadedQwen36Tensor,
    shared_gate: LoadedQwen36Tensor,
    shared_up: LoadedQwen36Tensor,
    shared_down: LoadedQwen36Tensor,
    layer: LayerId,
}

impl Qwen36StreamingMoe {
    pub fn open(loader: &Qwen36WeightLoader, layer: LayerId) -> Result<Self> {
        Ok(Self {
            router: loader.load(TensorRole::RouterGate, Some(layer))?,
            shared_input_gate: loader.load(TensorRole::SharedExpertInputGate, Some(layer))?,
            shared_gate: loader.load(TensorRole::SharedExpertGate, Some(layer))?,
            shared_up: loader.load(TensorRole::SharedExpertUp, Some(layer))?,
            shared_down: loader.load(TensorRole::SharedExpertDown, Some(layer))?,
            layer,
        })
    }

    pub fn forward(
        &mut self,
        loader: &Qwen36WeightLoader,
        cache: &mut WholeExpertLfuCache,
        broker: &MemoryBroker,
        input: &Qwen36Activation,
    ) -> Result<(Qwen36Activation, RouterResult)> {
        self.forward_with_observer(loader, cache, broker, input, |_, _| Ok(()))
    }

    pub fn forward_with_observer<F>(
        &mut self,
        loader: &Qwen36WeightLoader,
        cache: &mut WholeExpertLfuCache,
        broker: &MemoryBroker,
        input: &Qwen36Activation,
        mut observer: F,
    ) -> Result<(Qwen36Activation, RouterResult)>
    where
        F: FnMut(&'static str, &Qwen36Activation) -> Result<()>,
    {
        require_len("Qwen streaming MoE input", input.values.len(), HIDDEN)?;
        let route_logits = self.router.matvec(broker, &input.values)?;
        let route = RouterResult::from_logits(&route_logits.values)?;
        let plan = cache.prepare_exact_route(loader, self.layer, &route)?;
        let computed = (|| -> Result<Qwen36Activation> {
            let shared_gate = self.shared_gate.matvec(broker, &input.values)?;
            let shared_up = self.shared_up.matvec(broker, &input.values)?;
            let shared_hidden = Qwen36Activation::silu_mul(broker, &shared_gate, &shared_up)?;
            let mut output = self.shared_down.matvec(broker, &shared_hidden.values)?;
            let shared_gate_logit = self.shared_input_gate.dot(broker, &input.values)?;
            output.scale_in_place(1.0 / (1.0 + (-shared_gate_logit).exp()));
            observer("shared", &output)?;

            for (&expert, &weight) in plan.route.ids.iter().zip(&plan.route.weights) {
                let routed = cache
                    .planned_expert(&plan, expert)?
                    .forward(broker, &input.values)?;
                output.add_scaled_in_place(&routed, weight)?;
            }
            observer("combined", &output)?;
            Ok(output)
        })();
        // Release pins on both success and failure. The first computation
        // error remains authoritative after the cache returns to Ready state.
        let release = cache.finish_exact_route(&plan);
        let output = computed?;
        release?;
        Ok((output, plan.route))
    }
}

impl Qwen36ResidentMoe {
    /// Loads all fixed MoE tensors for one layer.  `Qwen36WeightLoader`
    /// reserves each extent before allocating and reading it; retaining the
    /// fields keeps those broker leases alive throughout the reference path.
    pub fn open(loader: &Qwen36WeightLoader, layer: crate::ids::LayerId) -> Result<Self> {
        Ok(Self {
            router: loader.load(TensorRole::RouterGate, Some(layer))?,
            shared_input_gate: loader.load(TensorRole::SharedExpertInputGate, Some(layer))?,
            shared_gate: loader.load(TensorRole::SharedExpertGate, Some(layer))?,
            shared_up: loader.load(TensorRole::SharedExpertUp, Some(layer))?,
            shared_down: loader.load(TensorRole::SharedExpertDown, Some(layer))?,
            routed_gate: loader.load(TensorRole::RoutedExpertGate, Some(layer))?,
            routed_up: loader.load(TensorRole::RoutedExpertUp, Some(layer))?,
            routed_down: loader.load(TensorRole::RoutedExpertDown, Some(layer))?,
        })
    }

    /// Executes the exact Qwen MoE computation in canonical order:
    /// router softmax/top-8, gated shared expert, then selected routed SwiGLU
    /// experts accumulated with their renormalized router weights.  Every
    /// temporary activation is broker-accounted by the typed tensor helpers.
    pub fn forward(
        &self,
        broker: &MemoryBroker,
        input: &Qwen36Activation,
    ) -> Result<(Qwen36Activation, RouterResult)> {
        require_len("Qwen MoE input", input.values.len(), HIDDEN)?;
        let route_logits = self.router.matvec(broker, &input.values)?;
        let route = RouterResult::from_logits(&route_logits.values)?;

        let shared_gate = self.shared_gate.matvec(broker, &input.values)?;
        let shared_up = self.shared_up.matvec(broker, &input.values)?;
        let shared_hidden = Qwen36Activation::silu_mul(broker, &shared_gate, &shared_up)?;
        let mut output = self.shared_down.matvec(broker, &shared_hidden.values)?;

        let shared_gate_logit = self.shared_input_gate.dot(broker, &input.values)?;
        output.scale_in_place(1.0 / (1.0 + (-shared_gate_logit).exp()));

        for (&expert, &weight) in route.ids.iter().zip(&route.weights) {
            let expert = expert.0 as usize;
            let gate = self
                .routed_gate
                .matvec_expert(broker, expert, &input.values)?;
            let up = self
                .routed_up
                .matvec_expert(broker, expert, &input.values)?;
            let hidden = Qwen36Activation::silu_mul(broker, &gate, &up)?;
            let routed = self
                .routed_down
                .matvec_expert(broker, expert, &hidden.values)?;
            output.add_scaled_in_place(&routed, weight)?;
        }
        Ok((output, route))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_top_k_is_stable_and_renormalized() {
        let mut logits = vec![-10.0; NUM_EXPERTS];
        logits[21] = 4.0;
        logits[3] = 4.0;
        logits[8] = 3.0;
        let route = RouterResult::from_logits(&logits).unwrap();
        assert_eq!(route.ids[0], ExpertId(3));
        assert_eq!(route.ids[1], ExpertId(21));
        assert!((route.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(route.weights[0] > route.weights[2]);
    }

    #[test]
    fn router_rejects_noncanonical_expert_count() {
        assert!(RouterResult::from_logits(&[0.0; NUM_EXPERTS - 1]).is_err());
    }

    #[test]
    fn router_rejects_nonfinite_logits_and_malformed_exact_plans() {
        let mut logits = [0.0; NUM_EXPERTS];
        logits[7] = f32::NAN;
        assert!(RouterResult::from_logits(&logits).is_err());

        let route = RouterResult {
            ids: [ExpertId(3); TOP_K],
            weights: [0.125; TOP_K],
        };
        assert!(route.validate().is_err());
    }
}
