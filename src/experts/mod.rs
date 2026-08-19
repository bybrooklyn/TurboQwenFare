//! MoE reference path and global whole-expert cache/streaming baseline.
//! Adaptive admission and parallel I/O remain later benchmark-gated phases.
//! Phase 14 intentionally starts with every routed expert resident so the
//! router and expert math have an unambiguous correctness oracle before any
//! I/O/cache policy can complicate failures (spec §286, §149-151).
//!
//! Phase 20 (spec §112 row 20) adds an A/B-able GPU-resident expert path:
//! with `TQF_EXPERT_GPU_RESIDENT=1` (or `WholeExpertLfuCache::set_gpu_enabled`)
//! a freshly loaded expert is uploaded once into broker-registered Metal
//! buffers (`backend::metal::GpuResidentExpert`) and the CPU copy is dropped,
//! so the same Q4_K bytes are charged to the broker exactly once. The GPU
//! path is a developer A/B control (invariant #10): absent a Metal device it
//! falls back to the CPU baseline without changing results.

pub mod policy;
pub mod prefetch;
pub mod tiling;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "metal")]
use crate::backend::metal::{GpuExecutionState, GpuResidentExpert};
use crate::backend::reference::{q4k_gemv, sigmoid, silu};
use crate::dev::inventory::TensorRole;
use crate::error::{ModelError, Result};
use crate::experts::policy::CachePolicyKind;
use crate::ids::{Bytes, ExpertId, LayerId};
use crate::io::ReadFanout;
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};
use crate::model::qwen36::geometry::Qwen36Geometry;
use crate::model::qwen36::weights::{
    LoadedQwen36Expert, LoadedQwen36Tensor, Qwen36Activation, Qwen36WeightLoader,
};

/// Placeholder so the cache's spawn/forward API compiles identically on
/// non-Metal backends; it is never constructed there.
#[cfg(not(feature = "metal"))]
pub(crate) type GpuExecutionState = ();

/// The backing store for one routed expert in the cache. `Cpu` keeps the
/// Phase 18 Q4_K payload bytes; `Gpu` (Phase 20, metal only) keeps the same
/// bytes uploaded once into persistent Metal buffers. Both compute the same
/// SwiGLU forward — the CPU and GPU variants must stay result-identical -
/// and both charge the same `stored_bytes` to the broker (a GPU value is
/// produced by moving the payload, never by copying it alongside the CPU
/// bytes).
pub enum ExpertValue {
    Cpu(LoadedQwen36Expert),
    #[cfg(feature = "metal")]
    Gpu(GpuResidentExpert),
}

impl ExpertValue {
    pub fn stored_bytes(&self) -> Bytes {
        match self {
            ExpertValue::Cpu(cpu) => cpu.stored_bytes(),
            #[cfg(feature = "metal")]
            ExpertValue::Gpu(gpu) => gpu.stored_bytes(),
        }
    }

    /// Executes the canonical SwiGLU expert forward on the backing store
    /// this value was admitted with. GPU values require the cache's GPU
    /// execution state (device/library/pipeline cache) to still exist;
    /// losing it mid-route is a violated internal invariant.
    #[cfg(feature = "metal")]
    pub fn forward(
        &self,
        broker: &MemoryBroker,
        gpu: Option<&mut GpuExecutionState>,
        input: &[f32],
    ) -> Result<Qwen36Activation> {
        match self {
            ExpertValue::Cpu(cpu) => cpu.forward(broker, input),
            ExpertValue::Gpu(expert) => {
                let state = gpu.ok_or_else(|| crate::error::InternalError {
                    incident_id: "expert-gpu-state-missing".to_string(),
                    message: "GPU-resident expert value without GPU execution state".to_string(),
                })?;
                let values = expert.forward(state, input)?;
                Qwen36Activation::from_slice(broker, &values)
            }
        }
    }

    #[cfg(not(feature = "metal"))]
    pub fn forward(
        &self,
        broker: &MemoryBroker,
        _gpu: Option<&mut GpuExecutionState>,
        input: &[f32],
    ) -> Result<Qwen36Activation> {
        match self {
            ExpertValue::Cpu(cpu) => cpu.forward(broker, input),
        }
    }
}

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
    /// Phase 23 (spec §295): speculative prefetch accounting.
    pub prefetched: u64,
    pub prefetch_hits: u64,
    pub prefetch_wasted_bytes: Bytes,
    pub prefetch_late: u64,
    /// Phase 25 (spec §297): wall time spent inside demand expert reads
    /// (the SSD-stall component of decode), for the optimization ledger.
    pub demand_io_nanos: u128,
}

impl Default for ExpertCacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            resident_bytes: Bytes(0),
            raw_miss_bytes: Bytes(0),
            prefetched: 0,
            prefetch_hits: 0,
            prefetch_wasted_bytes: Bytes(0),
            prefetch_late: 0,
            demand_io_nanos: 0,
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
    /// Phase 23: entry delivered speculatively by the prefetch path. It
    /// becomes a normal entry on first pin (which also counts the
    /// prefetch hit); probation entries evict before any demand entry.
    probation: bool,
    arrival_clock: u64,
    value: ExpertValue,
}

/// Phase 26 (spec §298): a layer/chunk prefill plan pins the *union* of
/// several exact routes so one distinct expert is fetched at most once
/// for the whole chunk, then re-used by every row that selected it.
/// Router IDs and weights stay exactly as the real router produced them
/// (invariant #7); this only changes fetch scheduling.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchExpertPlan {
    pub plan_id: u64,
    pub layer: LayerId,
    /// Distinct experts demanded by the chunk, in first-demand order.
    pub distinct: Vec<ExpertId>,
    /// Per-row route results, unchanged.
    pub routes: Vec<RouterResult>,
}

/// A speculative expert load completed by the Phase 23 prefetch worker,
/// waiting to be drained into the cache. The `LoadedQwen36Expert` owns
/// its broker lease, so the destination outlives the read (invariant #5)
/// even while the entry sits in this queue.
struct PrefetchDelivery {
    layer: LayerId,
    expert: ExpertId,
    value: LoadedQwen36Expert,
    issued_clock: u64,
}

/// Global, whole-expert cache. The cache plans from the *actual* router
/// result and never influences its IDs or weights. It has a dedicated byte
/// cap below the broker's global budget; entries retain their broker leases
/// and are dropped before a new expert allocation is attempted.
///
/// Eviction uses a pluggable `CachePolicyKind` (Phase 21, spec §112 row 21).
/// The type keeps its original name for continuity with the Phase 15-18
/// qualification history (cache policy only changes I/O volume, never a
/// computed result, so those correctness qualifications remain valid under
/// any policy) even though the *default* is no longer LFU: a real 128-token
/// route-trace replay (`docs/research/qualification/raw-a-128-route-trace-policy.md`)
/// measured LRU beating LFU by a wide margin, so `DEFAULT_CACHE_POLICY` is
/// now `Lru`. The admission/eviction utility function is the same one
/// `policy::replay_trace` uses offline, so a future benchmark winner can
/// replace the default by changing that one constant.
pub struct WholeExpertLfuCache {
    capacity: Bytes,
    resident_bytes: Bytes,
    clock: u64,
    next_plan_id: u64,
    active_plan: Option<ExpertLoadPlan>,
    active_batch_plan: Option<BatchExpertPlan>,
    stats: ExpertCacheStats,
    entries: Vec<CachedExpert>,
    io_fanout: ReadFanout,
    policy: CachePolicyKind,
    half_life_events: u64,
    /// Phase 20 GPU-resident expert path. The state (device/library/pipeline
    /// cache) is created lazily on first use and shared behind a mutex so
    /// the cache stays `Send` even though Metal objects are not. `None`
    /// once the CPU fallback is selected (no device available).
    #[cfg(feature = "metal")]
    gpu_state: Option<Arc<Mutex<GpuExecutionState>>>,
    /// Whether new cache admissions should be uploaded to GPU buffers.
    #[cfg(feature = "metal")]
    gpu_enabled: bool,
    /// Phase 23 predictive prefetch (spec §295). Off by default until the
    /// offline replay and a live A/B record a net win; `TQF_PREFETCH_ENABLED`
    /// turns it on for that measurement (invariant #10).
    prefetch_enabled: bool,
    prefetch_depth: usize,
    predictor: crate::experts::prefetch::TransitionPredictor,
    previous_route: Option<(LayerId, Vec<ExpertId>)>,
    last_demand_clock: std::collections::HashMap<(LayerId, ExpertId), u64>,
    prefetch_inbox: Arc<Mutex<Vec<PrefetchDelivery>>>,
    prefetch_in_flight: Arc<AtomicUsize>,
}

/// Env var read once per cache so a deployment/benchmark can force the
/// Phase 18 serial baseline back on without a code change (spec invariant
/// #10). Unset or unparseable values keep the Phase 19 parallel default.
const IO_FANOUT_ENV: &str = "TQF_EXPERT_IO_FANOUT";

/// Env var selecting the eviction policy (`lru`, `lfu`, `decayed-cost-aware`)
/// for the same A/B purpose. Unset or unparseable values keep the Phase
/// 15-18-proven LFU default.
const CACHE_POLICY_ENV: &str = "TQF_EXPERT_CACHE_POLICY";

/// Env var enabling the Phase 20 GPU-resident expert path (`1`/`on`/`true`).
/// This is the out-of-process A/B control (invariant #10); `set_gpu_enabled`
/// is the in-process one. Without a usable Metal device the flag silently
/// falls back to the CPU baseline rather than failing decode.
const EXPERT_GPU_ENV: &str = "TQF_EXPERT_GPU_RESIDENT";

/// Phase 23 A/B controls (spec §295, invariant #10): prefetch is off by
/// default until the offline replay and a live A/B record a net win.
/// `TQF_PREFETCH_DEPTH` bounds how many experts one prediction submits.
const PREFETCH_ENABLED_ENV: &str = "TQF_PREFETCH_ENABLED";
const PREFETCH_DEPTH_ENV: &str = "TQF_PREFETCH_DEPTH";
const DEFAULT_PREFETCH_DEPTH: usize = TOP_K;
/// At most one speculative batch in flight; more would contend with
/// demand reads on a single SSD queue without a measured win.
const MAX_PREFETCH_BATCHES_IN_FLIGHT: usize = 1;

fn env_enabled(var: &str) -> bool {
    match std::env::var(var).ok().as_deref() {
        Some(value) => {
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("true")
        }
        None => false,
    }
}

fn prefetch_depth_from_env() -> usize {
    std::env::var(PREFETCH_DEPTH_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PREFETCH_DEPTH)
        .min(TOP_K)
}

/// Default decay half-life for `DecayedCostAware`, in cache-route events;
/// matches the sweep in `policy::tests::qualification_trace_replays_all_phase21_policy_candidates`.
const DEFAULT_HALF_LIFE_EVENTS: u64 = 160;

/// Phase 21 benchmark-selected default (spec §112 row 21, "the benchmark
/// winner becomes default"): a real 128-token/40-layer route trace
/// (`docs/research/qualification/raw-a-128-route-trace-policy.md`) shows LRU
/// beating LFU by a wide margin at every cache capacity large enough to get
/// any reuse at all (e.g. 1 GiB: 35.9 GB vs 56.1 GB raw miss bytes over the
/// same 128-token run - a 36% reduction), with `DecayedCostAware` a close
/// second. LFU was the Phase 15-18 placeholder, not a measured choice.
const DEFAULT_CACHE_POLICY: CachePolicyKind = CachePolicyKind::Lru;

fn cache_policy_from_env() -> CachePolicyKind {
    match std::env::var(CACHE_POLICY_ENV).ok().as_deref() {
        Some(value) if value.eq_ignore_ascii_case("lfu") => CachePolicyKind::Lfu,
        Some(value) if value.eq_ignore_ascii_case("decayed-cost-aware") => {
            CachePolicyKind::DecayedCostAware
        }
        Some(value) if value.eq_ignore_ascii_case("lru") => CachePolicyKind::Lru,
        _ => DEFAULT_CACHE_POLICY,
    }
}

#[cfg(feature = "metal")]
fn gpu_enabled_from_env() -> bool {
    env_enabled(EXPERT_GPU_ENV)
}

impl WholeExpertLfuCache {
    pub fn new(capacity: Bytes) -> Self {
        Self {
            capacity,
            resident_bytes: Bytes(0),
            clock: 0,
            next_plan_id: 1,
            active_plan: None,
            active_batch_plan: None,
            stats: ExpertCacheStats::default(),
            entries: Vec::new(),
            io_fanout: ReadFanout::from_env(IO_FANOUT_ENV)
                .unwrap_or_else(ReadFanout::parallel_default),
            policy: cache_policy_from_env(),
            half_life_events: DEFAULT_HALF_LIFE_EVENTS,
            #[cfg(feature = "metal")]
            gpu_state: None,
            #[cfg(feature = "metal")]
            gpu_enabled: gpu_enabled_from_env(),
            prefetch_enabled: env_enabled(PREFETCH_ENABLED_ENV),
            prefetch_depth: prefetch_depth_from_env(),
            predictor: crate::experts::prefetch::TransitionPredictor::new(),
            previous_route: None,
            last_demand_clock: std::collections::HashMap::new(),
            prefetch_inbox: Arc::new(Mutex::new(Vec::new())),
            prefetch_in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Overrides the fan-out policy chosen at construction time. Exists for
    /// qualification/benchmark harnesses that A/B the Phase 19 parallel path
    /// against the Phase 18 serial baseline within one process.
    pub fn set_io_fanout(&mut self, fanout: ReadFanout) {
        self.io_fanout = fanout;
    }

    /// Overrides the eviction policy chosen at construction time. Same
    /// purpose as `set_io_fanout`: qualification/benchmark harnesses A/B
    /// Phase 21 candidates within one process.
    pub fn set_policy(&mut self, policy: CachePolicyKind, half_life_events: u64) {
        self.policy = policy;
        self.half_life_events = half_life_events.max(1);
    }

    /// Overrides the Phase 20 GPU-resident expert choice made (from
    /// `TQF_EXPERT_GPU_RESIDENT`) at construction time. Mirrors
    /// `set_io_fanout`/`set_policy`: the Phase 20 A/B must be runnable
    /// entirely within one process. Disabling keeps any already-resident
    /// GPU values (they stay correct to forward); it only stops new
    /// admissions from taking the GPU path.
    #[cfg(feature = "metal")]
    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.gpu_enabled = enabled;
    }

    #[cfg(not(feature = "metal"))]
    pub fn set_gpu_enabled(&mut self, _enabled: bool) {}

    #[cfg(feature = "metal")]
    pub(crate) fn gpu_expert_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(entry.value, ExpertValue::Gpu(_)))
            .count()
    }

    #[cfg(not(feature = "metal"))]
    pub(crate) fn gpu_expert_count(&self) -> usize {
        0
    }

    /// Lazily creates the shared GPU execution state (device, compiled
    /// kernel library, pipeline cache) on first use. Failure means no
    /// usable Metal device: the flag is cleared and the cache stays on the
    /// CPU baseline, so an A/B run never turns into a crash on a headless
    /// or sandboxed machine.
    #[cfg(feature = "metal")]
    fn init_gpu_state_if_needed(&mut self) {
        if !self.gpu_enabled || self.gpu_state.is_some() {
            return;
        }
        match GpuExecutionState::init() {
            Ok(state) => self.gpu_state = Some(Arc::new(Mutex::new(state))),
            Err(error) => {
                tracing::warn!(%error, "GPU-resident expert path unavailable; keeping CPU baseline");
                self.gpu_enabled = false;
            }
        }
    }

    /// Converts freshly staged CPU expert payloads into cache values,
    /// uploading each to persistent GPU buffers when the Phase 20 path is
    /// enabled and dropping the CPU copy so one expert's Q4_K bytes are
    /// charged to the broker exactly once (Phase 20 module doc's "sole
    /// backing store" variant, spec §50's shared-buffer expert slot shape).
    fn stage_expert_values(
        &mut self,
        staged: Vec<(ExpertId, LoadedQwen36Expert)>,
        broker: &MemoryBroker,
    ) -> Result<Vec<(ExpertId, ExpertValue)>> {
        #[cfg(feature = "metal")]
        {
            self.init_gpu_state_if_needed();
            if let Some(state) = &self.gpu_state {
                let guard = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                return staged
                    .into_iter()
                    .map(|(expert, cpu)| {
                        let [gate, up, down] = cpu.payload_parts();
                        let value = GpuResidentExpert::upload(
                            &guard.ctx,
                            broker,
                            gate,
                            up,
                            down,
                            EXPERT_WIDTH,
                            HIDDEN,
                        )?;
                        drop(cpu);
                        Ok((expert, ExpertValue::Gpu(value)))
                    })
                    .collect();
            }
        }
        Ok(staged
            .into_iter()
            .map(|(expert, cpu)| (expert, ExpertValue::Cpu(cpu)))
            .collect())
    }

    /// Same utility formula `policy::ReplayCache` uses offline, applied to a
    /// live cache entry. Higher is "keep"; the eviction candidate is the
    /// unpinned/unselected entry with the lowest utility, ties broken by
    /// (last_used, layer, expert) for determinism.
    fn utility(&self, entry: &CachedExpert) -> f64 {
        match self.policy {
            CachePolicyKind::Lru => entry.last_used as f64,
            CachePolicyKind::Lfu => entry.frequency as f64 * 1.0e12 + entry.last_used as f64,
            CachePolicyKind::DecayedCostAware => {
                let age = self.clock.saturating_sub(entry.last_used) as f64;
                let decay = 2.0f64.powf(-(age / self.half_life_events as f64));
                entry.frequency as f64 * decay * entry.value.stored_bytes().0 as f64
            }
        }
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let mut stats = self.stats;
        stats.resident_bytes = self.resident_bytes;
        stats
    }

    /// Phase 23: drains speculative loads completed by the prefetch
    /// worker. Entries demanded by the route being planned become regular
    /// (probation-flagged) entries so the plan sees them as hits and pins
    /// them; everything else is either kept as probation (inside the
    /// cache budget) or dropped as wasted traffic.
    fn drain_prefetch_inbox(&mut self, demanded: &[ExpertId; TOP_K]) {
        let deliveries = {
            let Ok(mut inbox) = self.prefetch_inbox.lock() else {
                return;
            };
            std::mem::take(&mut *inbox)
        };
        if deliveries.is_empty() {
            return;
        }
        for delivery in deliveries {
            self.stats.prefetched = self.stats.prefetched.saturating_add(1);
            let bytes = delivery.value.stored_bytes();
            let last_demand = self
                .last_demand_clock
                .get(&(delivery.layer, delivery.expert))
                .copied()
                .unwrap_or(0);
            if last_demand > delivery.issued_clock {
                // The demand it was issued for happened before it
                // arrived: a late speculative read.
                self.stats.prefetch_late = self.stats.prefetch_late.saturating_add(1);
            }
            let fit = self.resident_bytes.0.saturating_add(bytes.0) <= self.capacity.0;
            if !fit && !demanded.contains(&delivery.expert) {
                self.stats.prefetch_wasted_bytes.0 =
                    self.stats.prefetch_wasted_bytes.0.saturating_add(bytes.0);
                continue;
            }
            self.clock = self.clock.wrapping_add(1);
            self.resident_bytes.0 = self.resident_bytes.0.saturating_add(bytes.0);
            self.entries.push(CachedExpert {
                layer: delivery.layer,
                expert: delivery.expert,
                frequency: 1,
                last_used: self.clock,
                pin_count: 0,
                probation: true,
                arrival_clock: self.clock,
                value: ExpertValue::Cpu(delivery.value),
            });
        }
        // Capacity discipline stays in `prepare_exact_route`'s eviction
        // loop (which protects this route's selected experts); the drain
        // itself never evicts a delivery it just admitted.
    }

    /// Phase 23 live hook (spec §295): observes the previous route
    /// transition, predicts the next layer's expert set from the
    /// statistical transition table, and submits one bounded speculative
    /// batch to a worker thread. Prediction never touches the exact
    /// route - it only schedules bytes (invariant #7) - and a wrong
    /// prediction costs wasted SSD traffic, never a wrong result.
    pub fn advance_prefetch(
        &mut self,
        loader: &Arc<Qwen36WeightLoader>,
        _broker: &MemoryBroker,
        from_layer: LayerId,
        to_layer: LayerId,
        route: &RouterResult,
    ) {
        if let Some((previous_layer, previous_route)) = &self.previous_route {
            self.predictor
                .observe(*previous_layer, previous_route, from_layer, &route.ids);
        }
        self.previous_route = Some((from_layer, route.ids.to_vec()));
        for expert in route.ids {
            self.last_demand_clock
                .insert((from_layer, expert), self.clock);
        }
        if !self.prefetch_enabled
            || self.prefetch_depth == 0
            || self.prefetch_in_flight.load(Ordering::SeqCst) >= MAX_PREFETCH_BATCHES_IN_FLIGHT
        {
            return;
        }
        let is_resident = |expert: ExpertId| {
            self.entries
                .iter()
                .any(|entry| entry.layer == to_layer && entry.expert == expert)
        };
        let predicted = self.predictor.predict(
            from_layer,
            &route.ids,
            to_layer,
            self.prefetch_depth,
            is_resident,
        );
        if predicted.is_empty() {
            return;
        }
        self.prefetch_in_flight.fetch_add(1, Ordering::SeqCst);
        let inbox = Arc::clone(&self.prefetch_inbox);
        let in_flight = Arc::clone(&self.prefetch_in_flight);
        let loader = Arc::clone(loader);
        let issued_clock = self.clock;
        std::thread::spawn(move || {
            let mut deliveries = Vec::with_capacity(predicted.len());
            for expert in predicted {
                match loader.load_expert(to_layer, expert) {
                    Ok(value) => deliveries.push(PrefetchDelivery {
                        layer: to_layer,
                        expert,
                        value,
                        issued_clock,
                    }),
                    Err(error) => tracing::debug!(
                        layer = to_layer.0,
                        expert = expert.0,
                        %error,
                        "prefetch load failed; demand path will retry"
                    ),
                }
            }
            if let Ok(mut inbox) = inbox.lock() {
                inbox.extend(deliveries);
            }
            in_flight.fetch_sub(1, Ordering::SeqCst);
        });
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
        broker: &MemoryBroker,
        layer: LayerId,
        route: &RouterResult,
    ) -> Result<ExpertLoadPlan> {
        if self.active_plan.is_some() || self.active_batch_plan.is_some() {
            return Err(crate::error::InternalError {
                incident_id: "expert-plan-overlap".to_string(),
                message: "whole-expert baseline permits only one active exact route".to_string(),
            }
            .into());
        }

        // Phase 23: admit any speculative loads that finished since the
        // last route so the plan can treat them as hits.
        self.drain_prefetch_inbox(&route.ids);

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

        // Stage every miss first. Each miss reserves and reads an independent
        // destination (its own broker lease, its own expert), so Phase 19
        // fans these reads out across a bounded thread pool by default
        // (NVMAI R9 precedent) instead of issuing them one at a time; a
        // failed read still drops all staged broker leases and leaves no
        // partially valid cache entry behind. The staged CPU payloads then
        // become cache values (possibly GPU-resident uploads, Phase 20)
        // before any entry is visible to eviction.
        let io_started = std::time::Instant::now();
        let staged_cpu = crate::io::fetch_all(self.io_fanout, &misses, |&expert| {
            Ok((expert, loader.load_expert(layer, expert)?))
        })?;
        self.stats.demand_io_nanos = self
            .stats
            .demand_io_nanos
            .saturating_add(io_started.elapsed().as_nanos());
        let mut staged = self.stage_expert_values(staged_cpu, broker)?;

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
                if entry.probation {
                    entry.probation = false;
                    self.stats.prefetch_hits = self.stats.prefetch_hits.saturating_add(1);
                }
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
                probation: false,
                arrival_clock: self.clock,
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
    ) -> Result<&ExpertValue> {
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

    /// Executes one planned routed expert's forward on whatever backing
    /// store the cache chose for it (CPU Q4_K payload or Phase 20
    /// GPU-resident buffers), returning a broker-accounted activation. This
    /// is the live decode path's binding point: it replaces `planned_expert`
    /// at the streaming call site so the GPU path needs no Metal types
    /// outside the cache.
    pub(crate) fn forward_expert(
        &mut self,
        plan: &ExpertLoadPlan,
        expert: ExpertId,
        broker: &MemoryBroker,
        input: &[f32],
    ) -> Result<Qwen36Activation> {
        let entry = self.planned_expert(plan, expert)?;
        #[cfg(feature = "metal")]
        {
            if let ExpertValue::Gpu(gpu_expert) = entry {
                let state = self
                    .gpu_state
                    .as_ref()
                    .ok_or_else(|| crate::error::InternalError {
                        incident_id: "expert-gpu-state-missing".to_string(),
                        message: "GPU-resident expert value without GPU execution state"
                            .to_string(),
                    })?;
                let mut guard = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let values = gpu_expert.forward(&mut guard, input)?;
                return Qwen36Activation::from_slice(broker, &values);
            }
        }
        match entry {
            ExpertValue::Cpu(cpu) => cpu.forward(broker, input),
            #[cfg(feature = "metal")]
            ExpertValue::Gpu(_) => Err(crate::error::InternalError {
                incident_id: "expert-gpu-state-missing".to_string(),
                message: "GPU-resident expert value without GPU execution state".to_string(),
            }
            .into()),
        }
    }

    /// Phase 26 (spec §298): materializes one layer/chunk transaction
    /// whose pin set is the union of several exact routes. Each distinct
    /// absent expert is fetched exactly once - FlashMoE's "load each
    /// required expert on-demand exactly once per iteration" - and every
    /// selected entry is pinned before any row's computation begins.
    pub fn prepare_batch_route(
        &mut self,
        loader: &Qwen36WeightLoader,
        broker: &MemoryBroker,
        layer: LayerId,
        routes: &[RouterResult],
    ) -> Result<BatchExpertPlan> {
        if self.active_plan.is_some() || self.active_batch_plan.is_some() {
            return Err(crate::error::InternalError {
                incident_id: "expert-batch-plan-overlap".to_string(),
                message: "only one active exact-route batch is permitted".to_string(),
            }
            .into());
        }
        for route in routes {
            route.validate()?;
        }
        let mut distinct: Vec<ExpertId> = Vec::new();
        let mut resident_at_start = std::collections::HashSet::new();
        for route in routes {
            for expert in route.ids {
                if !distinct.contains(&expert) {
                    distinct.push(expert);
                }
            }
        }
        let mut miss_bytes = 0u64;
        let misses: Vec<ExpertId> = distinct
            .iter()
            .copied()
            .filter(|expert| {
                let resident = self
                    .entries
                    .iter()
                    .any(|entry| entry.layer == layer && entry.expert == *expert);
                if resident {
                    resident_at_start.insert(*expert);
                }
                !resident
            })
            .collect();
        for &expert in &misses {
            miss_bytes = miss_bytes.saturating_add(loader.expert_stored_bytes(layer, expert)?.0);
        }
        if miss_bytes > self.capacity.0 {
            return Err(ModelError::Unsupported(format!(
                "expert cache capacity {} bytes cannot hold the {} distinct experts in one prefill chunk ({} bytes)",
                self.capacity.0,
                misses.len(),
                miss_bytes
            ))
            .into());
        }
        while self.resident_bytes.0.saturating_add(miss_bytes) > self.capacity.0 {
            if self
                .evict_coldest_excluding_many(layer, &distinct)
                .is_none()
            {
                return Err(ModelError::Unsupported(format!(
                    "expert cache capacity {} bytes cannot pin all distinct experts in one prefill chunk",
                    self.capacity.0
                ))
                .into());
            }
        }

        let io_started = std::time::Instant::now();
        let staged_cpu = crate::io::fetch_all(self.io_fanout, &misses, |&expert| {
            Ok((expert, loader.load_expert(layer, expert)?))
        })?;
        self.stats.demand_io_nanos = self
            .stats
            .demand_io_nanos
            .saturating_add(io_started.elapsed().as_nanos());
        let mut staged = self.stage_expert_values(staged_cpu, broker)?;

        let plan_id = self.next_plan_id;
        self.next_plan_id = self.next_plan_id.wrapping_add(1).max(1);
        let plan = BatchExpertPlan {
            plan_id,
            layer,
            distinct: distinct.clone(),
            routes: routes.to_vec(),
        };
        for &expert in &plan.distinct {
            self.clock = self.clock.wrapping_add(1);
            if resident_at_start.contains(&expert) {
                let entry = self
                    .entries
                    .iter_mut()
                    .find(|entry| entry.layer == layer && entry.expert == expert)
                    .ok_or_else(|| crate::error::InternalError {
                        incident_id: format!("expert-batch-plan-{plan_id}-lost-hit"),
                        message: "batch hit disappeared before plan activation".to_string(),
                    })?;
                entry.frequency = entry.frequency.saturating_add(1);
                entry.last_used = self.clock;
                entry.pin_count = entry.pin_count.saturating_add(1);
                if entry.probation {
                    entry.probation = false;
                    self.stats.prefetch_hits = self.stats.prefetch_hits.saturating_add(1);
                }
                let demands = plan
                    .routes
                    .iter()
                    .filter(|route| route.ids.contains(&expert))
                    .count() as u64;
                self.stats.hits = self.stats.hits.saturating_add(demands);
                continue;
            }
            let index = staged
                .iter()
                .position(|(id, _)| *id == expert)
                .ok_or_else(|| crate::error::InternalError {
                    incident_id: format!("expert-batch-plan-{plan_id}-unstaged-miss"),
                    message: "batch miss was not present in the staged transaction".to_string(),
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
                probation: false,
                arrival_clock: self.clock,
                value,
            });
        }
        debug_assert!(staged.is_empty());
        self.active_batch_plan = Some(plan.clone());
        Ok(plan)
    }

    /// Executes one batch-planned expert forward (Phase 26). Same
    /// binding rules as `forward_expert`, against the batch plan's
    /// union.
    pub(crate) fn forward_batch_expert(
        &mut self,
        plan: &BatchExpertPlan,
        expert: ExpertId,
        broker: &MemoryBroker,
        input: &[f32],
    ) -> Result<Qwen36Activation> {
        if self.active_batch_plan.as_ref() != Some(plan) || !plan.distinct.contains(&expert) {
            return Err(crate::error::InternalError {
                incident_id: format!("expert-batch-plan-{}-binding", plan.plan_id),
                message: "batch computation requested an expert outside its active plan"
                    .to_string(),
            }
            .into());
        }
        let entry = self
            .entries
            .iter()
            .find(|entry| {
                entry.layer == plan.layer && entry.expert == expert && entry.pin_count > 0
            })
            .ok_or_else(|| crate::error::InternalError {
                incident_id: format!("expert-batch-plan-{}-missing", plan.plan_id),
                message: "active batch plan lost a pinned expert binding".to_string(),
            })?;
        #[cfg(feature = "metal")]
        {
            if let ExpertValue::Gpu(gpu_expert) = &entry.value {
                let state = self
                    .gpu_state
                    .as_ref()
                    .ok_or_else(|| crate::error::InternalError {
                        incident_id: "expert-gpu-state-missing".to_string(),
                        message: "GPU-resident expert value without GPU execution state"
                            .to_string(),
                    })?;
                let mut guard = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let values = gpu_expert.forward(&mut guard, input)?;
                return Qwen36Activation::from_slice(broker, &values);
            }
        }
        match &entry.value {
            ExpertValue::Cpu(cpu) => cpu.forward(broker, input),
            #[cfg(feature = "metal")]
            ExpertValue::Gpu(_) => Err(crate::error::InternalError {
                incident_id: "expert-gpu-state-missing".to_string(),
                message: "GPU-resident expert value without GPU execution state".to_string(),
            }
            .into()),
        }
    }

    pub(crate) fn finish_batch_route(&mut self, plan: &BatchExpertPlan) -> Result<()> {
        if self.active_batch_plan.as_ref() != Some(plan) {
            return Err(crate::error::InternalError {
                incident_id: format!("expert-batch-plan-{}-finish", plan.plan_id),
                message: "attempted to finish a batch route that is not active".to_string(),
            }
            .into());
        }
        for expert in &plan.distinct {
            let entry = self
                .entries
                .iter_mut()
                .find(|entry| entry.layer == plan.layer && entry.expert == *expert)
                .ok_or_else(|| crate::error::InternalError {
                    incident_id: format!("expert-batch-plan-{}-lost-pin", plan.plan_id),
                    message: "active batch plan lost a selected expert before release".to_string(),
                })?;
            entry.pin_count = entry.pin_count.saturating_sub(1);
        }
        self.active_batch_plan = None;
        Ok(())
    }

    /// Eviction candidate search for a batch plan: the whole distinct
    /// union is protected, and `Vec` pin counts continue to protect any
    /// other active bindings.
    fn evict_coldest_excluding_many(
        &mut self,
        selected_layer: LayerId,
        selected_experts: &[ExpertId],
    ) -> Option<()> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.pin_count == 0
                    && (entry.layer != selected_layer || !selected_experts.contains(&entry.expert))
            })
            .min_by(|(_, left), (_, right)| self.eviction_order(left, right))
            .map(|(index, _)| index)?;
        self.remove_entry(index);
        Some(())
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
        broker: &MemoryBroker,
        layer: LayerId,
        expert: ExpertId,
    ) -> Result<&ExpertValue> {
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
        // destination Vec. All cache evictions above happened first. The
        // loaded payload then becomes a cache value (Phase 20 GPU upload
        // when enabled) before it is exposed to eviction.
        let io_started = std::time::Instant::now();
        let cpu_value = loader.load_expert(layer, expert)?;
        self.stats.demand_io_nanos = self
            .stats
            .demand_io_nanos
            .saturating_add(io_started.elapsed().as_nanos());
        let (_, value) = self
            .stage_expert_values(vec![(expert, cpu_value)], broker)?
            .pop()
            .ok_or_else(|| crate::error::InternalError {
                incident_id: "expert-get-or-load-staged".to_string(),
                message: "get_or_load staging produced no value".to_string(),
            })?;
        self.stats.misses = self.stats.misses.saturating_add(1);
        self.stats.raw_miss_bytes.0 = self.stats.raw_miss_bytes.0.saturating_add(required.0);
        self.resident_bytes.0 = self.resident_bytes.0.saturating_add(value.stored_bytes().0);
        self.entries.push(CachedExpert {
            layer,
            expert,
            frequency: 1,
            last_used: self.clock,
            pin_count: 0,
            probation: false,
            arrival_clock: self.clock,
            value,
        });
        Ok(&self.entries.last().expect("just pushed cache entry").value)
    }

    fn evict_coldest(&mut self) -> Option<()> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.pin_count == 0)
            .min_by(|(_, left), (_, right)| self.eviction_order(left, right))
            .map(|(index, _)| index)?;
        self.remove_entry(index);
        Some(())
    }

    fn evict_coldest_excluding(
        &mut self,
        selected_layer: LayerId,
        selected_experts: &[ExpertId; TOP_K],
    ) -> Option<()> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.pin_count == 0
                    && (entry.layer != selected_layer || !selected_experts.contains(&entry.expert))
            })
            .min_by(|(_, left), (_, right)| self.eviction_order(left, right))
            .map(|(index, _)| index)?;
        self.remove_entry(index);
        Some(())
    }

    /// Total order for eviction candidates: probation (speculative Phase
    /// 23) entries first, then lowest policy utility, ties broken by
    /// (last_used, layer, expert) so eviction is deterministic regardless
    /// of `Vec` iteration/storage order.
    fn eviction_order(&self, left: &CachedExpert, right: &CachedExpert) -> std::cmp::Ordering {
        right
            .probation
            .cmp(&left.probation)
            .then_with(|| self.utility(left).total_cmp(&self.utility(right)))
            .then_with(|| left.last_used.cmp(&right.last_used))
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| left.expert.cmp(&right.expert))
    }

    fn remove_entry(&mut self, index: usize) {
        let entry = self.entries.swap_remove(index);
        self.resident_bytes.0 = self
            .resident_bytes
            .0
            .saturating_sub(entry.value.stored_bytes().0);
        if entry.probation {
            // A speculative load evicted before any demand: wasted SSD
            // traffic (spec §295's "wasted bytes" metric).
            self.stats.prefetch_wasted_bytes.0 = self
                .stats
                .prefetch_wasted_bytes
                .0
                .saturating_add(entry.value.stored_bytes().0);
        }
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
        loader: &Arc<Qwen36WeightLoader>,
        cache: &mut WholeExpertLfuCache,
        broker: &MemoryBroker,
        input: &Qwen36Activation,
    ) -> Result<(Qwen36Activation, RouterResult)> {
        self.forward_with_observer(loader, cache, broker, input, |_, _| Ok(()))
    }

    pub fn forward_with_observer<F>(
        &mut self,
        loader: &Arc<Qwen36WeightLoader>,
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
        let plan = cache.prepare_exact_route(loader, broker, self.layer, &route)?;
        // Phase 23: schedule speculative reads for the next layer's MoE
        // while this layer's routed experts compute. Purely an I/O hint -
        // the exact route above is authoritative (invariant #7).
        let to_layer = if self.layer.0 + 1 < Qwen36Geometry::NUM_LAYERS as u8 {
            LayerId(self.layer.0 + 1)
        } else {
            LayerId(0)
        };
        cache.advance_prefetch(loader, broker, self.layer, to_layer, &route);
        let computed = (|| -> Result<Qwen36Activation> {
            let shared_gate = self.shared_gate.matvec(broker, &input.values)?;
            let shared_up = self.shared_up.matvec(broker, &input.values)?;
            let shared_hidden = Qwen36Activation::silu_mul(broker, &shared_gate, &shared_up)?;
            let mut output = self.shared_down.matvec(broker, &shared_hidden.values)?;
            let shared_gate_logit = self.shared_input_gate.dot(broker, &input.values)?;
            output.scale_in_place(1.0 / (1.0 + (-shared_gate_logit).exp()));
            observer("shared", &output)?;

            for (&expert, &weight) in plan.route.ids.iter().zip(&plan.route.weights) {
                let routed = cache.forward_expert(&plan, expert, broker, &input.values)?;
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

impl Qwen36StreamingMoe {
    /// Phase 26 batched MoE (spec §298): routes every chunk row, plans
    /// the distinct-expert union once, fetches each absent expert once,
    /// and executes per-row routed accumulation in exact route order.
    /// The per-row shared-expert path is identical to the single-token
    /// forward; routed experts come from the shared batch plan.
    pub fn forward_batch(
        &mut self,
        loader: &Arc<Qwen36WeightLoader>,
        cache: &mut WholeExpertLfuCache,
        broker: &MemoryBroker,
        inputs: &[Qwen36Activation],
    ) -> Result<(Vec<Qwen36Activation>, Vec<RouterResult>)> {
        let mut routes = Vec::with_capacity(inputs.len());
        for input in inputs {
            require_len("Qwen batched MoE input", input.values.len(), HIDDEN)?;
            let route_logits = self.router.matvec(broker, &input.values)?;
            routes.push(RouterResult::from_logits(&route_logits.values)?);
        }
        let plan = cache.prepare_batch_route(loader, broker, self.layer, &routes)?;
        let computed = (|| -> Result<Vec<Qwen36Activation>> {
            let mut outputs = Vec::with_capacity(inputs.len());
            for (input, route) in inputs.iter().zip(&plan.routes) {
                let shared_gate = self.shared_gate.matvec(broker, &input.values)?;
                let shared_up = self.shared_up.matvec(broker, &input.values)?;
                let shared_hidden = Qwen36Activation::silu_mul(broker, &shared_gate, &shared_up)?;
                let mut output = self.shared_down.matvec(broker, &shared_hidden.values)?;
                let shared_gate_logit = self.shared_input_gate.dot(broker, &input.values)?;
                output.scale_in_place(1.0 / (1.0 + (-shared_gate_logit).exp()));
                for (&expert, &weight) in route.ids.iter().zip(&route.weights) {
                    let routed =
                        cache.forward_batch_expert(&plan, expert, broker, &input.values)?;
                    output.add_scaled_in_place(&routed, weight)?;
                }
                outputs.push(output);
            }
            Ok(outputs)
        })();
        let release = cache.finish_batch_route(&plan);
        let outputs = computed?;
        release?;
        Ok((outputs, plan.routes))
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

    fn synthetic_expert(broker: &MemoryBroker) -> LoadedQwen36Expert {
        // A minimal broker-accounted expert payload for cache tests; the
        // forward math is not exercised here, only residency accounting.
        LoadedQwen36Expert::synthetic_for_tests(broker)
    }

    #[test]
    fn probation_entries_evict_before_demand_entries() {
        let broker = MemoryBroker::new(Bytes(16 * 1024 * 1024));
        let mut cache = WholeExpertLfuCache::new(Bytes(8 * 1024));
        cache.clock = 10;
        let demand = CachedExpert {
            layer: LayerId(0),
            expert: ExpertId(1),
            frequency: 5,
            last_used: 1,
            pin_count: 0,
            probation: false,
            arrival_clock: 1,
            value: ExpertValue::Cpu(synthetic_expert(&broker)),
        };
        let probation = CachedExpert {
            layer: LayerId(0),
            expert: ExpertId(2),
            frequency: 50,
            last_used: 9,
            pin_count: 0,
            probation: true,
            arrival_clock: 9,
            value: ExpertValue::Cpu(synthetic_expert(&broker)),
        };
        cache.entries.push(demand);
        cache.entries.push(probation);
        cache.resident_bytes = Bytes(2048);
        // LRU would evict the demand entry (last_used 1 < 9); probation
        // priority must override recency.
        cache.evict_coldest().unwrap();
        assert_eq!(cache.entries.len(), 1);
        assert!(!cache.entries[0].probation, "demand entry survives");
        assert_eq!(cache.stats.prefetch_wasted_bytes.0, 1024);
    }

    #[test]
    fn prefetch_inbox_delivers_demanded_entries_and_wastes_the_rest() {
        let broker = MemoryBroker::new(Bytes(16 * 1024 * 1024));
        let mut cache = WholeExpertLfuCache::new(Bytes(2 * 1024));
        {
            let mut inbox = cache.prefetch_inbox.lock().unwrap();
            inbox.push(PrefetchDelivery {
                layer: LayerId(1),
                expert: ExpertId(7),
                value: synthetic_expert(&broker),
                issued_clock: 1,
            });
            inbox.push(PrefetchDelivery {
                layer: LayerId(1),
                expert: ExpertId(9),
                value: synthetic_expert(&broker),
                issued_clock: 1,
            });
        }
        let demanded = [
            ExpertId(7),
            ExpertId(0),
            ExpertId(0),
            ExpertId(0),
            ExpertId(0),
            ExpertId(0),
            ExpertId(0),
            ExpertId(0),
        ];
        cache.clock = 5;
        cache.drain_prefetch_inbox(&demanded);
        assert_eq!(cache.stats.prefetched, 2);
        assert_eq!(cache.entries.len(), 2);
        // Both fit (2 KiB in a 4 KiB cache); the demanded one is flagged
        // probation until its route pins it.
        assert!(cache.entries.iter().all(|entry| entry.probation));
        assert!(cache
            .entries
            .iter()
            .any(|entry| entry.expert == ExpertId(7) && entry.layer == LayerId(1)));

        // A second delivery beyond capacity for a non-demanded expert is
        // dropped as wasted traffic.
        {
            let mut inbox = cache.prefetch_inbox.lock().unwrap();
            inbox.push(PrefetchDelivery {
                layer: LayerId(1),
                expert: ExpertId(11),
                value: synthetic_expert(&broker),
                issued_clock: 2,
            });
        }
        cache.drain_prefetch_inbox(&demanded);
        assert_eq!(cache.stats.prefetched, 3);
        assert_eq!(cache.stats.prefetch_wasted_bytes.0, 1024);
        assert_eq!(cache.entries.len(), 2);
    }
}
