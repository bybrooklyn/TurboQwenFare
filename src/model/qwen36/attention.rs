//! Phase-13 full-attention correctness oracle for Qwen3.6.  It is deliberately
//! direct and allocation-conscious: BF16 K/V is the baseline, RoPE covers only
//! the first 64 dimensions, and grouped-query attention maps each query head
//! to its source KV head at consumption time instead of expanding cache pages.

use crate::backend::reference::q4k_gemv;
use crate::context::tqkv::{tqkv_enabled, tqkv_precision, TqkvPagedCache};
use crate::error::{ModelError, Result};
use crate::ids::{Bytes, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

use super::geometry::Qwen36Geometry;
use super::weights::Qwen36Activation;

const HEADS: usize = Qwen36Geometry::FULL_ATTENTION_HEADS;
const KV_HEADS: usize = Qwen36Geometry::FULL_KV_HEADS;
const HEAD_DIM: usize = Qwen36Geometry::FULL_HEAD_DIM;
const HIDDEN: usize = Qwen36Geometry::HIDDEN_SIZE;
const PROJECTED_Q_GATE: usize = HEADS * HEAD_DIM * 2;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;

/// Splits the canonical Qwen q-projection output. The checkpoint stores each
/// head as `[query_256, gate_256]`; it is not one global query half followed
/// by one global gate half.
pub(crate) fn split_query_gate_accounted(
    broker: &MemoryBroker,
    projected: &Qwen36Activation,
) -> Result<(Qwen36Activation, Qwen36Activation)> {
    checked_len(
        "attention q/gate projection",
        projected.values.len(),
        PROJECTED_Q_GATE,
    )?;
    let mut query = Qwen36Activation::zeros(broker, HEADS * HEAD_DIM)?;
    let mut gate = Qwen36Activation::zeros(broker, HEADS * HEAD_DIM)?;
    for head in 0..HEADS {
        let source = head * HEAD_DIM * 2;
        let target = head * HEAD_DIM;
        query.values[target..target + HEAD_DIM]
            .copy_from_slice(&projected.values[source..source + HEAD_DIM]);
        gate.values[target..target + HEAD_DIM]
            .copy_from_slice(&projected.values[source + HEAD_DIM..source + HEAD_DIM * 2]);
    }
    Ok((query, gate))
}

fn split_query_gate(projected: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    checked_len(
        "attention q/gate projection",
        projected.len(),
        PROJECTED_Q_GATE,
    )?;
    let mut query = vec![0.0; HEADS * HEAD_DIM];
    let mut gate = vec![0.0; HEADS * HEAD_DIM];
    for head in 0..HEADS {
        let source = head * HEAD_DIM * 2;
        let target = head * HEAD_DIM;
        query[target..target + HEAD_DIM].copy_from_slice(&projected[source..source + HEAD_DIM]);
        gate[target..target + HEAD_DIM]
            .copy_from_slice(&projected[source + HEAD_DIM..source + HEAD_DIM * 2]);
    }
    Ok((query, gate))
}

/// IEEE BF16 conversion with round-to-nearest-even. BF16 is intentionally the
/// reference cache representation; TQKV compression is a later phase.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn checked_len(name: &'static str, actual: usize, expected: usize) -> Result<()> {
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

/// BF16 post-RoPE K/V store for one of Qwen's ten full-attention layers.
/// A cache reserves all of its declared capacity before allocating its vectors.
pub struct Bf16KvCache {
    layer: LayerId,
    max_tokens: usize,
    keys: Vec<u16>,
    values: Vec<u16>,
    // Declared last so the physical vectors are dropped before their budget
    // is released.
    _lease: MemoryLease,
}

impl Bf16KvCache {
    pub fn bytes_for_tokens(tokens: usize) -> Result<Bytes> {
        let values = tokens
            .checked_mul(KV_HEADS)
            .and_then(|n| n.checked_mul(HEAD_DIM))
            .and_then(|n| n.checked_mul(2)) // K and V
            .and_then(|n| n.checked_mul(std::mem::size_of::<u16>()))
            .ok_or(ModelError::Shape {
                tensor: "BF16 KV capacity",
                expected: usize::MAX,
                actual: tokens,
            })?;
        Ok(Bytes(values as u64))
    }

    pub fn new(broker: &MemoryBroker, layer: LayerId, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 {
            return Err(ModelError::Shape {
                tensor: "BF16 KV capacity",
                expected: 1,
                actual: 0,
            }
            .into());
        }
        let elements = max_tokens.checked_mul(KV_WIDTH).ok_or(ModelError::Shape {
            tensor: "BF16 KV capacity",
            expected: usize::MAX,
            actual: max_tokens,
        })?;
        let lease = broker.reserve(
            MemoryOwner::ContextHot,
            MemoryClass::Protected,
            Self::bytes_for_tokens(max_tokens)?,
            64,
        )?;
        // The lease has already succeeded: this is the required
        // reserve-before-physical-allocation order.
        Ok(Self {
            layer,
            max_tokens,
            keys: Vec::with_capacity(elements),
            values: Vec::with_capacity(elements),
            _lease: lease,
        })
    }

    pub fn layer(&self) -> LayerId {
        self.layer
    }

    pub fn len(&self) -> usize {
        self.keys.len() / KV_WIDTH
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        checked_len("attention key", key.len(), KV_WIDTH)?;
        checked_len("attention value", value.len(), KV_WIDTH)?;
        if self.len() == self.max_tokens {
            return Err(ModelError::ContextCapacity {
                layer: self.layer.0,
                capacity: self.max_tokens,
            }
            .into());
        }
        self.keys.extend(key.iter().copied().map(f32_to_bf16));
        self.values.extend(value.iter().copied().map(f32_to_bf16));
        Ok(())
    }

    fn key(&self, token: usize, kv_head: usize) -> &[u16] {
        let start = (token * KV_HEADS + kv_head) * HEAD_DIM;
        &self.keys[start..start + HEAD_DIM]
    }

    fn value(&self, token: usize, kv_head: usize) -> &[u16] {
        let start = (token * KV_HEADS + kv_head) * HEAD_DIM;
        &self.values[start..start + HEAD_DIM]
    }
}

/// Reference Q4 projection tensors for one full-attention layer. The q
/// projection deliberately contains query and gate rows together, matching
/// the canonical checkpoint's 8192-row q_proj.
pub struct Q4FullAttentionWeights<'a> {
    pub q_proj: &'a [u8],
    pub k_proj: &'a [u8],
    pub v_proj: &'a [u8],
    pub o_proj: &'a [u8],
    pub q_norm_weight: &'a [f32],
    pub k_norm_weight: &'a [f32],
}

impl<'a> Q4FullAttentionWeights<'a> {
    pub fn validate(&self) -> Result<()> {
        let q4k_row = |cols: usize| cols / 256 * 144;
        checked_len(
            "q_proj",
            self.q_proj.len(),
            PROJECTED_Q_GATE * q4k_row(HIDDEN),
        )?;
        checked_len("k_proj", self.k_proj.len(), KV_WIDTH * q4k_row(HIDDEN))?;
        checked_len("v_proj", self.v_proj.len(), KV_WIDTH * q4k_row(HIDDEN))?;
        checked_len(
            "o_proj",
            self.o_proj.len(),
            HIDDEN * q4k_row(HEADS * HEAD_DIM),
        )?;
        checked_len("q_norm", self.q_norm_weight.len(), HEAD_DIM)?;
        checked_len("k_norm", self.k_norm_weight.len(), HEAD_DIM)?;
        Ok(())
    }
}

/// Backend selector for `FullAttentionLayer`'s K/V history. `Bf16` is the
/// Phase 13 correctness oracle and stays the default; `Tqkv` is the Phase 27
/// paged Q8/Q4 backend, opt-in via `TQF_TQKV_ENABLED` (crate invariant #10 —
/// every optimization must be A/B-disableable, and ordinary users never see
/// this switch).
enum KvCacheBackend {
    Bf16(Bf16KvCache),
    Tqkv(TqkvPagedCache),
}

#[derive(Clone, Copy)]
pub(crate) enum BackendChoice {
    Bf16,
    Tqkv(crate::context::tqkv::TqkvPrecision),
}

impl KvCacheBackend {
    fn new(broker: &MemoryBroker, layer: LayerId, max_tokens: usize) -> Result<Self> {
        Self::with_choice(
            broker,
            layer,
            max_tokens,
            if tqkv_enabled() {
                BackendChoice::Tqkv(tqkv_precision())
            } else {
                BackendChoice::Bf16
            },
        )
    }

    /// Explicit-choice constructor bypassing the process-global env var —
    /// used by differential tests so both backends can be exercised in one
    /// test binary regardless of process-wide `OnceLock` A/B state.
    fn with_choice(
        broker: &MemoryBroker,
        layer: LayerId,
        max_tokens: usize,
        choice: BackendChoice,
    ) -> Result<Self> {
        match choice {
            BackendChoice::Tqkv(precision) => Ok(KvCacheBackend::Tqkv(TqkvPagedCache::new(
                broker, layer, max_tokens, precision,
            )?)),
            BackendChoice::Bf16 => Ok(KvCacheBackend::Bf16(Bf16KvCache::new(
                broker, layer, max_tokens,
            )?)),
        }
    }

    fn len(&self) -> usize {
        match self {
            KvCacheBackend::Bf16(cache) => cache.len(),
            KvCacheBackend::Tqkv(cache) => cache.len(),
        }
    }

    fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        match self {
            KvCacheBackend::Bf16(cache) => cache.push(key, value),
            KvCacheBackend::Tqkv(cache) => cache.push(key, value),
        }
    }

    fn key(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        match self {
            KvCacheBackend::Bf16(cache) => {
                let mut out = [0f32; HEAD_DIM];
                for (dst, &src) in out.iter_mut().zip(cache.key(token, kv_head)) {
                    *dst = bf16_to_f32(src);
                }
                out
            }
            KvCacheBackend::Tqkv(cache) => cache.key(token, kv_head),
        }
    }

    fn value(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        match self {
            KvCacheBackend::Bf16(cache) => {
                let mut out = [0f32; HEAD_DIM];
                for (dst, &src) in out.iter_mut().zip(cache.value(token, kv_head)) {
                    *dst = bf16_to_f32(src);
                }
                out
            }
            KvCacheBackend::Tqkv(cache) => cache.value(token, kv_head),
        }
    }

    /// Clears logical content while keeping the same broker lease/capacity —
    /// mirrors the pre-existing `Bf16KvCache` reset behavior (same reserved
    /// bytes, just logically empty) rather than re-reserving.
    fn reset(&mut self) {
        match self {
            KvCacheBackend::Bf16(cache) => {
                cache.keys.clear();
                cache.values.clear();
            }
            KvCacheBackend::Tqkv(cache) => cache.reset(),
        }
    }
}

/// Stateful reference layer. The output of `decode_projected` is gated
/// attention output before `o_proj`; `decode_q4` includes the output
/// projection and residual addition.
pub struct FullAttentionLayer {
    cache: KvCacheBackend,
    position: u64,
}

impl FullAttentionLayer {
    pub fn new(broker: &MemoryBroker, layer: LayerId, max_tokens: usize) -> Result<Self> {
        Ok(Self {
            cache: KvCacheBackend::new(broker, layer, max_tokens)?,
            position: 0,
        })
    }

    /// Explicit-backend constructor for differential A/B tests (see
    /// `BackendChoice`); production call sites always use `new`, which
    /// reads the `TQF_TQKV_ENABLED`/`TQF_TQKV_PRECISION` A/B controls.
    pub(crate) fn new_with_backend(
        broker: &MemoryBroker,
        layer: LayerId,
        max_tokens: usize,
        choice: BackendChoice,
    ) -> Result<Self> {
        Ok(Self {
            cache: KvCacheBackend::with_choice(broker, layer, max_tokens, choice)?,
            position: 0,
        })
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Populates `tokens` synthetic post-RoPE Key/Value entries directly
    /// into the backend without computing attention on each one — real
    /// per-token decode is O(n) per step, so building up to a 128K-scale
    /// history that way is itself the Phase 29 problem. This exists purely
    /// for the Phase 29 populated-context attention-cost benchmark
    /// (`context::tqkv::scaling_bench`): it isolates "cost of one real
    /// attention step at depth n" from "cost of the n steps that built up
    /// that history", which the full decode loop cannot separate on its
    /// own. Not used by any production call site.
    pub(crate) fn seed_synthetic_history_for_benchmark(&mut self, tokens: usize) -> Result<()> {
        let mut state = 0x5EED_5EED_u64;
        let mut xorshift = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 3.0
        };
        let mut key = vec![0f32; KV_WIDTH];
        let mut value = vec![0f32; KV_WIDTH];
        for _ in 0..tokens {
            for slot in key.iter_mut().chain(value.iter_mut()) {
                *slot = xorshift();
            }
            self.cache.push(&key, &value)?;
        }
        self.position += tokens as u64;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.position = 0;
    }

    /// The current decode position (tokens pushed so far) — a prefix
    /// snapshot restore must set this so the next real token's RoPE angle
    /// is computed at the right absolute position.
    pub(crate) fn position(&self) -> u64 {
        self.position
    }

    /// Exposes the TQKV backend for `context::prefix` snapshot export;
    /// `None` when this layer is running the BF16 reference backend
    /// (prefix dedup is a TQKV-specific mechanism, spec §66).
    pub(crate) fn tqkv_cache(&self) -> Option<&TqkvPagedCache> {
        match &self.cache {
            KvCacheBackend::Tqkv(cache) => Some(cache),
            KvCacheBackend::Bf16(_) => None,
        }
    }

    /// Builds a `context::prefix` capture of this layer's current TQKV
    /// state (`None` for a BF16-backed layer — see `tqkv_cache`).
    pub(crate) fn capture_tqkv_for_snapshot(
        &self,
        layer: LayerId,
    ) -> Option<crate::context::prefix::FullAttentionCapture> {
        let cache = self.tqkv_cache()?;
        let pages = cache
            .sealed_pages()
            .iter()
            .map(|page| (page.content_id(), page.to_bytes()))
            .collect();
        let (tail_keys, tail_values) = cache.tail_raw();
        Some(crate::context::prefix::FullAttentionCapture {
            layer,
            precision: cache.precision(),
            position: self.position,
            pages,
            tail_keys: tail_keys.to_vec(),
            tail_values: tail_values.to_vec(),
        })
    }

    /// Restores this layer's TQKV backend from a snapshot and sets
    /// `position` to match (`context::prefix` restart-reuse path). Errors
    /// if this layer isn't currently running the TQKV backend.
    pub(crate) fn restore_tqkv_snapshot(
        &mut self,
        sealed: Vec<crate::context::tqkv::SealedPage>,
        tail_keys: Vec<f32>,
        tail_values: Vec<f32>,
        position: u64,
    ) -> Result<()> {
        match &mut self.cache {
            KvCacheBackend::Tqkv(cache) => {
                cache.restore_from_snapshot(sealed, tail_keys, tail_values);
                self.position = position;
                Ok(())
            }
            KvCacheBackend::Bf16(_) => Err(ModelError::Shape {
                tensor: "TQKV snapshot restore onto a BF16-backed layer",
                expected: 1,
                actual: 0,
            }
            .into()),
        }
    }

    /// Broker-accounted variant used by the real fixed-graph binder.  The
    /// historical `decode_projected` API below remains a small-vector oracle
    /// for unit tests, but must not be used by a model session because its
    /// convenience `Vec` inputs have no leases.
    pub fn decode_projected_accounted(
        &mut self,
        broker: &MemoryBroker,
        query: &mut Qwen36Activation,
        gate: &Qwen36Activation,
        key: &mut Qwen36Activation,
        value: &Qwen36Activation,
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
    ) -> Result<Qwen36Activation> {
        checked_len("attention query", query.values.len(), HEADS * HEAD_DIM)?;
        checked_len("attention gate", gate.values.len(), HEADS * HEAD_DIM)?;
        checked_len("attention key", key.values.len(), KV_WIDTH)?;
        checked_len("attention value", value.values.len(), KV_WIDTH)?;
        checked_len("q_norm", q_norm_weight.len(), HEAD_DIM)?;
        checked_len("k_norm", k_norm_weight.len(), HEAD_DIM)?;

        for head in 0..HEADS {
            rmsnorm_head(
                &mut query.values[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                q_norm_weight,
            );
            apply_partial_rope(
                &mut query.values[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                self.position,
            );
        }
        for head in 0..KV_HEADS {
            rmsnorm_head(
                &mut key.values[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                k_norm_weight,
            );
            apply_partial_rope(
                &mut key.values[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                self.position,
            );
        }
        self.cache.push(&key.values, &value.values)?;
        self.position += 1;

        let tokens = self.cache.len();
        let mut output = Qwen36Activation::zeros(broker, HEADS * HEAD_DIM)?;
        for q_head in 0..HEADS {
            // Reserve before the per-head causal-score allocation.  The
            // scratch values are discarded before proceeding to the next head.
            let score_bytes =
                tokens
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(ModelError::Shape {
                        tensor: "attention score scratch bytes",
                        expected: usize::MAX,
                        actual: tokens,
                    })?;
            let _scores_lease = broker.reserve(
                MemoryOwner::Scratch,
                MemoryClass::Transient,
                Bytes(score_bytes as u64),
                64,
            )?;
            let q = &query.values[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
            let mut scores = Vec::with_capacity(tokens);
            let mut max_score = f32::NEG_INFINITY;
            let kv_head = q_head / (HEADS / KV_HEADS);
            for token in 0..tokens {
                let score = q
                    .iter()
                    .zip(self.cache.key(token, kv_head))
                    .map(|(&q, k)| q * k)
                    .sum::<f32>()
                    * 0.0625;
                max_score = max_score.max(score);
                scores.push(score);
            }
            let denominator = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .sum::<f32>();
            let target = &mut output.values[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
            for (token, score) in scores.into_iter().enumerate() {
                let probability = (score - max_score).exp() / denominator;
                for (target, value) in target.iter_mut().zip(self.cache.value(token, kv_head)) {
                    *target += probability * value;
                }
            }
        }
        for (value, gate) in output.values.iter_mut().zip(&gate.values) {
            *value *= sigmoid(*gate);
        }
        Ok(output)
    }

    /// Full causal attention after Q/K projection and before O projection.
    pub fn decode_projected(
        &mut self,
        mut query: Vec<f32>,
        gate: &[f32],
        mut key: Vec<f32>,
        value: &[f32],
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
    ) -> Result<Vec<f32>> {
        checked_len("attention query", query.len(), HEADS * HEAD_DIM)?;
        checked_len("attention gate", gate.len(), HEADS * HEAD_DIM)?;
        checked_len("attention key", key.len(), KV_WIDTH)?;
        checked_len("attention value", value.len(), KV_WIDTH)?;
        checked_len("q_norm", q_norm_weight.len(), HEAD_DIM)?;
        checked_len("k_norm", k_norm_weight.len(), HEAD_DIM)?;

        for head in 0..HEADS {
            rmsnorm_head(
                &mut query[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                q_norm_weight,
            );
            apply_partial_rope(
                &mut query[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                self.position,
            );
        }
        for head in 0..KV_HEADS {
            rmsnorm_head(
                &mut key[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                k_norm_weight,
            );
            apply_partial_rope(
                &mut key[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                self.position,
            );
        }
        self.cache.push(&key, value)?;
        self.position += 1;

        let tokens = self.cache.len();
        let mut out = vec![0.0; HEADS * HEAD_DIM];
        for q_head in 0..HEADS {
            // 16 query heads / 2 KV heads = 8 virtual groups. This index is
            // the GQA mapping; K/V are never copied or expanded.
            let kv_head = q_head / (HEADS / KV_HEADS);
            let q = &query[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
            let mut scores = Vec::with_capacity(tokens);
            let mut max_score = f32::NEG_INFINITY;
            for token in 0..tokens {
                let score = q
                    .iter()
                    .zip(self.cache.key(token, kv_head))
                    .map(|(&q, k)| q * k)
                    .sum::<f32>()
                    * 0.0625; // 256^-0.5
                max_score = max_score.max(score);
                scores.push(score);
            }
            let denom = scores.iter().map(|s| (*s - max_score).exp()).sum::<f32>();
            let target = &mut out[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
            for (token, score) in scores.into_iter().enumerate() {
                let probability = (score - max_score).exp() / denom;
                for (target, v) in target.iter_mut().zip(self.cache.value(token, kv_head)) {
                    *target += probability * v;
                }
            }
        }
        for (value, &gate) in out.iter_mut().zip(gate) {
            *value *= sigmoid(gate);
        }
        Ok(out)
    }

    /// Complete Phase-13 Q4 path: normalize/projection -> attention -> O
    /// projection -> residual. It is intentionally a CPU oracle; Metal/CUDA
    /// kernels must prove parity against this path rather than share it.
    pub fn decode_q4(
        &mut self,
        hidden: &[f32],
        weights: &Q4FullAttentionWeights<'_>,
    ) -> Result<Vec<f32>> {
        checked_len("attention hidden", hidden.len(), HIDDEN)?;
        weights.validate()?;
        let q_gate = q4k_gemv(weights.q_proj, hidden, PROJECTED_Q_GATE, HIDDEN);
        let (query, gate) = split_query_gate(&q_gate)?;
        let key = q4k_gemv(weights.k_proj, hidden, KV_WIDTH, HIDDEN);
        let value = q4k_gemv(weights.v_proj, hidden, KV_WIDTH, HIDDEN);
        let attended = self.decode_projected(
            query,
            &gate,
            key,
            &value,
            weights.q_norm_weight,
            weights.k_norm_weight,
        )?;
        let projected = q4k_gemv(weights.o_proj, &attended, HIDDEN, HEADS * HEAD_DIM);
        Ok(hidden.iter().zip(projected).map(|(a, b)| a + b).collect())
    }
}

fn rmsnorm_head(values: &mut [f32], weight: &[f32]) {
    let inv_rms = 1.0 / (values.iter().map(|v| v * v).sum::<f32>() / HEAD_DIM as f32 + 1e-6).sqrt();
    for (value, &weight) in values.iter_mut().zip(weight) {
        // llama.cpp's Qwen3.5 GGUF conversion stores the already-folded
        // `1 + source_weight` multiplier.
        *value *= inv_rms * weight;
    }
}

/// Applies Qwen's split-half rotary embedding only to the canonical first 64
/// dimensions. `rotate_half` maps `[x0..x31, x32..x63]` to
/// `[-x32..-x63, x0..x31]`; adjacent-pair rotation is a different layout.
/// The remaining 192 dimensions are provably untouched.
pub fn apply_partial_rope(values: &mut [f32], position: u64) {
    debug_assert_eq!(values.len(), HEAD_DIM);
    let half = Qwen36Geometry::ROTARY_SUBDIM / 2;
    for index in 0..half {
        let exponent = (2 * index) as f32 / Qwen36Geometry::ROTARY_SUBDIM as f32;
        let angle = position as f32 / Qwen36Geometry::ROPE_THETA.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let first = values[index];
        let second = values[index + half];
        values[index] = first * cos - second * sin;
        values[index + half] = second * cos + first * sin;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn broker() -> MemoryBroker {
        MemoryBroker::new(Bytes(32 * 1024 * 1024))
    }

    #[test]
    fn cache_reserves_exact_bf16_kv_bytes_before_allocating() {
        let broker = broker();
        let cache = Bf16KvCache::new(&broker, LayerId(3), 8).unwrap();
        assert_eq!(cache.layer(), LayerId(3));
        assert_eq!(
            broker.snapshot().reserved,
            Bf16KvCache::bytes_for_tokens(8).unwrap()
        );
        drop(cache);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn all_ten_full_attention_layers_fit_the_bf16_reference_accounting() {
        let broker = broker();
        let mut layers = Vec::new();
        for layer in 0..Qwen36Geometry::NUM_LAYERS {
            if Qwen36Geometry::layer_kind(LayerId(layer as u8))
                == crate::ids::LayerKind::FullAttention
            {
                layers.push(FullAttentionLayer::new(&broker, LayerId(layer as u8), 8).unwrap());
            }
        }
        assert_eq!(layers.len(), Qwen36Geometry::FULL_ATTENTION_LAYERS);
        assert_eq!(
            broker.snapshot().reserved,
            Bytes(Bf16KvCache::bytes_for_tokens(8).unwrap().0 * 10)
        );
    }

    #[test]
    fn partial_rope_rotates_only_the_first_64_dimensions() {
        let mut value = vec![0.0; HEAD_DIM];
        value[0] = 1.0;
        value[64] = 7.0;
        apply_partial_rope(&mut value, 1);
        assert_ne!(value[0], 1.0);
        assert_ne!(value[32], 0.0);
        assert_eq!(value[1], 0.0);
        assert_eq!(value[64], 7.0);
        assert!(value[65..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn q_projection_splits_query_and_gate_within_each_head() {
        let broker = broker();
        let mut projected = Qwen36Activation::zeros(&broker, PROJECTED_Q_GATE).unwrap();
        for head in 0..HEADS {
            let source = head * HEAD_DIM * 2;
            projected.values[source] = 1000.0 + head as f32;
            projected.values[source + HEAD_DIM] = 2000.0 + head as f32;
        }
        let (query, gate) = split_query_gate_accounted(&broker, &projected).unwrap();
        for head in 0..HEADS {
            assert_eq!(query.values[head * HEAD_DIM], 1000.0 + head as f32);
            assert_eq!(gate.values[head * HEAD_DIM], 2000.0 + head as f32);
        }
    }

    #[test]
    fn qwen_head_rmsnorm_uses_gguf_folded_weight() {
        let mut values = vec![2.0; HEAD_DIM];
        rmsnorm_head(&mut values, &vec![1.0; HEAD_DIM]);
        assert!(values.iter().all(|&value| (value - 1.0).abs() < 1e-5));
    }

    #[test]
    fn gqa_is_virtual_and_all_query_heads_observe_their_source_kv_head() {
        let mut layer = FullAttentionLayer::new(&broker(), LayerId(3), 4).unwrap();
        let q = vec![1.0; HEADS * HEAD_DIM];
        let gate = vec![20.0; HEADS * HEAD_DIM];
        // Identical zero keys make the two causal scores equal at both
        // positions, independently of their different RoPE positions.
        let k = vec![0.0; KV_WIDTH];
        let mut v = vec![0.0; KV_WIDTH];
        v[..HEAD_DIM].fill(3.0);
        v[HEAD_DIM..].fill(9.0);
        let out = layer
            .decode_projected(q, &gate, k, &v, &vec![1.0; HEAD_DIM], &vec![1.0; HEAD_DIM])
            .unwrap();
        assert_eq!(layer.cache_len(), 1);
        assert!((out[0] - 3.0).abs() < 0.02);
        assert!((out[7 * HEAD_DIM] - 3.0).abs() < 0.02);
        assert!((out[8 * HEAD_DIM] - 9.0).abs() < 0.02);
        assert!((out[15 * HEAD_DIM] - 9.0).abs() < 0.02);
    }

    #[test]
    fn attention_is_causal_over_bf16_kv_history() {
        let mut layer = FullAttentionLayer::new(&broker(), LayerId(3), 4).unwrap();
        let weights = vec![1.0; HEAD_DIM];
        let q = vec![1.0; HEADS * HEAD_DIM];
        let gate = vec![20.0; HEADS * HEAD_DIM];
        // Identical zero keys make the two causal scores equal at both
        // positions, independently of their different RoPE positions.
        let k = vec![0.0; KV_WIDTH];
        let first = vec![2.0; KV_WIDTH];
        let second = vec![6.0; KV_WIDTH];
        let one = layer
            .decode_projected(q.clone(), &gate, k.clone(), &first, &weights, &weights)
            .unwrap();
        let two = layer
            .decode_projected(q, &gate, k, &second, &weights, &weights)
            .unwrap();
        assert!((one[0] - 2.0).abs() < 0.02);
        assert!((two[0] - 4.0).abs() < 0.02);
    }

    fn xorshift_activation(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 3.0
    }

    /// Phase 27 production-path differential test: the same
    /// `FullAttentionLayer::decode_projected` call sequence, run once against
    /// each backend via `new_with_backend` (so both share the exact
    /// normalize/RoPE/softmax/gate code — only the K/V storage differs).
    /// The sequence spans one full sealed page plus a partial tail so both
    /// the quantized-page and mutable-tail decode paths are exercised.
    #[test]
    fn tqkv_q8_backend_matches_bf16_reference_within_tolerance() {
        let steps = crate::context::tqkv::PAGE_TOKENS + 5;
        let broker_bf16 = broker();
        let broker_tqkv = broker();
        let mut bf16_layer =
            FullAttentionLayer::new_with_backend(&broker_bf16, LayerId(3), steps, BackendChoice::Bf16)
                .unwrap();
        let mut tqkv_layer = FullAttentionLayer::new_with_backend(
            &broker_tqkv,
            LayerId(3),
            steps,
            BackendChoice::Tqkv(crate::context::tqkv::TqkvPrecision::Q8),
        )
        .unwrap();

        let q_norm = vec![1.0; HEAD_DIM];
        let k_norm = vec![1.0; HEAD_DIM];
        let mut state = 0xC0FFEEu64;
        let mut max_abs_diff = 0f32;
        for _ in 0..steps {
            let q: Vec<f32> = (0..HEADS * HEAD_DIM)
                .map(|_| xorshift_activation(&mut state))
                .collect();
            let gate: Vec<f32> = (0..HEADS * HEAD_DIM).map(|_| 4.0).collect();
            let k: Vec<f32> = (0..KV_WIDTH).map(|_| xorshift_activation(&mut state)).collect();
            let v: Vec<f32> = (0..KV_WIDTH).map(|_| xorshift_activation(&mut state)).collect();

            let bf16_out = bf16_layer
                .decode_projected(q.clone(), &gate, k.clone(), &v, &q_norm, &k_norm)
                .unwrap();
            let tqkv_out = tqkv_layer
                .decode_projected(q, &gate, k, &v, &q_norm, &k_norm)
                .unwrap();
            for (a, b) in bf16_out.iter().zip(&tqkv_out) {
                max_abs_diff = max_abs_diff.max((a - b).abs());
            }
        }
        assert_eq!(bf16_layer.cache_len(), steps);
        assert_eq!(tqkv_layer.cache_len(), steps);
        // Q8 is a compressed *oracle*, not bit-exact (spec section 158); the
        // gated output stays close across a page-boundary-crossing sequence.
        assert!(
            max_abs_diff < 0.05,
            "TQKV-Q8 vs BF16 output diverged too far: {max_abs_diff}"
        );
    }
}
