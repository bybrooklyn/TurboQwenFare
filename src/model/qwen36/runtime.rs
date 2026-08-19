//! Fixed Phase-15/18 Qwen3.6 execution graphs. These are the only model
//! runtimes: both bind the canonical forty-layer topology directly to
//! validated TQF tensors, with the Phase-14 resident-expert profile retained
//! as an explicit high-memory oracle alongside the bounded streaming cache.

use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::context::prefix::{GdnCapture, PrefixSnapshotStore};
use crate::dev::inventory::TensorRole;
use crate::error::{ModelError, Result};
use crate::experts::{Qwen36ResidentMoe, Qwen36StreamingMoe, RouterResult, WholeExpertLfuCache};
use crate::ids::{Bytes, LayerId, LayerKind};
use crate::memory::MemoryBroker;
use crate::runtime::decode::top_logit_candidates;
use crate::runtime::{DecodeDiagnostics, DecodeTimings, DecodeToken, LayerHash, RouterTrace};
use crate::sampling::Sampler;

use super::attention::{split_query_gate_accounted, FullAttentionLayer};
use super::gdn::{
    gated_norm_accounted, qk_norm_in_place, recurrent_step_accounted, GdnRecurrentParameters,
    GdnState,
};
use super::geometry::Qwen36Geometry;
use super::weights::{LoadedQwen36Tensor, Qwen36Activation, Qwen36WeightLoader};

struct CommonLayerWeights {
    attn_norm: LoadedQwen36Tensor,
    ffn_norm: LoadedQwen36Tensor,
    moe: BoundMoe,
}

enum BoundMoe {
    Resident(Qwen36ResidentMoe),
    Streaming(Qwen36StreamingMoe),
}

struct GdnLayerWeights {
    qkv: LoadedQwen36Tensor,
    z: LoadedQwen36Tensor,
    alpha: LoadedQwen36Tensor,
    beta: LoadedQwen36Tensor,
    conv: LoadedQwen36Tensor,
    a_neg_exp: LoadedQwen36Tensor,
    dt_bias: LoadedQwen36Tensor,
    gated_norm: LoadedQwen36Tensor,
    out: LoadedQwen36Tensor,
    state: GdnState,
}

struct FullAttentionWeights {
    q: LoadedQwen36Tensor,
    k: LoadedQwen36Tensor,
    v: LoadedQwen36Tensor,
    out: LoadedQwen36Tensor,
    q_norm: LoadedQwen36Tensor,
    k_norm: LoadedQwen36Tensor,
    state: FullAttentionLayer,
}

enum AttentionLayer {
    Gdn(GdnLayerWeights),
    Full(FullAttentionWeights),
}

struct Qwen36Layer {
    id: LayerId,
    common: CommonLayerWeights,
    attention: AttentionLayer,
}

/// High-memory, fixed-graph reference runtime for exactly Qwen3.6-35B-A3B.
/// `open_resident` intentionally retains all expert matrices for correctness;
/// it can therefore fail honestly against a normal 4 GiB runtime budget.
/// The bounded out-of-core profile does not replace this oracle.
pub struct Qwen36ReferenceRuntime {
    broker: MemoryBroker,
    loader: Arc<Qwen36WeightLoader>,
    expert_cache: Option<WholeExpertLfuCache>,
    embedding: LoadedQwen36Tensor,
    final_norm: LoadedQwen36Tensor,
    lm_head: LoadedQwen36Tensor,
    layers: Vec<Qwen36Layer>,
    decode_index: u64,
}

/// Bounded-memory reference graph. It keeps only session state and the
/// global routed-expert cache resident; every embedding, LM-head, norm, and
/// layer projection lease is acquired immediately before use and released at
/// the next fixed-graph boundary. This is intentionally slower than the
/// resident oracle, but it makes the broker contract mechanically true while
/// later phases select the fast mapped/GPU policy.
pub struct Qwen36BoundedReferenceRuntime {
    broker: MemoryBroker,
    loader: Arc<Qwen36WeightLoader>,
    expert_cache: WholeExpertLfuCache,
    layers: Vec<BoundedLayerState>,
    decode_index: u64,
}

enum BoundedLayerState {
    Gdn {
        id: LayerId,
        state: GdnState,
    },
    Full {
        id: LayerId,
        state: FullAttentionLayer,
    },
}

impl Qwen36ReferenceRuntime {
    pub fn open_resident(path: &Path, broker: MemoryBroker, max_context: usize) -> Result<Self> {
        let loader = Arc::new(Qwen36WeightLoader::open(path, broker.clone())?);
        let embedding = loader.load(TensorRole::TokenEmbedding, None)?;
        let final_norm = loader.load(TensorRole::FinalNorm, None)?;
        let lm_head = loader.load(TensorRole::LmHead, None)?;
        let mut layers = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        for index in 0..Qwen36Geometry::NUM_LAYERS {
            let layer = LayerId(index as u8);
            let common = CommonLayerWeights {
                attn_norm: loader.load(TensorRole::AttnNorm, Some(layer))?,
                ffn_norm: loader.load(TensorRole::FfnNorm, Some(layer))?,
                moe: BoundMoe::Resident(Qwen36ResidentMoe::open(&loader, layer)?),
            };
            let attention = match Qwen36Geometry::layer_kind(layer) {
                LayerKind::GatedDeltaNet => AttentionLayer::Gdn(GdnLayerWeights {
                    qkv: loader.load(TensorRole::GdnInProjQkv, Some(layer))?,
                    z: loader.load(TensorRole::GdnInProjZ, Some(layer))?,
                    alpha: loader.load(TensorRole::GdnInProjA, Some(layer))?,
                    beta: loader.load(TensorRole::GdnInProjB, Some(layer))?,
                    conv: loader.load(TensorRole::GdnConv1d, Some(layer))?,
                    a_neg_exp: loader.load(TensorRole::GdnALog, Some(layer))?,
                    dt_bias: loader.load(TensorRole::GdnDtBias, Some(layer))?,
                    gated_norm: loader.load(TensorRole::GdnGatedNorm, Some(layer))?,
                    out: loader.load(TensorRole::GdnOutProj, Some(layer))?,
                    state: GdnState::new(&broker, layer)?,
                }),
                LayerKind::FullAttention => AttentionLayer::Full(FullAttentionWeights {
                    q: loader.load(TensorRole::AttnQProj, Some(layer))?,
                    k: loader.load(TensorRole::AttnKProj, Some(layer))?,
                    v: loader.load(TensorRole::AttnVProj, Some(layer))?,
                    out: loader.load(TensorRole::AttnOProj, Some(layer))?,
                    q_norm: loader.load(TensorRole::AttnQNorm, Some(layer))?,
                    k_norm: loader.load(TensorRole::AttnKNorm, Some(layer))?,
                    state: FullAttentionLayer::new(&broker, layer, max_context)?,
                }),
            };
            layers.push(Qwen36Layer {
                id: layer,
                common,
                attention,
            });
        }
        Ok(Self {
            broker,
            loader,
            expert_cache: None,
            embedding,
            final_norm,
            lm_head,
            layers,
            decode_index: 0,
        })
    }

    /// Opens the fixed graph against canonical v2 whole-expert storage.
    /// The shared weights remain resident for this reference path, while the
    /// exact router binds selected Q4_K experts through one global LFU cache.
    /// `expert_cache_bytes` is a strict sub-budget: it must be chosen so the
    /// broker can still reserve activations and context alongside cache hits.
    pub fn open_streaming(
        path: &Path,
        broker: MemoryBroker,
        max_context: usize,
        expert_cache_bytes: Bytes,
    ) -> Result<Self> {
        let loader = Arc::new(Qwen36WeightLoader::open(path, broker.clone())?);
        if !loader.manifest().uses_expert_superextents() {
            return Err(ModelError::Unsupported(
                "streaming reference runtime requires canonical expert-superextent TQF conversion"
                    .to_string(),
            )
            .into());
        }
        let embedding = loader.load(TensorRole::TokenEmbedding, None)?;
        let final_norm = loader.load(TensorRole::FinalNorm, None)?;
        let lm_head = loader.load(TensorRole::LmHead, None)?;
        let mut layers = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        for index in 0..Qwen36Geometry::NUM_LAYERS {
            let layer = LayerId(index as u8);
            let common = CommonLayerWeights {
                attn_norm: loader.load(TensorRole::AttnNorm, Some(layer))?,
                ffn_norm: loader.load(TensorRole::FfnNorm, Some(layer))?,
                moe: BoundMoe::Streaming(Qwen36StreamingMoe::open(&loader, layer)?),
            };
            let attention = match Qwen36Geometry::layer_kind(layer) {
                LayerKind::GatedDeltaNet => AttentionLayer::Gdn(GdnLayerWeights {
                    qkv: loader.load(TensorRole::GdnInProjQkv, Some(layer))?,
                    z: loader.load(TensorRole::GdnInProjZ, Some(layer))?,
                    alpha: loader.load(TensorRole::GdnInProjA, Some(layer))?,
                    beta: loader.load(TensorRole::GdnInProjB, Some(layer))?,
                    conv: loader.load(TensorRole::GdnConv1d, Some(layer))?,
                    a_neg_exp: loader.load(TensorRole::GdnALog, Some(layer))?,
                    dt_bias: loader.load(TensorRole::GdnDtBias, Some(layer))?,
                    gated_norm: loader.load(TensorRole::GdnGatedNorm, Some(layer))?,
                    out: loader.load(TensorRole::GdnOutProj, Some(layer))?,
                    state: GdnState::new(&broker, layer)?,
                }),
                LayerKind::FullAttention => AttentionLayer::Full(FullAttentionWeights {
                    q: loader.load(TensorRole::AttnQProj, Some(layer))?,
                    k: loader.load(TensorRole::AttnKProj, Some(layer))?,
                    v: loader.load(TensorRole::AttnVProj, Some(layer))?,
                    out: loader.load(TensorRole::AttnOProj, Some(layer))?,
                    q_norm: loader.load(TensorRole::AttnQNorm, Some(layer))?,
                    k_norm: loader.load(TensorRole::AttnKNorm, Some(layer))?,
                    state: FullAttentionLayer::new(&broker, layer, max_context)?,
                }),
            };
            layers.push(Qwen36Layer {
                id: layer,
                common,
                attention,
            });
        }
        Ok(Self {
            broker,
            loader,
            expert_cache: Some(WholeExpertLfuCache::new(expert_cache_bytes)),
            embedding,
            final_norm,
            lm_head,
            layers,
            decode_index: 0,
        })
    }

    pub fn expert_cache_stats(&self) -> Option<crate::experts::ExpertCacheStats> {
        self.expert_cache.as_ref().map(WholeExpertLfuCache::stats)
    }

    /// Phase 26 chunked prefill (spec §298): layer-outer processing of a
    /// prompt with per-(layer, chunk) expert-set dedup. Each layer's
    /// attention/recurrent state advances per token in exact order, so
    /// the result is identical to the per-token loop; the MoE tail
    /// instead routes every chunk row, fetches each *distinct* absent
    /// expert once, and re-uses it for every row that selected it.
    /// The chunk size auto-halves when the broker cannot reserve chunk
    /// scratch (spec §152: "chunk size auto-reduces when context/scratch
    /// pressure would violate --memory"). `TQF_PREFILL_CHUNK` overrides
    /// the seed (default 4096).
    pub fn prefill_greedy(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() {
            return Err(ModelError::Unsupported(
                "chunked prefill requires a nonempty prompt".to_string(),
            )
            .into());
        }
        let seed: usize = std::env::var("TQF_PREFILL_CHUNK")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4096)
            .max(1);
        for chunk in prompt[..prompt.len() - 1].chunks(seed) {
            let mut chunk_size = chunk.len();
            loop {
                match self.prefill_chunk(&chunk[..chunk_size]) {
                    Ok(()) => break,
                    Err(crate::error::TqfError::Memory(_)) if chunk_size > 1 => {
                        chunk_size /= 2;
                    }
                    Err(error) => return Err(error),
                }
            }
            if chunk_size < chunk.len() {
                for remainder in chunk[chunk_size..].chunks(chunk_size.max(1)) {
                    self.prefill_chunk(remainder)?;
                }
            }
        }
        // The final prompt token must produce real logits for greedy
        // selection; one ordinary decode step does that.
        let decoded = self.decode_greedy(prompt[prompt.len() - 1])?;
        Ok(decoded.token)
    }

    fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut hiddens: Vec<Qwen36Activation> = tokens
            .iter()
            .map(|token| {
                if *token as usize >= Qwen36Geometry::VOCAB_SIZE {
                    return Err(ModelError::Shape {
                        tensor: "Qwen prefill token",
                        expected: Qwen36Geometry::VOCAB_SIZE,
                        actual: *token as usize,
                    }
                    .into());
                }
                self.embedding.row(&self.broker, *token as usize)
            })
            .collect::<Result<_>>()?;
        for layer in &mut self.layers {
            let mut ffn_inputs = Vec::with_capacity(hiddens.len());
            let mut after_attention = Vec::with_capacity(hiddens.len());
            for hidden in &hiddens {
                let attn_weight = layer.common.attn_norm.vector(&self.broker)?;
                let normalized =
                    Qwen36Activation::qwen_rmsnorm(&self.broker, hidden, &attn_weight.values)?;
                let attended = match &mut layer.attention {
                    AttentionLayer::Gdn(gdn) => run_gdn(&self.broker, gdn, &normalized)?,
                    AttentionLayer::Full(full) => {
                        run_full_attention(&self.broker, full, &normalized)?
                    }
                };
                let after = Qwen36Activation::residual_add(&self.broker, hidden, &attended)?;
                let ffn_weight = layer.common.ffn_norm.vector(&self.broker)?;
                ffn_inputs.push(Qwen36Activation::qwen_rmsnorm(
                    &self.broker,
                    &after,
                    &ffn_weight.values,
                )?);
                after_attention.push(after);
            }
            let moe_outputs = match &mut layer.common.moe {
                BoundMoe::Streaming(moe) => {
                    let cache = self.expert_cache.as_mut().ok_or_else(|| {
                        ModelError::Unsupported(
                            "streaming MoE missing global expert cache".to_string(),
                        )
                    })?;
                    moe.forward_batch(&self.loader, cache, &self.broker, &ffn_inputs)?
                        .0
                }
                BoundMoe::Resident(moe) => ffn_inputs
                    .iter()
                    .map(|input| moe.forward(&self.broker, input).map(|(output, _)| output))
                    .collect::<Result<Vec<_>>>()?,
            };
            for ((hidden, after), moe) in hiddens.iter_mut().zip(after_attention).zip(moe_outputs) {
                *hidden = Qwen36Activation::residual_add(&self.broker, &after, &moe)?;
            }
        }
        self.decode_index = self.decode_index.saturating_add(tokens.len() as u64);
        Ok(())
    }

    /// Phase 20 A/B seam: toggles the GPU-resident expert path on the
    /// streaming whole-expert cache (a no-op on the resident runtime, which
    /// has no expert cache). Call before decode; already-resident values
    /// stay correct either way — this only steers future admissions, the
    /// same contract as `WholeExpertLfuCache::set_gpu_enabled`.
    pub fn set_expert_gpu_enabled(&mut self, enabled: bool) {
        if let Some(cache) = self.expert_cache.as_mut() {
            cache.set_gpu_enabled(enabled);
        }
    }

    /// Greedy decode: the signature and behavior every qualification
    /// harness in `docs/research/qualification/` was recorded against.
    /// It delegates to `decode_step` with `Sampler::Greedy`, which returns
    /// the argmax verbatim, so parity records stay valid by construction.
    pub fn decode_greedy(&mut self, input_token: u32) -> Result<DecodeToken> {
        self.decode_step(input_token, &mut Sampler::Greedy, &[])
    }

    /// Runs one actual checkpoint token through embedding, all forty
    /// attention/GDN+MoE layers, final norm, Q6_K LM head, and token
    /// selection.  The returned diagnostics are the Phase-15 qualification
    /// artifacts; no generic callback or protocol code enters this loop.
    pub fn decode_step(
        &mut self,
        input_token: u32,
        sampler: &mut Sampler,
        history: &[u32],
    ) -> Result<DecodeToken> {
        if input_token as usize >= Qwen36Geometry::VOCAB_SIZE {
            return Err(ModelError::Shape {
                tensor: "Qwen input token",
                expected: Qwen36Geometry::VOCAB_SIZE,
                actual: input_token as usize,
            }
            .into());
        }
        let mut timings = DecodeTimings::default();
        let decode_index = self.decode_index;
        let start = Instant::now();
        let mut hidden = self.embedding.row(&self.broker, input_token as usize)?;
        timings.embedding = start.elapsed();

        let mut per_layer_hashes = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        let mut router_trace = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        for layer in &mut self.layers {
            let start = Instant::now();
            let (next, route) = layer.forward(
                &self.broker,
                &self.loader,
                self.expert_cache.as_mut(),
                &hidden,
            )?;
            timings.layers.push((layer.id, start.elapsed()));
            per_layer_hashes.push(LayerHash {
                layer: layer.id,
                hash: hash_activation(&next),
            });
            maybe_dump_activation(decode_index, input_token, layer.id, &next)?;
            router_trace.push(RouterTrace {
                layer: layer.id,
                route,
            });
            hidden = next;
        }

        let start = Instant::now();
        let final_weight = self.final_norm.vector(&self.broker)?;
        hidden = Qwen36Activation::qwen_rmsnorm(&self.broker, &hidden, &final_weight.values)?;
        timings.final_norm = start.elapsed();

        let start = Instant::now();
        let logits = self.lm_head.matvec(&self.broker, &hidden.values)?;
        timings.lm_head = start.elapsed();
        let start = Instant::now();
        // `top_logits` stays unconditional: it is a diagnostics artifact
        // the qualification harnesses read, and it is also the exact argmax
        // `Sampler::Greedy` returns — so the greedy path's arithmetic is
        // unchanged, not merely equivalent.
        let top_logits = top_logit_candidates(&logits.values);
        let token = sampler.select(&logits.values, history, top_logits[0].token);
        timings.sampling = start.elapsed();
        self.decode_index = self.decode_index.saturating_add(1);
        Ok(DecodeToken {
            token,
            diagnostics: DecodeDiagnostics {
                per_layer_hashes,
                router_trace,
                top_logits,
                timings,
            },
        })
    }

    /// Begins an isolated model session.  The resident weights remain pinned,
    /// while every per-session GDN recurrence and BF16 attention cache is
    /// cleared before a new normalized request is prefixed.
    pub fn reset_session(&mut self) {
        self.decode_index = 0;
        for layer in &mut self.layers {
            match &mut layer.attention {
                AttentionLayer::Gdn(gdn) => gdn.state.reset(),
                AttentionLayer::Full(attention) => attention.state.reset(),
            }
        }
    }
}

impl Qwen36BoundedReferenceRuntime {
    pub fn open(
        path: &Path,
        broker: MemoryBroker,
        max_context: usize,
        expert_cache_bytes: Bytes,
    ) -> Result<Self> {
        let loader = Arc::new(Qwen36WeightLoader::open(path, broker.clone())?);
        if !loader.manifest().uses_expert_superextents() {
            return Err(ModelError::Unsupported(
                "bounded runtime requires canonical expert-superextent TQF conversion".to_string(),
            )
            .into());
        }
        let mut layers = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        for index in 0..Qwen36Geometry::NUM_LAYERS {
            let id = LayerId(index as u8);
            let state = match Qwen36Geometry::layer_kind(id) {
                LayerKind::GatedDeltaNet => BoundedLayerState::Gdn {
                    id,
                    state: GdnState::new(&broker, id)?,
                },
                LayerKind::FullAttention => BoundedLayerState::Full {
                    id,
                    state: FullAttentionLayer::new(&broker, id, max_context)?,
                },
            };
            layers.push(state);
        }
        // Cache capacity is a ceiling, not an up-front reservation. Keep the
        // largest single core tensor plus bounded activation/I/O slack
        // available so a warm expert cache cannot make embedding or LM-head
        // execution fail under the same hard broker budget.
        let largest_core_extent = [TensorRole::TokenEmbedding, TensorRole::LmHead]
            .into_iter()
            .map(|role| loader.stored_bytes(role, None))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|bytes| bytes.0)
            .max()
            .expect("fixed graph always has embedding and LM head");
        let transient_slack = 64 * 1024 * 1024_u64;
        let snapshot = broker.snapshot();
        let available_after_state = snapshot.budget.0.saturating_sub(snapshot.reserved.0);
        let required_headroom = largest_core_extent.saturating_add(transient_slack);
        let cache_capacity = expert_cache_bytes
            .0
            .min(available_after_state.saturating_sub(required_headroom));
        let one_expert = loader
            .expert_stored_bytes(LayerId(0), crate::ids::ExpertId(0))?
            .0;
        if cache_capacity < one_expert {
            return Err(crate::error::MemoryError::BudgetExceeded {
                requested: one_expert.saturating_add(required_headroom),
                available: available_after_state,
                owner: "ExpertPinned".to_string(),
                suggestion: "reduce --context or increase --memory so one expert and the largest core tensor fit together".to_string(),
            }
            .into());
        }
        Ok(Self {
            broker,
            loader,
            expert_cache: WholeExpertLfuCache::new(Bytes(cache_capacity)),
            layers,
            decode_index: 0,
        })
    }

    pub fn expert_cache_stats(&self) -> crate::experts::ExpertCacheStats {
        self.expert_cache.stats()
    }

    /// Phase 20 A/B seam: toggles the GPU-resident expert path on the
    /// whole-expert cache. Call before decode; already-resident values stay
    /// correct either way — this only steers future admissions, the same
    /// contract as `WholeExpertLfuCache::set_gpu_enabled`.
    pub fn set_expert_gpu_enabled(&mut self, enabled: bool) {
        self.expert_cache.set_gpu_enabled(enabled);
    }

    /// Greedy decode: the signature and behavior every qualification
    /// harness in `docs/research/qualification/` was recorded against.
    /// It delegates to `decode_step` with `Sampler::Greedy`, which returns
    /// the argmax verbatim, so parity records stay valid by construction.
    pub fn decode_greedy(&mut self, input_token: u32) -> Result<DecodeToken> {
        self.decode_step(input_token, &mut Sampler::Greedy, &[])
    }

    pub fn decode_step(
        &mut self,
        input_token: u32,
        sampler: &mut Sampler,
        history: &[u32],
    ) -> Result<DecodeToken> {
        if input_token as usize >= Qwen36Geometry::VOCAB_SIZE {
            return Err(ModelError::Shape {
                tensor: "Qwen input token",
                expected: Qwen36Geometry::VOCAB_SIZE,
                actual: input_token as usize,
            }
            .into());
        }
        let mut timings = DecodeTimings::default();
        let decode_index = self.decode_index;
        let start = Instant::now();
        let mut hidden = {
            let embedding = self.loader.load(TensorRole::TokenEmbedding, None)?;
            embedding.row(&self.broker, input_token as usize)?
        };
        timings.embedding = start.elapsed();
        let mut per_layer_hashes = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        let mut router_trace = Vec::with_capacity(Qwen36Geometry::NUM_LAYERS);
        for layer in &mut self.layers {
            let start = Instant::now();
            let (id, next, route) = bounded_layer_forward(
                &self.broker,
                &self.loader,
                &mut self.expert_cache,
                layer,
                &hidden,
                decode_index,
                input_token,
            )?;
            timings.layers.push((id, start.elapsed()));
            per_layer_hashes.push(LayerHash {
                layer: id,
                hash: hash_activation(&next),
            });
            maybe_dump_activation(decode_index, input_token, id, &next)?;
            router_trace.push(RouterTrace { layer: id, route });
            hidden = next;
        }
        let start = Instant::now();
        let final_weight = self.loader.load(TensorRole::FinalNorm, None)?;
        let final_weight = final_weight.vector(&self.broker)?;
        hidden = Qwen36Activation::qwen_rmsnorm(&self.broker, &hidden, &final_weight.values)?;
        timings.final_norm = start.elapsed();
        let start = Instant::now();
        let logits = {
            let lm_head = self.loader.load(TensorRole::LmHead, None)?;
            lm_head.matvec(&self.broker, &hidden.values)?
        };
        timings.lm_head = start.elapsed();
        let start = Instant::now();
        // `top_logits` stays unconditional: it is a diagnostics artifact
        // the qualification harnesses read, and it is also the exact argmax
        // `Sampler::Greedy` returns — so the greedy path's arithmetic is
        // unchanged, not merely equivalent.
        let top_logits = top_logit_candidates(&logits.values);
        let token = sampler.select(&logits.values, history, top_logits[0].token);
        timings.sampling = start.elapsed();
        self.decode_index = self.decode_index.saturating_add(1);
        Ok(DecodeToken {
            token,
            diagnostics: DecodeDiagnostics {
                per_layer_hashes,
                router_trace,
                top_logits,
                timings,
            },
        })
    }

    pub fn reset_session(&mut self) {
        self.decode_index = 0;
        for layer in &mut self.layers {
            match layer {
                BoundedLayerState::Gdn { state, .. } => state.reset(),
                BoundedLayerState::Full { state, .. } => state.reset(),
            }
        }
    }

    /// Phase 30 prefix reuse (spec §66-67): captures every layer's current
    /// TQKV/GDN state and persists it under `tokens`' exact prefix hash.
    /// Full-attention layers running the BF16 backend contribute nothing
    /// (`capture_tqkv_for_snapshot` returns `None` for them — prefix dedup
    /// is a TQKV-specific mechanism, `TQF_TQKV_ENABLED=1` is required for
    /// this to capture anything useful).
    pub fn snapshot_session(
        &self,
        store: &PrefixSnapshotStore,
        tokens: &[u32],
    ) -> Result<[u8; 32]> {
        let mut full_attention = Vec::new();
        let mut gdn = Vec::new();
        for layer in &self.layers {
            match layer {
                BoundedLayerState::Full { id, state } => {
                    if let Some(capture) = state.capture_tqkv_for_snapshot(*id) {
                        full_attention.push(capture);
                    }
                }
                BoundedLayerState::Gdn { id, state } => {
                    gdn.push(GdnCapture {
                        layer: *id,
                        bytes: state.to_bytes(),
                    });
                }
            }
        }
        store.store(tokens, &full_attention, &gdn)
    }

    /// Restores session state from a stored exact-prefix snapshot, if one
    /// exists. Returns `true` (and leaves `decode_index` at the snapshot's
    /// token count, ready to continue from there) on a hit, `false` on a
    /// miss with no state changed.
    pub fn restore_session(&mut self, store: &PrefixSnapshotStore, tokens: &[u32]) -> Result<bool> {
        let Some(loaded) = store.load(tokens, &self.broker)? else {
            return Ok(false);
        };
        for restored in loaded.full_attention {
            for layer in &mut self.layers {
                if let BoundedLayerState::Full { id, state } = layer {
                    if *id == restored.layer {
                        state.restore_tqkv_snapshot(
                            restored.sealed,
                            restored.tail_keys,
                            restored.tail_values,
                            restored.position,
                        )?;
                        break;
                    }
                }
            }
        }
        for restored in loaded.gdn {
            for layer in &mut self.layers {
                if let BoundedLayerState::Gdn { id, state } = layer {
                    if *id == restored.layer {
                        *state = restored.state;
                        break;
                    }
                }
            }
        }
        self.decode_index = loaded.token_count as u64;
        Ok(true)
    }
}

fn bounded_layer_forward(
    broker: &MemoryBroker,
    loader: &Arc<Qwen36WeightLoader>,
    expert_cache: &mut WholeExpertLfuCache,
    layer: &mut BoundedLayerState,
    input: &Qwen36Activation,
    decode_index: u64,
    input_token: u32,
) -> Result<(LayerId, Qwen36Activation, RouterResult)> {
    let id = match layer {
        BoundedLayerState::Gdn { id, .. } | BoundedLayerState::Full { id, .. } => *id,
    };
    let attn_weight = loader.load(TensorRole::AttnNorm, Some(id))?;
    let attn_weight = attn_weight.vector(broker)?;
    maybe_dump_stage(decode_index, input_token, id, "layer_input", &input.values)?;
    let normalized = Qwen36Activation::qwen_rmsnorm(broker, input, &attn_weight.values)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "attn_norm",
        &normalized.values,
    )?;
    let attended = match layer {
        BoundedLayerState::Gdn { state, .. } => bounded_gdn(
            broker,
            loader,
            id,
            state,
            &normalized,
            decode_index,
            input_token,
        )?,
        BoundedLayerState::Full { state, .. } => bounded_full(
            broker,
            loader,
            id,
            state,
            &normalized,
            decode_index,
            input_token,
        )?,
    };
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "attn_output",
        &attended.values,
    )?;
    let after_attention = Qwen36Activation::residual_add(broker, input, &attended)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "attn_residual",
        &after_attention.values,
    )?;
    let ffn_weight = loader.load(TensorRole::FfnNorm, Some(id))?;
    let ffn_weight = ffn_weight.vector(broker)?;
    let ffn_input = Qwen36Activation::qwen_rmsnorm(broker, &after_attention, &ffn_weight.values)?;
    maybe_dump_stage(decode_index, input_token, id, "ffn_norm", &ffn_input.values)?;
    let mut moe = Qwen36StreamingMoe::open(loader, id)?;
    let (moe, route) = moe.forward_with_observer(
        loader,
        expert_cache,
        broker,
        &ffn_input,
        |stage, activation| {
            maybe_dump_stage(
                decode_index,
                input_token,
                id,
                match stage {
                    "shared" => "moe_shared_output",
                    "combined" => "moe_combined_output",
                    _ => "moe_unknown_output",
                },
                &activation.values,
            )
        },
    )?;
    maybe_dump_stage(decode_index, input_token, id, "moe_output", &moe.values)?;
    let output = Qwen36Activation::residual_add(broker, &after_attention, &moe)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "layer_output",
        &output.values,
    )?;
    Ok((id, output, route))
}

fn bounded_gdn(
    broker: &MemoryBroker,
    loader: &Qwen36WeightLoader,
    id: LayerId,
    state: &mut GdnState,
    input: &Qwen36Activation,
    decode_index: u64,
    input_token: u32,
) -> Result<Qwen36Activation> {
    let qkv = loader
        .load(TensorRole::GdnInProjQkv, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_qkv", &qkv.values)?;
    let conv = loader
        .load(TensorRole::GdnConv1d, Some(id))?
        .gdn_conv1d_weights(broker)?;
    let mut convolved = Qwen36Activation::zeros(broker, Qwen36Geometry::GDN_CONV_CHANNELS)?;
    state
        .conv_tail_mut()
        .step_without_bias_into(&qkv.values, &conv.values, &mut convolved.values);
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "gdn_conv_silu",
        &convolved.values,
    )?;
    let mut q =
        Qwen36Activation::from_slice(broker, &convolved.values[..Qwen36Geometry::GDN_KEY_DIM])?;
    let mut k = Qwen36Activation::from_slice(
        broker,
        &convolved.values[Qwen36Geometry::GDN_KEY_DIM..Qwen36Geometry::GDN_KEY_DIM * 2],
    )?;
    let v =
        Qwen36Activation::from_slice(broker, &convolved.values[Qwen36Geometry::GDN_KEY_DIM * 2..])?;
    qk_norm_in_place(&mut q, &mut k)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_q_norm", &q.values)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_k_norm", &k.values)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_v", &v.values)?;
    let z = loader
        .load(TensorRole::GdnInProjZ, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_z", &z.values)?;
    let alpha = loader
        .load(TensorRole::GdnInProjA, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_alpha", &alpha.values)?;
    let beta = loader
        .load(TensorRole::GdnInProjB, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(decode_index, input_token, id, "gdn_beta", &beta.values)?;
    let a_neg_exp = loader.load(TensorRole::GdnALog, Some(id))?.vector(broker)?;
    let dt_bias = loader
        .load(TensorRole::GdnDtBias, Some(id))?
        .vector(broker)?;
    let recurrent = recurrent_step_accounted(
        broker,
        state,
        &q,
        &k,
        &v,
        GdnRecurrentParameters {
            alpha: &alpha.values,
            beta: &beta.values,
            a_neg_exp: &a_neg_exp.values,
            dt_bias: &dt_bias.values,
        },
    )?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "gdn_recurrent",
        &recurrent.values,
    )?;
    let gated_weight = loader
        .load(TensorRole::GdnGatedNorm, Some(id))?
        .vector(broker)?;
    let gated = gated_norm_accounted(broker, &recurrent, &z, &gated_weight.values)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "gdn_gated_norm",
        &gated.values,
    )?;
    let output = loader
        .load(TensorRole::GdnOutProj, Some(id))?
        .matvec(broker, &gated.values)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "gdn_out_projection",
        &output.values,
    )?;
    Ok(output)
}

fn bounded_full(
    broker: &MemoryBroker,
    loader: &Qwen36WeightLoader,
    id: LayerId,
    state: &mut FullAttentionLayer,
    input: &Qwen36Activation,
    decode_index: u64,
    input_token: u32,
) -> Result<Qwen36Activation> {
    let projected = loader
        .load(TensorRole::AttnQProj, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "full_q_gate_projection",
        &projected.values,
    )?;
    let (mut query, gate) = split_query_gate_accounted(broker, &projected)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "full_query_raw",
        &query.values,
    )?;
    maybe_dump_stage(decode_index, input_token, id, "full_gate_raw", &gate.values)?;
    let mut key = loader
        .load(TensorRole::AttnKProj, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(decode_index, input_token, id, "full_key_raw", &key.values)?;
    let value = loader
        .load(TensorRole::AttnVProj, Some(id))?
        .matvec(broker, &input.values)?;
    maybe_dump_stage(decode_index, input_token, id, "full_value", &value.values)?;
    let q_norm = loader
        .load(TensorRole::AttnQNorm, Some(id))?
        .vector(broker)?;
    let k_norm = loader
        .load(TensorRole::AttnKNorm, Some(id))?
        .vector(broker)?;
    let attended = state.decode_projected_accounted(
        broker,
        &mut query,
        &gate,
        &mut key,
        &value,
        &q_norm.values,
        &k_norm.values,
    )?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "full_query_rope",
        &query.values,
    )?;
    maybe_dump_stage(decode_index, input_token, id, "full_key_rope", &key.values)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "full_attn_gated",
        &attended.values,
    )?;
    let output = loader
        .load(TensorRole::AttnOProj, Some(id))?
        .matvec(broker, &attended.values)?;
    maybe_dump_stage(
        decode_index,
        input_token,
        id,
        "full_out_projection",
        &output.values,
    )?;
    Ok(output)
}

impl Qwen36Layer {
    fn forward(
        &mut self,
        broker: &MemoryBroker,
        loader: &Arc<Qwen36WeightLoader>,
        expert_cache: Option<&mut WholeExpertLfuCache>,
        input: &Qwen36Activation,
    ) -> Result<(Qwen36Activation, RouterResult)> {
        let attn_weight = self.common.attn_norm.vector(broker)?;
        let normalized = Qwen36Activation::qwen_rmsnorm(broker, input, &attn_weight.values)?;
        let attended = match &mut self.attention {
            AttentionLayer::Gdn(layer) => run_gdn(broker, layer, &normalized)?,
            AttentionLayer::Full(layer) => run_full_attention(broker, layer, &normalized)?,
        };
        let after_attention = Qwen36Activation::residual_add(broker, input, &attended)?;
        let ffn_weight = self.common.ffn_norm.vector(broker)?;
        let ffn_input =
            Qwen36Activation::qwen_rmsnorm(broker, &after_attention, &ffn_weight.values)?;
        let (moe, route) = match &mut self.common.moe {
            BoundMoe::Resident(moe) => moe.forward(broker, &ffn_input)?,
            BoundMoe::Streaming(moe) => moe.forward(
                loader,
                expert_cache.ok_or_else(|| {
                    ModelError::Unsupported("streaming MoE missing global expert cache".to_string())
                })?,
                broker,
                &ffn_input,
            )?,
        };
        Ok((
            Qwen36Activation::residual_add(broker, &after_attention, &moe)?,
            route,
        ))
    }
}

fn run_gdn(
    broker: &MemoryBroker,
    layer: &mut GdnLayerWeights,
    input: &Qwen36Activation,
) -> Result<Qwen36Activation> {
    let qkv = layer.qkv.matvec(broker, &input.values)?;
    // Canonical GGUF stores ssm_conv1d.weight as rank-two {4, 8192}
    // consumed as channel-major depthwise weights; the naive vector view
    // was replaced by this decoding in the Phase 16 parity work.
    let conv_weight = layer.conv.gdn_conv1d_weights(broker)?;
    let mut convolved = Qwen36Activation::zeros(broker, Qwen36Geometry::GDN_CONV_CHANNELS)?;
    layer.state.conv_tail_mut().step_without_bias_into(
        &qkv.values,
        &conv_weight.values,
        &mut convolved.values,
    );
    let mut q =
        Qwen36Activation::from_slice(broker, &convolved.values[..Qwen36Geometry::GDN_KEY_DIM])?;
    let mut k = Qwen36Activation::from_slice(
        broker,
        &convolved.values[Qwen36Geometry::GDN_KEY_DIM..Qwen36Geometry::GDN_KEY_DIM * 2],
    )?;
    let v =
        Qwen36Activation::from_slice(broker, &convolved.values[Qwen36Geometry::GDN_KEY_DIM * 2..])?;
    qk_norm_in_place(&mut q, &mut k)?;
    let z = layer.z.matvec(broker, &input.values)?;
    let alpha = layer.alpha.matvec(broker, &input.values)?;
    let beta = layer.beta.matvec(broker, &input.values)?;
    let a_neg_exp = layer.a_neg_exp.vector(broker)?;
    let dt_bias = layer.dt_bias.vector(broker)?;
    let recurrent = recurrent_step_accounted(
        broker,
        &mut layer.state,
        &q,
        &k,
        &v,
        GdnRecurrentParameters {
            alpha: &alpha.values,
            beta: &beta.values,
            a_neg_exp: &a_neg_exp.values,
            dt_bias: &dt_bias.values,
        },
    )?;
    let gated_weight = layer.gated_norm.vector(broker)?;
    let gated = gated_norm_accounted(broker, &recurrent, &z, &gated_weight.values)?;
    layer.out.matvec(broker, &gated.values)
}

fn run_full_attention(
    broker: &MemoryBroker,
    layer: &mut FullAttentionWeights,
    input: &Qwen36Activation,
) -> Result<Qwen36Activation> {
    let projected = layer.q.matvec(broker, &input.values)?;
    let (mut query, gate) = split_query_gate_accounted(broker, &projected)?;
    let mut key = layer.k.matvec(broker, &input.values)?;
    let value = layer.v.matvec(broker, &input.values)?;
    let q_norm = layer.q_norm.vector(broker)?;
    let k_norm = layer.k_norm.vector(broker)?;
    let attended = layer.state.decode_projected_accounted(
        broker,
        &mut query,
        &gate,
        &mut key,
        &value,
        &q_norm.values,
        &k_norm.values,
    )?;
    layer.out.matvec(broker, &attended.values)
}

fn hash_activation(activation: &Qwen36Activation) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for value in &activation.values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn maybe_dump_activation(
    decode_index: u64,
    input_token: u32,
    layer: LayerId,
    activation: &Qwen36Activation,
) -> Result<()> {
    let Some(directory) = std::env::var_os("TQF_DEV_TENSOR_DUMP_DIR") else {
        return Ok(());
    };
    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!(
        "decode-{decode_index:06}-input-{input_token}-layer-{:02}.f32le",
        layer.0
    ));
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for value in &activation.values {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn maybe_dump_stage(
    decode_index: u64,
    input_token: u32,
    layer: LayerId,
    stage: &'static str,
    values: &[f32],
) -> Result<()> {
    let Some(directory) = std::env::var_os("TQF_DEV_STAGE_DUMP_DIR") else {
        return Ok(());
    };
    let selected_layer = std::env::var("TQF_DEV_STAGE_DUMP_LAYER")
        .map_err(|_| {
            ModelError::Unsupported(
                "TQF_DEV_STAGE_DUMP_DIR requires TQF_DEV_STAGE_DUMP_LAYER".to_string(),
            )
        })?
        .parse::<u8>()
        .map_err(|_| {
            ModelError::Unsupported(
                "TQF_DEV_STAGE_DUMP_LAYER must be an integer from 0 through 39".to_string(),
            )
        })?;
    if selected_layer as usize >= Qwen36Geometry::NUM_LAYERS {
        return Err(ModelError::Unsupported(
            "TQF_DEV_STAGE_DUMP_LAYER must be an integer from 0 through 39".to_string(),
        )
        .into());
    }
    if layer.0 != selected_layer {
        return Ok(());
    }
    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!(
        "decode-{decode_index:06}-input-{input_token}-layer-{:02}-stage-{stage}.f32le",
        layer.0
    ));
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for value in values {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod canonical_checkpoint_tests {
    use super::*;
    use crate::format::gguf;
    use crate::tokenizer::TqfTokenizer;

    /// Release-only qualification seam for comparing one exact greedy token
    /// with an independent runtime. `"A"` is required to encode as one token,
    /// so both runtimes begin at position zero with empty GDN/KV state.
    #[test]
    #[ignore = "requires a converted canonical checkpoint and several minutes of reference decode"]
    fn canonical_single_token_greedy_decode() {
        let tqf_path = std::env::var("TQF_CANONICAL_TQF")
            .expect("set TQF_CANONICAL_TQF to the converted canonical container");
        let gguf_path = std::env::var("TQF_CANONICAL_GGUF")
            .expect("set TQF_CANONICAL_GGUF to the verified canonical source");
        let broker = MemoryBroker::new(Bytes(4 * 1024 * 1024 * 1024));
        let tokenizer_source = gguf::open_with_broker(Path::new(&gguf_path), &broker).unwrap();
        let tokenizer =
            TqfTokenizer::from_gguf(&tokenizer_source).expect("canonical tokenizer must load");
        let input = tokenizer.encode("A", false).unwrap();
        assert_eq!(input.len(), 1, "qualification prompt must be one token");

        let mut runtime = Qwen36BoundedReferenceRuntime::open(
            Path::new(&tqf_path),
            broker,
            8,
            Bytes(256 * 1024 * 1024),
        )
        .unwrap();
        let decoded = runtime.decode_greedy(input[0]).unwrap();
        assert_eq!(decoded.diagnostics.per_layer_hashes.len(), 40);
        assert_eq!(decoded.diagnostics.router_trace.len(), 40);
        println!("input_token={} output_token={}", input[0], decoded.token);
        println!(
            "output_text={:?}",
            tokenizer.decode(&[decoded.token], false).unwrap()
        );
        println!("expert_cache={:?}", runtime.expert_cache_stats());

        if let Ok(expected) = std::env::var("TQF_EXPECTED_GREEDY_TOKEN") {
            assert_eq!(decoded.token, expected.parse::<u32>().unwrap());
        }
    }

    /// Phase 20 decode-loop A/B: the same greedy continuation run twice —
    /// once with the streaming expert cache's GPU-resident path enabled
    /// (`TQF_EXPERT_GPU_RESIDENT` semantics, in-process via
    /// `set_expert_gpu_enabled`), once with the CPU baseline — comparing
    /// per-token wall time and the emitted greedy token sequence. Spec §1005:
    /// the isolated microbenchmark's 2.07x staged16 win must survive a real
    /// decode loop before any default flip is considered. Token parity is
    /// asserted exactly: the staged16 kernel's real-weight parity is
    /// effectively exact, so a divergence would be a genuine A/B finding,
    /// not an expected rounding artifact.
    ///
    /// Start token 32 ("A", per `docs/research/oracles/raw-a-16.json`
    /// prompt_tokens) so no tokenizer is needed. `TQF_DECODE_AB_TOKENS`
    /// overrides the token count (default 16).
    /// Phase 26 prefill A/B (spec §298): tokenize a fixed multi-token
    /// prompt, then run it through (a) the per-token decode loop and
    /// (b) chunked prefill, asserting identical greedy continuation and
    /// reporting TTFT for both. The per-token loop is the Phase 26
    /// baseline; chunked prefill is expected to win via expert-set dedup
    /// (fewer distinct fetches per layer/chunk).
    #[test]
    #[ignore = "requires the canonical .tqf checkpoint; Phase 26 prefill A/B"]
    fn chunked_prefill_parity_and_ttft() {
        let tqf_path = std::env::var("TQF_CANONICAL_TQF")
            .expect("set TQF_CANONICAL_TQF to the converted canonical container");
        let gguf_path = std::env::var("TQF_CANONICAL_GGUF")
            .expect("set TQF_CANONICAL_GGUF to the verified canonical source");
        let prompt = std::env::var("TQF_PREFILL_PROMPT").unwrap_or_else(|_| {
            "Once upon a time, in a quiet valley between two mountains, there lived a              small village of craftsmen who built everything they needed with their own              hands, from wooden carts to iron tools, and they believed that patience was              the finest skill of all."
                .to_string()
        });
        let broker = MemoryBroker::new(Bytes(4 * 1024 * 1024 * 1024));
        let tokenizer_source = gguf::open_with_broker(Path::new(&gguf_path), &broker).unwrap();
        let tokenizer = TqfTokenizer::from_gguf(&tokenizer_source).unwrap();
        let tokens = tokenizer.encode(&prompt, false).unwrap();
        assert!(
            tokens.len() >= 8,
            "A/B prompt must tokenize to several tokens"
        );

        let open = || {
            Qwen36ReferenceRuntime::open_streaming(
                Path::new(&tqf_path),
                MemoryBroker::new(Bytes(4 * 1024 * 1024 * 1024)),
                tokens.len() + 16,
                Bytes(1024 * 1024 * 1024),
            )
            .unwrap()
        };

        let started = Instant::now();
        let mut per_token_runtime = open();
        let mut per_token_next = 0;
        for &token in &tokens {
            per_token_next = per_token_runtime.decode_greedy(token).unwrap().token;
        }
        let per_token_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_token_cache = per_token_runtime.expert_cache_stats().unwrap();

        let started = Instant::now();
        let mut chunked_runtime = open();
        let chunked_next = chunked_runtime.prefill_greedy(&tokens).unwrap();
        let chunked_ms = started.elapsed().as_secs_f64() * 1e3;
        let chunked_cache = chunked_runtime.expert_cache_stats().unwrap();

        println!(
            "phase26_prefill prompt_tokens={} per_token_ms={per_token_ms:.1} chunked_ms={chunked_ms:.1} speedup={:.2}x",
            tokens.len(),
            per_token_ms / chunked_ms
        );
        println!(
            "phase26_prefill per_token_misses={} per_token_bytes={}",
            per_token_cache.misses, per_token_cache.raw_miss_bytes.0
        );
        println!(
            "phase26_prefill chunked_misses={} chunked_bytes={}",
            chunked_cache.misses, chunked_cache.raw_miss_bytes.0
        );
        assert_eq!(
            per_token_next, chunked_next,
            "chunked prefill diverged from the per-token loop"
        );
    }

    /// Phase 25 M4 assault harness (spec §297): the resident-core
    /// streaming profile - attention/GDN/router/shared weights resident,
    /// routed Q4_K experts streamed through the global cache - timed
    /// token-by-token against the pinned raw-a-16 greedy oracle. Prints
    /// per-stage wall times, per-token milliseconds, expert-cache I/O
    /// counters, and the sustained tok/s, so the optimization ledger can
    /// record exactly where the time goes. `TQF_DECODE_AB_TOKENS`
    /// overrides the token count (default 16).
    #[test]
    #[ignore = "requires the canonical .tqf checkpoint; Phase 25 M4 assault"]
    fn resident_streaming_decode_benchmark() {
        let tqf_path = std::env::var("TQF_CANONICAL_TQF")
            .expect("set TQF_CANONICAL_TQF to the converted canonical container");
        let tokens: usize = std::env::var("TQF_DECODE_AB_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(16);
        let budget = Bytes(4 * 1024 * 1024 * 1024);
        let mut runtime = Qwen36ReferenceRuntime::open_streaming(
            Path::new(&tqf_path),
            MemoryBroker::new(budget),
            64,
            Bytes(1024 * 1024 * 1024),
        )
        .unwrap();
        // Pinned raw-a-16 oracle: prompt token 32 -> 220, 16, 15, 15, ...
        let oracle: [u32; 16] = [
            220, 16, 15, 15, 15, 20332, 1740, 369, 6992, 506, 220, 17, 15, 295, 2600, 948,
        ];
        let mut current = 32u32;
        let mut per_token_ms = Vec::with_capacity(tokens);
        let mut embedding_ms = Vec::with_capacity(tokens);
        let mut layer_ms = Vec::with_capacity(tokens);
        let mut lm_head_ms = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let start = Instant::now();
            let decoded = runtime.decode_greedy(current).unwrap();
            let elapsed = start.elapsed();
            per_token_ms.push(elapsed.as_secs_f64() * 1e3);
            embedding_ms.push(decoded.diagnostics.timings.embedding.as_secs_f64() * 1e3);
            layer_ms.push(
                decoded
                    .diagnostics
                    .timings
                    .layers
                    .iter()
                    .map(|(_, duration)| duration.as_secs_f64() * 1e3)
                    .sum::<f64>(),
            );
            lm_head_ms.push(decoded.diagnostics.timings.lm_head.as_secs_f64() * 1e3);
            let expected = if step < oracle.len() { oracle[step] } else { 0 };
            if step < oracle.len() {
                assert_eq!(
                    decoded.token, expected,
                    "resident-core streaming decode diverged from the raw-a-16 oracle at step {step}"
                );
            }
            let cache = runtime.expert_cache_stats().unwrap();
            let mut slowest: Vec<(u8, f64)> = decoded
                .diagnostics
                .timings
                .layers
                .iter()
                .map(|(layer, duration)| (layer.0, duration.as_secs_f64() * 1e3))
                .collect();
            slowest.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top_layers: Vec<String> = slowest
                .iter()
                .take(5)
                .map(|(layer, ms)| format!("L{layer}={ms:.0}ms"))
                .collect();
            println!(
                "phase25_resident_stream step={step} input={current} next={} wall_ms={:.1} embedding_ms={:.1} layers_ms={:.1} lm_head_ms={:.1} cache_hits={} cache_misses={} resident_bytes={} slowest=[{}]",
                decoded.token,
                per_token_ms[step],
                embedding_ms[step],
                layer_ms[step],
                lm_head_ms[step],
                cache.hits,
                cache.misses,
                cache.resident_bytes.0,
                top_layers.join(", "),
            );
            current = decoded.token;
        }
        let total_ms: f64 = per_token_ms.iter().sum();
        let tok_per_s = tokens as f64 / (total_ms / 1000.0);
        println!(
            "phase25_resident_stream summary tokens={tokens} total_ms={total_ms:.1} tok_per_s={tok_per_s:.2} avg_token_ms={:.1} avg_layers_ms={:.1} avg_lm_head_ms={:.1}",
            total_ms / tokens as f64,
            layer_ms.iter().sum::<f64>() / tokens as f64,
            lm_head_ms.iter().sum::<f64>() / tokens as f64,
        );
        let cache = runtime.expert_cache_stats().unwrap();
        println!("phase25_resident_stream cache={:?}", cache);
        println!(
            "phase25_resident_stream io_total_ms={:.0} io_avg_ms_per_token={:.1}",
            cache.demand_io_nanos as f64 / 1e6,
            cache.demand_io_nanos as f64 / 1e6 / tokens as f64,
        );
    }

    #[test]
    #[ignore = "requires the canonical .tqf checkpoint; Phase 20 decode-loop GPU/CPU A/B"]
    fn decode_loop_ab_gpu_vs_cpu_experts() {
        let tqf_path = std::env::var("TQF_CANONICAL_TQF")
            .expect("set TQF_CANONICAL_TQF to the converted canonical container");
        let tokens: usize = std::env::var("TQF_DECODE_AB_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);

        let open = |gpu: bool| {
            let broker = MemoryBroker::new(Bytes(4 * 1024 * 1024 * 1024));
            let mut runtime = Qwen36BoundedReferenceRuntime::open(
                Path::new(&tqf_path),
                broker,
                64,
                Bytes(1024 * 1024 * 1024),
            )
            .unwrap();
            runtime.set_expert_gpu_enabled(gpu);
            (runtime, gpu)
        };

        let (mut gpu_runtime, _) = open(true);
        let (mut cpu_runtime, _) = open(false);
        let mut current = 32u32;
        let mut gpu_tokens = Vec::with_capacity(tokens);
        let mut cpu_tokens = Vec::with_capacity(tokens);
        let mut gpu_ms = Vec::with_capacity(tokens);
        let mut cpu_ms = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let start = Instant::now();
            let decoded = gpu_runtime.decode_greedy(current).unwrap();
            gpu_ms.push(start.elapsed().as_secs_f64() * 1e3);
            let gpu_token = decoded.token;

            let start = Instant::now();
            let decoded = cpu_runtime.decode_greedy(current).unwrap();
            cpu_ms.push(start.elapsed().as_secs_f64() * 1e3);
            let cpu_token = decoded.token;

            println!(
                "phase20_decode_ab step={step} token={current} gpu_ms={:.1} cpu_ms={:.1} gpu_next={gpu_token} cpu_next={cpu_token}",
                gpu_ms[step],
                cpu_ms[step]
            );
            gpu_tokens.push(gpu_token);
            cpu_tokens.push(cpu_token);
            current = cpu_token;
            assert_eq!(
                gpu_token, cpu_token,
                "GPU/CPU expert paths diverged at decode step {step}: gpu={gpu_token} cpu={cpu_token}"
            );
        }

        let gpu_total: f64 = gpu_ms.iter().sum();
        let cpu_total: f64 = cpu_ms.iter().sum();
        let speedup = cpu_total / gpu_total.max(f64::MIN_POSITIVE);
        println!(
            "phase20_decode_ab summary tokens={tokens} gpu_total_ms={gpu_total:.1} cpu_total_ms={cpu_total:.1} speedup={speedup:.2}x"
        );
        println!(
            "phase20_decode_ab gpu_cache={:?}",
            gpu_runtime.expert_cache_stats()
        );
        println!(
            "phase20_decode_ab cpu_cache={:?}",
            cpu_runtime.expert_cache_stats()
        );
        println!("phase20_decode_ab gpu_tokens={gpu_tokens:?}");
        println!("phase20_decode_ab cpu_tokens={cpu_tokens:?}");
    }
}
