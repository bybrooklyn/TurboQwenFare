//! Fixed Phase-15/18 Qwen3.6 execution graphs. These are the only model
//! runtimes: both bind the canonical forty-layer topology directly to
//! validated TQF tensors, with the Phase-14 resident-expert profile retained
//! as an explicit high-memory oracle alongside the bounded streaming cache.

use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::dev::inventory::TensorRole;
use crate::error::{ModelError, Result};
use crate::experts::{Qwen36ResidentMoe, Qwen36StreamingMoe, RouterResult, WholeExpertLfuCache};
use crate::ids::{Bytes, LayerId, LayerKind};
use crate::memory::MemoryBroker;
use crate::runtime::{DecodeDiagnostics, DecodeTimings, DecodeToken, LayerHash, RouterTrace};

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
    loader: Qwen36WeightLoader,
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
    loader: Qwen36WeightLoader,
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
        let loader = Qwen36WeightLoader::open(path, broker.clone())?;
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
        let loader = Qwen36WeightLoader::open(path, broker.clone())?;
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

    /// Runs one actual checkpoint token through embedding, all forty
    /// attention/GDN+MoE layers, final norm, Q6_K LM head, and greedy
    /// selection.  The returned diagnostics are the Phase-15 qualification
    /// artifacts; no generic callback or protocol code enters this loop.
    pub fn decode_greedy(&mut self, input_token: u32) -> Result<DecodeToken> {
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
        let token = logits
            .values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .expect("Qwen LM head has fixed nonzero vocabulary");
        timings.sampling = start.elapsed();
        self.decode_index = self.decode_index.saturating_add(1);
        Ok(DecodeToken {
            token,
            diagnostics: DecodeDiagnostics {
                per_layer_hashes,
                router_trace,
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
        let loader = Qwen36WeightLoader::open(path, broker.clone())?;
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

    pub fn decode_greedy(&mut self, input_token: u32) -> Result<DecodeToken> {
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
        let token = logits
            .values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .expect("Qwen LM head has fixed nonzero vocabulary");
        timings.sampling = start.elapsed();
        self.decode_index = self.decode_index.saturating_add(1);
        Ok(DecodeToken {
            token,
            diagnostics: DecodeDiagnostics {
                per_layer_hashes,
                router_trace,
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
}

fn bounded_layer_forward(
    broker: &MemoryBroker,
    loader: &Qwen36WeightLoader,
    expert_cache: &mut WholeExpertLfuCache,
    layer: &mut BoundedLayerState,
    input: &Qwen36Activation,
) -> Result<(LayerId, Qwen36Activation, RouterResult)> {
    let id = match layer {
        BoundedLayerState::Gdn { id, .. } | BoundedLayerState::Full { id, .. } => *id,
    };
    let attn_weight = loader.load(TensorRole::AttnNorm, Some(id))?;
    let attn_weight = attn_weight.vector(broker)?;
    let normalized = Qwen36Activation::qwen_rmsnorm(broker, input, &attn_weight.values)?;
    let attended = match layer {
        BoundedLayerState::Gdn { state, .. } => {
            bounded_gdn(broker, loader, id, state, &normalized)?
        }
        BoundedLayerState::Full { state, .. } => {
            bounded_full(broker, loader, id, state, &normalized)?
        }
    };
    let after_attention = Qwen36Activation::residual_add(broker, input, &attended)?;
    let ffn_weight = loader.load(TensorRole::FfnNorm, Some(id))?;
    let ffn_weight = ffn_weight.vector(broker)?;
    let ffn_input = Qwen36Activation::qwen_rmsnorm(broker, &after_attention, &ffn_weight.values)?;
    let mut moe = Qwen36StreamingMoe::open(loader, id)?;
    let (moe, route) = moe.forward(loader, expert_cache, broker, &ffn_input)?;
    Ok((
        id,
        Qwen36Activation::residual_add(broker, &after_attention, &moe)?,
        route,
    ))
}

fn bounded_gdn(
    broker: &MemoryBroker,
    loader: &Qwen36WeightLoader,
    id: LayerId,
    state: &mut GdnState,
    input: &Qwen36Activation,
) -> Result<Qwen36Activation> {
    let qkv = loader
        .load(TensorRole::GdnInProjQkv, Some(id))?
        .matvec(broker, &input.values)?;
    let conv = loader
        .load(TensorRole::GdnConv1d, Some(id))?
        .gdn_conv1d_weights(broker)?;
    let mut convolved = Qwen36Activation::zeros(broker, Qwen36Geometry::GDN_CONV_CHANNELS)?;
    state
        .conv_tail_mut()
        .step_without_bias_into(&qkv.values, &conv.values, &mut convolved.values);
    let mut q =
        Qwen36Activation::from_slice(broker, &convolved.values[..Qwen36Geometry::GDN_KEY_DIM])?;
    let mut k = Qwen36Activation::from_slice(
        broker,
        &convolved.values[Qwen36Geometry::GDN_KEY_DIM..Qwen36Geometry::GDN_KEY_DIM * 2],
    )?;
    let v =
        Qwen36Activation::from_slice(broker, &convolved.values[Qwen36Geometry::GDN_KEY_DIM * 2..])?;
    qk_norm_in_place(&mut q, &mut k)?;
    let z = loader
        .load(TensorRole::GdnInProjZ, Some(id))?
        .matvec(broker, &input.values)?;
    let alpha = loader
        .load(TensorRole::GdnInProjA, Some(id))?
        .matvec(broker, &input.values)?;
    let beta = loader
        .load(TensorRole::GdnInProjB, Some(id))?
        .matvec(broker, &input.values)?;
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
    let gated_weight = loader
        .load(TensorRole::GdnGatedNorm, Some(id))?
        .vector(broker)?;
    let gated = gated_norm_accounted(broker, &recurrent, &z, &gated_weight.values)?;
    loader
        .load(TensorRole::GdnOutProj, Some(id))?
        .matvec(broker, &gated.values)
}

fn bounded_full(
    broker: &MemoryBroker,
    loader: &Qwen36WeightLoader,
    id: LayerId,
    state: &mut FullAttentionLayer,
    input: &Qwen36Activation,
) -> Result<Qwen36Activation> {
    let projected = loader
        .load(TensorRole::AttnQProj, Some(id))?
        .matvec(broker, &input.values)?;
    let (mut query, gate) = split_query_gate_accounted(broker, &projected)?;
    let mut key = loader
        .load(TensorRole::AttnKProj, Some(id))?
        .matvec(broker, &input.values)?;
    let value = loader
        .load(TensorRole::AttnVProj, Some(id))?
        .matvec(broker, &input.values)?;
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
    loader
        .load(TensorRole::AttnOProj, Some(id))?
        .matvec(broker, &attended.values)
}

impl Qwen36Layer {
    fn forward(
        &mut self,
        broker: &MemoryBroker,
        loader: &Qwen36WeightLoader,
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
    let conv_weight = layer.conv.vector(broker)?;
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
}
