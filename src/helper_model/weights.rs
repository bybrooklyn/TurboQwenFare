//! Loads a converted pplx-embed `.tqf` container under a transient broker
//! lease (spec §37, spec §115 invariant #4: "every large allocation is
//! registered with the memory broker before physical allocation"; spec
//! §115 item 7 in the memory design: "transient helper model while its
//! current operation is executing"). Every tensor is reserved under
//! `MemoryOwner::HelperModel`/`MemoryClass::Transient` and released the
//! moment `PplxEmbedWeights` is dropped — unlike the main model's
//! `MemoryOwner::Core`, this owner is never resident across requests.

use std::path::Path;

use crate::error::{ModelError, Result};
use crate::format::tqf::TqfReader;
use crate::ids::{Bytes, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

use super::geometry::PplxEmbedGeometry;
use super::roles::PplxTensorRole;

pub struct LoadedTensor {
    pub dims: Vec<u64>,
    pub values: Vec<f32>,
    _lease: MemoryLease,
}

/// All of the helper model's weights, decoded to F32 and resident for the
/// lifetime of one embedding operation. ~2.2 GiB for the 0.6B checkpoint —
/// deliberately loaded whole (REFERENCE BASELINE, matching the "load
/// everything, no per-tensor streaming" approach the main model started
/// with before Phase 18's out-of-core work): the helper model is small
/// enough, and used rarely and briefly enough, that out-of-core streaming
/// for it is not justified without a measured need.
pub struct PplxEmbedWeights {
    pub token_embedding: LoadedTensor,
    pub final_norm: LoadedTensor,
    pub layers: Vec<PplxLayerWeights>,
}

pub struct PplxLayerWeights {
    pub input_layernorm: LoadedTensor,
    pub post_attention_layernorm: LoadedTensor,
    pub q_proj: LoadedTensor,
    pub k_proj: LoadedTensor,
    pub v_proj: LoadedTensor,
    pub o_proj: LoadedTensor,
    pub q_norm: LoadedTensor,
    pub k_norm: LoadedTensor,
    pub gate_proj: LoadedTensor,
    pub up_proj: LoadedTensor,
    pub down_proj: LoadedTensor,
}

fn load_one(
    reader: &TqfReader,
    broker: &MemoryBroker,
    role: PplxTensorRole,
    layer: Option<LayerId>,
) -> Result<LoadedTensor> {
    let extent = reader.tensor(role as u32, layer)?;
    let lease = broker.reserve(
        MemoryOwner::HelperModel,
        MemoryClass::Transient,
        Bytes(extent.stored_bytes),
        64,
    )?;
    let mut bytes = vec![0u8; extent.stored_bytes as usize];
    reader.read_extent_into(extent, &mut bytes)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(ModelError::Shape {
            tensor: "pplx-embed F32 extent",
            expected: 0,
            actual: bytes.len() % 4,
        }
        .into());
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")));
    }
    Ok(LoadedTensor {
        dims: extent.dims[..extent.rank as usize].to_vec(),
        values,
        _lease: lease,
    })
}

impl PplxEmbedWeights {
    pub fn load(path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let reader = TqfReader::open_validated_with_broker(path, broker)?;
        let token_embedding = load_one(&reader, broker, PplxTensorRole::TokenEmbedding, None)?;
        let final_norm = load_one(&reader, broker, PplxTensorRole::FinalNorm, None)?;

        let mut layers = Vec::with_capacity(PplxEmbedGeometry::NUM_LAYERS);
        for layer in 0..PplxEmbedGeometry::NUM_LAYERS as u8 {
            let id = Some(LayerId(layer));
            layers.push(PplxLayerWeights {
                input_layernorm: load_one(&reader, broker, PplxTensorRole::InputLayernorm, id)?,
                post_attention_layernorm: load_one(
                    &reader,
                    broker,
                    PplxTensorRole::PostAttentionLayernorm,
                    id,
                )?,
                q_proj: load_one(&reader, broker, PplxTensorRole::AttnQProj, id)?,
                k_proj: load_one(&reader, broker, PplxTensorRole::AttnKProj, id)?,
                v_proj: load_one(&reader, broker, PplxTensorRole::AttnVProj, id)?,
                o_proj: load_one(&reader, broker, PplxTensorRole::AttnOProj, id)?,
                q_norm: load_one(&reader, broker, PplxTensorRole::AttnQNorm, id)?,
                k_norm: load_one(&reader, broker, PplxTensorRole::AttnKNorm, id)?,
                gate_proj: load_one(&reader, broker, PplxTensorRole::MlpGateProj, id)?,
                up_proj: load_one(&reader, broker, PplxTensorRole::MlpUpProj, id)?,
                down_proj: load_one(&reader, broker, PplxTensorRole::MlpDownProj, id)?,
            });
        }

        Ok(Self {
            token_embedding,
            final_norm,
            layers,
        })
    }
}
