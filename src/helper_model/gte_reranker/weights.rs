//! Loads a converted GTE reranker `.tqf` container under a transient
//! broker lease (same `MemoryOwner::HelperModel`/`MemoryClass::
//! Transient` pattern Phase 37 established for pplx-embed — spec §115
//! item 7, "transient helper model while its current operation is
//! executing").

use std::path::Path;

use crate::error::Result;
use crate::format::tqf::TqfReader;
use crate::ids::{Bytes, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

use super::geometry::GteRerankerGeometry;
use super::roles::GteTensorRole;

pub struct LoadedTensor {
    pub dims: Vec<u64>,
    pub values: Vec<f32>,
    _lease: MemoryLease,
}

pub struct GteRerankerWeights {
    pub token_embedding: LoadedTensor,
    pub embedding_norm: LoadedTensor,
    pub final_norm: LoadedTensor,
    pub head_dense: LoadedTensor,
    pub head_norm: LoadedTensor,
    pub classifier_weight: LoadedTensor,
    pub classifier_bias: LoadedTensor,
    pub layers: Vec<GteLayerWeights>,
}

pub struct GteLayerWeights {
    /// `None` only for layer 0, whose real checkpoint has no
    /// `attn_norm` tensor at all (identity, per the architecture doc).
    pub attn_norm: Option<LoadedTensor>,
    pub attn_wqkv: LoadedTensor,
    pub attn_wo: LoadedTensor,
    pub mlp_norm: LoadedTensor,
    pub mlp_wi: LoadedTensor,
    pub mlp_wo: LoadedTensor,
}

fn load_one(
    reader: &TqfReader,
    broker: &MemoryBroker,
    role: GteTensorRole,
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

impl GteRerankerWeights {
    pub fn load(path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let reader = TqfReader::open_validated_with_broker(path, broker)?;

        let token_embedding = load_one(&reader, broker, GteTensorRole::TokenEmbedding, None)?;
        let embedding_norm = load_one(&reader, broker, GteTensorRole::EmbeddingNorm, None)?;
        let final_norm = load_one(&reader, broker, GteTensorRole::FinalNorm, None)?;
        let head_dense = load_one(&reader, broker, GteTensorRole::HeadDense, None)?;
        let head_norm = load_one(&reader, broker, GteTensorRole::HeadNorm, None)?;
        let classifier_weight = load_one(&reader, broker, GteTensorRole::ClassifierWeight, None)?;
        let classifier_bias = load_one(&reader, broker, GteTensorRole::ClassifierBias, None)?;

        let mut layers = Vec::with_capacity(GteRerankerGeometry::NUM_LAYERS);
        for layer in 0..GteRerankerGeometry::NUM_LAYERS as u8 {
            let id = Some(LayerId(layer));
            let attn_norm = if layer == 0 {
                None
            } else {
                Some(load_one(&reader, broker, GteTensorRole::AttnNorm, id)?)
            };
            layers.push(GteLayerWeights {
                attn_norm,
                attn_wqkv: load_one(&reader, broker, GteTensorRole::AttnWqkv, id)?,
                attn_wo: load_one(&reader, broker, GteTensorRole::AttnWo, id)?,
                mlp_norm: load_one(&reader, broker, GteTensorRole::MlpNorm, id)?,
                mlp_wi: load_one(&reader, broker, GteTensorRole::MlpWi, id)?,
                mlp_wo: load_one(&reader, broker, GteTensorRole::MlpWo, id)?,
            });
        }

        Ok(Self {
            token_embedding,
            embedding_norm,
            final_norm,
            head_dense,
            head_norm,
            classifier_weight,
            classifier_bias,
            layers,
        })
    }
}
