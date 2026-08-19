//! Loads a converted vision-tower `.tqf` container under a transient
//! broker lease (`MemoryOwner::HelperModel`/`MemoryClass::Transient`,
//! the same pattern Phase 37/43 established — spec §115 item 7).

use std::path::Path;

use crate::error::Result;
use crate::format::tqf::TqfReader;
use crate::ids::{Bytes, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

use super::geometry::VisionGeometry;
use super::roles::VisionTensorRole;

pub struct LoadedTensor {
    pub dims: Vec<u64>,
    pub values: Vec<f32>,
    _lease: MemoryLease,
}

pub struct VisionLayerWeights {
    pub ln1_weight: LoadedTensor,
    pub ln1_bias: LoadedTensor,
    pub ln2_weight: LoadedTensor,
    pub ln2_bias: LoadedTensor,
    pub attn_qkv_weight: LoadedTensor,
    pub attn_qkv_bias: LoadedTensor,
    pub attn_out_weight: LoadedTensor,
    pub attn_out_bias: LoadedTensor,
    pub ffn_up_weight: LoadedTensor,
    pub ffn_up_bias: LoadedTensor,
    pub ffn_down_weight: LoadedTensor,
    pub ffn_down_bias: LoadedTensor,
}

pub struct VisionWeights {
    pub patch_embed_weight0: LoadedTensor,
    pub patch_embed_weight1: LoadedTensor,
    pub patch_embed_bias: LoadedTensor,
    pub position_embed: LoadedTensor,
    pub post_ln_weight: LoadedTensor,
    pub post_ln_bias: LoadedTensor,
    pub merger_fc1_weight: LoadedTensor,
    pub merger_fc1_bias: LoadedTensor,
    pub merger_fc2_weight: LoadedTensor,
    pub merger_fc2_bias: LoadedTensor,
    pub layers: Vec<VisionLayerWeights>,
}

fn load_one(
    reader: &TqfReader,
    broker: &MemoryBroker,
    role: VisionTensorRole,
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

impl VisionWeights {
    pub fn load(path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let reader = TqfReader::open_validated_with_broker(path, broker)?;

        let patch_embed_weight0 =
            load_one(&reader, broker, VisionTensorRole::PatchEmbedWeight0, None)?;
        let patch_embed_weight1 =
            load_one(&reader, broker, VisionTensorRole::PatchEmbedWeight1, None)?;
        let patch_embed_bias = load_one(&reader, broker, VisionTensorRole::PatchEmbedBias, None)?;
        let position_embed = load_one(&reader, broker, VisionTensorRole::PositionEmbed, None)?;
        let post_ln_weight = load_one(&reader, broker, VisionTensorRole::PostLnWeight, None)?;
        let post_ln_bias = load_one(&reader, broker, VisionTensorRole::PostLnBias, None)?;
        let merger_fc1_weight = load_one(&reader, broker, VisionTensorRole::MergerFc1Weight, None)?;
        let merger_fc1_bias = load_one(&reader, broker, VisionTensorRole::MergerFc1Bias, None)?;
        let merger_fc2_weight = load_one(&reader, broker, VisionTensorRole::MergerFc2Weight, None)?;
        let merger_fc2_bias = load_one(&reader, broker, VisionTensorRole::MergerFc2Bias, None)?;

        let mut layers = Vec::with_capacity(VisionGeometry::LAYERS);
        for layer in 0..VisionGeometry::LAYERS as u8 {
            let id = Some(LayerId(layer));
            layers.push(VisionLayerWeights {
                ln1_weight: load_one(&reader, broker, VisionTensorRole::Ln1Weight, id)?,
                ln1_bias: load_one(&reader, broker, VisionTensorRole::Ln1Bias, id)?,
                ln2_weight: load_one(&reader, broker, VisionTensorRole::Ln2Weight, id)?,
                ln2_bias: load_one(&reader, broker, VisionTensorRole::Ln2Bias, id)?,
                attn_qkv_weight: load_one(&reader, broker, VisionTensorRole::AttnQkvWeight, id)?,
                attn_qkv_bias: load_one(&reader, broker, VisionTensorRole::AttnQkvBias, id)?,
                attn_out_weight: load_one(&reader, broker, VisionTensorRole::AttnOutWeight, id)?,
                attn_out_bias: load_one(&reader, broker, VisionTensorRole::AttnOutBias, id)?,
                ffn_up_weight: load_one(&reader, broker, VisionTensorRole::FfnUpWeight, id)?,
                ffn_up_bias: load_one(&reader, broker, VisionTensorRole::FfnUpBias, id)?,
                ffn_down_weight: load_one(&reader, broker, VisionTensorRole::FfnDownWeight, id)?,
                ffn_down_bias: load_one(&reader, broker, VisionTensorRole::FfnDownBias, id)?,
            });
        }

        Ok(Self {
            patch_embed_weight0,
            patch_embed_weight1,
            patch_embed_bias,
            position_embed,
            post_ln_weight,
            post_ln_bias,
            merger_fc1_weight,
            merger_fc1_bias,
            merger_fc2_weight,
            merger_fc2_bias,
            layers,
        })
    }
}
