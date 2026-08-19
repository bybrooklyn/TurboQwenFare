//! Tensor role IDs for the vision `.tqf` container, mapped to the real
//! GGUF tensor names confirmed by direct inspection of the pinned
//! `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` (334 tensors, `general.architecture
//! = clip`, `clip.projector_type = qwen3vl_merger`) on 2026-08-19. A
//! separate small enum from every other model family's role list, same
//! convention as `helper_model::gte_reranker::roles::GteTensorRole`.

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionTensorRole {
    /// `v.patch_embd.weight` — first of the two summed patch-embedding
    /// conv kernels (spec: `clip_graph_qwen2vl::build_inp_with_temporal_merge`,
    /// reused unmodified by `clip_graph_qwen3vl`, sums two independent
    /// convs over the same still-image input).
    PatchEmbedWeight0 = 0,
    /// `v.patch_embd.weight.1` — the second summed conv kernel.
    PatchEmbedWeight1 = 1,
    PatchEmbedBias = 2,
    /// `v.position_embd.weight` — the native 48x48 learned absolute
    /// position table, bilinear-resized at runtime for other grid
    /// sizes.
    PositionEmbed = 3,
    PostLnWeight = 4,
    PostLnBias = 5,
    /// `mm.0.weight`/`mm.0.bias` — merger FC1 (4608 -> 4608).
    MergerFc1Weight = 6,
    MergerFc1Bias = 7,
    /// `mm.2.weight`/`mm.2.bias` — merger FC2 (4608 -> 2048). (`mm.1` is
    /// the GELU activation, no weights.)
    MergerFc2Weight = 8,
    MergerFc2Bias = 9,
    Ln1Weight = 10,
    Ln1Bias = 11,
    Ln2Weight = 12,
    Ln2Bias = 13,
    AttnQkvWeight = 14,
    AttnQkvBias = 15,
    AttnOutWeight = 16,
    AttnOutBias = 17,
    FfnUpWeight = 18,
    FfnUpBias = 19,
    FfnDownWeight = 20,
    FfnDownBias = 21,
}

impl VisionTensorRole {
    /// Per-layer roles that exist on every layer `0..LAYERS`.
    pub const ALL_PER_LAYER: [VisionTensorRole; 12] = [
        VisionTensorRole::Ln1Weight,
        VisionTensorRole::Ln1Bias,
        VisionTensorRole::Ln2Weight,
        VisionTensorRole::Ln2Bias,
        VisionTensorRole::AttnQkvWeight,
        VisionTensorRole::AttnQkvBias,
        VisionTensorRole::AttnOutWeight,
        VisionTensorRole::AttnOutBias,
        VisionTensorRole::FfnUpWeight,
        VisionTensorRole::FfnUpBias,
        VisionTensorRole::FfnDownWeight,
        VisionTensorRole::FfnDownBias,
    ];

    /// The real GGUF tensor name for this role, at the given layer (or
    /// the layer-independent name for patch/position/post-ln/merger
    /// roles).
    pub fn gguf_name(self, layer: Option<u8>) -> String {
        match self {
            VisionTensorRole::PatchEmbedWeight0 => "v.patch_embd.weight".to_string(),
            VisionTensorRole::PatchEmbedWeight1 => "v.patch_embd.weight.1".to_string(),
            VisionTensorRole::PatchEmbedBias => "v.patch_embd.bias".to_string(),
            VisionTensorRole::PositionEmbed => "v.position_embd.weight".to_string(),
            VisionTensorRole::PostLnWeight => "v.post_ln.weight".to_string(),
            VisionTensorRole::PostLnBias => "v.post_ln.bias".to_string(),
            VisionTensorRole::MergerFc1Weight => "mm.0.weight".to_string(),
            VisionTensorRole::MergerFc1Bias => "mm.0.bias".to_string(),
            VisionTensorRole::MergerFc2Weight => "mm.2.weight".to_string(),
            VisionTensorRole::MergerFc2Bias => "mm.2.bias".to_string(),
            VisionTensorRole::Ln1Weight => format!("v.blk.{}.ln1.weight", layer.expect("layer")),
            VisionTensorRole::Ln1Bias => format!("v.blk.{}.ln1.bias", layer.expect("layer")),
            VisionTensorRole::Ln2Weight => format!("v.blk.{}.ln2.weight", layer.expect("layer")),
            VisionTensorRole::Ln2Bias => format!("v.blk.{}.ln2.bias", layer.expect("layer")),
            VisionTensorRole::AttnQkvWeight => {
                format!("v.blk.{}.attn_qkv.weight", layer.expect("layer"))
            }
            VisionTensorRole::AttnQkvBias => {
                format!("v.blk.{}.attn_qkv.bias", layer.expect("layer"))
            }
            VisionTensorRole::AttnOutWeight => {
                format!("v.blk.{}.attn_out.weight", layer.expect("layer"))
            }
            VisionTensorRole::AttnOutBias => {
                format!("v.blk.{}.attn_out.bias", layer.expect("layer"))
            }
            VisionTensorRole::FfnUpWeight => {
                format!("v.blk.{}.ffn_up.weight", layer.expect("layer"))
            }
            VisionTensorRole::FfnUpBias => format!("v.blk.{}.ffn_up.bias", layer.expect("layer")),
            VisionTensorRole::FfnDownWeight => {
                format!("v.blk.{}.ffn_down.weight", layer.expect("layer"))
            }
            VisionTensorRole::FfnDownBias => {
                format!("v.blk.{}.ffn_down.bias", layer.expect("layer"))
            }
        }
    }
}
