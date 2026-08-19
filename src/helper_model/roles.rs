//! Tensor role IDs for the pplx-embed `.tqf` container. Deliberately a
//! separate small enum from `dev::inventory::TensorRole` — that one is
//! wired to Qwen3.6's specific MoE/GDN graph and its variant list is a
//! stable on-disk contract for *that* model; overloading it here would
//! either break its stability or force cross-model role IDs to collide.

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PplxTensorRole {
    TokenEmbedding = 0,
    FinalNorm = 1,
    InputLayernorm = 2,
    PostAttentionLayernorm = 3,
    AttnQProj = 4,
    AttnKProj = 5,
    AttnVProj = 6,
    AttnOProj = 7,
    AttnQNorm = 8,
    AttnKNorm = 9,
    MlpGateProj = 10,
    MlpUpProj = 11,
    MlpDownProj = 12,
}

impl PplxTensorRole {
    pub const ALL_PER_LAYER: [PplxTensorRole; 11] = [
        PplxTensorRole::InputLayernorm,
        PplxTensorRole::PostAttentionLayernorm,
        PplxTensorRole::AttnQProj,
        PplxTensorRole::AttnKProj,
        PplxTensorRole::AttnVProj,
        PplxTensorRole::AttnOProj,
        PplxTensorRole::AttnQNorm,
        PplxTensorRole::AttnKNorm,
        PplxTensorRole::MlpGateProj,
        PplxTensorRole::MlpUpProj,
        PplxTensorRole::MlpDownProj,
    ];

    /// The safetensors tensor name for this role at the given layer (or
    /// the layer-independent name for `TokenEmbedding`/`FinalNorm`).
    pub fn safetensors_name(self, layer: Option<u8>) -> String {
        match self {
            PplxTensorRole::TokenEmbedding => "embed_tokens.weight".to_string(),
            PplxTensorRole::FinalNorm => "norm.weight".to_string(),
            PplxTensorRole::InputLayernorm => {
                format!("layers.{}.input_layernorm.weight", layer.expect("layer"))
            }
            PplxTensorRole::PostAttentionLayernorm => format!(
                "layers.{}.post_attention_layernorm.weight",
                layer.expect("layer")
            ),
            PplxTensorRole::AttnQProj => {
                format!("layers.{}.self_attn.q_proj.weight", layer.expect("layer"))
            }
            PplxTensorRole::AttnKProj => {
                format!("layers.{}.self_attn.k_proj.weight", layer.expect("layer"))
            }
            PplxTensorRole::AttnVProj => {
                format!("layers.{}.self_attn.v_proj.weight", layer.expect("layer"))
            }
            PplxTensorRole::AttnOProj => {
                format!("layers.{}.self_attn.o_proj.weight", layer.expect("layer"))
            }
            PplxTensorRole::AttnQNorm => {
                format!("layers.{}.self_attn.q_norm.weight", layer.expect("layer"))
            }
            PplxTensorRole::AttnKNorm => {
                format!("layers.{}.self_attn.k_norm.weight", layer.expect("layer"))
            }
            PplxTensorRole::MlpGateProj => {
                format!("layers.{}.mlp.gate_proj.weight", layer.expect("layer"))
            }
            PplxTensorRole::MlpUpProj => {
                format!("layers.{}.mlp.up_proj.weight", layer.expect("layer"))
            }
            PplxTensorRole::MlpDownProj => {
                format!("layers.{}.mlp.down_proj.weight", layer.expect("layer"))
            }
        }
    }
}
