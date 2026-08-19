//! Tensor role IDs for the GTE reranker `.tqf` container. A separate
//! small enum from both `dev::inventory::TensorRole` (Qwen3.6's own
//! stable role list) and `helper_model::roles::PplxTensorRole` (a
//! different architecture family) — each model's on-disk role IDs are
//! independent, matching the pattern already established there.

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GteTensorRole {
    TokenEmbedding = 0,
    EmbeddingNorm = 1,
    FinalNorm = 2,
    HeadDense = 3,
    HeadNorm = 4,
    ClassifierWeight = 5,
    ClassifierBias = 6,
    AttnNorm = 7,
    AttnWqkv = 8,
    AttnWo = 9,
    MlpNorm = 10,
    MlpWi = 11,
    MlpWo = 12,
}

impl GteTensorRole {
    /// Per-layer roles that exist on *every* layer 0..21.
    pub const ALL_PER_LAYER: [GteTensorRole; 5] = [
        GteTensorRole::AttnWqkv,
        GteTensorRole::AttnWo,
        GteTensorRole::MlpNorm,
        GteTensorRole::MlpWi,
        GteTensorRole::MlpWo,
    ];

    /// The real checkpoint's safetensors tensor name for this role at
    /// the given layer (or the layer-independent name for the
    /// embedding/head/classifier roles). `AttnNorm` at layer 0 has no
    /// real tensor (confirmed empirically against the checkpoint) —
    /// callers must not call this for `(AttnNorm, Some(0))`.
    pub fn safetensors_name(self, layer: Option<u8>) -> String {
        match self {
            GteTensorRole::TokenEmbedding => "model.embeddings.tok_embeddings.weight".to_string(),
            GteTensorRole::EmbeddingNorm => "model.embeddings.norm.weight".to_string(),
            GteTensorRole::FinalNorm => "model.final_norm.weight".to_string(),
            GteTensorRole::HeadDense => "head.dense.weight".to_string(),
            GteTensorRole::HeadNorm => "head.norm.weight".to_string(),
            GteTensorRole::ClassifierWeight => "classifier.weight".to_string(),
            GteTensorRole::ClassifierBias => "classifier.bias".to_string(),
            GteTensorRole::AttnNorm => {
                format!("model.layers.{}.attn_norm.weight", layer.expect("layer"))
            }
            GteTensorRole::AttnWqkv => {
                format!("model.layers.{}.attn.Wqkv.weight", layer.expect("layer"))
            }
            GteTensorRole::AttnWo => {
                format!("model.layers.{}.attn.Wo.weight", layer.expect("layer"))
            }
            GteTensorRole::MlpNorm => {
                format!("model.layers.{}.mlp_norm.weight", layer.expect("layer"))
            }
            GteTensorRole::MlpWi => {
                format!("model.layers.{}.mlp.Wi.weight", layer.expect("layer"))
            }
            GteTensorRole::MlpWo => {
                format!("model.layers.{}.mlp.Wo.weight", layer.expect("layer"))
            }
        }
    }
}
