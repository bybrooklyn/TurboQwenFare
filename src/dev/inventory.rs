//! Tensor-inventory generator (spec §118, REFERENCE BASELINE; spec Phase 0
//! §272 deliverable "Build tensor-inventory generator against source
//! metadata"). Reads a GGUF file's tensor descriptors (via the Phase 5
//! reader) and classifies each into a `TensorInventoryEntry`, emitting a
//! reviewable JSON artifact. Per §118: "The generator must fail if the
//! source checkpoint contains a production-language tensor that cannot be
//! classified" — `generate_inventory` does exactly that rather than
//! silently skipping unknown tensors.
//!
//! **Known limitation, stated plainly:** this environment has never
//! downloaded the real 20+ GB canonical checkpoint (see
//! `docs/research/canonical-source-manifest.md` — the pin is resolved, but
//! fetching 20GB wasn't part of this research pass), so the *actual*
//! on-disk tensor names `ggml-org`'s conversion uses are not independently
//! confirmed here. Classification below is grounded in two sources only:
//! (1) the exact Transformers-style names spec §117 quotes directly
//! (`q_proj`, `in_proj_qkv`, `gate`/`up`/`down`, ...), and (2) llama.cpp's
//! well-established GGUF naming convention for embedding/norm/full-
//! attention/MoE-expert tensors (`token_embd`, `output_norm`, `attn_q`,
//! `ffn_gate_exps`, ...) — standard across essentially all llama.cpp
//! conversions, not specific to this model. Gated DeltaNet's *llama.cpp*
//! names are not confirmed by either source; only the Transformers-style
//! GDN names are recognized here. Running this generator against the real
//! file (once downloaded) is expected to surface new patterns to add —
//! that's the generator doing its job, not a bug in it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{GgufError, Result};
use crate::format::gguf;
use crate::format::quant::GgmlType;
use crate::format::tqf::TqfSectionKind;
use crate::ids::LayerId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorRole {
    TokenEmbedding,
    FinalNorm,
    LmHead,
    VisionProjector,
    AttnQProj,
    AttnKProj,
    AttnVProj,
    AttnOProj,
    AttnQNorm,
    AttnKNorm,
    AttnNorm,
    GdnInProjQkv,
    GdnInProjZ,
    GdnInProjA,
    GdnInProjB,
    GdnConv1d,
    GdnALog,
    GdnDtBias,
    GdnGatedNorm,
    GdnOutProj,
    RouterGate,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    RoutedExpertGate,
    RoutedExpertUp,
    RoutedExpertDown,
    FfnNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyClass {
    /// Always resident — never evicted by the memory broker (spec Part VI).
    ResidentCore,
    /// Streamed from SSD on demand; lives in the global expert cache.
    ColdExpertStore,
    /// Loaded only on the first multimodal request (spec §3: `--enable-vision`).
    Lazy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelConsumer {
    Embedding,
    LmHead,
    Norm,
    Attention,
    GatedDeltaNet,
    Router,
    Expert,
    Vision,
}

/// One row of the tensor inventory (spec §118's `TensorInventoryEntry`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInventoryEntry {
    pub canonical_name: String,
    pub logical_role: TensorRole,
    pub layer: Option<u8>,
    pub shape: Vec<u64>,
    pub source_quant: GgmlType,
    pub source_bytes: u64,
    pub tqf_section: TqfSectionKind,
    pub residency: ResidencyClass,
    pub consumer: KernelConsumer,
}

struct Classification {
    role: TensorRole,
    residency: ResidencyClass,
    consumer: KernelConsumer,
    section: TqfSectionKind,
}

fn c(
    role: TensorRole,
    residency: ResidencyClass,
    consumer: KernelConsumer,
    section: TqfSectionKind,
) -> Option<Classification> {
    Some(Classification {
        role,
        residency,
        consumer,
        section,
    })
}

/// Classifies one tensor name into a logical role. Order matters — more
/// specific patterns are checked before broader ones so e.g. `q_norm`
/// isn't swallowed by a looser `norm` match.
fn classify(name: &str) -> Option<Classification> {
    use KernelConsumer as K;
    use ResidencyClass as R;
    use TensorRole as T;
    use TqfSectionKind as S;

    if name.contains("token_embd") || name.ends_with("embed_tokens.weight") {
        return c(
            T::TokenEmbedding,
            R::ResidentCore,
            K::Embedding,
            S::Embeddings,
        );
    }
    if name == "output_norm.weight" || name.ends_with("model.norm.weight") {
        return c(T::FinalNorm, R::ResidentCore, K::Norm, S::Embeddings);
    }
    if name == "output.weight" || name.ends_with("lm_head.weight") {
        return c(T::LmHead, R::ResidentCore, K::LmHead, S::LmHead);
    }
    if name.contains("mmproj") || name.contains("vision") {
        return c(T::VisionProjector, R::Lazy, K::Vision, S::VisionLink);
    }

    if name.ends_with("q_proj.weight") || name.ends_with("attn_q.weight") {
        return c(T::AttnQProj, R::ResidentCore, K::Attention, S::ResidentCore);
    }
    if name.ends_with("k_proj.weight") || name.ends_with("attn_k.weight") {
        return c(T::AttnKProj, R::ResidentCore, K::Attention, S::ResidentCore);
    }
    if name.ends_with("v_proj.weight") || name.ends_with("attn_v.weight") {
        return c(T::AttnVProj, R::ResidentCore, K::Attention, S::ResidentCore);
    }
    if name.ends_with("o_proj.weight") || name.ends_with("attn_output.weight") {
        return c(T::AttnOProj, R::ResidentCore, K::Attention, S::ResidentCore);
    }
    if name.ends_with("q_norm.weight") {
        return c(T::AttnQNorm, R::ResidentCore, K::Attention, S::ResidentCore);
    }
    if name.ends_with("k_norm.weight") {
        return c(T::AttnKNorm, R::ResidentCore, K::Attention, S::ResidentCore);
    }
    if name.ends_with("input_layernorm.weight") || name.ends_with("attn_norm.weight") {
        return c(T::AttnNorm, R::ResidentCore, K::Norm, S::ResidentCore);
    }

    // Gated DeltaNet — Transformers-style names only, per module doc.
    if name.ends_with("in_proj_qkv.weight") {
        return c(
            T::GdnInProjQkv,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("in_proj_z.weight") {
        return c(
            T::GdnInProjZ,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("in_proj_a.weight") {
        return c(
            T::GdnInProjA,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("in_proj_b.weight") {
        return c(
            T::GdnInProjB,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("conv1d.weight") {
        return c(
            T::GdnConv1d,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("A_log") {
        return c(
            T::GdnALog,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("dt_bias") {
        return c(
            T::GdnDtBias,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("gated_norm.weight") {
        return c(
            T::GdnGatedNorm,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }
    if name.ends_with("out_proj.weight") {
        return c(
            T::GdnOutProj,
            R::ResidentCore,
            K::GatedDeltaNet,
            S::ResidentCore,
        );
    }

    // Router / experts.
    if name.contains("ffn_gate_inp") || name.ends_with("mlp.gate.weight") {
        return c(T::RouterGate, R::ResidentCore, K::Router, S::ResidentCore);
    }
    if name.contains("shared_expert") {
        if name.contains("gate") {
            return c(
                T::SharedExpertGate,
                R::ResidentCore,
                K::Expert,
                S::ResidentCore,
            );
        }
        if name.contains("up") {
            return c(
                T::SharedExpertUp,
                R::ResidentCore,
                K::Expert,
                S::ResidentCore,
            );
        }
        if name.contains("down") {
            return c(
                T::SharedExpertDown,
                R::ResidentCore,
                K::Expert,
                S::ResidentCore,
            );
        }
    }
    if name.contains("ffn_gate_exps") || (name.contains("experts") && name.contains("gate")) {
        return c(
            T::RoutedExpertGate,
            R::ColdExpertStore,
            K::Expert,
            S::RoutedExperts,
        );
    }
    if name.contains("ffn_up_exps") || (name.contains("experts") && name.contains("up")) {
        return c(
            T::RoutedExpertUp,
            R::ColdExpertStore,
            K::Expert,
            S::RoutedExperts,
        );
    }
    if name.contains("ffn_down_exps") || (name.contains("experts") && name.contains("down")) {
        return c(
            T::RoutedExpertDown,
            R::ColdExpertStore,
            K::Expert,
            S::RoutedExperts,
        );
    }
    if name.ends_with("ffn_norm.weight") || name.ends_with("post_attention_layernorm.weight") {
        return c(T::FfnNorm, R::ResidentCore, K::Norm, S::ResidentCore);
    }

    None
}

/// Finds the first purely-numeric dot-separated path segment (`blk.5.` /
/// `model.layers.12.` both work) and treats it as the layer index. Not
/// validated against `Qwen36Geometry::NUM_LAYERS` here — that cross-check
/// belongs to whichever phase first validates a full real manifest.
fn extract_layer_index(name: &str) -> Option<LayerId> {
    name.split('.')
        .find_map(|segment| segment.parse::<u32>().ok())
        .and_then(|n| u8::try_from(n).ok())
        .map(LayerId)
}

/// Reads `gguf_path`'s tensor descriptors and classifies every one. Fails
/// on the first tensor that doesn't match a known role (spec §118) rather
/// than silently dropping it from the inventory.
pub fn generate_inventory(gguf_path: &Path) -> Result<Vec<TensorInventoryEntry>> {
    let file = gguf::open(gguf_path)?;
    let mut entries = Vec::with_capacity(file.tensors.len());
    for tensor in &file.tensors {
        let classification = classify(&tensor.name)
            .ok_or_else(|| GgufError::UnclassifiedTensor(tensor.name.clone()))?;
        entries.push(TensorInventoryEntry {
            canonical_name: tensor.name.clone(),
            logical_role: classification.role,
            layer: extract_layer_index(&tensor.name).map(|l| l.0),
            shape: tensor.dims.clone(),
            source_quant: tensor.ggml_type,
            source_bytes: tensor.byte_size,
            tqf_section: classification.section,
            residency: classification.residency,
            consumer: classification.consumer,
        });
    }
    Ok(entries)
}

/// Serializes `entries` as pretty JSON, writes it to `path`, and returns a
/// SHA-256 of the serialized bytes — the "generated inventory hash stable"
/// property spec §272 asks Phase 0 to test is just determinism of this
/// function on the same input.
pub fn write_inventory_json(entries: &[TensorInventoryEntry], path: &Path) -> Result<String> {
    let json = serde_json::to_string_pretty(entries).map_err(|e| {
        GgufError::UnclassifiedTensor(format!("failed to serialize inventory: {e}"))
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json.as_bytes())?;
    Ok(crate::source::checksum::hex_digest(json.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformers_style_names_are_all_classified() {
        for name in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.1.linear_attn.in_proj_qkv.weight",
            "model.layers.1.linear_attn.in_proj_z.weight",
            "model.layers.1.linear_attn.in_proj_a.weight",
            "model.layers.1.linear_attn.in_proj_b.weight",
            "model.layers.1.linear_attn.conv1d.weight",
            "model.layers.1.linear_attn.A_log",
            "model.layers.1.linear_attn.dt_bias",
            "model.layers.1.linear_attn.gated_norm.weight",
            "model.layers.1.linear_attn.out_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
        ] {
            assert!(classify(name).is_some(), "expected {name:?} to classify");
        }
    }

    #[test]
    fn llama_cpp_style_names_are_all_classified() {
        for name in [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_norm.weight",
        ] {
            assert!(classify(name).is_some(), "expected {name:?} to classify");
        }
    }

    #[test]
    fn unknown_tensor_name_does_not_classify() {
        assert!(classify("totally.unknown.tensor.weight").is_none());
    }

    #[test]
    fn layer_index_extracted_from_both_naming_conventions() {
        assert_eq!(extract_layer_index("blk.7.attn_q.weight"), Some(LayerId(7)));
        assert_eq!(
            extract_layer_index("model.layers.23.self_attn.q_proj.weight"),
            Some(LayerId(23))
        );
        assert_eq!(extract_layer_index("output.weight"), None);
    }

    // --- Generator-level tests using a small synthetic GGUF fixture ---

    fn write_synthetic_fixture(path: &std::path::Path) {
        write_gguf_fixture(
            path,
            &[
                ("token_embd.weight", vec![32]),
                ("blk.0.attn_q.weight", vec![32]),
                ("blk.3.ffn_gate_exps.weight", vec![32]),
                ("output.weight", vec![32]),
            ],
        );
    }

    /// Hand-rolled GGUF byte builder — same pattern as
    /// `src/format/gguf/tests.rs`, kept local/minimal here rather than
    /// sharing code across a private test module boundary. Every tensor
    /// gets one Q4_0 block (32 elements, 18 bytes) of placeholder data.
    fn write_gguf_fixture(path: &std::path::Path, tensors: &[(&str, Vec<u64>)]) {
        fn write_string(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        fn align_up(v: u64, a: u64) -> u64 {
            v.div_ceil(a) * a
        }

        let block = vec![0xABu8; 18]; // one Q4_0 block per tensor

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // no metadata entries

        let alignment = 32u64;
        let mut relative_offsets = Vec::new();
        let mut cursor = 0u64;
        for _ in tensors.iter() {
            let aligned = align_up(cursor, alignment);
            relative_offsets.push(aligned);
            cursor = aligned + block.len() as u64;
        }
        for ((name, dims), rel_off) in tensors.iter().zip(&relative_offsets) {
            write_string(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
            out.extend_from_slice(&rel_off.to_le_bytes());
        }
        let data_start = align_up(out.len() as u64, alignment);
        out.resize(data_start as usize, 0);
        for (_, rel_off) in tensors.iter().zip(&relative_offsets) {
            let target_len = (data_start + rel_off) as usize;
            if out.len() < target_len {
                out.resize(target_len, 0);
            }
            out.extend_from_slice(&block);
        }

        std::fs::write(path, out).unwrap();
    }

    fn fixture_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tqf-inventory-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn generates_inventory_from_synthetic_fixture() {
        let path = fixture_dir("synthetic.gguf");
        write_synthetic_fixture(&path);

        let entries = generate_inventory(&path).unwrap();
        assert_eq!(entries.len(), 4);

        let embed = entries
            .iter()
            .find(|e| e.canonical_name == "token_embd.weight")
            .unwrap();
        assert_eq!(embed.logical_role, TensorRole::TokenEmbedding);
        assert_eq!(embed.residency, ResidencyClass::ResidentCore);
        assert_eq!(embed.layer, None);

        let attn_q = entries
            .iter()
            .find(|e| e.canonical_name == "blk.0.attn_q.weight")
            .unwrap();
        assert_eq!(attn_q.layer, Some(0));
        assert_eq!(attn_q.consumer, KernelConsumer::Attention);

        let expert = entries
            .iter()
            .find(|e| e.canonical_name == "blk.3.ffn_gate_exps.weight")
            .unwrap();
        assert_eq!(expert.residency, ResidencyClass::ColdExpertStore);
        assert_eq!(expert.tqf_section, TqfSectionKind::RoutedExperts);
    }

    #[test]
    fn unclassified_tensor_fails_the_whole_generation() {
        let path = fixture_dir("with-unknown.gguf");
        // Reuse the fixture writer's shape but inject one unclassifiable
        // name by writing a minimal one-tensor file inline.
        fn write_string(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        fn align_up(v: u64, a: u64) -> u64 {
            v.div_ceil(a) * a
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut out, "mystery.tensor.weight");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&32u64.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
        out.extend_from_slice(&0u64.to_le_bytes()); // relative_offset = 0
        let data_start = align_up(out.len() as u64, 32);
        out.resize(data_start as usize, 0);
        out.extend_from_slice(&[0u8; 18]); // one Q4_0 block
        std::fs::write(&path, out).unwrap();

        let err = generate_inventory(&path).unwrap_err();
        assert!(matches!(
            err,
            crate::error::TqfError::Format(crate::error::FormatError::Gguf(
                GgufError::UnclassifiedTensor(_)
            ))
        ));
    }

    #[test]
    fn generated_inventory_hash_is_stable_across_runs() {
        let fixture_path = fixture_dir("hash-stable.gguf");
        write_synthetic_fixture(&fixture_path);

        let entries_a = generate_inventory(&fixture_path).unwrap();
        let entries_b = generate_inventory(&fixture_path).unwrap();

        let out_a = fixture_dir("inventory-a.json");
        let out_b = fixture_dir("inventory-b.json");
        let hash_a = write_inventory_json(&entries_a, &out_a).unwrap();
        let hash_b = write_inventory_json(&entries_b, &out_b).unwrap();

        assert_eq!(hash_a, hash_b);
        assert_eq!(
            std::fs::read(&out_a).unwrap(),
            std::fs::read(&out_b).unwrap()
        );
    }

    /// Regenerates the committed `dev/generated/qwen36_tensor_inventory.json`
    /// artifact (spec §272: "generated inventory hash stable" — stability
    /// is what `generated_inventory_hash_is_stable_across_runs` checks;
    /// this test just (re)produces the checked-in file). `#[ignore]`d so a
    /// normal `cargo test` run never writes into the repo tree — run it
    /// explicitly with `cargo test -- --ignored regenerate_committed_tensor_inventory_artifact`
    /// after changing the classifier or this fixture's tensor list.
    ///
    /// **This fixture is synthetic**, built from llama.cpp's well-known
    /// GGUF naming convention plus one representative tensor per logical
    /// role — not the real per-tensor shapes from the actual 20+ GB
    /// checkpoint, which this environment has not downloaded (see this
    /// module's top-level doc comment and
    /// `docs/research/canonical-source-manifest.md`). Shapes below are
    /// illustrative placeholders `[32]`, not the real §117 projection
    /// shapes — re-run this generator against the real file once one is
    /// available to get real shapes classified.
    #[test]
    #[ignore]
    fn regenerate_committed_tensor_inventory_artifact() {
        let path = fixture_dir("full-representative.gguf");
        write_gguf_fixture(
            &path,
            &[
                ("token_embd.weight", vec![32]),
                ("output_norm.weight", vec![32]),
                ("output.weight", vec![32]),
                ("mmproj.weight", vec![32]),
                // One full-attention layer (layer 3, per the 3:1 pattern).
                ("blk.3.attn_q.weight", vec![32]),
                ("blk.3.attn_k.weight", vec![32]),
                ("blk.3.attn_v.weight", vec![32]),
                ("blk.3.attn_output.weight", vec![32]),
                ("blk.3.attn_norm.weight", vec![32]),
                // One Gated DeltaNet layer (layer 0) — Transformers-style
                // names only, see module doc caveat.
                ("model.layers.0.linear_attn.in_proj_qkv.weight", vec![32]),
                ("model.layers.0.linear_attn.in_proj_z.weight", vec![32]),
                ("model.layers.0.linear_attn.in_proj_a.weight", vec![32]),
                ("model.layers.0.linear_attn.in_proj_b.weight", vec![32]),
                ("model.layers.0.linear_attn.conv1d.weight", vec![32]),
                ("model.layers.0.linear_attn.A_log", vec![32]),
                ("model.layers.0.linear_attn.dt_bias", vec![32]),
                ("model.layers.0.linear_attn.gated_norm.weight", vec![32]),
                ("model.layers.0.linear_attn.out_proj.weight", vec![32]),
                // Router + shared + routed experts (layer 0).
                ("blk.0.ffn_gate_inp.weight", vec![32]),
                ("blk.0.mlp.shared_expert.gate_proj.weight", vec![32]),
                ("blk.0.mlp.shared_expert.up_proj.weight", vec![32]),
                ("blk.0.mlp.shared_expert.down_proj.weight", vec![32]),
                ("blk.0.ffn_gate_exps.weight", vec![32]),
                ("blk.0.ffn_up_exps.weight", vec![32]),
                ("blk.0.ffn_down_exps.weight", vec![32]),
                ("blk.0.ffn_norm.weight", vec![32]),
            ],
        );

        let entries = generate_inventory(&path).unwrap();
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let out_path = manifest_dir.join("dev/generated/qwen36_tensor_inventory.json");
        let hash = write_inventory_json(&entries, &out_path).unwrap();
        println!(
            "wrote {} entries to {out_path:?}, sha256={hash}",
            entries.len()
        );
    }
}
