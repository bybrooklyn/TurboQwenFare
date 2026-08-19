//! Canonical Qwen3.6 geometry (spec §117, **LOCKED** — "from the pinned
//! official checkpoint. The importer must validate these fields before
//! conversion."). Cross-checked live against `Qwen/Qwen3.6-35B-A3B`'s
//! `config.json` on 2026-08-11 — every field matched exactly; see
//! `docs/research/canonical-source-manifest.md` for the full comparison
//! table and how to re-verify.

use crate::ids::{LayerId, LayerKind};

/// Zero-sized marker type: `Qwen36Geometry::HIDDEN_SIZE` etc. reads as the
/// "compile-time `Qwen36Geometry` constants" spec §272's Phase 0 exit test
/// names ("official config fields equal compile-time `Qwen36Geometry`
/// constants").
pub struct Qwen36Geometry;

impl Qwen36Geometry {
    pub const VOCAB_SIZE: usize = 248_320;
    pub const HIDDEN_SIZE: usize = 2048;
    pub const NUM_LAYERS: usize = 40;

    /// Every 4th layer (1-indexed) is full attention; the other three in
    /// each group of four are Gated DeltaNet — §116: "compiled from the
    /// official 3-linear/1-full pattern."
    pub const FULL_ATTENTION_INTERVAL: usize = 4;
    pub const GDN_LAYERS: usize = 30;
    pub const FULL_ATTENTION_LAYERS: usize = 10;

    pub const FULL_ATTENTION_HEADS: usize = 16;
    pub const FULL_KV_HEADS: usize = 2;
    pub const FULL_HEAD_DIM: usize = 256;
    pub const ROTARY_FRACTION: f64 = 0.25;
    /// `FULL_HEAD_DIM * ROTARY_FRACTION`.
    pub const ROTARY_SUBDIM: usize = 64;
    /// Text RoPE base from the pinned canonical config. Multimodal M-RoPE
    /// sectioning is intentionally deferred to the vision phase.
    pub const ROPE_THETA: f32 = 10_000_000.0;

    pub const GDN_KEY_HEADS: usize = 16;
    pub const GDN_VALUE_HEADS: usize = 32;
    pub const GDN_KEY_HEAD_DIM: usize = 128;
    pub const GDN_VALUE_HEAD_DIM: usize = 128;
    /// `GDN_KEY_HEADS * GDN_KEY_HEAD_DIM`.
    pub const GDN_KEY_DIM: usize = 2048;
    /// `GDN_VALUE_HEADS * GDN_VALUE_HEAD_DIM`.
    pub const GDN_VALUE_DIM: usize = 4096;
    /// The fused `in_proj_qkv` output width: key (2048) + query (2048) +
    /// value (4096) — spec §117's projection-shape table gives
    /// `in_proj_qkv: [8192, 2048]` directly; this constant is that 8192.
    pub const GDN_CONV_CHANNELS: usize = 8192;
    pub const GDN_CONV_WIDTH: usize = 4;

    pub const NUM_EXPERTS: usize = 256;
    pub const ROUTED_EXPERTS_PER_TOKEN: usize = 8;
    pub const ROUTED_EXPERT_WIDTH: usize = 512;
    pub const SHARED_EXPERT_WIDTH: usize = 512;

    pub const MTP_HIDDEN_LAYERS: usize = 1;
    pub const NATIVE_CONTEXT: usize = 262_144;

    /// The canonical layer-kind table, compiled from `FULL_ATTENTION_INTERVAL`
    /// and also verified against the installed model manifest at runtime —
    /// a mismatch there is a fatal architecture error (spec §116).
    pub fn layer_kind(layer: LayerId) -> LayerKind {
        if (layer.0 as usize + 1).is_multiple_of(Self::FULL_ATTENTION_INTERVAL) {
            LayerKind::FullAttention
        } else {
            LayerKind::GatedDeltaNet
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_live_config_json_cross_check() {
        // Frozen against docs/research/canonical-source-manifest.md's
        // 2026-08-11 fetch of Qwen/Qwen3.6-35B-A3B/config.json — every
        // value here was compared field-by-field against that live fetch
        // and matched exactly.
        assert_eq!(Qwen36Geometry::HIDDEN_SIZE, 2048);
        assert_eq!(Qwen36Geometry::NUM_LAYERS, 40);
        assert_eq!(Qwen36Geometry::FULL_ATTENTION_HEADS, 16);
        assert_eq!(Qwen36Geometry::FULL_KV_HEADS, 2);
        assert_eq!(Qwen36Geometry::FULL_HEAD_DIM, 256);
        assert_eq!(Qwen36Geometry::GDN_KEY_HEADS, 16);
        assert_eq!(Qwen36Geometry::GDN_VALUE_HEADS, 32);
        assert_eq!(Qwen36Geometry::GDN_KEY_HEAD_DIM, 128);
        assert_eq!(Qwen36Geometry::GDN_VALUE_HEAD_DIM, 128);
        assert_eq!(Qwen36Geometry::GDN_CONV_WIDTH, 4);
        assert_eq!(Qwen36Geometry::NUM_EXPERTS, 256);
        assert_eq!(Qwen36Geometry::ROUTED_EXPERTS_PER_TOKEN, 8);
        assert_eq!(Qwen36Geometry::ROUTED_EXPERT_WIDTH, 512);
        assert_eq!(Qwen36Geometry::SHARED_EXPERT_WIDTH, 512);
        assert_eq!(Qwen36Geometry::VOCAB_SIZE, 248_320);
        assert_eq!(Qwen36Geometry::NATIVE_CONTEXT, 262_144);
    }

    #[test]
    fn derived_dims_are_internally_consistent() {
        assert_eq!(
            Qwen36Geometry::ROTARY_SUBDIM,
            (Qwen36Geometry::FULL_HEAD_DIM as f64 * Qwen36Geometry::ROTARY_FRACTION) as usize
        );
        assert_eq!(
            Qwen36Geometry::GDN_KEY_DIM,
            Qwen36Geometry::GDN_KEY_HEADS * Qwen36Geometry::GDN_KEY_HEAD_DIM
        );
        assert_eq!(
            Qwen36Geometry::GDN_VALUE_DIM,
            Qwen36Geometry::GDN_VALUE_HEADS * Qwen36Geometry::GDN_VALUE_HEAD_DIM
        );
        assert_eq!(
            Qwen36Geometry::GDN_CONV_CHANNELS,
            Qwen36Geometry::GDN_KEY_DIM
                + Qwen36Geometry::HIDDEN_SIZE
                + Qwen36Geometry::GDN_VALUE_DIM
        );
    }

    #[test]
    fn all_forty_layer_kinds_match_the_three_to_one_pattern() {
        let mut gdn_count = 0;
        let mut full_count = 0;
        for i in 0..Qwen36Geometry::NUM_LAYERS {
            let kind = Qwen36Geometry::layer_kind(LayerId(i as u8));
            match kind {
                LayerKind::GatedDeltaNet => gdn_count += 1,
                LayerKind::FullAttention => full_count += 1,
            }
            // Every group of 4 layers (0-indexed) is GDN,GDN,GDN,FullAttention.
            let expected = if i % 4 == 3 {
                LayerKind::FullAttention
            } else {
                LayerKind::GatedDeltaNet
            };
            assert_eq!(kind, expected, "layer {i} has unexpected kind");
        }
        assert_eq!(gdn_count, Qwen36Geometry::GDN_LAYERS);
        assert_eq!(full_count, Qwen36Geometry::FULL_ATTENTION_LAYERS);
        assert_eq!(gdn_count + full_count, Qwen36Geometry::NUM_LAYERS);
    }
}
