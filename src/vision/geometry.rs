//! Fixed graph for the lazy vision tower (spec Part XIII, phase 48):
//! Qwen3.6's CLIP-style ViT encoder + `qwen3vl_merger` projector, shipped
//! as a separate `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` sidecar
//! (`source::pinned::VISION_PROJECTOR_FILENAME`). Every constant below
//! was cross-checked two ways on 2026-08-19: (1) a direct read of the
//! real pinned mmproj GGUF's `clip.vision.*` metadata keys via Python's
//! `gguf` library, and (2) the real llama.cpp reference architecture —
//! `tools/mtmd/models/qwen3vl.cpp`'s `clip_graph_qwen3vl::build()` (which
//! this crate does not link against, only reads as a source-of-truth
//! reference) plus `tools/mtmd/clip.cpp`'s `resize_position_embeddings`
//! and `ggml-cpu/ops.cpp`'s `ggml_mrope_cache_init`/`rotate_pairs` for
//! the exact 2D vision-RoPE and bilinear-align-corners position
//! interpolation formulas. A real `llama-mtmd-debug` oracle run against
//! the pinned checkpoint (96x96 synthetic "gray" image) is the
//! validation target in `tests.rs`.

pub struct VisionGeometry;

impl VisionGeometry {
    /// `clip.vision.embedding_length`.
    pub const HIDDEN: usize = 1152;
    /// `clip.vision.attention.head_count`.
    pub const HEADS: usize = 16;
    pub const HEAD_DIM: usize = Self::HIDDEN / Self::HEADS; // 72
    /// `clip.vision.block_count`.
    pub const LAYERS: usize = 27;
    /// `clip.vision.feed_forward_length`.
    pub const INTERMEDIATE: usize = 4304;
    /// `clip.vision.attention.layer_norm_epsilon` (real value
    /// 9.999999974752427e-07, an f32-rounded 1e-6).
    pub const LN_EPS: f32 = 1e-6;
    /// `clip.vision.patch_size`.
    pub const PATCH_SIZE: usize = 16;
    /// `clip.vision.image_size` — the native square grid the learned
    /// absolute position embedding table was trained at
    /// (768 / 16 = 48 patches per side, 48*48 = 2304 stored positions).
    /// Any other input resolution bilinear-interpolates this table
    /// (`resize_position_embeddings`, `LN_EPS`-independent).
    pub const NATIVE_IMAGE_SIZE: usize = 768;
    pub const NATIVE_PATCHES_PER_SIDE: usize = Self::NATIVE_IMAGE_SIZE / Self::PATCH_SIZE; // 48
    /// `clip.vision.spatial_merge_size` — 2x2 patches merge into one
    /// projected token.
    pub const SPATIAL_MERGE: usize = 2;
    pub const MERGED_HIDDEN: usize = Self::HIDDEN * Self::SPATIAL_MERGE * Self::SPATIAL_MERGE; // 4608
    /// `clip.vision.projection_dim` — the merger's final output width.
    pub const PROJECTION_DIM: usize = 2048;
    /// `clip.vision.image_mean` / `clip.vision.image_std` (all three
    /// RGB channels identical: 0.5).
    pub const IMAGE_MEAN: f32 = 0.5;
    pub const IMAGE_STD: f32 = 0.5;

    /// M-RoPE (`GGML_ROPE_TYPE_VISION`) frequency base
    /// (`ggml_rope_multi(..., GGML_ROPE_TYPE_VISION, 32768, 10000, ...)`
    /// in `qwen3vl.cpp`).
    pub const ROPE_FREQ_BASE: f32 = 10000.0;
    /// The `n_dims` argument to `ggml_rope_multi` is `d_head/2` = 36 —
    /// *not* `d_head` — confirmed against `ggml_mrope_cache_init`'s
    /// `theta_scale = freq_base^(-2/n_dims)` and `is_vision`'s
    /// `GGML_ASSERT(n_dims == ne0/2)`. Each of the two position axes
    /// (row, column) gets `ROPE_PAIRS_PER_AXIS` independent frequencies,
    /// and together they cover the *entire* head dim via paired
    /// rotation `(i, i+36)` for `i` in `0..36` — vision RoPE has no
    /// pass-through tail, unlike the text NEOX path.
    pub const ROPE_N_DIMS: usize = Self::HEAD_DIM / 2; // 36
    pub const ROPE_PAIRS_PER_AXIS: usize = Self::ROPE_N_DIMS / 2; // 18

    pub const fn patch_grid_side(image_size: usize) -> usize {
        image_size / Self::PATCH_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_times_heads_matches_hidden() {
        assert_eq!(
            VisionGeometry::HEAD_DIM * VisionGeometry::HEADS,
            VisionGeometry::HIDDEN
        );
    }

    #[test]
    fn merged_hidden_matches_spatial_merge_of_hidden() {
        assert_eq!(
            VisionGeometry::MERGED_HIDDEN,
            VisionGeometry::HIDDEN * VisionGeometry::SPATIAL_MERGE * VisionGeometry::SPATIAL_MERGE
        );
    }

    #[test]
    fn native_grid_matches_position_table_size() {
        assert_eq!(
            VisionGeometry::NATIVE_PATCHES_PER_SIDE * VisionGeometry::NATIVE_PATCHES_PER_SIDE,
            2304
        );
    }

    #[test]
    fn rope_pairs_cover_the_full_head_dim() {
        assert_eq!(
            VisionGeometry::ROPE_PAIRS_PER_AXIS * 2 * 2,
            VisionGeometry::HEAD_DIM
        );
    }
}
