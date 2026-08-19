//! Ties the vision tower together: load a converted `.tqf` under a
//! transient broker lease, encode one already-preprocessed image.
//! Same transient-lease-by-construction pattern as
//! `helper_model::gte_reranker::runtime::GteRerankerRuntime` — dropping
//! `VisionRuntime` releases the `MemoryOwner::HelperModel` reservation,
//! so a text-only session that never sets `--enable-vision` never pays
//! for this tower at all.

use std::path::Path;

use crate::error::Result;
use crate::memory::MemoryBroker;

use super::forward::encode_image;
use super::weights::VisionWeights;

pub struct VisionRuntime {
    weights: VisionWeights,
}

impl VisionRuntime {
    pub fn load(tqf_path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let weights = VisionWeights::load(tqf_path, broker)?;
        Ok(Self { weights })
    }

    /// `image`: normalized pixels, row-major `[height][width][channel=3]`,
    /// already `(raw - IMAGE_MEAN) / IMAGE_STD`. `image_w`/`image_h` must
    /// each be a multiple of `PATCH_SIZE * SPATIAL_MERGE` (32). Returns
    /// one `PROJECTION_DIM`-wide row per merged 2x2-patch token.
    pub fn encode(&self, image: &[f32], image_w: usize, image_h: usize) -> Vec<Vec<f32>> {
        encode_image(&self.weights, image, image_w, image_h)
    }
}
