//! Lazy vision encoder/projector, unloaded unless `--enable-vision` is set
//! (spec Part XIII phase 48; Part I section 1). A CLIP-style ViT encoder
//! plus `qwen3vl_merger` projector for the pinned
//! `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` sidecar — see `geometry.rs` for how
//! every constant here was cross-checked against the real checkpoint and
//! the real llama.cpp reference implementation, and `forward.rs` for the
//! per-step architecture derivation.

mod convert;
mod forward;
mod geometry;
mod roles;
mod runtime;
mod weights;

#[cfg(test)]
mod tests;

// Module facade. `tqf` is a bin-only crate (spec §23: one crate, one
// binary, no `[lib]` target), so rustc reachability-analyses every
// `pub use` from `main` and reports the ones the product surface does not
// yet consume. These re-exports are the module's real interface — keeping
// them is deliberate; the allows go away as each is wired up.
#[allow(unused_imports)]
pub use convert::{convert_vision_gguf, VisionConversionReport};
#[allow(unused_imports)]
pub use geometry::VisionGeometry;
#[allow(unused_imports)]
pub use runtime::VisionRuntime;
