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
