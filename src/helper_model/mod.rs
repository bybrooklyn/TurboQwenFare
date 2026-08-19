//! Phase 37: the pplx-embed-v1-0.6b helper model runtime (spec §37, §86).
//! A transiently loaded, dense bidirectional Qwen3-architecture encoder
//! used only to serve `/v1/embeddings` — architecturally and
//! operationally distinct from `model::qwen36`'s always-resident MoE
//! decode core (spec's dependency-firewall table lists "helper-model
//! runtime" as its own thing retrieval may depend on, separate from
//! `model`/`runtime`). Kept as a sibling top-level module rather than
//! nested under `model` so that distinction stays visible in the module
//! tree, and so `retrieval` can depend on it without depending on the
//! inference core.

pub mod convert;
pub mod forward;
pub mod geometry;
pub mod pooling;
pub mod quantize;
pub mod roles;
pub mod runtime;
pub mod safetensors;
pub mod weights;

#[cfg(test)]
mod tests;

pub use runtime::{PplxEmbedRuntime, PplxEmbedding};
