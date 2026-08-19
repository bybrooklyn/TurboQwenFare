//! Phase 43: the GTE reranker helper model (spec §43, §93):
//! `Alibaba-NLP/gte-reranker-modernbert-base`, a transient ModernBERT
//! cross-encoder used only to rerank a bounded candidate set (spec
//! §196). A sibling of `helper_model`'s pplx-embed implementation, not
//! a variant of it — different architecture family entirely (LayerNorm
//! not RMSNorm, GeGLU not SwiGLU, alternating global/local sliding-
//! window attention, a joint (query, document) cross-encoder input
//! rather than independent embeddings).

pub mod convert;
pub mod forward;
pub mod geometry;
pub mod roles;
pub mod runtime;
pub mod weights;

#[cfg(test)]
mod tests;

// Module facade. `tqf` is a bin-only crate (spec §23: one crate, one
// binary, no `[lib]` target), so rustc reachability-analyses every
// `pub use` from `main` and reports the ones the product surface does not
// yet consume. These re-exports are the module's real interface — keeping
// them is deliberate; the allows go away as each is wired up.
#[allow(unused_imports)]
pub use runtime::GteRerankerRuntime;
