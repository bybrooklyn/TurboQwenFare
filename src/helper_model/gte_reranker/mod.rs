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

