//! Ollama-owned blob locator (spec §276: "later Ollama blob locator").
//! Deferred: Ollama stores blobs content-addressed under its own manifest
//! layout, and resolving that layout doesn't exist in this crate yet. When
//! it lands it plugs in next to `HfRangeSource`/`LocalFileSource` at the
//! same `Box<dyn ModelSource>` call sites. Per spec §29/§127, an
//! Ollama-owned blob must never be deleted by TQF regardless of how
//! ownership bookkeeping evolves elsewhere in this module — it is always
//! `SourceOwnership::UserOwned`-equivalent.
