//! Internal benchmark harness: workload set, performance/quality protocol
//! (spec Part XII sections 106-108).
//!
//! Phase 10 (§282) adds one concrete mode: a synthetic Metal bandwidth/
//! GEMV microbenchmark proving the `backend::metal` plumbing (device,
//! buffers, pipeline cache, dispatch, timing) works end to end, wired to
//! `tqf optimize` rather than a second binary.

#[cfg(tqf_metal)]
pub mod metal_synthetic;
