//! Apple Metal backend: device/queue ownership, shared buffers, pipeline
//! cache, MSL kernels (spec Part VII sections 49-53). Primary platform.
//!
//! This phase (§282, phase 10) builds the plumbing — device/queue,
//! buffer leases, a pipeline cache, event timing, and baseline
//! metallib/MSL loading — that later phases' real Q4/GDN/attention/MoE
//! kernels (§283 onward) will be built on top of. Nothing here implements
//! an actual model kernel yet; `shaderlib::BASELINE_MSL_SOURCE` is a
//! synthetic bandwidth-copy/GEMV pair used only to prove the plumbing
//! works end to end (exercised by `tqf optimize`, spec §3).

pub mod buffer;
pub mod context;
pub mod pipeline;
pub mod shaderlib;
pub mod timing;

pub use buffer::BufferLease;
pub use context::MetalContext;
pub use pipeline::PipelineCache;
pub use timing::{EventFence, GpuStopwatch};
