//! Apple Metal backend: device/queue ownership, shared buffers, pipeline
//! cache, MSL kernels (spec Part VII sections 49-53). Primary platform.
//!
//! Phase 10 (§282) built the plumbing — device/queue, buffer leases, a
//! pipeline cache, event timing, and baseline metallib/MSL loading
//! (`shaderlib::BASELINE_MSL_SOURCE`'s synthetic bandwidth-copy/GEMV pair,
//! exercised by `tqf optimize`, spec §3). Phase 11 (§283) builds the first
//! real model kernels on top of it: `kernels` holds the reference Q4_K
//! GEMV/GEMM, RMSNorm, elementwise, and LM-head kernels. These are
//! deliberately unoptimized "slow-clear" implementations — a correctness
//! oracle later specialization work (spec §51) is validated against, not a
//! throughput target.

pub mod buffer;
pub mod context;
pub mod expert;
pub mod kernels;
pub mod pipeline;
pub mod shaderlib;
pub mod timing;

pub use buffer::BufferLease;
pub use context::MetalContext;
pub use expert::GpuResidentExpert;
pub use pipeline::PipelineCache;
pub use timing::{EventFence, GpuStopwatch};
