//! Backend interface: one execution-plan abstraction, platform-specific
//! implementations underneath (spec Part VII section 48).

#[cfg(tqf_metal)]
pub mod metal;

#[cfg(tqf_cuda)]
pub mod cuda;

pub mod reference;
