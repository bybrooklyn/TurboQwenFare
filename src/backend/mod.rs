//! Backend interface: one execution-plan abstraction, platform-specific
//! implementations underneath (spec Part VII section 48).

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

pub mod reference;
