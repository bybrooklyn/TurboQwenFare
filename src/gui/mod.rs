//! Native GUI. Talks to local server/control endpoints only; does not
//! duplicate inference logic (spec Part IV section 22; Part XI).

#[cfg(target_os = "macos")]
pub mod macos;
