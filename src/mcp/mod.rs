//! Retrieval/index MCP server, stdio and HTTP transports (spec Part X
//! section 95, Phase 44). Read-only tools only — "File edits/execution
//! belong to the client harness." Works with `IndexState: None` (no
//! index built) without erroring the protocol, since spec §44 requires
//! "the server works normally without an index."

pub mod protocol;
pub mod server;
pub mod stdio;
pub mod tools;

#[cfg(test)]
mod tests;

// Module facade. `tqf` is a bin-only crate (spec §23: one crate, one
// binary, no `[lib]` target), so rustc reachability-analyses every
// `pub use` from `main` and reports the ones the product surface does not
// yet consume. These re-exports are the module's real interface — keeping
// them is deliberate; the allows go away as each is wired up.
#[allow(unused_imports)]
pub use server::handle_request;
#[allow(unused_imports)]
pub use tools::IndexState;
