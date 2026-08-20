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
