//! `--open <client>` launchers: ephemeral provider/MCP config for opencode,
//! Claude, Codex (spec Part XI sections 99-100). Compatibility glue only;
//! TQF does not run agent loops itself (spec Part I section 2).

pub mod config;
pub mod launch;

// Module facade. `tqf` is a bin-only crate (spec §23: one crate, one
// binary, no `[lib]` target), so rustc reachability-analyses every
// `pub use` from `main` and reports the ones the product surface does not
// yet consume. These re-exports are the module's real interface — keeping
// them is deliberate; the allows go away as each is wired up.
#[allow(unused_imports)]
pub use config::{build_ephemeral_config, ClientKind, EphemeralConfig};
#[allow(unused_imports)]
pub use launch::{
    build_launch_command, ensure_client_available, find_binary_on_path, run_to_completion,
    write_ephemeral_config, LaunchError, WrittenConfig,
};
