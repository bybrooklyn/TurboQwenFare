//! `--open <client>` launchers: ephemeral provider/MCP config for opencode,
//! Claude, Codex (spec Part XI sections 99-100). Compatibility glue only;
//! TQF does not run agent loops itself (spec Part I section 2).

pub mod config;
pub mod launch;

pub use config::{build_ephemeral_config, ClientKind, EphemeralConfig};
pub use launch::{
    build_launch_command, ensure_client_available, find_binary_on_path, run_to_completion,
    write_ephemeral_config, LaunchError, WrittenConfig,
};
