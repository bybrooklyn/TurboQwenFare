//! Ephemeral provider/MCP configuration generation (spec §99's table).
//! Every mechanism here was confirmed against each client's real, live
//! documentation before writing this — not guessed from convention:
//!
//! - **OpenCode**: `OPENCODE_CONFIG=<path>` env var loads a config file
//!   from a caller-chosen path (confirmed: opencode.ai/docs/config).
//! - **Claude Code**: `ANTHROPIC_BASE_URL` env var redirects the
//!   gateway (spec's own citation), and `--mcp-config <path>` loads an
//!   MCP config file for one run without touching `.mcp.json` or
//!   `~/.claude.json` (confirmed: code.claude.com/docs/en/mcp).
//! - **Codex**: `CODEX_HOME=<dir>` env var redirects Codex's entire
//!   config directory (so `$CODEX_HOME/config.toml` is never the
//!   user's real `~/.codex/config.toml`), and `[model_providers.tqf]`
//!   with `base_url`/`wire_api = "responses"` matches spec's own
//!   citation of the Responses wire API.

use std::path::PathBuf;

/// Stands in for the ephemeral config directory in an env var value,
/// substituted for the real absolute path by
/// `launch::build_launch_command`.
///
/// The directory does not exist yet when a config is built, so its path
/// cannot be written here. This used to be spelled `"."`, which worked
/// only because the launcher moved the client's working directory into
/// the config directory — and that broke the client, whose whole job is
/// to operate on the project the user is in. A placeholder no path could
/// be confused with keeps the substitution explicit.
pub const CONFIG_DIR: &str = "<tqf-ephemeral-config-dir>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientKind {
    OpenCode,
    Claude,
    Codex,
}

impl ClientKind {
    pub fn binary_name(self) -> &'static str {
        match self {
            ClientKind::OpenCode => "opencode",
            ClientKind::Claude => "claude",
            ClientKind::Codex => "codex",
        }
    }

    /// spec §100: "TQF may use a known official installation recipe."
    /// A recipe is only ever *offered*, never executed automatically —
    /// see `integrations::ensure_client_available`.
    pub fn install_recipe(self) -> &'static str {
        match self {
            ClientKind::OpenCode => "curl -fsSL https://opencode.ai/install | bash",
            ClientKind::Claude => "npm install -g @anthropic-ai/claude-code",
            ClientKind::Codex => "npm install -g @openai/codex",
        }
    }
}

/// One file to write to a private ephemeral directory (never the
/// client's real config location) plus the environment variables that
/// point the client at it.
#[derive(Debug, Clone)]
pub struct EphemeralConfig {
    pub env_vars: Vec<(String, String)>,
    /// `(relative_filename, contents)` — the caller writes these under
    /// a fresh temp directory it fully controls and deletes on exit.
    pub files: Vec<(PathBuf, String)>,
    pub extra_args: Vec<String>,
}

/// Builds the ephemeral config for one client, pointed at TQF's local
/// server (`server_base_url`, e.g. `http://127.0.0.1:11535`) and one
/// MCP server entry naming `mcp_binary`/`mcp_args` as the command a
/// stdio MCP client should launch to reach `mcp::stdio`'s transport.
pub fn build_ephemeral_config(
    kind: ClientKind,
    server_base_url: &str,
    mcp_binary: &str,
    mcp_args: &[&str],
) -> EphemeralConfig {
    match kind {
        ClientKind::OpenCode => {
            let mcp_args_json: Vec<String> = mcp_args.iter().map(|a| format!("{a:?}")).collect();
            let config = format!(
                r#"{{
  "provider": {{
    "tqf": {{
      "npm": "@ai-sdk/openai-compatible",
      "name": "TurboQwenFare (local)",
      "options": {{ "baseURL": {server_base_url:?} }},
      "models": {{ "tqf": {{ "name": "TurboQwenFare" }} }}
    }}
  }},
  "mcp": {{
    "tqf": {{
      "type": "local",
      "command": [{mcp_binary:?}, {}]
    }}
  }}
}}
"#,
                mcp_args_json.join(", ")
            );
            EphemeralConfig {
                env_vars: vec![("OPENCODE_CONFIG".to_string(), "opencode.json".to_string())],
                files: vec![(PathBuf::from("opencode.json"), config)],
                extra_args: vec![],
            }
        }
        ClientKind::Claude => {
            let mcp_args_json: Vec<String> = mcp_args.iter().map(|a| format!("{a:?}")).collect();
            let mcp_config = format!(
                r#"{{
  "mcpServers": {{
    "tqf": {{
      "command": {mcp_binary:?},
      "args": [{}]
    }}
  }}
}}
"#,
                mcp_args_json.join(", ")
            );
            EphemeralConfig {
                env_vars: vec![
                    (
                        "ANTHROPIC_BASE_URL".to_string(),
                        server_base_url.to_string(),
                    ),
                    // Claude Code requires a non-empty key to start;
                    // TQF's local-only server doesn't validate it.
                    ("ANTHROPIC_API_KEY".to_string(), "tqf-local".to_string()),
                ],
                files: vec![(PathBuf::from("mcp.json"), mcp_config)],
                extra_args: vec!["--mcp-config".to_string(), "mcp.json".to_string()],
            }
        }
        ClientKind::Codex => {
            let mcp_args_toml: Vec<String> = mcp_args.iter().map(|a| format!("{a:?}")).collect();
            let config = format!(
                r#"[model_providers.tqf]
name = "TurboQwenFare (local)"
base_url = {server_base_url:?}
env_key = "TQF_API_KEY"
wire_api = "responses"

[mcp_servers.tqf]
command = {mcp_binary:?}
args = [{}]
"#,
                mcp_args_toml.join(", ")
            );
            EphemeralConfig {
                env_vars: vec![
                    ("CODEX_HOME".to_string(), CONFIG_DIR.to_string()),
                    ("TQF_API_KEY".to_string(), "tqf-local".to_string()),
                ],
                files: vec![(PathBuf::from("config.toml"), config)],
                extra_args: vec![],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencode_config_points_baseurl_at_the_local_server() {
        let config = build_ephemeral_config(
            ClientKind::OpenCode,
            "http://127.0.0.1:11535",
            "tqf",
            &["mcp", "stdio"],
        );
        assert!(config
            .env_vars
            .contains(&("OPENCODE_CONFIG".to_string(), "opencode.json".to_string())));
        let (_, contents) = &config.files[0];
        assert!(contents.contains("http://127.0.0.1:11535"));
        assert!(contents.contains("\"mcp\""));
    }

    #[test]
    fn claude_config_uses_anthropic_base_url_and_mcp_config_flag() {
        let config = build_ephemeral_config(
            ClientKind::Claude,
            "http://127.0.0.1:11535",
            "tqf",
            &["mcp", "stdio"],
        );
        assert!(config
            .env_vars
            .iter()
            .any(|(k, v)| k == "ANTHROPIC_BASE_URL" && v == "http://127.0.0.1:11535"));
        assert_eq!(config.extra_args, vec!["--mcp-config", "mcp.json"]);
    }

    #[test]
    fn codex_config_uses_codex_home_and_responses_wire_api() {
        let config = build_ephemeral_config(
            ClientKind::Codex,
            "http://127.0.0.1:11535",
            "tqf",
            &["mcp", "stdio"],
        );
        assert!(config.env_vars.iter().any(|(k, _)| k == "CODEX_HOME"));
        let (_, contents) = &config.files[0];
        assert!(contents.contains(r#"wire_api = "responses""#));
        assert!(contents.contains("http://127.0.0.1:11535"));
    }

    #[test]
    fn no_client_config_ever_names_a_real_permanent_config_path() {
        for kind in [ClientKind::OpenCode, ClientKind::Claude, ClientKind::Codex] {
            let config = build_ephemeral_config(kind, "http://127.0.0.1:11535", "tqf", &["mcp"]);
            for (path, _) in &config.files {
                let s = path.to_string_lossy();
                assert!(
                    !s.contains(".claude.json")
                        && !s.contains(".codex/config")
                        && !s.starts_with('/'),
                    "config file path must be relative to a caller-owned temp dir, got {s:?}"
                );
            }
        }
    }
}
