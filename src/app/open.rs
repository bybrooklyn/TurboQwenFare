//! `tqf --open {opencode,claude,codex}` (spec §99-100, §224).
//!
//! Starts the server, waits until it is genuinely answering, writes an
//! ephemeral provider/MCP config for the chosen client, launches it, and
//! deletes the config when it exits. The user's own client configuration
//! is never touched.
//!
//! The pieces this drives (`integrations::{config,launch}`) were built and
//! tested long before anything called them; this module is the call site
//! they were missing.

use std::io::{IsTerminal, Write};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use crate::cli::Cli;
use crate::error::{ConfigError, Result, SetupError};
use crate::integrations::config::{build_ephemeral_config, ClientKind};
use crate::integrations::launch::{
    build_launch_command, ensure_client_available, run_to_completion, write_ephemeral_config,
    LaunchError,
};

/// How long to wait for the server to answer before giving up. Startup
/// includes opening and validating the container, which is not instant on
/// a cold cache.
const READY_TIMEOUT: Duration = Duration::from_secs(300);

pub fn parse_client(name: &str) -> Result<ClientKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "opencode" => Ok(ClientKind::OpenCode),
        "claude" | "claude-code" => Ok(ClientKind::Claude),
        "codex" => Ok(ClientKind::Codex),
        other => Err(ConfigError::InvalidClient(other.to_string()).into()),
    }
}

/// Runs the client against an already-bound server.
///
/// `addr` is the address the server actually bound, not the default one:
/// under port fallback those differ, and writing the default into a
/// client's config would point it at whatever else holds that port.
pub fn launch(cli: &Cli, kind: ClientKind, addr: SocketAddr) -> Result<()> {
    let binary = match ensure_client_available(kind, |recipe| confirm_install(cli, kind, recipe)) {
        Ok(path) => path,
        Err(LaunchError::ClientNotFound { kind, declined }) => {
            // Spec §100: offer the recipe, never run it. "Ollama-easy,
            // not magical system mutation."
            let message = if declined {
                format!(
                    "tqf: {} is not installed and installation was declined.",
                    kind.binary_name()
                )
            } else {
                format!(
                    "tqf: {} is not installed.\n     Install it with:\n       {}\n     \
                     then re-run `tqf --open {}`.",
                    kind.binary_name(),
                    kind.install_recipe(),
                    kind.binary_name()
                )
            };
            println!("{message}");
            return Ok(());
        }
        Err(error) => return Err(SetupError::ClientLaunch(error.to_string()).into()),
    };

    let base_url = format!("http://{addr}/v1");
    let executable = std::env::current_exe().map_err(|error| {
        SetupError::ClientLaunch(format!("cannot locate the running tqf binary: {error}"))
    })?;
    let executable = executable.to_string_lossy().to_string();

    // The MCP command points back at this same binary's hidden stdio
    // entrypoint, which is why that flag has to exist for `--open` to
    // write a config that works.
    let config = build_ephemeral_config(kind, &base_url, &executable, &["--mcp-stdio"]);
    let written = write_ephemeral_config(&config, &std::env::temp_dir())
        .map_err(|error| SetupError::ClientLaunch(error.to_string()))?;

    println!(
        "tqf: launching {} against http://{addr} (ephemeral config in {}).",
        kind.binary_name(),
        written.dir.display()
    );

    let command = build_launch_command(&binary, &config, &written);
    // `run_to_completion` drops `written`, deleting the ephemeral
    // directory on every exit path.
    let status = run_to_completion(command, written)
        .map_err(|error| SetupError::ClientLaunch(error.to_string()))?;

    if !status.success() {
        println!(
            "tqf: {} exited with {}.",
            kind.binary_name(),
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "a signal".to_string())
        );
    }
    Ok(())
}

/// Asks before installing anything. `--yes` is the documented
/// non-interactive consent; with no tty and no `--yes`, refuse rather
/// than guess.
fn confirm_install(cli: &Cli, kind: ClientKind, recipe: &str) -> bool {
    if cli.yes {
        return true;
    }
    if !std::io::stdin().is_terminal() {
        return false;
    }
    print!(
        "tqf: {} is not installed. Install it with:\n       {recipe}\n     Proceed? [y/N] ",
        kind.binary_name()
    );
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() {
        return false;
    }
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

/// Blocks until the server answers `/health`, so the client is never
/// launched against a port that is not serving yet.
pub async fn wait_until_ready(addr: SocketAddr) -> bool {
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if let Some(body) = crate::server::bind::http_get(addr, "/health").await {
            if body.contains(r#""status":"ok""#) {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_documented_client_name_parses() {
        assert_eq!(parse_client("opencode").unwrap(), ClientKind::OpenCode);
        assert_eq!(parse_client("claude").unwrap(), ClientKind::Claude);
        assert_eq!(parse_client("codex").unwrap(), ClientKind::Codex);
        // Case and surrounding whitespace should not matter.
        assert_eq!(parse_client("  Codex ").unwrap(), ClientKind::Codex);
        assert_eq!(parse_client("claude-code").unwrap(), ClientKind::Claude);
    }

    /// An unknown client must name the real options rather than just
    /// failing — this is a flag people mistype.
    #[test]
    fn an_unknown_client_lists_the_supported_ones() {
        let error = parse_client("cursor").expect_err("must reject").to_string();
        assert!(error.contains("opencode"), "{error}");
        assert!(error.contains("claude"), "{error}");
        assert!(error.contains("codex"), "{error}");
    }

    /// The config must carry the *actual* bound address. Under port
    /// fallback the default and the real port differ, and writing the
    /// default would point the client at whatever else holds it.
    #[test]
    fn the_ephemeral_config_points_at_the_real_bound_address() {
        let config = build_ephemeral_config(
            ClientKind::Codex,
            "http://127.0.0.1:11435/v1",
            "/usr/local/bin/tqf",
            &["--mcp-stdio"],
        );
        let rendered: String = config
            .files
            .iter()
            .map(|(_, contents)| contents.as_str())
            .collect();
        assert!(rendered.contains("11435"), "{rendered}");
        assert!(!rendered.contains("11434"), "{rendered}");
    }

    /// `--open` writes an MCP command naming this binary's hidden stdio
    /// entrypoint, so that flag existing is load-bearing rather than
    /// decorative.
    #[test]
    fn the_ephemeral_config_launches_this_binarys_mcp_entrypoint() {
        for kind in [ClientKind::OpenCode, ClientKind::Claude, ClientKind::Codex] {
            let config = build_ephemeral_config(
                kind,
                "http://127.0.0.1:11434/v1",
                "/bin/tqf",
                &["--mcp-stdio"],
            );
            let rendered: String = config
                .files
                .iter()
                .map(|(_, contents)| contents.as_str())
                .collect();
            assert!(
                rendered.contains("--mcp-stdio"),
                "{kind:?} config must launch the MCP entrypoint: {rendered}"
            );
        }
    }
}
