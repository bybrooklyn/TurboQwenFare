//! Real process launch/cleanup mechanics for `tqf --open <client>`
//! (spec §99: "ensure the server is running, synchronize the
//! associated index if one is registered, construct ephemeral
//! provider/MCP configuration, launch the client as a child, and
//! remove the temporary environment/config when it exits"; spec §100:
//! "If a requested client binary is absent, ask permission... it must
//! not silently install external software").

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use super::config::{ClientKind, EphemeralConfig};

#[derive(Debug)]
pub enum LaunchError {
    ClientNotFound { kind: ClientKind, declined: bool },
    Io(std::io::Error),
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::ClientNotFound {
                kind,
                declined: true,
            } => {
                write!(
                    f,
                    "{} not found and installation was declined",
                    kind.binary_name()
                )
            }
            LaunchError::ClientNotFound {
                kind,
                declined: false,
            } => {
                write!(f, "{} not found", kind.binary_name())
            }
            LaunchError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LaunchError {}

/// Real, read-only `PATH` search — never executes anything. Generic
/// over the binary name so tests can search for a name that
/// deliberately doesn't exist rather than depending on whether the
/// real client CLIs happen to be installed on the machine running the
/// tests.
pub fn find_binary_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(metadata) = candidate.metadata() {
                    if metadata.permissions().mode() & 0o111 != 0 {
                        return Some(candidate);
                    }
                    continue;
                }
            }
            #[cfg(not(unix))]
            return Some(candidate);
        }
    }
    None
}

/// spec §100's confirmation gate. `confirm` is given the client's real
/// install recipe and decides whether to run it — this function never
/// runs the recipe itself; that stays the caller's explicit choice, one
/// more layer away from "silently install external software."
pub fn ensure_client_available(
    kind: ClientKind,
    confirm: impl FnOnce(&str) -> bool,
) -> Result<PathBuf, LaunchError> {
    if let Some(path) = find_binary_on_path(kind.binary_name()) {
        return Ok(path);
    }
    if confirm(kind.install_recipe()) {
        // Confirmed: the caller (a real interactive CLI session) is
        // responsible for actually running the recipe and re-checking
        // PATH — this function's contract is "find or ask," not
        // "find, ask, and mutate the system," keeping the actual
        // `Command::new(installer).spawn()` an explicit, visible step
        // at the call site rather than hidden in this confirmation gate.
        Err(LaunchError::ClientNotFound {
            kind,
            declined: false,
        })
    } else {
        Err(LaunchError::ClientNotFound {
            kind,
            declined: true,
        })
    }
}

pub struct WrittenConfig {
    pub dir: PathBuf,
}

impl Drop for WrittenConfig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Writes every file in `config` under a fresh directory inside
/// `parent_temp_dir` and returns a handle that deletes the whole
/// directory when dropped — "remove the temporary environment/config
/// when it exits" (spec §99), tied to Rust's own scope rules rather
/// than a separate cleanup step a caller could forget.
pub fn write_ephemeral_config(
    config: &EphemeralConfig,
    parent_temp_dir: &Path,
) -> std::io::Result<WrittenConfig> {
    // `cargo test` runs tests in parallel within one process, so a
    // directory name keyed only by PID collides across concurrent
    // calls (a real, found-and-fixed race — see Phase 42's identical
    // class of bug with real-repo scratch directories). A monotonic
    // counter makes every call's directory unique regardless of
    // concurrency.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = parent_temp_dir.join(format!("tqf-open-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    for (relative_path, contents) in &config.files {
        let full_path = dir.join(relative_path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(full_path, contents)?;
    }
    Ok(WrittenConfig { dir })
}

/// Builds the real `Command` for launching `binary` with `config`'s
/// env vars (resolved relative to the written config directory) and
/// extra args. Split out from the actual `spawn()` call so tests can
/// inspect/run it against a harmless real binary instead of the real
/// AI client CLIs.
pub fn build_launch_command(
    binary: &Path,
    config: &EphemeralConfig,
    written: &WrittenConfig,
) -> Command {
    let mut command = Command::new(binary);
    for (key, value) in &config.env_vars {
        // Three cases. A written config *file* name resolves to its
        // real path under the config directory
        // (`OPENCODE_CONFIG=opencode.json`); the config *directory*
        // placeholder resolves to the directory itself
        // (`CODEX_HOME`); everything else passes through unchanged
        // (`ANTHROPIC_BASE_URL=http://...`).
        //
        // All three produce absolute values, because the child is not
        // started in the config directory — see below.
        let resolved = if value == super::config::CONFIG_DIR {
            written.dir.to_string_lossy().to_string()
        } else if PathBuf::from(value).is_relative()
            && config.files.iter().any(|(p, _)| p == Path::new(value))
        {
            written.dir.join(value).to_string_lossy().to_string()
        } else {
            value.clone()
        };
        command.env(key, resolved);
    }
    // Deliberately *not* `current_dir(&written.dir)`. A coding client
    // works on the directory it was started in, and `tqf --open claude`
    // is run from inside a project — pointing it at the ephemeral config
    // directory hands it an empty temp folder as its workspace. The
    // config directory is where the config file lives, not where work
    // happens, and the relative-value resolution above already accounts
    // for it without moving the child.
    //
    // It also decides which index the MCP server serves: it selects the
    // registered root containing its working directory, so a child
    // rooted in /tmp resolves to no index at all.
    command.args(&config.extra_args);
    command
}

/// Runs `command` to completion. The ephemeral config directory is
/// deleted (via `written`'s `Drop`) once this returns, regardless of
/// whether the child exited successfully — spec §99's "remove the
/// temporary environment/config when it exits" applies on every exit
/// path, not just the happy one.
pub fn run_to_completion(
    mut command: Command,
    written: WrittenConfig,
) -> Result<ExitStatus, LaunchError> {
    let status = command.status().map_err(LaunchError::Io)?;
    drop(written);
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::super::config::build_ephemeral_config;
    use super::*;

    #[test]
    fn find_binary_on_path_returns_none_for_a_name_that_does_not_exist() {
        assert!(find_binary_on_path("definitely-not-a-real-tqf-test-binary-xyz").is_none());
    }

    #[test]
    fn find_binary_on_path_finds_a_real_common_binary() {
        // `sh` exists on every real macOS/Linux machine this crate
        // targets (spec's own base M4/RTX reference platforms).
        assert!(find_binary_on_path("sh").is_some());
    }

    #[test]
    fn ensure_client_available_respects_a_declined_confirmation() {
        // Force the "not found" path deterministically by asking about
        // a fabricated client kind's binary name indirectly: OpenCode's
        // real binary is unlikely to be installed in a CI/test sandbox,
        // but to stay independent of that, exercise the confirm-declined
        // path directly against `ensure_client_available`'s contract.
        let result = ensure_client_available(ClientKind::OpenCode, |_recipe| false);
        // Either OpenCode happens to be installed on this machine (Ok),
        // or it's absent and declining must produce `declined: true`.
        if let Err(LaunchError::ClientNotFound { declined, .. }) = result {
            assert!(declined);
        }
    }

    #[test]
    fn ensure_client_available_shows_the_real_install_recipe_when_confirming() {
        let mut seen_recipe = String::new();
        let result = ensure_client_available(ClientKind::Codex, |recipe| {
            seen_recipe = recipe.to_string();
            true
        });
        if matches!(
            result,
            Err(LaunchError::ClientNotFound {
                declined: false,
                ..
            })
        ) {
            assert_eq!(seen_recipe, "npm install -g @openai/codex");
        }
    }

    /// Real end-to-end test of the write -> spawn -> wait -> cleanup
    /// mechanics using a real but harmless process (`sh`) standing in
    /// for a real AI client CLI. Confirms: the ephemeral config file
    /// actually exists on disk while the child runs, the child actually
    /// observes the real env vars this crate set, and the config
    /// directory is actually gone once the child exits — proving spec
    /// §99's full lifecycle, not just asserting it in a comment.
    #[test]
    fn real_spawn_observes_env_vars_and_cleans_up_on_exit() {
        let sh = find_binary_on_path("sh").expect("sh must exist on this platform");
        let config = build_ephemeral_config(
            ClientKind::OpenCode,
            "http://127.0.0.1:11535",
            "tqf",
            &["mcp", "stdio"],
        );
        let temp_root = std::env::temp_dir();
        let written = write_ephemeral_config(&config, &temp_root).unwrap();
        let config_dir = written.dir.clone();
        assert!(
            config_dir.join("opencode.json").exists(),
            "config file must exist before the child runs"
        );

        let mut command = build_launch_command(&sh, &config, &written);
        // Real child process: print the env var TQF set, confirm the
        // config is readable through it, and report the child's own
        // working directory.
        //
        // It reads the config via `$OPENCODE_CONFIG`, not via a bare
        // relative `opencode.json`. The relative form used to work only
        // because the child was being relocated into the config
        // directory, which is the bug this test now guards against: a
        // coding client must start in the user's project.
        command.arg("-c");
        command.arg(
            "echo \"OPENCODE_CONFIG=$OPENCODE_CONFIG\"; \
             cat \"$OPENCODE_CONFIG\" > /dev/null && echo CONFIG_READABLE; \
             echo \"CHILD_CWD=$PWD\"",
        );
        command.stdout(std::process::Stdio::piped());
        let output = command.output().expect("spawn sh");
        let stdout = String::from_utf8_lossy(&output.stdout);
        // build_launch_command resolves the relative config filename to
        // an absolute path under the written config directory (not the
        // literal "opencode.json" from EphemeralConfig) — assert the
        // real resolved value the child actually observed.
        let expected_var = format!(
            "OPENCODE_CONFIG={}",
            config_dir.join("opencode.json").display()
        );
        assert!(stdout.contains(&expected_var), "{stdout}");
        assert!(stdout.contains("CONFIG_READABLE"), "{stdout}");

        // The child inherits this process's working directory. `tqf
        // --open claude` is run from inside a project, and a client
        // started in the ephemeral config directory would see an empty
        // temp folder as its workspace — and, since the MCP server picks
        // its index by working directory, no index either.
        let parent_cwd = std::env::current_dir().unwrap();
        let child_cwd = stdout
            .lines()
            .find_map(|line| line.strip_prefix("CHILD_CWD="))
            .unwrap_or_else(|| panic!("{stdout}"));
        assert_ne!(
            std::path::Path::new(child_cwd),
            config_dir,
            "the client must not be relocated into the config directory: {stdout}"
        );
        assert_eq!(
            std::fs::canonicalize(child_cwd).unwrap(),
            std::fs::canonicalize(&parent_cwd).unwrap(),
            "{stdout}"
        );

        drop(written);
        assert!(
            !config_dir.exists(),
            "ephemeral config directory must be removed after the child exits"
        );
    }

    #[test]
    fn run_to_completion_returns_the_real_exit_status_and_cleans_up() {
        let sh = find_binary_on_path("sh").expect("sh must exist on this platform");
        let config =
            build_ephemeral_config(ClientKind::Codex, "http://127.0.0.1:11535", "tqf", &["mcp"]);
        let written = write_ephemeral_config(&config, &std::env::temp_dir()).unwrap();
        let config_dir = written.dir.clone();
        let mut command = build_launch_command(&sh, &config, &written);
        command.arg("-c").arg("exit 7");
        let status = run_to_completion(command, written).unwrap();
        assert_eq!(status.code(), Some(7));
        assert!(!config_dir.exists());
    }
}
