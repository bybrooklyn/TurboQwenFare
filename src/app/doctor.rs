//! `tqf doctor` (spec §3): environment and installation diagnostics.
//!
//! Every check reports one of three verdicts and, when it is not a pass,
//! what to do about it. A check that cannot determine an answer says so
//! rather than passing by default — a diagnostic that reports "all good"
//! because it failed to look is worse than no diagnostic.

use std::fmt::Write as _;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use crate::config::paths;
use crate::error::Result;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;
use crate::server::bind::{self, Occupant, DEFAULT_PORT};
use crate::setup::hardware;
use crate::setup::receipt;
use crate::source::pinned;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Warn,
    Fail,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "ok",
            Self::Warn => "warn",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug)]
pub struct Check {
    pub name: &'static str,
    pub verdict: Verdict,
    pub detail: String,
    /// What the user can do. Empty for a pass.
    pub remedy: String,
}

impl Check {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            verdict: Verdict::Pass,
            detail: detail.into(),
            remedy: String::new(),
        }
    }

    fn warn(name: &'static str, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            name,
            verdict: Verdict::Warn,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }

    fn fail(name: &'static str, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            name,
            verdict: Verdict::Fail,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

pub fn run(config: &crate::config::Config) -> Result<std::process::ExitCode> {
    let checks = collect(config);
    print!("{}", render(&checks));
    // A nonzero exit is what makes `tqf doctor` usable in a script.
    Ok(if checks.iter().any(|c| c.verdict == Verdict::Fail) {
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    })
}

fn collect(config: &crate::config::Config) -> Vec<Check> {
    let mut checks = vec![check_hardware(), check_backend(), check_data_root()];
    checks.push(check_disk_space());
    let (receipt_check, receipt) = check_receipt();
    checks.push(receipt_check);
    checks.push(check_container(receipt.as_ref()));
    checks.push(check_tokenizer(receipt.as_ref()));
    checks.push(check_port(config));
    checks.push(check_memory_plan(config));
    checks
}

fn check_hardware() -> Check {
    let profile = hardware::detect();
    Check::pass(
        "hardware",
        format!(
            "{} {}, {} cores",
            profile.os, profile.arch, profile.cpu_cores
        ),
    )
}

fn check_backend() -> Check {
    let profile = hardware::detect();
    match profile.backend.as_str() {
        "metal" | "cuda" => Check::pass("backend", format!("{} compiled in", profile.backend)),
        other => Check::warn(
            "backend",
            format!("{other}: no accelerated backend is compiled into this binary"),
            "Rebuild on macOS for Metal. CUDA (spec phases 50-51) is not implemented; \
             the reference path is correct but far below the performance floor.",
        ),
    }
}

fn check_data_root() -> Check {
    match paths::ensure_layout() {
        Ok(root) => {
            // Existing is not the same as writable, and the failure only
            // shows up much later during install if it isn't checked here.
            let probe = root.join(".tqf-doctor-write-probe");
            match std::fs::write(&probe, b"") {
                Ok(()) => {
                    let _ = std::fs::remove_file(&probe);
                    Check::pass("data root", root.display().to_string())
                }
                Err(error) => Check::fail(
                    "data root",
                    format!("{} is not writable: {error}", root.display()),
                    "Fix the directory's permissions, or set TQF_HOME to a writable path.",
                ),
            }
        }
        Err(error) => Check::fail(
            "data root",
            error.to_string(),
            "Set $HOME, or point TQF_HOME at a writable directory.",
        ),
    }
}

/// Free space against the real pinned checkpoint size plus conversion
/// headroom — the container is written alongside the source before the
/// source can be released.
fn check_disk_space() -> Check {
    let Ok(models) = paths::models_dir() else {
        return Check::warn(
            "disk space",
            "could not resolve the models directory",
            "Check TQF_HOME and $HOME.",
        );
    };
    let _ = std::fs::create_dir_all(&models);

    let Some(available) = available_bytes(&models) else {
        return Check::warn(
            "disk space",
            "could not read filesystem capacity on this platform",
            "Check free space manually before installing.",
        );
    };

    // Source GGUF plus the converted container, both resident during
    // conversion.
    let needed = pinned::LANGUAGE_CHECKPOINT_SIZE_BYTES * 2;
    let detail = format!(
        "{:.1} GiB free, {:.1} GiB needed to install",
        available as f64 / 1e9,
        needed as f64 / 1e9
    );
    if available >= needed {
        Check::pass("disk space", detail)
    } else {
        Check::warn(
            "disk space",
            detail,
            "Free space before running setup: an interrupted conversion leaves a resumable \
             partial install, but it will not finish.",
        )
    }
}

// The casts below are redundant on glibc, where both fields are already
// `u64`, but not on macOS, where `f_bavail` is `c_uint` and `f_frsize` is
// `c_ulong`. Removing them to satisfy the lint on Linux would break the
// macOS build, which is this project's primary reference platform.
#[allow(clippy::unnecessary_cast)]
#[cfg(unix)]
fn available_bytes(path: &std::path::Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a
    // correctly sized, zeroed output buffer.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn available_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

fn check_receipt() -> (Check, Option<receipt::ModelReceipt>) {
    let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));
    let Ok(dir) = paths::receipts_dir() else {
        return (
            Check::fail(
                "model receipt",
                "could not resolve the receipts directory",
                "Check TQF_HOME and $HOME.",
            ),
            None,
        );
    };
    match receipt::load_trusted_receipt(&dir, &broker) {
        Some(receipt) => {
            let detail = format!("{} ({})", receipt.model_family, receipt.tqf_path.display());
            (Check::pass("model receipt", detail), Some(receipt))
        }
        None => (
            Check::warn(
                "model receipt",
                "no valid receipt: no model is installed, or its receipt failed validation",
                "Run `tqf` to install the pinned checkpoint, or `tqf --model <path.gguf>` \
                 to import one you already have.",
            ),
            None,
        ),
    }
}

/// Opens the container and validates its superblock and topology — what
/// `finish_install` does after conversion. Deliberately not a full
/// content rehash: reading 20 GiB at every `doctor` run would make the
/// command unusable.
fn check_container(receipt: Option<&receipt::ModelReceipt>) -> Check {
    let Some(receipt) = receipt else {
        return Check::warn(
            "model container",
            "skipped: no installed model to check",
            "Install a model first.",
        );
    };
    let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
    match crate::model::qwen36::weights::Qwen36WeightManifest::open_with_broker(
        &receipt.tqf_path,
        &broker,
    ) {
        Ok(_) => Check::pass(
            "model container",
            "superblock and tensor topology validate (payload checksums verify lazily on load)",
        ),
        Err(error) => Check::fail(
            "model container",
            format!("{} failed to open: {error}", receipt.tqf_path.display()),
            "Delete the receipt and reinstall: the container is unusable, and a stale \
             receipt pointing at it will keep the server from starting.",
        ),
    }
}

fn check_tokenizer(receipt: Option<&receipt::ModelReceipt>) -> Check {
    let Some(receipt) = receipt else {
        return Check::warn(
            "tokenizer",
            "skipped: no installed model to check",
            "Install a model first.",
        );
    };
    if !receipt.tokenizer_gguf_path.exists() {
        return Check::fail(
            "tokenizer",
            format!("{} is missing", receipt.tokenizer_gguf_path.display()),
            "The verified GGUF is the authoritative vocab/merge source and cannot be \
             deleted after conversion. Reinstall.",
        );
    }
    let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
    match crate::format::gguf::open_with_broker(&receipt.tokenizer_gguf_path, &broker)
        .and_then(|gguf| crate::tokenizer::TqfTokenizer::from_gguf(&gguf))
    {
        Ok(_) => Check::pass("tokenizer", "builds from the verified GGUF metadata"),
        Err(error) => Check::fail(
            "tokenizer",
            format!("failed to build: {error}"),
            "Reinstall: the GGUF's vocab/merge metadata is unusable.",
        ),
    }
}

/// Whether the port a client would use is free — and if not, whether the
/// occupant is another tqf or a real Ollama, which is the single most
/// likely collision on 11434.
fn check_port(config: &crate::config::Config) -> Check {
    let host: IpAddr = config
        .host
        .as_deref()
        .and_then(|h| h.parse().ok())
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let port = config.port.unwrap_or(DEFAULT_PORT);
    let addr = SocketAddr::new(host, port);

    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return Check::warn("port", "could not start a probe runtime", "");
    };

    runtime.block_on(async {
        if tokio::net::TcpListener::bind(addr).await.is_ok() {
            return Check::pass("port", format!("{addr} is available"));
        }
        match bind::identify_occupant(addr).await {
            Occupant::Tqf => Check::pass("port", format!("{addr} is already serving tqf")),
            Occupant::Ollama => Check::warn(
                "port",
                format!("{addr} is held by a real Ollama server"),
                "Stop Ollama, or start tqf with `--port 11435` and point clients there. \
                 Clients using the default URL will otherwise reach Ollama, not tqf.",
            ),
            Occupant::Unknown => Check::warn(
                "port",
                format!("{addr} is held by an unidentified process"),
                "Free the port, or use `--port` to choose another.",
            ),
        }
    })
}

/// Spec §76 asks that an impossible `--memory`/`--context` combination be
/// caught before startup. There is no estimator that maps a context
/// length to a reservation without constructing it, so this reports what
/// it can and is explicit that it is not the full validation.
fn check_memory_plan(config: &crate::config::Config) -> Check {
    // One definition of the floor, shared with the flag that rejects it,
    // so `doctor` cannot disagree with `--memory` about what is allowed.
    const MINIMUM_BUDGET: u64 = crate::config::MINIMUM_MEMORY_BUDGET_BYTES;
    let budget = config.memory_budget_bytes.unwrap_or(4 * 1024 * 1024 * 1024);
    let context = config.context_limit_tokens.unwrap_or(128 * 1024);

    if budget < MINIMUM_BUDGET {
        return Check::fail(
            "memory plan",
            format!(
                "--memory {:.2} GiB is below the 2 GiB experimental floor",
                budget as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
            "Raise --memory to at least 2G (4G is the supported default).",
        );
    }
    Check::warn(
        "memory plan",
        format!(
            "{:.0} GiB budget with a {context}-token context — combination validation is not \
             implemented",
            budget as f64 / (1024.0 * 1024.0 * 1024.0)
        ),
        "Startup reserves through the broker and will fail loudly rather than overcommit, \
         but it cannot tell you in advance that a combination will not fit.",
    )
}

pub fn render(checks: &[Check]) -> String {
    let mut out = String::new();
    for check in checks {
        let _ = writeln!(
            out,
            "{:>4}  {:<16} {}",
            check.verdict.label(),
            check.name,
            check.detail
        );
        if !check.remedy.is_empty() {
            for line in wrap(&check.remedy, 68) {
                let _ = writeln!(out, "      {:<16} {line}", "");
            }
        }
    }

    let failures = checks.iter().filter(|c| c.verdict == Verdict::Fail).count();
    let warnings = checks.iter().filter(|c| c.verdict == Verdict::Warn).count();
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{} checks, {failures} failed, {warnings} warned",
        checks.len()
    );
    out
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failing_check_renders_its_remedy() {
        let checks = vec![Check::fail("data root", "not writable", "Fix permissions.")];
        let rendered = render(&checks);
        assert!(rendered.contains("FAIL"), "{rendered}");
        assert!(rendered.contains("not writable"), "{rendered}");
        assert!(rendered.contains("Fix permissions."), "{rendered}");
        assert!(rendered.contains("1 failed"), "{rendered}");
    }

    #[test]
    fn a_passing_check_has_nothing_to_advise() {
        let rendered = render(&[Check::pass("hardware", "linux x86_64, 8 cores")]);
        assert!(rendered.contains("ok"), "{rendered}");
        assert!(rendered.contains("0 failed, 0 warned"), "{rendered}");
    }

    /// The whole point of the exit code: `tqf doctor` has to be usable in
    /// a script, and a warning is not a failure.
    #[test]
    fn only_failures_are_counted_as_failures() {
        let checks = vec![
            Check::pass("a", "fine"),
            Check::warn("b", "questionable", "consider this"),
            Check::fail("c", "broken", "fix this"),
        ];
        assert_eq!(
            checks.iter().filter(|c| c.verdict == Verdict::Fail).count(),
            1
        );
        let rendered = render(&checks);
        assert!(
            rendered.contains("3 checks, 1 failed, 1 warned"),
            "{rendered}"
        );
    }

    /// A doctor that reports "all good" because it could not look is
    /// worse than no doctor.
    #[test]
    fn an_undeterminable_check_warns_rather_than_passing() {
        let check = check_memory_plan(&crate::config::Config::default());
        assert_eq!(check.verdict, Verdict::Warn);
        assert!(check.detail.contains("not implemented"), "{check:?}");
    }

    #[test]
    fn a_budget_below_the_documented_floor_fails() {
        let config = crate::config::Config {
            memory_budget_bytes: Some(512 * 1024 * 1024),
            ..crate::config::Config::default()
        };
        assert_eq!(check_memory_plan(&config).verdict, Verdict::Fail);
    }

    #[test]
    fn remedies_wrap_instead_of_running_off_the_terminal() {
        let long = "word ".repeat(40);
        for line in wrap(&long, 68) {
            assert!(line.len() <= 68, "line too long: {line:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn free_space_is_a_real_reading_from_the_filesystem() {
        let available =
            available_bytes(std::path::Path::new(".")).expect("statvfs must work on this platform");
        assert!(available > 0, "implausible free space: {available}");
    }
}
