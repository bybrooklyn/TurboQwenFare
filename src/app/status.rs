//! `tqf status` (spec §3): what is installed, what is running, and what
//! this machine is configured to do.
//!
//! Deliberately works whether or not a server is up. If one is listening
//! it reports live numbers from `/health` and `/v1/tqf/metrics`; if not,
//! it reports installation and configuration state from disk rather than
//! erroring, because "is anything installed" is exactly what a user asks
//! when the server *isn't* running.

use std::fmt::Write as _;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use crate::config::paths;
use crate::config::persisted::PersistedConfig;
use crate::error::Result;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;
use crate::server::bind::{self, DEFAULT_PORT, FALLBACK_PORT};
use crate::setup::hardware::HardwareProfile;
use crate::setup::receipt::{self, ModelReceipt};

/// Everything `tqf status` reports, gathered before any of it is
/// rendered. Keeping collection and rendering apart is what lets the
/// renderer be tested without a server or a filesystem.
#[derive(Debug, Default)]
pub struct StatusSnapshot {
    pub server: Option<RunningServer>,
    pub receipt: Option<ModelReceipt>,
    pub receipt_error: Option<String>,
    pub config: PersistedConfig,
    pub hardware: Option<HardwareProfile>,
    pub home: Option<String>,
}

#[derive(Debug)]
pub struct RunningServer {
    pub addr: SocketAddr,
    pub version: String,
    pub uptime_seconds: u64,
    pub model_installed: bool,
    pub resident_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
}

pub fn run(config: &crate::config::Config) -> Result<()> {
    let snapshot = collect(config)?;
    print!("{}", render(&snapshot));
    Ok(())
}

fn collect(config: &crate::config::Config) -> Result<StatusSnapshot> {
    let mut snapshot = StatusSnapshot {
        home: paths::home_dir().ok().map(|p| p.display().to_string()),
        ..StatusSnapshot::default()
    };

    if let Ok(path) = paths::config_path() {
        snapshot.config = PersistedConfig::load(&path).unwrap_or_default();
    }
    if let Ok(path) = paths::profile_path() {
        snapshot.hardware = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| toml::from_str(&text).ok());
    }

    // A small broker just for receipt validation's own bounded reads; it
    // is not the server's budget and never sees a model allocation.
    let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));
    match paths::receipts_dir() {
        Ok(dir) => match receipt::load_trusted_receipt(&dir, &broker) {
            Some(receipt) => snapshot.receipt = Some(receipt),
            None => {
                snapshot.receipt_error =
                    Some("no valid model receipt found in the receipts directory".to_string())
            }
        },
        Err(error) => snapshot.receipt_error = Some(error.to_string()),
    }

    snapshot.server = probe_server(config);
    Ok(snapshot)
}

/// Tries the configured port first, then the default and its documented
/// fallback — the same three places a client would look.
fn probe_server(config: &crate::config::Config) -> Option<RunningServer> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    let host: IpAddr = config
        .host
        .as_deref()
        .and_then(|h| h.parse().ok())
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));

    let mut ports = Vec::new();
    if let Some(port) = config.port {
        ports.push(port);
    }
    ports.extend([DEFAULT_PORT, FALLBACK_PORT]);

    runtime.block_on(async {
        for port in ports {
            let addr = SocketAddr::new(host, port);
            let Some(health) = bind::http_get(addr, "/health").await else {
                continue;
            };
            let Some(health) = json_body(&health) else {
                continue;
            };
            if health.get("status").and_then(|v| v.as_str()) != Some("ok") {
                continue;
            }

            let metrics = bind::http_get(addr, "/v1/tqf/metrics")
                .await
                .and_then(|text| json_body(&text));

            return Some(RunningServer {
                addr,
                version: health
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                uptime_seconds: health
                    .get("uptime_seconds")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                model_installed: health
                    .get("model_installed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                resident_bytes: metrics
                    .as_ref()
                    .and_then(|m| m.get("resident_bytes")?.as_u64()),
                peak_bytes: metrics
                    .as_ref()
                    .and_then(|m| m.get("resident_peak_bytes")?.as_u64()),
            });
        }
        None
    })
}

/// `bind::http_get` returns the whole response including headers, since
/// its other caller only substring-matches.
fn json_body(response: &str) -> Option<serde_json::Value> {
    let body = response.split("\r\n\r\n").nth(1)?;
    serde_json::from_str(body.trim()).ok()
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn mib(bytes: u64) -> String {
    format!("{:.0} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn duration(seconds: u64) -> String {
    match seconds {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m {}s", s / 60, s % 60),
        s => format!("{}h {}m", s / 3600, (s % 3600) / 60),
    }
}

pub fn render(snapshot: &StatusSnapshot) -> String {
    let mut out = String::new();

    match &snapshot.server {
        Some(server) => {
            let _ = writeln!(out, "server:    running on http://{}", server.addr);
            let _ = writeln!(
                out,
                "           tqf {}, up {}",
                server.version,
                duration(server.uptime_seconds)
            );
            if let Some(resident) = server.resident_bytes {
                let peak = server
                    .peak_bytes
                    .map(|p| format!(", peak {}", mib(p)))
                    .unwrap_or_default();
                let _ = writeln!(out, "           resident {}{peak}", mib(resident));
            }
        }
        None => {
            let _ = writeln!(out, "server:    not running");
        }
    }

    match &snapshot.receipt {
        Some(receipt) => {
            let _ = writeln!(out, "model:     {} (installed)", receipt.model_family);
            let _ = writeln!(out, "           {}", receipt.tqf_path.display());
            if let Ok(meta) = std::fs::metadata(&receipt.tqf_path) {
                let _ = writeln!(out, "           {}", gib(meta.len()));
            }
            if let Some(revision) = &receipt.source_revision {
                let _ = writeln!(out, "           source revision {revision}");
            }
        }
        None => {
            let _ = writeln!(out, "model:     not installed");
            if let Some(error) = &snapshot.receipt_error {
                let _ = writeln!(out, "           {error}");
            }
            let _ = writeln!(out, "           run `tqf` to install the pinned checkpoint");
        }
    }

    let memory = snapshot
        .config
        .memory_budget_bytes
        .map(gib)
        .unwrap_or_else(|| "4.00 GiB (default)".to_string());
    let context = snapshot
        .config
        .context_limit_tokens
        .map(|tokens| format!("{} tokens", tokens))
        .unwrap_or_else(|| "131072 tokens (default)".to_string());
    let _ = writeln!(out, "memory:    {memory}");
    let _ = writeln!(out, "context:   {context}");
    let _ = writeln!(
        out,
        "bind:      {}:{}",
        snapshot.config.host.as_deref().unwrap_or("127.0.0.1"),
        snapshot.config.port.unwrap_or(DEFAULT_PORT)
    );
    let _ = writeln!(
        out,
        "vision:    {}",
        if snapshot.config.enable_vision {
            "enabled (encoder is not wired into the request path yet)"
        } else {
            "disabled"
        }
    );

    if let Some(hardware) = &snapshot.hardware {
        let _ = writeln!(
            out,
            "hardware:  {} {}, {} cores, backend {}",
            hardware.os, hardware.arch, hardware.cpu_cores, hardware.backend
        );
        if let Some(tune) = &hardware.quick_tune {
            let _ = writeln!(
                out,
                "           {} — {:.1} GB/s copy, {:.2} GFLOP/s GEMV",
                tune.device_name, tune.bandwidth_gigabytes_per_second, tune.naive_gemv_gflops
            );
        }
    }

    if let Some(home) = &snapshot.home {
        let _ = writeln!(out, "data:      {home}");
    }

    // Read from the registry rather than from a running server: `tqf
    // status` has to answer when nothing is listening, which is the case
    // this whole function is written for.
    let resolved = crate::retrieval::tqi::registry::resolve();
    if resolved.live.is_empty() && resolved.stale.is_empty() {
        let _ = writeln!(out, "indexes:   none registered (run `tqf sync <path>`)");
    } else {
        // The label goes on whichever row prints first, live or stale —
        // a registry holding only stale roots is still a registry.
        let mut first = true;
        let mut label = || {
            let label = if first { "indexes:  " } else { "          " };
            first = false;
            label
        };
        for root in &resolved.live {
            // The generation comes from the root's own project file, so a
            // root whose index was deleted by hand still reports as
            // registered — which is the truth, and what `tqf unsync`
            // fixes.
            match crate::retrieval::tqi::registry::read_project_file(root) {
                Some(project) => {
                    let _ = writeln!(
                        out,
                        "{} {} (generation {})",
                        label(),
                        root.display(),
                        project.generation
                    );
                }
                None => {
                    let _ = writeln!(out, "{} {} (no project file)", label(), root.display());
                }
            }
        }
        for root in &resolved.stale {
            let _ = writeln!(
                out,
                "{} {} (registered but gone; `tqf unsync` to forget it)",
                label(),
                root.display()
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with_server() -> StatusSnapshot {
        StatusSnapshot {
            server: Some(RunningServer {
                addr: "127.0.0.1:11434".parse().unwrap(),
                version: "0.0.1".to_string(),
                uptime_seconds: 3725,
                model_installed: true,
                resident_bytes: Some(1_800_000_000),
                peak_bytes: Some(1_900_000_000),
            }),
            ..StatusSnapshot::default()
        }
    }

    /// `tqf status` reads the registry directly, because it has to
    /// answer when no server is running — the case the whole command
    /// exists for.
    ///
    /// This line used to read "index persistence is not implemented".
    /// It stayed convincing for a while after it stopped being true,
    /// which is the failure this test exists to prevent; both branches
    /// are pinned so neither can quietly go stale again.
    #[test]
    fn the_index_line_reports_the_real_registry() {
        let home = std::env::temp_dir().join(format!("tqf-status-registry-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        // SAFETY: same contract as `config::paths`'s own test — the suite
        // runs single-threaded (`just test` passes --test-threads=1)
        // precisely because TQF_HOME is process-global.
        unsafe {
            std::env::set_var("TQF_HOME", &home);
        }

        let empty = render(&StatusSnapshot::default());
        assert!(
            empty.contains("none registered (run `tqf sync <path>`)"),
            "{empty}"
        );

        std::fs::write(
            home.join("roots.toml"),
            "roots = [\"/nonexistent/definitely-gone\"]\n",
        )
        .unwrap();
        let stale = render(&StatusSnapshot::default());
        assert!(
            stale.contains("indexes:   /nonexistent/definitely-gone"),
            "a registry holding only stale roots still gets the label: {stale}"
        );
        assert!(stale.contains("`tqf unsync`"), "{stale}");

        unsafe {
            std::env::remove_var("TQF_HOME");
        }
        std::fs::remove_dir_all(&home).ok();
    }

    /// The case that matters most: a user runs `tqf status` precisely
    /// because nothing is working, so it must report state rather than
    /// fail.
    #[test]
    fn reports_a_stopped_server_and_missing_model_without_erroring() {
        let rendered = render(&StatusSnapshot::default());
        assert!(rendered.contains("server:    not running"), "{rendered}");
        assert!(rendered.contains("model:     not installed"), "{rendered}");
        assert!(
            rendered.contains("run `tqf`"),
            "must say what to do next: {rendered}"
        );
    }

    #[test]
    fn reports_a_running_server_with_its_real_address_and_memory() {
        let rendered = render(&snapshot_with_server());
        assert!(rendered.contains("http://127.0.0.1:11434"), "{rendered}");
        assert!(rendered.contains("up 1h 2m"), "{rendered}");
        assert!(rendered.contains("1717 MiB"), "{rendered}");
    }

    #[test]
    fn falls_back_to_documented_defaults_when_nothing_was_configured() {
        let rendered = render(&StatusSnapshot::default());
        assert!(rendered.contains("4.00 GiB (default)"), "{rendered}");
        assert!(rendered.contains("131072 tokens (default)"), "{rendered}");
        assert!(rendered.contains("127.0.0.1:11434"), "{rendered}");
    }

    #[test]
    fn configured_values_win_over_defaults() {
        let snapshot = StatusSnapshot {
            config: PersistedConfig {
                memory_budget_bytes: Some(8 * 1024 * 1024 * 1024),
                context_limit_tokens: Some(1_048_576),
                port: Some(11435),
                enable_vision: true,
                ..PersistedConfig::default()
            },
            ..StatusSnapshot::default()
        };
        let rendered = render(&snapshot);
        assert!(rendered.contains("8.00 GiB"), "{rendered}");
        assert!(rendered.contains("1048576 tokens"), "{rendered}");
        assert!(rendered.contains("127.0.0.1:11435"), "{rendered}");
        assert!(rendered.contains("enabled"), "{rendered}");
    }

    #[test]
    fn durations_read_naturally_at_every_scale() {
        assert_eq!(duration(45), "45s");
        assert_eq!(duration(90), "1m 30s");
        assert_eq!(duration(3725), "1h 2m");
    }

    /// `bind::http_get` hands back the whole response, headers included.
    #[test]
    fn a_json_body_is_extracted_from_a_full_http_response() {
        let response =
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"status\":\"ok\"}";
        assert_eq!(
            json_body(response).and_then(|v| v["status"].as_str().map(str::to_string)),
            Some("ok".to_string())
        );
        assert!(json_body("HTTP/1.1 500 Internal\r\n\r\nnot json").is_none());
    }
}
