//! Hardware/OS/backend profile (spec Part IX section 77). Initial setup
//! runs this short detection pass; `tqf optimize` later runs a deeper
//! benchmark matrix that refines measured kernel variants, I/O strategy,
//! and cache-policy parameters — none of which exist yet, so this only
//! captures the coarse facts those future passes will key off of.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    /// `None` when detection failed, rather than reporting a fabricated 0.
    pub total_memory_bytes: Option<u64>,
    pub backend: &'static str,
    pub detected_at_unix: u64,
}

pub fn detect() -> HardwareProfile {
    HardwareProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        total_memory_bytes: detect_total_memory(),
        backend: compiled_backend(),
        detected_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }
}

fn compiled_backend() -> &'static str {
    if cfg!(feature = "metal") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "none"
    }
}

/// REFERENCE BASELINE: shells out to `sysctl` rather than adding an FFI/
/// crate dependency for one integer. Direct `sysctl()` FFI is worth
/// revisiting once the Metal backend already needs its own FFI boundary
/// (spec Part VII).
#[cfg(target_os = "macos")]
fn detect_total_memory() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn detect_total_memory() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_total_memory() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_reports_at_least_one_core() {
        let profile = detect();
        assert!(profile.cpu_cores >= 1);
        assert!(!profile.os.is_empty());
    }
}
