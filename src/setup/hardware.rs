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
    pub backend: String,
    pub detected_at_unix: u64,
    #[serde(default)]
    pub quick_tune: Option<QuickTuneProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickTuneProfile {
    pub device_name: String,
    pub bandwidth_gigabytes_per_second: f64,
    pub naive_gemv_gflops: f64,
    pub tuned_at_unix: u64,
}

pub fn detect() -> HardwareProfile {
    HardwareProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        total_memory_bytes: detect_total_memory(),
        backend: compiled_backend().to_string(),
        detected_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        quick_tune: None,
    }
}

impl HardwareProfile {
    pub fn preserve_compatible_quick_tune(&mut self, previous: &Self) {
        if self.os == previous.os && self.arch == previous.arch && self.backend == previous.backend
        {
            self.quick_tune.clone_from(&previous.quick_tune);
        }
    }
}

#[cfg(tqf_metal)]
pub fn run_short_autotune(profile: &mut HardwareProfile) -> crate::error::Result<()> {
    let report = crate::bench::metal_synthetic::run_synthetic_bandwidth_gemv()?;
    profile.quick_tune = Some(QuickTuneProfile {
        device_name: report.device_name,
        bandwidth_gigabytes_per_second: report.bandwidth.gigabytes_per_second,
        naive_gemv_gflops: report.gemv.gflops,
        tuned_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
    });
    Ok(())
}

#[cfg(not(tqf_metal))]
pub fn run_short_autotune(_profile: &mut HardwareProfile) -> crate::error::Result<()> {
    Ok(())
}

/// Which compute backend this binary can actually reach on this target.
/// `reference` is not a shipping inference backend (spec §48 expects one
/// of Metal/CUDA per build) — it is the portable scalar path the crate
/// is validated against, and reporting it honestly is what lets
/// `tqf doctor` say "this build has no GPU backend" instead of "none".
fn compiled_backend() -> &'static str {
    if cfg!(tqf_metal) {
        "metal"
    } else if cfg!(tqf_cuda) {
        "cuda"
    } else {
        "reference"
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

    #[test]
    fn compatible_redetection_preserves_the_short_tune() {
        let mut previous = detect();
        previous.quick_tune = Some(QuickTuneProfile {
            device_name: "fixture".to_string(),
            bandwidth_gigabytes_per_second: 1.0,
            naive_gemv_gflops: 2.0,
            tuned_at_unix: 3,
        });
        let mut current = detect();
        current.preserve_compatible_quick_tune(&previous);
        assert_eq!(
            current
                .quick_tune
                .as_ref()
                .map(|tune| tune.device_name.as_str()),
            Some("fixture")
        );
    }
}
