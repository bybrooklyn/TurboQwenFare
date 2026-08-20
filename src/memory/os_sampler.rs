//! Phase 24 OS-observed memory qualification (spec §132): samples the
//! process's real OS footprint alongside the broker's own reserved
//! accounting. The governing rule is blunt: a configuration is not "4G
//! certified" because steady-state decode is 3.9G if loading or
//! admission spikes the OS-observed footprint to 4.7G - so every
//! qualification harness samples both numbers, never one.
//!
//! macOS uses `task_info(MACH_TASK_BASIC_INFO)` (resident/virtual set);
//! Linux falls back to `/proc/self/statm`. The sampler is read-only and
//! never reserves broker budget itself.

use crate::error::{InternalError, Result};
use crate::ids::Bytes;
use crate::memory::MemoryBroker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsFootprintSample {
    /// OS-observed resident set (bytes).
    pub resident_bytes: u64,
    /// OS-observed virtual size (bytes).
    pub virtual_bytes: u64,
    /// Peak resident set observed by the OS so far (bytes).
    pub resident_peak_bytes: u64,
    /// The broker's own reserved total at sample time.
    pub broker_reserved_bytes: u64,
    /// The broker's peak reserved total so far.
    pub broker_peak_bytes: u64,
}

impl OsFootprintSample {
    /// How far the OS-observed resident set exceeds the broker's own
    /// accounting at sample time. Non-model overhead (code, stacks,
    /// tokenizer metadata, allocator slack) shows up here; the Phase 24
    /// qualification records this as the certified overhead envelope.
    pub fn observed_over_broker(&self) -> u64 {
        self.resident_bytes
            .saturating_sub(self.broker_reserved_bytes)
    }
}

#[cfg(target_os = "macos")]
fn sample_native() -> Option<(u64, u64, u64)> {
    #[allow(deprecated)]
    let task = unsafe { libc::mach_task_self_ };
    let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::zeroed();
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    let status = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as libc::task_info_t,
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS {
        return None;
    }
    let info = unsafe { info.assume_init() };
    Some((
        info.resident_size,
        info.virtual_size,
        info.resident_size_max,
    ))
}

#[cfg(target_os = "linux")]
fn sample_native() -> Option<(u64, u64, u64)> {
    // `/proc/self/status` is preferred over `/proc/self/statm` because it
    // is the only one of the two that reports a *peak* resident set
    // (`VmHWM`). Phase 24's whole point is that a configuration is not
    // "4G certified" because steady-state decode is 3.9G if admission
    // spiked to 4.7G — a sampler that reports peak as 0 cannot make that
    // judgement, and the macOS branch above (`resident_size_max`) does.
    // Values are in kB per the kernel's own formatting.
    if let Some(sample) = sample_proc_status() {
        return Some(sample);
    }
    // `/proc/self/statm` fallback for kernels/sandboxes without a status
    // file. Peak is genuinely unavailable here, so it is reported as the
    // current resident set rather than 0: peak is by definition at least
    // the current value, and 0 would understate it in a way callers
    // (`assert_footprint_within`) treat as real evidence.
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut fields = statm.split_whitespace();
    let pages = fields.next()?.parse::<u64>().ok()?;
    let resident_pages = fields.next()?.parse::<u64>().ok()?;
    let page = page_size();
    let resident = resident_pages * page;
    Some((resident, pages * page, resident))
}

/// Parses `VmRSS`, `VmSize`, and `VmHWM` (peak resident) out of
/// `/proc/self/status`. Returns `None` if the file is unreadable or any
/// of the three fields is absent, so the caller can fall back rather
/// than report a partially-zero sample.
#[cfg(target_os = "linux")]
fn sample_proc_status() -> Option<(u64, u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut resident = None;
    let mut virtual_bytes = None;
    let mut peak = None;
    for line in status.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let target = match key {
            "VmRSS" => &mut resident,
            "VmSize" => &mut virtual_bytes,
            "VmHWM" => &mut peak,
            _ => continue,
        };
        // "VmRSS:\t   12345 kB"
        let kilobytes = value.split_whitespace().next()?.parse::<u64>().ok()?;
        *target = Some(kilobytes.saturating_mul(1024));
    }
    Some((resident?, virtual_bytes?, peak?))
}

/// The real OS page size rather than an assumed 4096: aarch64 Linux and
/// several distributions ship 16K/64K pages, which would silently scale
/// every `/proc/self/statm` reading by 4x or 16x.
#[cfg(target_os = "linux")]
fn page_size() -> u64 {
    // SAFETY: `sysconf` is thread-safe and takes no pointers.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if raw > 0 {
        raw as u64
    } else {
        4096
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn sample_native() -> Option<(u64, u64, u64)> {
    None
}

/// Real OS-observed resident/virtual/peak-resident bytes for the
/// current process, with no broker dependency — for callers (like the
/// GUI's inspector metrics endpoint, spec §47) that want a genuine
/// process footprint reading without needing a live `MemoryBroker`
/// instance. `None` on platforms without a sampler (same fallback
/// `sample_os_footprint` uses).
pub fn sample_process_footprint() -> Option<(Bytes, Bytes, Bytes)> {
    let (resident, virtual_bytes, resident_peak) = sample_native()?;
    Some((Bytes(resident), Bytes(virtual_bytes), Bytes(resident_peak)))
}

/// Samples the OS footprint and the broker accounting in one place. On
/// platforms without a sampler, the native fields are zero and the
/// sample still carries the broker numbers.
pub fn sample_os_footprint(broker: &MemoryBroker) -> Result<OsFootprintSample> {
    let (resident, virtual_bytes, resident_peak) =
        sample_native().ok_or_else(|| InternalError {
            incident_id: "os-footprint-sampler".to_string(),
            message: "the OS footprint sampler is unavailable on this platform".to_string(),
        })?;
    let snapshot = broker.snapshot();
    Ok(OsFootprintSample {
        resident_bytes: resident,
        virtual_bytes,
        resident_peak_bytes: resident_peak,
        broker_reserved_bytes: snapshot.reserved.0,
        broker_peak_bytes: snapshot.peak.0,
    })
}

/// Convenience: asserts the sampled footprint of `decode` stays within
/// `budget` plus a measured overhead envelope. Returns the sample for
/// the qualification record.
pub fn assert_footprint_within(
    broker: &MemoryBroker,
    budget: Bytes,
    overhead: Bytes,
    mut decode: impl FnMut() -> Result<()>,
) -> Result<OsFootprintSample> {
    let before = sample_os_footprint(broker)?;
    decode()?;
    let after = sample_os_footprint(broker)?;
    let sample = OsFootprintSample {
        resident_bytes: before.resident_bytes.max(after.resident_bytes),
        virtual_bytes: before.virtual_bytes.max(after.virtual_bytes),
        resident_peak_bytes: before.resident_peak_bytes.max(after.resident_peak_bytes),
        broker_reserved_bytes: before
            .broker_reserved_bytes
            .max(after.broker_reserved_bytes),
        broker_peak_bytes: before.broker_peak_bytes.max(after.broker_peak_bytes),
    };
    if sample.resident_bytes > budget.0.saturating_add(overhead.0) {
        return Err(crate::error::MemoryError::BudgetExceeded {
            requested: sample.resident_bytes,
            available: budget.0.saturating_add(overhead.0),
            owner: "OS-observed process footprint".to_string(),
            suggestion: "reduce --memory, cache capacity, or context so peak resident fits the qualified envelope".to_string(),
        }
        .into());
    }
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryClass, MemoryOwner};

    #[test]
    fn sampler_reports_a_plausible_footprint_above_broker_reservation() {
        let broker = MemoryBroker::new(Bytes(1024 * 1024 * 1024));
        let lease = broker
            .reserve(
                MemoryOwner::Scratch,
                MemoryClass::Transient,
                Bytes(16 * 1024 * 1024),
                64,
            )
            .unwrap();
        let mut buffer = vec![0u8; 16 * 1024 * 1024];
        for (index, byte) in buffer.iter_mut().enumerate().take(1 << 20) {
            *byte = (index & 0xFF) as u8; // touch pages so RSS reflects them
        }
        let sample = sample_os_footprint(&broker).unwrap();
        assert!(sample.resident_bytes > 0, "resident set must be nonzero");
        assert!(
            sample.resident_bytes >= 1024 * 1024,
            "touched pages must be resident (got {})",
            sample.resident_bytes
        );
        assert!(sample.broker_reserved_bytes >= 16 * 1024 * 1024);
        assert!(sample.resident_peak_bytes >= sample.resident_bytes);
        drop(buffer);
        drop(lease);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    /// The Linux branch previously read `/proc/self/statm`, which has no
    /// peak field at all, and reported peak as a hardcoded 0 — so the
    /// `resident_peak_bytes >= resident_bytes` invariant every caller
    /// relies on was false on every Linux run. This pins the real
    /// `/proc/self/status` reader instead.
    #[cfg(target_os = "linux")]
    #[test]
    fn proc_status_reports_a_real_peak_at_or_above_the_current_resident_set() {
        let (resident, virtual_bytes, peak) =
            sample_proc_status().expect("/proc/self/status must be readable on Linux");
        assert!(resident > 0, "VmRSS must be nonzero for a live process");
        assert!(
            virtual_bytes >= resident,
            "VmSize ({virtual_bytes}) must be at least VmRSS ({resident})"
        );
        assert!(
            peak >= resident,
            "VmHWM ({peak}) must be at least VmRSS ({resident})"
        );
    }

    /// The page size must come from the OS, not a hardcoded 4096: a
    /// 16K-page kernel would otherwise scale every `statm` reading by 4x.
    #[cfg(target_os = "linux")]
    #[test]
    fn page_size_is_a_real_power_of_two_from_the_os() {
        let page = page_size();
        assert!(page >= 4096, "implausible page size {page}");
        assert!(
            page.is_power_of_two(),
            "page size {page} is not a power of two"
        );
    }

    #[test]
    fn sampler_tracks_broker_peak_across_churn() {
        let broker = MemoryBroker::new(Bytes(1024 * 1024 * 1024));
        let first = broker
            .reserve(
                MemoryOwner::IoStaging,
                MemoryClass::Elastic,
                Bytes(64 * 1024 * 1024),
                64,
            )
            .unwrap();
        drop(first);
        let _second = broker
            .reserve(
                MemoryOwner::IoStaging,
                MemoryClass::Elastic,
                Bytes(32 * 1024 * 1024),
                64,
            )
            .unwrap();
        let sample = sample_os_footprint(&broker).unwrap();
        assert_eq!(sample.broker_reserved_bytes, 32 * 1024 * 1024);
        assert_eq!(sample.broker_peak_bytes, 64 * 1024 * 1024);
    }
}
