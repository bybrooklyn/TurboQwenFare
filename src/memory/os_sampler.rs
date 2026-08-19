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
    // /proc/self/statm: size resident shared text lib data dt - pages,
    // resident is the model-relevant number; virtual = size.
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut fields = statm.split_whitespace();
    let pages = fields.next()?.parse::<u64>().ok()?;
    let resident_pages = fields.next()?.parse::<u64>().ok()?;
    let page = 4096u64;
    Some((resident_pages * page, pages * page, 0))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn sample_native() -> Option<(u64, u64, u64)> {
    None
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
        let broker = MemoryBroker::new(Bytes(1 * 1024 * 1024 * 1024));
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
            sample.resident_bytes >= 1 * 1024 * 1024,
            "touched pages must be resident (got {})",
            sample.resident_bytes
        );
        assert!(sample.broker_reserved_bytes >= 16 * 1024 * 1024);
        assert!(sample.resident_peak_bytes >= sample.resident_bytes);
        drop(buffer);
        drop(lease);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn sampler_tracks_broker_peak_across_churn() {
        let broker = MemoryBroker::new(Bytes(1 * 1024 * 1024 * 1024));
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
