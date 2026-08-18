//! Device/queue ownership (spec §49 "macOS/Metal ownership": "Rust owns
//! inference. Use mature Objective-C/Metal bindings for device/queue/
//! resource management..."). One `MetalContext` per process — Metal
//! devices/queues are meant to be long-lived and reused, never
//! re-acquired per request (spec §50: "no token-loop allocation churn").

use metal_sys::{CommandQueue, Device, MTLResourceOptions};

use crate::error::{BackendError, Result};
use crate::ids::Bytes;
use crate::memory::{MemoryBroker, MemoryClass, MemoryOwner};

use super::buffer::BufferLease;

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
}

impl MetalContext {
    /// Acquires the system default Metal device and one command queue.
    /// `None` from `Device::system_default()` means no usable GPU is
    /// present (e.g. a headless/sandboxed CI runner) — a typed error, not
    /// a panic, since callers (like `tqf optimize`) must be able to report
    /// this cleanly rather than crash.
    pub fn init() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| BackendError::Gpu("no Metal device available".to_string()))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Allocates a `StorageModeShared` buffer — the "baseline expert slot"
    /// shape from spec §50: an aligned CPU-visible allocation the GPU
    /// reads directly via Apple's unified memory, no explicit copy step.
    ///
    /// This does not yet register the allocation with a memory broker
    /// (spec §115 invariant #4: "every large allocation is registered
    /// with the memory broker before physical allocation") because the
    /// broker itself is Part VI/phase 21+ — `BufferLease` exists as its
    /// own type specifically so that registration can be inserted here
    /// later without changing this method's signature or callers.
    pub fn allocate_buffer(&self, length: u64, label: &str) -> BufferLease {
        let buffer = self
            .device
            .new_buffer(length, MTLResourceOptions::StorageModeShared);
        buffer.set_label(label);
        BufferLease::new(buffer, label.to_string())
    }

    /// Same as `allocate_buffer`, but copies `data` in immediately —
    /// convenient for the synthetic benchmark harness and future
    /// reference-path tests; the hot expert-streaming path fills buffers
    /// via `pread` into an already-leased buffer instead (spec §50).
    pub fn allocate_buffer_with_data(&self, data: &[u8], label: &str) -> BufferLease {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        buffer.set_label(label);
        BufferLease::new(buffer, label.to_string())
    }

    /// Broker-registered counterpart to `allocate_buffer`: reserves `length`
    /// bytes with `broker` *before* the physical Metal allocation happens
    /// (spec invariant #4), and ties the reservation's lifetime to the
    /// returned `BufferLease` so dropping the buffer frees the budget too.
    /// This is the allocation path Phase 20 GPU-resident state should use;
    /// the un-registered `allocate_buffer*` methods above remain only for
    /// the pre-broker synthetic-benchmark/kernel-parity call sites.
    pub fn allocate_broker_buffer(
        &self,
        broker: &MemoryBroker,
        owner: MemoryOwner,
        class: MemoryClass,
        length: u64,
        label: &str,
    ) -> Result<BufferLease> {
        let lease = broker.reserve(owner, class, Bytes(length), 64)?;
        let buffer = self
            .device
            .new_buffer(length, MTLResourceOptions::StorageModeShared);
        buffer.set_label(label);
        Ok(BufferLease::new_with_lease(
            buffer,
            label.to_string(),
            lease,
        ))
    }

    /// Same as `allocate_broker_buffer`, but copies `data` in immediately.
    pub fn allocate_broker_buffer_with_data(
        &self,
        broker: &MemoryBroker,
        owner: MemoryOwner,
        class: MemoryClass,
        data: &[u8],
        label: &str,
    ) -> Result<BufferLease> {
        let lease = broker.reserve(owner, class, Bytes(data.len() as u64), 64)?;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        buffer.set_label(label);
        Ok(BufferLease::new_with_lease(
            buffer,
            label.to_string(),
            lease,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal devices aren't guaranteed to exist in every CI/sandbox
    /// environment — this suite skips (not fails) when
    /// `Device::system_default()` returns `None`, rather than asserting a
    /// GPU must be present.
    fn context_or_skip() -> Option<MetalContext> {
        match MetalContext::init() {
            Ok(ctx) => Some(ctx),
            Err(_) => {
                eprintln!("skipping Metal test: no device available in this environment");
                None
            }
        }
    }

    #[test]
    fn acquires_a_device_and_queue_when_available() {
        let Some(ctx) = context_or_skip() else {
            return;
        };
        assert!(!ctx.device_name().is_empty());
    }

    #[test]
    fn allocates_a_shared_buffer() {
        let Some(ctx) = context_or_skip() else {
            return;
        };
        let lease = ctx.allocate_buffer(4096, "test-buffer");
        assert_eq!(lease.length(), 4096);
    }

    #[test]
    fn allocates_a_shared_buffer_with_initial_data() {
        let Some(ctx) = context_or_skip() else {
            return;
        };
        let data = vec![0xABu8; 256];
        let lease = ctx.allocate_buffer_with_data(&data, "test-buffer-data");
        assert_eq!(lease.length(), 256);
        assert_eq!(lease.as_slice(), data.as_slice());
    }

    #[test]
    fn broker_buffer_reserves_before_allocating_and_releases_on_drop() {
        let Some(ctx) = context_or_skip() else {
            return;
        };
        let broker = crate::memory::MemoryBroker::new(Bytes(4096));
        let lease = ctx
            .allocate_broker_buffer(
                &broker,
                MemoryOwner::ExpertPinned,
                MemoryClass::Elastic,
                1024,
                "broker-buffer",
            )
            .unwrap();
        assert_eq!(lease.length(), 1024);
        assert_eq!(broker.snapshot().reserved, Bytes(1024));
        drop(lease);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn broker_buffer_over_budget_is_rejected_before_any_metal_allocation() {
        let Some(ctx) = context_or_skip() else {
            return;
        };
        let broker = crate::memory::MemoryBroker::new(Bytes(512));
        let error = match ctx.allocate_broker_buffer(
            &broker,
            MemoryOwner::ExpertPinned,
            MemoryClass::Elastic,
            4096,
            "too-big",
        ) {
            Err(error) => error,
            Ok(_) => panic!("expected an over-budget reservation to fail"),
        };
        assert!(
            error.to_string().to_lowercase().contains("budget")
                || error.to_string().to_lowercase().contains("memory")
        );
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn broker_buffer_with_data_round_trips_contents() {
        let Some(ctx) = context_or_skip() else {
            return;
        };
        let broker = crate::memory::MemoryBroker::new(Bytes(4096));
        let data = vec![0x5Au8; 256];
        let lease = ctx
            .allocate_broker_buffer_with_data(
                &broker,
                MemoryOwner::ExpertPinned,
                MemoryClass::Elastic,
                &data,
                "broker-buffer-data",
            )
            .unwrap();
        assert_eq!(lease.as_slice(), data.as_slice());
        assert_eq!(broker.snapshot().reserved, Bytes(256));
    }
}
