//! Buffer leases (spec §144 "Metal buffer ownership", §37 "the memory
//! broker is the law"). `BufferLease` is deliberately its own type,
//! distinct from a bare `metal_sys::Buffer`, so that once the memory
//! broker exists (Part VI, phase 21+) a lease can carry a broker
//! reservation handle without changing any call site's signature — spec
//! §115 invariant #4 ("every large allocation is registered with the
//! memory broker *before* physical allocation") and invariant #5 ("every
//! async I/O op owns/borrows a destination lease that outlives
//! completion") both describe the *shape* this type exists to satisfy,
//! even though the broker side isn't wired in yet.

use metal_sys::Buffer;

pub struct BufferLease {
    buffer: Buffer,
    label: String,
}

impl BufferLease {
    pub(super) fn new(buffer: Buffer, label: String) -> Self {
        Self { buffer, label }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn length(&self) -> u64 {
        self.buffer.length()
    }

    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Reads the buffer's current contents as a byte slice. Only sound to
    /// call when no GPU command writing to this buffer is still in
    /// flight (spec §115 invariant #6's buffer-lease-outlives-completion
    /// posture, applied to CPU-side reads of a `StorageModeShared`
    /// buffer) — a synchronous benchmark harness that always
    /// `wait_until_completed()`s before reading satisfies this trivially;
    /// a real async decode loop must not.
    pub fn as_slice(&self) -> &[u8] {
        let len = self.buffer.length() as usize;
        let ptr = self.buffer.contents() as *const u8;
        if len == 0 {
            &[]
        } else {
            // Safety: `contents()` is valid for `length()` bytes for a
            // `StorageModeShared` buffer, and this is a shared (`&self`)
            // borrow of `self`, upholding the no-concurrent-GPU-write
            // precondition documented above.
            unsafe { std::slice::from_raw_parts(ptr, len) }
        }
    }

    /// Writes `data` into the buffer starting at byte 0. Same in-flight
    /// caveat as `as_slice`, mirrored for writes.
    pub fn write(&self, data: &[u8]) {
        assert!(
            data.len() as u64 <= self.buffer.length(),
            "write of {} bytes exceeds buffer length {}",
            data.len(),
            self.buffer.length()
        );
        let ptr = self.buffer.contents() as *mut u8;
        // Safety: same as `as_slice` — no GPU command may be concurrently
        // reading/writing this buffer while this `&self` write happens.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }
}
