//! The memory broker: the single source of truth for every large allocation
//! (spec Part VI sections 37-40; Part XIV sections 128-132). `--memory` is a
//! hard contract, not an advisory cache size.
//!
//! This intentionally small first implementation provides the part of the
//! broker every correctness implementation needs: an allocation must obtain a
//! RAII reservation *before* it performs its physical allocation. Cache
//! owners reclaim their own elastic leases; later parallel-I/O work builds
//! asynchronous admission on this accounting rather than bypassing it.

use std::sync::{Arc, Mutex};

use crate::error::{MemoryError, Result};
use crate::ids::Bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryClass {
    Fixed,
    Protected,
    Elastic,
    Transient,
    Backing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOwner {
    Core,
    GdnState,
    ContextHot,
    ContextCold,
    ExpertPinned,
    ExpertProbation,
    IoStaging,
    Scratch,
    ServerReserve,
}

impl std::fmt::Display for MemoryOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub budget: Bytes,
    pub reserved: Bytes,
}

#[derive(Debug)]
struct MemoryBrokerInner {
    budget: u64,
    reserved: Mutex<u64>,
}

/// Global accounting handle.  Cloning it does not duplicate budget.
#[derive(Debug, Clone)]
pub struct MemoryBroker {
    inner: Arc<MemoryBrokerInner>,
}

/// RAII proof that a caller owns budget.  It deliberately contains no data
/// pointer: callers acquire this first, then allocate their own storage.
#[derive(Debug)]
pub struct MemoryLease {
    bytes: u64,
    _owner: MemoryOwner,
    _class: MemoryClass,
    inner: Arc<MemoryBrokerInner>,
}

impl MemoryBroker {
    pub fn new(budget: Bytes) -> Self {
        Self {
            inner: Arc::new(MemoryBrokerInner {
                budget: budget.0,
                reserved: Mutex::new(0),
            }),
        }
    }

    /// Reserves first; callers must only allocate after this succeeds.
    pub fn reserve(
        &self,
        owner: MemoryOwner,
        class: MemoryClass,
        bytes: Bytes,
        alignment: u64,
    ) -> Result<MemoryLease> {
        if bytes.0 == 0 || alignment == 0 || !alignment.is_power_of_two() {
            return Err(MemoryError::BudgetExceeded {
                requested: bytes.0,
                available: self
                    .snapshot()
                    .budget
                    .0
                    .saturating_sub(self.snapshot().reserved.0),
                owner: owner.to_string(),
                suggestion: "request a nonzero size with power-of-two alignment".to_string(),
            }
            .into());
        }

        let mut reserved = self
            .inner
            .reserved
            .lock()
            .expect("memory accounting mutex poisoned");
        let available = self.inner.budget.saturating_sub(*reserved);
        if bytes.0 > available {
            return Err(MemoryError::BudgetExceeded {
                requested: bytes.0,
                available,
                owner: owner.to_string(),
                suggestion: "increase --memory or reduce the requested context/cache capacity"
                    .to_string(),
            }
            .into());
        }
        *reserved += bytes.0;
        drop(reserved);
        Ok(MemoryLease {
            bytes: bytes.0,
            _owner: owner,
            _class: class,
            inner: Arc::clone(&self.inner),
        })
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        let reserved = *self
            .inner
            .reserved
            .lock()
            .expect("memory accounting mutex poisoned");
        MemorySnapshot {
            budget: Bytes(self.inner.budget),
            reserved: Bytes(reserved),
        }
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        let mut reserved = self
            .inner
            .reserved
            .lock()
            .expect("memory accounting mutex poisoned");
        *reserved = reserved.saturating_sub(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_released_on_drop() {
        let broker = MemoryBroker::new(Bytes(128));
        let lease = broker
            .reserve(
                MemoryOwner::ContextHot,
                MemoryClass::Protected,
                Bytes(96),
                64,
            )
            .unwrap();
        assert_eq!(broker.snapshot().reserved, Bytes(96));
        assert!(broker
            .reserve(MemoryOwner::Scratch, MemoryClass::Transient, Bytes(64), 64)
            .is_err());
        drop(lease);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }
}
