//! The memory broker: the single source of truth for every large allocation
//! (spec Part VI sections 37-40; Part XIV sections 128-132). `--memory` is a
//! hard contract, not an advisory cache size.
//!
//! This intentionally small first implementation provides the part of the
//! broker every correctness implementation needs: an allocation must obtain a
//! RAII reservation *before* it performs its physical allocation. Cache
//! owners reclaim their own elastic leases; later parallel-I/O work builds
//! asynchronous admission on this accounting rather than bypassing it.

pub mod os_sampler;

use std::sync::atomic::{AtomicU64, Ordering};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Highest reservation total observed since construction (Phase 24).
    pub peak: Bytes,
    /// Per-owner reserved breakdown (Phase 24 sampler reporting).
    pub by_owner: OwnerReserved,
}

/// Fixed-size per-owner reservation table; `MemoryOwner` has nine
/// variants so a small array avoids a heap map in the hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerReserved {
    pub core: u64,
    pub gdn_state: u64,
    pub context_hot: u64,
    pub context_cold: u64,
    pub expert_pinned: u64,
    pub expert_probation: u64,
    pub io_staging: u64,
    pub scratch: u64,
    pub server_reserve: u64,
}

impl OwnerReserved {
    fn add(&mut self, owner: MemoryOwner, bytes: u64) {
        let slot = match owner {
            MemoryOwner::Core => &mut self.core,
            MemoryOwner::GdnState => &mut self.gdn_state,
            MemoryOwner::ContextHot => &mut self.context_hot,
            MemoryOwner::ContextCold => &mut self.context_cold,
            MemoryOwner::ExpertPinned => &mut self.expert_pinned,
            MemoryOwner::ExpertProbation => &mut self.expert_probation,
            MemoryOwner::IoStaging => &mut self.io_staging,
            MemoryOwner::Scratch => &mut self.scratch,
            MemoryOwner::ServerReserve => &mut self.server_reserve,
        };
        *slot = slot.saturating_add(bytes);
    }

    fn sub(&mut self, owner: MemoryOwner, bytes: u64) {
        let slot = match owner {
            MemoryOwner::Core => &mut self.core,
            MemoryOwner::GdnState => &mut self.gdn_state,
            MemoryOwner::ContextHot => &mut self.context_hot,
            MemoryOwner::ContextCold => &mut self.context_cold,
            MemoryOwner::ExpertPinned => &mut self.expert_pinned,
            MemoryOwner::ExpertProbation => &mut self.expert_probation,
            MemoryOwner::IoStaging => &mut self.io_staging,
            MemoryOwner::Scratch => &mut self.scratch,
            MemoryOwner::ServerReserve => &mut self.server_reserve,
        };
        *slot = slot.saturating_sub(bytes);
    }
}

impl Default for OwnerReserved {
    fn default() -> Self {
        Self {
            core: 0,
            gdn_state: 0,
            context_hot: 0,
            context_cold: 0,
            expert_pinned: 0,
            expert_probation: 0,
            io_staging: 0,
            scratch: 0,
            server_reserve: 0,
        }
    }
}

#[derive(Debug)]
struct MemoryBrokerInner {
    budget: u64,
    reserved: Mutex<u64>,
    by_owner: Mutex<OwnerReserved>,
    peak: AtomicU64,
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
    owner: MemoryOwner,
    _class: MemoryClass,
    inner: Arc<MemoryBrokerInner>,
}

impl MemoryBroker {
    pub fn new(budget: Bytes) -> Self {
        Self {
            inner: Arc::new(MemoryBrokerInner {
                budget: budget.0,
                reserved: Mutex::new(0),
                by_owner: Mutex::new(OwnerReserved::default()),
                peak: AtomicU64::new(0),
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
        self.inner
            .by_owner
            .lock()
            .expect("memory owner-accounting mutex poisoned")
            .add(owner, bytes.0);
        self.inner.peak.fetch_max(*reserved, Ordering::SeqCst);
        drop(reserved);
        Ok(MemoryLease {
            bytes: bytes.0,
            owner,
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
        let by_owner = *self
            .inner
            .by_owner
            .lock()
            .expect("memory owner-accounting mutex poisoned");
        MemorySnapshot {
            budget: Bytes(self.inner.budget),
            reserved: Bytes(reserved),
            peak: Bytes(self.inner.peak.load(Ordering::SeqCst)),
            by_owner,
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
        self.inner
            .by_owner
            .lock()
            .expect("memory owner-accounting mutex poisoned")
            .sub(self.owner, self.bytes);
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
        assert_eq!(broker.snapshot().by_owner.context_hot, 96);
        assert_eq!(broker.snapshot().peak, Bytes(96));
        assert!(broker
            .reserve(MemoryOwner::Scratch, MemoryClass::Transient, Bytes(64), 64)
            .is_err());
        drop(lease);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        assert_eq!(broker.snapshot().by_owner.context_hot, 0);
        // Peak is monotone across the churn.
        assert_eq!(broker.snapshot().peak, Bytes(96));
    }

    /// Phase 24 adversarial churn (spec §296, §132): random reservation
    /// bursts across every owner/class must never exceed the budget,
    /// must release exactly, and must keep the per-owner breakdown in
    /// agreement with the total at all times.
    #[test]
    fn adversarial_reservation_churn_stays_within_budget() {
        const BUDGET: u64 = 64 * 1024 * 1024;
        let broker = MemoryBroker::new(Bytes(BUDGET));
        let owners = [
            MemoryOwner::Core,
            MemoryOwner::GdnState,
            MemoryOwner::ContextHot,
            MemoryOwner::ContextCold,
            MemoryOwner::ExpertPinned,
            MemoryOwner::ExpertProbation,
            MemoryOwner::IoStaging,
            MemoryOwner::Scratch,
            MemoryOwner::ServerReserve,
        ];
        let classes = [
            MemoryClass::Fixed,
            MemoryClass::Protected,
            MemoryClass::Elastic,
            MemoryClass::Transient,
            MemoryClass::Backing,
        ];
        let mut held: Vec<MemoryLease> = Vec::new();
        // Deterministic xorshift so the stress is reproducible.
        let mut state = 0x9E3779B97F4A7C15u64;
        for step in 0..200_000u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let size = 4096u64.saturating_mul((state % 32) + 1);
            let owner = owners[(state as usize >> 3) % owners.len()];
            let class = classes[(state as usize >> 6) % classes.len()];
            if state % 3 == 0 {
                held.clear();
            }
            match broker.reserve(owner, class, Bytes(size), 64) {
                Ok(lease) => {
                    let snapshot = broker.snapshot();
                    assert!(snapshot.reserved.0 <= BUDGET, "step {step}");
                    assert!(snapshot.peak.0 <= BUDGET);
                    assert_eq!(snapshot.peak.0 >= snapshot.reserved.0, true);
                    // Per-owner breakdown sums to the total.
                    let by_owner = snapshot.by_owner;
                    let sum = by_owner.core
                        + by_owner.gdn_state
                        + by_owner.context_hot
                        + by_owner.context_cold
                        + by_owner.expert_pinned
                        + by_owner.expert_probation
                        + by_owner.io_staging
                        + by_owner.scratch
                        + by_owner.server_reserve;
                    assert_eq!(sum, snapshot.reserved.0, "step {step}");
                    if held.len() >= 16 {
                        held.remove((state as usize) % held.len());
                    }
                    held.push(lease);
                }
                Err(error) => {
                    assert!(
                        matches!(error, crate::error::TqfError::Memory(_)),
                        "step {step}: {error:?}"
                    );
                }
            }
        }
        drop(held);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        assert_eq!(
            broker.snapshot().by_owner,
            OwnerReserved::default(),
            "per-owner breakdown must drain to zero"
        );
    }
}
