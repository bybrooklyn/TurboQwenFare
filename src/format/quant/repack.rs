//! Lossless "TQF passthrough" packing candidate (spec §279 REFERENCE
//! BASELINE): repacks a source tensor's quant blocks into `.tqf` extent
//! bytes byte-for-byte identical to the GGUF source. Kernel-specific
//! reordering (spec §146: "`TQF-Q4` packing candidates may reorder the
//! packed block to remove shifts/gathers") is a later optimization phase
//! (§283, reference Q4 kernels); this phase's job is a correct lossless
//! baseline plus the validation machinery (`format::quant::validate`) any
//! future reordering candidate must also pass.

use std::ops::{Deref, DerefMut};

use crate::error::{ContainerError, Result};
use crate::format::gguf::QuantBlockReader;
use crate::format::quant::GgmlType;
use crate::ids::Bytes;
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

/// TQF quant-layout IDs (spec §123 `quant_layout_id` field) — this
/// format's own namespace, distinct from GGML type IDs
/// (`GgmlType::from_ggml_id`), so a future non-passthrough layout for the
/// same source type gets a new ID without colliding with this one.
pub const TQF_QUANT_PASSTHROUGH_Q4_0: u32 = 1;
pub const TQF_QUANT_PASSTHROUGH_Q4_K: u32 = 2;
pub const TQF_QUANT_PASSTHROUGH_Q8_0: u32 = 3;
pub const TQF_QUANT_PASSTHROUGH_F32: u32 = 4;
pub const TQF_QUANT_PASSTHROUGH_F16: u32 = 5;
pub const TQF_QUANT_PASSTHROUGH_BF16: u32 = 6;
/// The official Q4_K_M language checkpoint stores `output.weight` as Q6_K;
/// rejecting it would make a supposedly canonical conversion unable to
/// produce a runnable LM head.
pub const TQF_QUANT_PASSTHROUGH_Q6_K: u32 = 7;

/// `None` for GGML types this phase's passthrough packer does not (yet)
/// carry into `.tqf` — the importer must reject those tensors rather than
/// pack unrecognized bytes under a made-up layout ID.
pub fn tqf_quant_layout_id(ggml_type: GgmlType) -> Option<u32> {
    Some(match ggml_type {
        GgmlType::Q4_0 => TQF_QUANT_PASSTHROUGH_Q4_0,
        GgmlType::Q4K => TQF_QUANT_PASSTHROUGH_Q4_K,
        GgmlType::Q8_0 => TQF_QUANT_PASSTHROUGH_Q8_0,
        GgmlType::F32 => TQF_QUANT_PASSTHROUGH_F32,
        GgmlType::F16 => TQF_QUANT_PASSTHROUGH_F16,
        GgmlType::Bf16 => TQF_QUANT_PASSTHROUGH_BF16,
        GgmlType::Q6K => TQF_QUANT_PASSTHROUGH_Q6_K,
        _ => return None,
    })
}

/// Inverse of `tqf_quant_layout_id`, for readers that need to know how to
/// interpret a `.tqf` extent's bytes.
pub fn ggml_type_for_quant_layout(layout_id: u32) -> Option<GgmlType> {
    Some(match layout_id {
        TQF_QUANT_PASSTHROUGH_Q4_0 => GgmlType::Q4_0,
        TQF_QUANT_PASSTHROUGH_Q4_K => GgmlType::Q4K,
        TQF_QUANT_PASSTHROUGH_Q8_0 => GgmlType::Q8_0,
        TQF_QUANT_PASSTHROUGH_F32 => GgmlType::F32,
        TQF_QUANT_PASSTHROUGH_F16 => GgmlType::F16,
        TQF_QUANT_PASSTHROUGH_BF16 => GgmlType::Bf16,
        TQF_QUANT_PASSTHROUGH_Q6_K => GgmlType::Q6K,
        _ => return None,
    })
}

/// Exact passthrough bytes plus the broker reservation that must outlive
/// them. The byte vector is declared first so it is physically released
/// before the lease returns its budget.
pub struct AccountedRepack {
    bytes: Vec<u8>,
    _lease: MemoryLease,
}

impl Deref for AccountedRepack {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl DerefMut for AccountedRepack {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bytes
    }
}

/// Streams every block of `reader` into one contiguous byte buffer,
/// unchanged. The complete output and the largest temporary read batch are
/// reserved before either allocation; exact preallocation prevents `Vec`
/// growth from silently exceeding the registered size.
pub fn repack_passthrough(
    reader: &mut QuantBlockReader,
    broker: &MemoryBroker,
) -> Result<AccountedRepack> {
    let total_bytes = reader.total_bytes();
    let reserved_bytes = total_bytes
        .checked_add(reader.max_batch_bytes())
        .ok_or(ContainerError::IntegerOverflow)?;
    let lease = broker.reserve(
        MemoryOwner::IoStaging,
        MemoryClass::Transient,
        Bytes(reserved_bytes),
        64,
    )?;
    let capacity: usize = total_bytes
        .try_into()
        .map_err(|_| ContainerError::IntegerOverflow)?;
    let mut out = Vec::with_capacity(capacity);
    while let Some(batch) = reader.next_batch()? {
        out.extend_from_slice(&batch);
    }
    if out.len() != capacity {
        return Err(ContainerError::MalformedRecord {
            table: "quant passthrough byte count",
        }
        .into());
    }
    Ok(AccountedRepack {
        bytes: out,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_id_round_trips_for_pinned_checkpoint_types() {
        for t in [
            GgmlType::Q4_0,
            GgmlType::Q4K,
            GgmlType::Q8_0,
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q6K,
        ] {
            let id = tqf_quant_layout_id(t).unwrap();
            assert_eq!(ggml_type_for_quant_layout(id), Some(t));
        }
    }

    #[test]
    fn layout_ids_are_pairwise_distinct() {
        let ids = [
            TQF_QUANT_PASSTHROUGH_Q4_0,
            TQF_QUANT_PASSTHROUGH_Q4_K,
            TQF_QUANT_PASSTHROUGH_Q8_0,
            TQF_QUANT_PASSTHROUGH_F32,
            TQF_QUANT_PASSTHROUGH_F16,
            TQF_QUANT_PASSTHROUGH_BF16,
            TQF_QUANT_PASSTHROUGH_Q6_K,
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }

    #[test]
    fn unrecognized_type_has_no_layout_id() {
        assert!(tqf_quant_layout_id(GgmlType::Q5K).is_none());
        assert!(ggml_type_for_quant_layout(9999).is_none());
    }
}
