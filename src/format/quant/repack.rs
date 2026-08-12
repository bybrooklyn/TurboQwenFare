//! Lossless "TQF passthrough" packing candidate (spec §279 REFERENCE
//! BASELINE): repacks a source tensor's quant blocks into `.tqf` extent
//! bytes byte-for-byte identical to the GGUF source. Kernel-specific
//! reordering (spec §146: "`TQF-Q4` packing candidates may reorder the
//! packed block to remove shifts/gathers") is a later optimization phase
//! (§283, reference Q4 kernels); this phase's job is a correct lossless
//! baseline plus the validation machinery (`format::quant::validate`) any
//! future reordering candidate must also pass.

use crate::error::Result;
use crate::format::gguf::QuantBlockReader;
use crate::format::quant::GgmlType;

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
        _ => return None,
    })
}

/// Streams every block of `reader` into one contiguous byte buffer,
/// unchanged. Bounded-memory in the sense that it reuses `QuantBlockReader`'s
/// own on-disk batching rather than doing a separate large read, but the
/// *output* buffer holds one whole tensor at a time — acceptable at
/// individual-tensor granularity for the pinned checkpoint geometry (the
/// largest single tensor is well under the 4 GiB default budget); this
/// must be revisited if that stops being true for a future model.
pub fn repack_passthrough(reader: &mut QuantBlockReader) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(batch) = reader.next_batch()? {
        out.extend_from_slice(&batch);
    }
    Ok(out)
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
