//! Canonical GGUF-to-TQF installation conversion. This is deliberately a
//! strict, lossless bridge: every source tensor must classify, every carried
//! quantization type must have an explicit TQF layout ID, and finalization is
//! delegated to `ConversionTransaction` for its durable journal/rename path.
//!
//! Routed-expert tensors are rewritten losslessly into checksummed
//! whole-expert superextents. This lets the Phase-18 cache read exactly a
//! router-selected expert without retaining its 255 siblings.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::dev::inventory::{generate_inventory_from_file, TensorInventoryEntry, TensorRole};
use crate::error::{ContainerError, Result};
use crate::format::gguf;
use crate::format::quant::repack::{repack_passthrough, tqf_quant_layout_id};
use crate::format::tqf::conversion::{BeginOutcome, ConversionTransaction};
use crate::format::tqf::{TqfHeaderInfo, TqfReader};
use crate::ids::{ExpertId, LayerId};
use crate::memory::MemoryBroker;
use crate::model::qwen36::geometry::Qwen36Geometry;

/// Conversion implementation identity. It is part of the receipt and makes
/// upgrades observable rather than silently reinterpreting installed bytes.
pub const CONVERSION_FINGERPRINT_LABEL: &[u8] = b"tqf-qwen36-expert-superextents-v2";
pub const MODEL_FAMILY_LABEL: &[u8] = b"qwen3.6-35b-a3b";

#[derive(Debug, Clone)]
pub struct ConversionReport {
    pub path: PathBuf,
    pub extent_count: usize,
    pub verified_output_bytes: u64,
    pub source_sha256: [u8; 32],
    pub metadata_root_blake3: [u8; 32],
}

fn hex_digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(ContainerError::MalformedRecord {
            table: "source SHA-256 length",
        }
        .into());
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16).map_err(|_| {
            ContainerError::MalformedRecord {
                table: "source SHA-256 hex",
            }
        })?;
    }
    Ok(out)
}

pub fn canonical_header(source_sha256_hex: &str) -> Result<TqfHeaderInfo> {
    let family_hash = blake3::hash(MODEL_FAMILY_LABEL);
    let conversion_hash = blake3::hash(CONVERSION_FINGERPRINT_LABEL);
    let mut model_family_id = [0u8; 16];
    model_family_id.copy_from_slice(&family_hash.as_bytes()[..16]);
    Ok(TqfHeaderInfo {
        backend_id: 0,
        feature_bits: 0,
        model_family_id,
        source_sha256: hex_digest(source_sha256_hex)?,
        conversion_fingerprint: *conversion_hash.as_bytes(),
    })
}

/// Converts a verified source GGUF. `source_sha256_hex` must be the checksum
/// established by the source resolver, never a hash supplied by a caller who
/// has not verified the bytes. `AlreadyInstalled` is re-opened and validated
/// rather than treated as an unchecked cache hit.
pub fn convert_canonical_gguf(
    source_path: &Path,
    source_sha256_hex: &str,
    destination: &Path,
    broker: &MemoryBroker,
) -> Result<ConversionReport> {
    let source = gguf::open_with_broker(source_path, broker)?;
    let inventory = generate_inventory_from_file(&source)?;
    if inventory.len() != source.tensors.len() {
        return Err(ContainerError::MalformedRecord {
            table: "GGUF inventory/tensor count",
        }
        .into());
    }
    let header = canonical_header(source_sha256_hex)?;
    match ConversionTransaction::begin(destination, header.clone(), source_sha256_hex)? {
        BeginOutcome::AlreadyInstalled => {
            let existing = TqfReader::open_validated_with_broker(destination, broker)?;
            if existing.superblock.model_family_id != header.model_family_id
                || existing.superblock.source_sha256 != header.source_sha256
                || existing.superblock.conversion_fingerprint != header.conversion_fingerprint
            {
                return Err(ContainerError::MalformedRecord {
                    table: "existing TQF provenance does not match source",
                }
                .into());
            }
        }
        BeginOutcome::Transaction(mut transaction) => {
            let mut routed = HashMap::<
                (LayerId, TensorRole),
                (&gguf::TensorDescriptor, &TensorInventoryEntry),
            >::new();
            for (descriptor, inventory_entry) in source.tensors.iter().zip(inventory.iter()) {
                if matches!(
                    inventory_entry.logical_role,
                    TensorRole::RoutedExpertGate
                        | TensorRole::RoutedExpertUp
                        | TensorRole::RoutedExpertDown
                ) {
                    let layer = inventory_entry
                        .layer
                        .ok_or(ContainerError::MalformedRecord {
                            table: "routed expert without layer",
                        })?;
                    routed.insert(
                        (LayerId(layer), inventory_entry.logical_role),
                        (descriptor, inventory_entry),
                    );
                    continue;
                }
                if transaction.has_extent(&descriptor.name) {
                    continue;
                }
                let quant_layout_id = tqf_quant_layout_id(descriptor.ggml_type).ok_or(
                    ContainerError::MalformedRecord {
                        table: "unsupported canonical GGUF quantization layout",
                    },
                )?;
                let mut blocks = source.quant_block_reader(descriptor)?;
                let packed = repack_passthrough(&mut blocks, broker)?;
                transaction.write_extent(
                    inventory_entry.logical_role as u32,
                    &descriptor.name,
                    inventory_entry.layer.map(crate::ids::LayerId),
                    inventory_entry.tqf_section,
                    &descriptor.dims,
                    descriptor.ggml_type.ggml_id(),
                    quant_layout_id,
                    64,
                    &packed,
                )?;
            }
            if !routed.is_empty() {
                write_routed_expert_superextents(&source, &routed, &mut transaction, broker)?;
            }
            transaction.finish()?;
        }
    }
    let reader = TqfReader::open_validated_with_broker(destination, broker)?;
    Ok(ConversionReport {
        path: destination.to_path_buf(),
        extent_count: reader.superblock.extent_count as usize,
        verified_output_bytes: reader.superblock.file_length,
        source_sha256: reader.superblock.source_sha256,
        metadata_root_blake3: reader.superblock.metadata_root_blake3,
    })
}

/// Rewrites Qwen's three 3D routed-expert tensors into 40*256 contiguous,
/// checksummed superextents.  The payload remains lossless Q4_K; only
/// placement changes, so a cache miss can validate and retain one selected
/// expert without loading the other 255 planes.
fn write_routed_expert_superextents(
    source: &gguf::GgufFile,
    routed: &HashMap<(LayerId, TensorRole), (&gguf::TensorDescriptor, &TensorInventoryEntry)>,
    transaction: &mut ConversionTransaction,
    broker: &MemoryBroker,
) -> Result<()> {
    for layer_index in 0..Qwen36Geometry::NUM_LAYERS {
        let layer = LayerId(layer_index as u8);
        if (0..Qwen36Geometry::NUM_EXPERTS)
            .all(|expert| transaction.has_expert(layer, ExpertId(expert as u16)))
        {
            continue;
        }
        let (gate, gate_entry) = routed
            .get(&(layer, TensorRole::RoutedExpertGate))
            .copied()
            .ok_or(ContainerError::MalformedRecord {
                table: "missing routed expert gate tensor",
            })?;
        let (up, up_entry) = routed
            .get(&(layer, TensorRole::RoutedExpertUp))
            .copied()
            .ok_or(ContainerError::MalformedRecord {
                table: "missing routed expert up tensor",
            })?;
        let (down, down_entry) = routed
            .get(&(layer, TensorRole::RoutedExpertDown))
            .copied()
            .ok_or(ContainerError::MalformedRecord {
                table: "missing routed expert down tensor",
            })?;
        validate_routed_tensor(gate, gate_entry, TensorRole::RoutedExpertGate)?;
        validate_routed_tensor(up, up_entry, TensorRole::RoutedExpertUp)?;
        validate_routed_tensor(down, down_entry, TensorRole::RoutedExpertDown)?;
        if gate.ggml_type != up.ggml_type || gate.ggml_type != down.ggml_type {
            return Err(ContainerError::MalformedRecord {
                table: "routed expert matrix quantization mismatch",
            }
            .into());
        }
        let layout: u16 = tqf_quant_layout_id(gate.ggml_type)
            .ok_or(ContainerError::MalformedRecord {
                table: "routed expert quantization layout",
            })?
            .try_into()
            .map_err(|_| ContainerError::IntegerOverflow)?;

        // The existing lossless packer is tensor-granular. These three
        // buffers are conversion-only temporaries; the emitted TQF layout is
        // per expert and never holds this pool resident at runtime.
        let mut gate_reader = source.quant_block_reader(gate)?;
        let gate_bytes = repack_passthrough(&mut gate_reader, broker)?;
        let mut up_reader = source.quant_block_reader(up)?;
        let up_bytes = repack_passthrough(&mut up_reader, broker)?;
        let mut down_reader = source.quant_block_reader(down)?;
        let down_bytes = repack_passthrough(&mut down_reader, broker)?;
        let gate_plane = expert_plane_bytes(&gate_bytes, layer, "gate")?;
        let up_plane = expert_plane_bytes(&up_bytes, layer, "up")?;
        let down_plane = expert_plane_bytes(&down_bytes, layer, "down")?;
        for expert in 0..Qwen36Geometry::NUM_EXPERTS {
            let id = ExpertId(expert as u16);
            if transaction.has_expert(layer, id) {
                continue;
            }
            let gate_range = expert * gate_plane..(expert + 1) * gate_plane;
            let up_range = expert * up_plane..(expert + 1) * up_plane;
            let down_range = expert * down_plane..(expert + 1) * down_plane;
            transaction.write_expert_parts(
                layer,
                id,
                layout,
                &gate_bytes[gate_range],
                &up_bytes[up_range],
                &down_bytes[down_range],
            )?;
        }
    }
    Ok(())
}

fn validate_routed_tensor(
    descriptor: &gguf::TensorDescriptor,
    entry: &TensorInventoryEntry,
    role: TensorRole,
) -> Result<()> {
    let expected = match role {
        TensorRole::RoutedExpertGate | TensorRole::RoutedExpertUp => &[2048, 512, 256][..],
        TensorRole::RoutedExpertDown => &[512, 2048, 256][..],
        _ => unreachable!("only routed expert roles are passed here"),
    };
    if descriptor.dims != expected || descriptor.ggml_type != crate::format::quant::GgmlType::Q4K {
        return Err(ContainerError::MalformedRecord {
            table: "canonical routed expert shape or type",
        }
        .into());
    }
    if entry.logical_role != role {
        return Err(ContainerError::MalformedRecord {
            table: "routed expert inventory role",
        }
        .into());
    }
    Ok(())
}

fn expert_plane_bytes(bytes: &[u8], layer: LayerId, matrix: &'static str) -> Result<usize> {
    if bytes.len() % Qwen36Geometry::NUM_EXPERTS != 0 {
        return Err(ContainerError::MalformedRecord {
            table: "routed expert plane byte alignment",
        }
        .into());
    }
    let plane = bytes.len() / Qwen36Geometry::NUM_EXPERTS;
    if plane == 0 {
        return Err(ContainerError::MalformedRecord {
            table: "empty routed expert plane",
        }
        .into());
    }
    let _ = (layer, matrix); // retained for future per-layer diagnostics.
    Ok(plane)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::inventory::TensorRole;
    use crate::ids::Bytes;

    fn write_string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    /// One well-formed Q4_0 tensor is sufficient to test the complete
    /// importer transaction: GGUF parse -> inventory -> passthrough -> TQF
    /// finalization -> validated reader. Production shape validation happens
    /// against the real canonical inventory, not this tiny fixture.
    fn write_minimal_gguf(path: &Path) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut bytes, "token_embd.weight");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&32u64.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes()); // Q4_0
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned, 0);
        let payload = vec![0x11; 18];
        bytes.extend_from_slice(&payload);
        std::fs::write(path, &bytes).unwrap();
        payload
    }

    #[test]
    fn canonical_header_is_stable_and_binds_the_source_digest() {
        let header = canonical_header(&"ab".repeat(32)).unwrap();
        assert_eq!(header.source_sha256, [0xAB; 32]);
        assert_eq!(
            header.conversion_fingerprint,
            *blake3::hash(CONVERSION_FINGERPRINT_LABEL).as_bytes()
        );
        assert!(canonical_header("not-a-sha").is_err());
    }

    #[test]
    fn converts_a_classified_gguf_through_to_a_validated_tqf() {
        let directory = std::env::temp_dir().join(format!(
            "tqf-importer-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("fixture.gguf");
        let destination = directory.join("fixture.tqf");
        let payload = write_minimal_gguf(&source);
        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        let report =
            convert_canonical_gguf(&source, &"cd".repeat(32), &destination, &broker).unwrap();
        assert_eq!(report.extent_count, 1);
        assert!(report.verified_output_bytes > payload.len() as u64);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        let reader = TqfReader::open_validated(&destination).unwrap();
        let extent = reader
            .tensor(TensorRole::TokenEmbedding as u32, None)
            .unwrap();
        assert_eq!(reader.read_extent_bytes(extent).unwrap(), payload);
        std::fs::remove_dir_all(directory).ok();
    }
}
