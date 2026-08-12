//! Fixed-size on-disk records for the `.tqf` container (spec Part XIV
//! sections 122-124). The superblock/section/tensor-extent byte layouts
//! are given exactly by the spec (§121-123) and reproduced faithfully
//! here. The expert index/tile records (§124) and checksum-table entries
//! are given only as Rust field lists with no prescribed byte offsets, so
//! this module defines a concrete compact layout for them — documented
//! inline where invented rather than copied from the spec.
//!
//! Every record is hand-serialized field-by-field (spec §120: "`.tqf` ...
//! does not depend on Rust `repr(C)` memory layout for persistence") so
//! layout is controlled explicitly regardless of Rust struct field order
//! or padding.

use serde::{Deserialize, Serialize};

use crate::error::{ContainerError, Result};
use crate::ids::{ExpertId, LayerId, TileId};

pub const SECTION_RECORD_SIZE: usize = 64;
pub const TENSOR_EXTENT_RECORD_SIZE: usize = 96;
pub const EXPERT_INDEX_RECORD_SIZE: usize = 32;
pub const EXPERT_TILE_RECORD_SIZE: usize = 24;
pub const CHECKSUM_ENTRY_SIZE: usize = 40;

/// A tensor extent's `flags` field packs an optional layer ID into its low
/// byte (bit 0 = "has a layer", bits 8-15 = the `LayerId` value) — the
/// spec's §123 record has no dedicated layer field, and the taskbook's
/// `TqfReader::tensor(role/layer)` API needs a way to disambiguate
/// per-layer tensors sharing one logical `role_id` (e.g. `q_proj` across
/// 40 layers) from layer-independent ones (e.g. the embedding table).
const FLAG_HAS_LAYER: u32 = 0x1;

pub fn encode_layer_flag(layer: Option<LayerId>) -> u32 {
    match layer {
        Some(l) => FLAG_HAS_LAYER | ((l.0 as u32) << 8),
        None => 0,
    }
}

pub fn decode_layer_flag(flags: u32) -> Option<LayerId> {
    if flags & FLAG_HAS_LAYER != 0 {
        Some(LayerId(((flags >> 8) & 0xFF) as u8))
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum TqfSectionKind {
    Architecture = 1,
    Tokenizer = 2,
    StringTable = 3,
    ResidentCore = 4,
    Embeddings = 5,
    LmHead = 6,
    RoutedExperts = 7,
    Mtp = 8,
    VisionLink = 9,
    DuplicateLayouts = 10,
    Extents = 11,
    ExpertIndex = 12,
    Checksums = 13,
    Provenance = 14,
}

impl TqfSectionKind {
    pub fn from_u32(value: u32) -> Result<Self> {
        Ok(match value {
            1 => Self::Architecture,
            2 => Self::Tokenizer,
            3 => Self::StringTable,
            4 => Self::ResidentCore,
            5 => Self::Embeddings,
            6 => Self::LmHead,
            7 => Self::RoutedExperts,
            8 => Self::Mtp,
            9 => Self::VisionLink,
            10 => Self::DuplicateLayouts,
            11 => Self::Extents,
            12 => Self::ExpertIndex,
            13 => Self::Checksums,
            14 => Self::Provenance,
            other => return Err(ContainerError::UnknownSectionKind(other).into()),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExpertMatrix {
    GateUp = 0,
    Down = 1,
}

impl ExpertMatrix {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::GateUp),
            1 => Some(Self::Down),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SectionRecord {
    pub kind: u32,
    pub flags: u32,
    pub file_offset: u64,
    pub stored_bytes: u64,
    pub logical_bytes: u64,
    pub required_alignment: u32,
    pub checksum_index: u32,
    pub element_count: u64,
    pub aux_offset: u64,
    pub aux_count: u64,
}

impl SectionRecord {
    pub fn encode(&self) -> [u8; SECTION_RECORD_SIZE] {
        let mut buf = [0u8; SECTION_RECORD_SIZE];
        buf[0..4].copy_from_slice(&self.kind.to_le_bytes());
        buf[4..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.file_offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.stored_bytes.to_le_bytes());
        buf[24..32].copy_from_slice(&self.logical_bytes.to_le_bytes());
        buf[32..36].copy_from_slice(&self.required_alignment.to_le_bytes());
        buf[36..40].copy_from_slice(&self.checksum_index.to_le_bytes());
        buf[40..48].copy_from_slice(&self.element_count.to_le_bytes());
        buf[48..56].copy_from_slice(&self.aux_offset.to_le_bytes());
        buf[56..64].copy_from_slice(&self.aux_count.to_le_bytes());
        buf
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < SECTION_RECORD_SIZE {
            return None;
        }
        Some(Self {
            kind: u32::from_le_bytes(buf[0..4].try_into().ok()?),
            flags: u32::from_le_bytes(buf[4..8].try_into().ok()?),
            file_offset: u64::from_le_bytes(buf[8..16].try_into().ok()?),
            stored_bytes: u64::from_le_bytes(buf[16..24].try_into().ok()?),
            logical_bytes: u64::from_le_bytes(buf[24..32].try_into().ok()?),
            required_alignment: u32::from_le_bytes(buf[32..36].try_into().ok()?),
            checksum_index: u32::from_le_bytes(buf[36..40].try_into().ok()?),
            element_count: u64::from_le_bytes(buf[40..48].try_into().ok()?),
            aux_offset: u64::from_le_bytes(buf[48..56].try_into().ok()?),
            aux_count: u64::from_le_bytes(buf[56..64].try_into().ok()?),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorExtentRecord {
    pub role_id: u32,
    pub flags: u32,
    pub name_string_offset: u64,
    pub file_offset: u64,
    pub stored_bytes: u64,
    pub logical_elements: u64,
    pub rank: u32,
    pub quant_layout_id: u32,
    pub dtype_id: u32,
    pub required_alignment: u32,
    pub dims: [u64; 4],
    pub checksum_index: u32,
}

impl TensorExtentRecord {
    pub fn layer(&self) -> Option<LayerId> {
        decode_layer_flag(self.flags)
    }

    pub fn encode(&self) -> [u8; TENSOR_EXTENT_RECORD_SIZE] {
        let mut buf = [0u8; TENSOR_EXTENT_RECORD_SIZE];
        buf[0..4].copy_from_slice(&self.role_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.name_string_offset.to_le_bytes());
        buf[16..24].copy_from_slice(&self.file_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.stored_bytes.to_le_bytes());
        buf[32..40].copy_from_slice(&self.logical_elements.to_le_bytes());
        buf[40..44].copy_from_slice(&self.rank.to_le_bytes());
        buf[44..48].copy_from_slice(&self.quant_layout_id.to_le_bytes());
        buf[48..52].copy_from_slice(&self.dtype_id.to_le_bytes());
        buf[52..56].copy_from_slice(&self.required_alignment.to_le_bytes());
        buf[56..64].copy_from_slice(&self.dims[0].to_le_bytes());
        buf[64..72].copy_from_slice(&self.dims[1].to_le_bytes());
        buf[72..80].copy_from_slice(&self.dims[2].to_le_bytes());
        buf[80..88].copy_from_slice(&self.dims[3].to_le_bytes());
        buf[88..92].copy_from_slice(&self.checksum_index.to_le_bytes());
        buf
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < TENSOR_EXTENT_RECORD_SIZE {
            return None;
        }
        Some(Self {
            role_id: u32::from_le_bytes(buf[0..4].try_into().ok()?),
            flags: u32::from_le_bytes(buf[4..8].try_into().ok()?),
            name_string_offset: u64::from_le_bytes(buf[8..16].try_into().ok()?),
            file_offset: u64::from_le_bytes(buf[16..24].try_into().ok()?),
            stored_bytes: u64::from_le_bytes(buf[24..32].try_into().ok()?),
            logical_elements: u64::from_le_bytes(buf[32..40].try_into().ok()?),
            rank: u32::from_le_bytes(buf[40..44].try_into().ok()?),
            quant_layout_id: u32::from_le_bytes(buf[44..48].try_into().ok()?),
            dtype_id: u32::from_le_bytes(buf[48..52].try_into().ok()?),
            required_alignment: u32::from_le_bytes(buf[52..56].try_into().ok()?),
            dims: [
                u64::from_le_bytes(buf[56..64].try_into().ok()?),
                u64::from_le_bytes(buf[64..72].try_into().ok()?),
                u64::from_le_bytes(buf[72..80].try_into().ok()?),
                u64::from_le_bytes(buf[80..88].try_into().ok()?),
            ],
            checksum_index: u32::from_le_bytes(buf[88..92].try_into().ok()?),
        })
    }
}

/// Layout invented for Phase 6 (spec §124 gives only a Rust field sketch,
/// no byte offsets). `tile_first` indexes into the tile-record array that
/// immediately follows all `ExpertIndexRecord`s within the superblock's
/// `expert_index_offset`/`expert_index_bytes` blob — the boundary between
/// the two is recovered via the `ExpertIndex` section record's
/// `element_count` (see `writer.rs`/`reader.rs`), since the superblock
/// itself has no separate tile-table offset field.
#[derive(Debug, Clone)]
pub struct ExpertIndexRecord {
    pub layer: LayerId,
    pub expert: ExpertId,
    pub flags: u16,
    pub layout_id: u16,
    pub file_offset: u64,
    pub stored_bytes: u32,
    pub tile_first: u32,
    pub tile_count: u16,
    pub checksum_index: u32,
}

impl ExpertIndexRecord {
    pub fn encode(&self) -> [u8; EXPERT_INDEX_RECORD_SIZE] {
        let mut buf = [0u8; EXPERT_INDEX_RECORD_SIZE];
        buf[0] = self.layer.0;
        buf[2..4].copy_from_slice(&self.expert.0.to_le_bytes());
        buf[4..6].copy_from_slice(&self.flags.to_le_bytes());
        buf[6..8].copy_from_slice(&self.layout_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.file_offset.to_le_bytes());
        buf[16..20].copy_from_slice(&self.stored_bytes.to_le_bytes());
        buf[20..24].copy_from_slice(&self.tile_first.to_le_bytes());
        buf[24..26].copy_from_slice(&self.tile_count.to_le_bytes());
        buf[28..32].copy_from_slice(&self.checksum_index.to_le_bytes());
        buf
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < EXPERT_INDEX_RECORD_SIZE {
            return None;
        }
        Some(Self {
            layer: LayerId(buf[0]),
            expert: ExpertId(u16::from_le_bytes(buf[2..4].try_into().ok()?)),
            flags: u16::from_le_bytes(buf[4..6].try_into().ok()?),
            layout_id: u16::from_le_bytes(buf[6..8].try_into().ok()?),
            file_offset: u64::from_le_bytes(buf[8..16].try_into().ok()?),
            stored_bytes: u32::from_le_bytes(buf[16..20].try_into().ok()?),
            tile_first: u32::from_le_bytes(buf[20..24].try_into().ok()?),
            tile_count: u16::from_le_bytes(buf[24..26].try_into().ok()?),
            checksum_index: u32::from_le_bytes(buf[28..32].try_into().ok()?),
        })
    }
}

/// REFERENCE BASELINE per §124: this phase writes one whole-region tile
/// per matrix (`GateUp`, `Down`) rather than splitting into 128-neuron
/// sub-tiles — "Phase 6 only needs the metadata shape to exist, not
/// multiple tile widths implemented."
#[derive(Debug, Clone)]
pub struct ExpertTileRecord {
    pub matrix: ExpertMatrix,
    pub tile_id: TileId,
    pub neuron_start: u16,
    pub neuron_count: u16,
    pub relative_offset: u32,
    pub stored_bytes: u32,
    pub quant_layout_id: u16,
    pub flags: u16,
}

impl ExpertTileRecord {
    pub fn encode(&self) -> [u8; EXPERT_TILE_RECORD_SIZE] {
        let mut buf = [0u8; EXPERT_TILE_RECORD_SIZE];
        buf[0] = self.matrix as u8;
        buf[2..4].copy_from_slice(&self.tile_id.0.to_le_bytes());
        buf[4..6].copy_from_slice(&self.neuron_start.to_le_bytes());
        buf[6..8].copy_from_slice(&self.neuron_count.to_le_bytes());
        buf[8..12].copy_from_slice(&self.relative_offset.to_le_bytes());
        buf[12..16].copy_from_slice(&self.stored_bytes.to_le_bytes());
        buf[16..18].copy_from_slice(&self.quant_layout_id.to_le_bytes());
        buf[18..20].copy_from_slice(&self.flags.to_le_bytes());
        buf
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < EXPERT_TILE_RECORD_SIZE {
            return None;
        }
        Some(Self {
            matrix: ExpertMatrix::from_u8(buf[0])?,
            tile_id: TileId(u16::from_le_bytes(buf[2..4].try_into().ok()?)),
            neuron_start: u16::from_le_bytes(buf[4..6].try_into().ok()?),
            neuron_count: u16::from_le_bytes(buf[6..8].try_into().ok()?),
            relative_offset: u32::from_le_bytes(buf[8..12].try_into().ok()?),
            stored_bytes: u32::from_le_bytes(buf[12..16].try_into().ok()?),
            quant_layout_id: u16::from_le_bytes(buf[16..18].try_into().ok()?),
            flags: u16::from_le_bytes(buf[18..20].try_into().ok()?),
        })
    }
}

/// Layout invented for Phase 6 — the spec names a checksum table
/// (superblock `checksum_table_offset`/`checksum_table_bytes`, §121) and
/// requires BLAKE3-256 for internal integrity (§120) but gives no entry
/// byte layout.
#[derive(Debug, Clone)]
pub struct ChecksumEntry {
    pub hash_kind: u32,
    pub digest: [u8; 32],
}

impl ChecksumEntry {
    pub const BLAKE3_256: u32 = 1;

    pub fn encode(&self) -> [u8; CHECKSUM_ENTRY_SIZE] {
        let mut buf = [0u8; CHECKSUM_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&self.hash_kind.to_le_bytes());
        buf[8..40].copy_from_slice(&self.digest);
        buf
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < CHECKSUM_ENTRY_SIZE {
            return None;
        }
        Some(Self {
            hash_kind: u32::from_le_bytes(buf[0..4].try_into().ok()?),
            digest: buf[8..40].try_into().ok()?,
        })
    }
}

/// String table: a flat concatenation of `(u32 length, utf8 bytes)`
/// entries. `name_string_offset` in a `TensorExtentRecord` is a byte
/// offset relative to the string table's own start (not an absolute file
/// offset) — compact, and the table can be relocated without rewriting
/// extent records.
pub fn append_string(table: &mut Vec<u8>, s: &str) -> u64 {
    let offset = table.len() as u64;
    table.extend_from_slice(&(s.len() as u32).to_le_bytes());
    table.extend_from_slice(s.as_bytes());
    offset
}

pub fn read_string_at(table: &[u8], offset: u64) -> Result<String> {
    let table_len = table.len() as u64;
    let bounds_err = |offset: u64, len: u64| ContainerError::StringTableOutOfBounds {
        offset,
        len,
        table_len,
    };

    let offset_usize: usize = offset.try_into().map_err(|_| bounds_err(offset, 4))?;
    let len_bytes = table
        .get(offset_usize..)
        .and_then(|s| s.get(..4))
        .ok_or_else(|| bounds_err(offset, 4))?;
    let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as u64;

    let start = offset
        .checked_add(4)
        .ok_or_else(|| bounds_err(offset, len))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| bounds_err(start, len))?;
    let start_usize: usize = start.try_into().map_err(|_| bounds_err(start, len))?;
    let end_usize: usize = end.try_into().map_err(|_| bounds_err(start, len))?;
    let bytes = table
        .get(start_usize..end_usize)
        .ok_or_else(|| bounds_err(start, len))?;

    String::from_utf8(bytes.to_vec()).map_err(|_| {
        ContainerError::MalformedRecord {
            table: "string table",
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_record_round_trips() {
        let rec = SectionRecord {
            kind: 4,
            flags: 0,
            file_offset: 4096,
            stored_bytes: 1024,
            logical_bytes: 1024,
            required_alignment: 64,
            checksum_index: 2,
            element_count: 3,
            aux_offset: 0,
            aux_count: 0,
        };
        let encoded = rec.encode();
        assert_eq!(encoded.len(), SECTION_RECORD_SIZE);
        let decoded = SectionRecord::decode(&encoded).unwrap();
        assert_eq!(decoded.file_offset, 4096);
        assert_eq!(decoded.element_count, 3);
    }

    #[test]
    fn tensor_extent_record_round_trips_with_layer_flag() {
        let rec = TensorExtentRecord {
            role_id: 7,
            flags: encode_layer_flag(Some(LayerId(12))),
            name_string_offset: 100,
            file_offset: 8192,
            stored_bytes: 2048,
            logical_elements: 4096,
            rank: 2,
            quant_layout_id: 12,
            dtype_id: 12,
            required_alignment: 64,
            dims: [2048, 2, 0, 0],
            checksum_index: 1,
        };
        let encoded = rec.encode();
        assert_eq!(encoded.len(), TENSOR_EXTENT_RECORD_SIZE);
        let decoded = TensorExtentRecord::decode(&encoded).unwrap();
        assert_eq!(decoded.layer(), Some(LayerId(12)));
        assert_eq!(decoded.dims, [2048, 2, 0, 0]);
    }

    #[test]
    fn layer_flag_none_round_trips() {
        assert_eq!(decode_layer_flag(encode_layer_flag(None)), None);
        assert_eq!(
            decode_layer_flag(encode_layer_flag(Some(LayerId(39)))),
            Some(LayerId(39))
        );
    }

    #[test]
    fn expert_index_and_tile_records_round_trip() {
        let idx = ExpertIndexRecord {
            layer: LayerId(5),
            expert: ExpertId(200),
            flags: 0,
            layout_id: 1,
            file_offset: 4096 * 10,
            stored_bytes: 4096,
            tile_first: 3,
            tile_count: 2,
            checksum_index: 9,
        };
        let decoded = ExpertIndexRecord::decode(&idx.encode()).unwrap();
        assert_eq!(decoded.layer, LayerId(5));
        assert_eq!(decoded.expert, ExpertId(200));
        assert_eq!(decoded.file_offset, 4096 * 10);

        let tile = ExpertTileRecord {
            matrix: ExpertMatrix::Down,
            tile_id: TileId(3),
            neuron_start: 0,
            neuron_count: 512,
            relative_offset: 1024,
            stored_bytes: 512,
            quant_layout_id: 2,
            flags: 0,
        };
        let decoded_tile = ExpertTileRecord::decode(&tile.encode()).unwrap();
        assert_eq!(decoded_tile.matrix, ExpertMatrix::Down);
        assert_eq!(decoded_tile.neuron_count, 512);
    }

    #[test]
    fn string_table_round_trips_multiple_entries() {
        let mut table = Vec::new();
        let off_a = append_string(&mut table, "gate");
        let off_b = append_string(&mut table, "up_proj");
        assert_eq!(read_string_at(&table, off_a).unwrap(), "gate");
        assert_eq!(read_string_at(&table, off_b).unwrap(), "up_proj");
    }

    #[test]
    fn string_table_out_of_bounds_is_an_error() {
        let table = vec![0u8; 3];
        assert!(read_string_at(&table, 0).is_err());
    }

    #[test]
    fn section_kind_rejects_unknown_value() {
        assert!(TqfSectionKind::from_u32(999).is_err());
        assert_eq!(
            TqfSectionKind::from_u32(7).unwrap(),
            TqfSectionKind::RoutedExperts
        );
    }
}
