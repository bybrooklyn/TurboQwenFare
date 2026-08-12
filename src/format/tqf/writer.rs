//! `TqfWriter`: builds a `.tqf` v1 container (spec §278, phase 6). Writes
//! to a `.partial` sibling file and only becomes the real file on success
//! (spec §115 invariant #9: temp + fsync + atomic-rename). The full
//! journal/resume machinery in spec §126 is Phase 8's job (streaming
//! conversion); this phase's `create_partial`/`write_extent`/`commit` is a
//! simpler, non-resumable slice of that state machine sufficient for a
//! synthetic-fixture roundtrip.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ContainerError, Result};
use crate::ids::{ExpertId, LayerId, TileId};

use super::records::{
    append_string, ChecksumEntry, ExpertIndexRecord, ExpertMatrix, ExpertTileRecord, SectionRecord,
    TensorExtentRecord, TqfSectionKind, CHECKSUM_ENTRY_SIZE, EXPERT_INDEX_RECORD_SIZE,
    EXPERT_TILE_RECORD_SIZE, SECTION_RECORD_SIZE, TENSOR_EXTENT_RECORD_SIZE,
};
use super::superblock::{Superblock, FORMAT_MAJOR, FORMAT_MINOR, SUPERBLOCK_SIZE};

/// Whole routed-expert superextents are 4096-byte aligned per spec §120;
/// everything else defaults to 64-byte metadata-table alignment.
const EXPERT_SUPEREXTENT_ALIGNMENT: u64 = 4096;
const METADATA_TABLE_ALIGNMENT: u64 = 64;

pub struct TqfHeaderInfo {
    pub backend_id: u32,
    pub feature_bits: u64,
    pub model_family_id: [u8; 16],
    pub source_sha256: [u8; 32],
    pub conversion_fingerprint: [u8; 32],
}

struct PendingExtent {
    role_id: u32,
    name: String,
    section_kind: TqfSectionKind,
    file_offset: u64,
    stored_bytes: u64,
    logical_elements: u64,
    rank: u32,
    quant_layout_id: u32,
    dtype_id: u32,
    required_alignment: u32,
    dims: [u64; 4],
    layer: Option<LayerId>,
    digest: [u8; 32],
}

struct PendingExpert {
    layer: LayerId,
    expert: ExpertId,
    file_offset: u64,
    stored_bytes: u32,
    tiles: Vec<ExpertTileRecord>,
    digest: [u8; 32],
}

/// Plain-data, journal-serializable snapshot of a `PendingExtent` (spec
/// §126, phase 8): enough to reconstruct writer state on resume *without*
/// re-reading payload bytes back off disk — the journal, not the file, is
/// the source of truth for what has been verified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveredExtent {
    pub role_id: u32,
    pub name: String,
    pub section_kind: u32,
    pub file_offset: u64,
    pub stored_bytes: u64,
    pub logical_elements: u64,
    pub rank: u32,
    pub quant_layout_id: u32,
    pub dtype_id: u32,
    pub required_alignment: u32,
    pub dims: [u64; 4],
    pub layer: Option<u8>,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveredTile {
    pub matrix: u8,
    pub tile_id: u16,
    pub neuron_start: u16,
    pub neuron_count: u16,
    pub relative_offset: u32,
    pub stored_bytes: u32,
    pub quant_layout_id: u16,
    pub flags: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveredExpert {
    pub layer: u8,
    pub expert: u16,
    pub file_offset: u64,
    pub stored_bytes: u32,
    pub tiles: Vec<RecoveredTile>,
    pub digest: [u8; 32],
}

impl PendingExtent {
    fn to_recovered(&self) -> RecoveredExtent {
        RecoveredExtent {
            role_id: self.role_id,
            name: self.name.clone(),
            section_kind: self.section_kind as u32,
            file_offset: self.file_offset,
            stored_bytes: self.stored_bytes,
            logical_elements: self.logical_elements,
            rank: self.rank,
            quant_layout_id: self.quant_layout_id,
            dtype_id: self.dtype_id,
            required_alignment: self.required_alignment,
            dims: self.dims,
            layer: self.layer.map(|l| l.0),
            digest: self.digest,
        }
    }

    fn from_recovered(r: &RecoveredExtent) -> Result<Self> {
        Ok(Self {
            role_id: r.role_id,
            name: r.name.clone(),
            section_kind: TqfSectionKind::from_u32(r.section_kind)?,
            file_offset: r.file_offset,
            stored_bytes: r.stored_bytes,
            logical_elements: r.logical_elements,
            rank: r.rank,
            quant_layout_id: r.quant_layout_id,
            dtype_id: r.dtype_id,
            required_alignment: r.required_alignment,
            dims: r.dims,
            layer: r.layer.map(LayerId),
            digest: r.digest,
        })
    }
}

impl PendingExpert {
    fn to_recovered(&self) -> RecoveredExpert {
        RecoveredExpert {
            layer: self.layer.0,
            expert: self.expert.0,
            file_offset: self.file_offset,
            stored_bytes: self.stored_bytes,
            tiles: self
                .tiles
                .iter()
                .map(|t| RecoveredTile {
                    matrix: t.matrix as u8,
                    tile_id: t.tile_id.0,
                    neuron_start: t.neuron_start,
                    neuron_count: t.neuron_count,
                    relative_offset: t.relative_offset,
                    stored_bytes: t.stored_bytes,
                    quant_layout_id: t.quant_layout_id,
                    flags: t.flags,
                })
                .collect(),
            digest: self.digest,
        }
    }

    fn from_recovered(r: &RecoveredExpert) -> Result<Self> {
        let tiles = r
            .tiles
            .iter()
            .map(|t| {
                Ok(ExpertTileRecord {
                    matrix: ExpertMatrix::from_u8(t.matrix).ok_or(
                        ContainerError::MalformedRecord {
                            table: "conversion journal expert tile",
                        },
                    )?,
                    tile_id: TileId(t.tile_id),
                    neuron_start: t.neuron_start,
                    neuron_count: t.neuron_count,
                    relative_offset: t.relative_offset,
                    stored_bytes: t.stored_bytes,
                    quant_layout_id: t.quant_layout_id,
                    flags: t.flags,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layer: LayerId(r.layer),
            expert: ExpertId(r.expert),
            file_offset: r.file_offset,
            stored_bytes: r.stored_bytes,
            tiles,
            digest: r.digest,
        })
    }
}

pub struct TqfWriter {
    partial_path: PathBuf,
    final_path: PathBuf,
    file: File,
    next_offset: u64,
    header: TqfHeaderInfo,
    extents: Vec<PendingExtent>,
    experts: Vec<PendingExpert>,
}

impl TqfWriter {
    /// Creates the `.partial` sibling file and reserves the 4096-byte
    /// superblock region (written for real only in `commit`).
    pub fn create_partial(final_path: impl Into<PathBuf>, header: TqfHeaderInfo) -> Result<Self> {
        let final_path = final_path.into();
        let partial_path = partial_path_for(&final_path);

        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&partial_path)?;
        file.write_all(&[0u8; SUPERBLOCK_SIZE])?;

        Ok(Self {
            partial_path,
            final_path,
            file,
            next_offset: SUPERBLOCK_SIZE as u64,
            header,
            extents: Vec::new(),
            experts: Vec::new(),
        })
    }

    /// Resumes writing an existing `.partial` file using journal-recovered
    /// extent/expert metadata (spec §126 phase 8: "A completed extent is
    /// never trusted solely because its bytes exist; its journal hash must
    /// validate" — so `next_offset` and the in-memory tables are derived
    /// entirely from `recovered_extents`/`recovered_experts`, never from
    /// the `.partial` file's own current length). Callers own validating
    /// that the journal actually belongs to this `final_path`/header
    /// before calling in.
    pub fn resume_partial(
        final_path: impl Into<PathBuf>,
        header: TqfHeaderInfo,
        recovered_extents: &[RecoveredExtent],
        recovered_experts: &[RecoveredExpert],
    ) -> Result<Self> {
        let final_path = final_path.into();
        let partial_path = partial_path_for(&final_path);

        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&partial_path)?;

        let extents = recovered_extents
            .iter()
            .map(PendingExtent::from_recovered)
            .collect::<Result<Vec<_>>>()?;
        let experts = recovered_experts
            .iter()
            .map(PendingExpert::from_recovered)
            .collect::<Result<Vec<_>>>()?;

        let next_offset = extents
            .iter()
            .map(|e| e.file_offset + e.stored_bytes)
            .chain(
                experts
                    .iter()
                    .map(|e| e.file_offset + e.stored_bytes as u64),
            )
            .max()
            .unwrap_or(SUPERBLOCK_SIZE as u64);

        // `write_extent`/`write_expert` assume the OS file cursor already
        // sits at `next_offset` (they write sequentially via `write_all`
        // without seeking first) — a freshly `open()`ed file starts its
        // cursor at 0, so resume must seek explicitly before any further
        // writes, or they'd silently overwrite already-verified bytes.
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(next_offset))?;

        Ok(Self {
            partial_path,
            final_path,
            file,
            next_offset,
            header,
            extents,
            experts,
        })
    }

    pub fn has_extent(&self, name: &str) -> bool {
        self.extents.iter().any(|e| e.name == name)
    }

    pub fn has_expert(&self, layer: LayerId, expert: ExpertId) -> bool {
        self.experts
            .iter()
            .any(|e| e.layer == layer && e.expert == expert)
    }

    /// Appends one tensor's bytes at the next `required_alignment`-aligned
    /// offset and records its extent. `layer` is `None` for layer-
    /// independent tensors (e.g. the embedding table); `role_id` is a
    /// caller-defined logical-role tag shared across layers.
    #[allow(clippy::too_many_arguments)]
    pub fn write_extent(
        &mut self,
        role_id: u32,
        name: &str,
        layer: Option<LayerId>,
        section_kind: TqfSectionKind,
        dims: &[u64],
        dtype_id: u32,
        quant_layout_id: u32,
        required_alignment: u32,
        data: &[u8],
    ) -> Result<RecoveredExtent> {
        if self.extents.iter().any(|e| e.name == name) {
            return Err(ContainerError::DuplicateExtent(name.to_string()).into());
        }
        if dims.is_empty() || dims.len() > 4 {
            return Err(ContainerError::UnsupportedRank(dims.len() as u32).into());
        }
        let logical_elements = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or(ContainerError::IntegerOverflow)?;
        let mut fixed_dims = [0u64; 4];
        fixed_dims[..dims.len()].copy_from_slice(dims);

        let aligned_offset = align_up(self.next_offset, required_alignment.max(1) as u64)?;
        pad_to(&mut self.file, self.next_offset, aligned_offset)?;
        self.file.write_all(data)?;

        self.extents.push(PendingExtent {
            role_id,
            name: name.to_string(),
            section_kind,
            file_offset: aligned_offset,
            stored_bytes: data.len() as u64,
            logical_elements,
            rank: dims.len() as u32,
            quant_layout_id,
            dtype_id,
            required_alignment,
            dims: fixed_dims,
            layer,
            digest: *blake3::hash(data).as_bytes(),
        });
        self.next_offset = aligned_offset + data.len() as u64;
        Ok(self.extents.last().expect("just pushed").to_recovered())
    }

    /// Writes one routed expert as a single 4096-aligned superextent
    /// (`gate_up` bytes followed immediately by `down` bytes), with two
    /// whole-region tile records (REFERENCE BASELINE per §124 — no
    /// 128-neuron sub-tiling in this phase).
    pub fn write_expert(
        &mut self,
        layer: LayerId,
        expert: ExpertId,
        quant_layout_id: u16,
        gate_up: &[u8],
        down: &[u8],
    ) -> Result<RecoveredExpert> {
        if self
            .experts
            .iter()
            .any(|e| e.layer == layer && e.expert == expert)
        {
            return Err(ContainerError::DuplicateExtent(format!(
                "expert layer={} expert={}",
                layer.0, expert.0
            ))
            .into());
        }

        let aligned_offset = align_up(self.next_offset, EXPERT_SUPEREXTENT_ALIGNMENT)?;
        pad_to(&mut self.file, self.next_offset, aligned_offset)?;
        self.file.write_all(gate_up)?;
        self.file.write_all(down)?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(gate_up);
        hasher.update(down);
        let stored_bytes = (gate_up.len() + down.len()) as u32;

        let tiles = vec![
            ExpertTileRecord {
                matrix: ExpertMatrix::GateUp,
                tile_id: TileId(0),
                neuron_start: 0,
                neuron_count: 0,
                relative_offset: 0,
                stored_bytes: gate_up.len() as u32,
                quant_layout_id,
                flags: 0,
            },
            ExpertTileRecord {
                matrix: ExpertMatrix::Down,
                tile_id: TileId(1),
                neuron_start: 0,
                neuron_count: 0,
                relative_offset: gate_up.len() as u32,
                stored_bytes: down.len() as u32,
                quant_layout_id,
                flags: 0,
            },
        ];

        self.experts.push(PendingExpert {
            layer,
            expert,
            file_offset: aligned_offset,
            stored_bytes,
            tiles,
            digest: *hasher.finalize().as_bytes(),
        });
        self.next_offset = aligned_offset + stored_bytes as u64;
        Ok(self.experts.last().expect("just pushed").to_recovered())
    }

    /// Finalizes all metadata tables, writes the real superblock, fsyncs,
    /// and atomically renames the `.partial` file to its final name (spec
    /// §115 invariant #9).
    pub fn commit(mut self) -> Result<()> {
        let mut string_table = Vec::new();
        let name_offsets: Vec<u64> = self
            .extents
            .iter()
            .map(|e| append_string(&mut string_table, &e.name))
            .collect();

        let mut checksums = Vec::new();
        let extent_checksum_idx: Vec<u32> = self
            .extents
            .iter()
            .map(|e| {
                let idx = checksums.len() as u32;
                checksums.push(ChecksumEntry {
                    hash_kind: ChecksumEntry::BLAKE3_256,
                    digest: e.digest,
                });
                idx
            })
            .collect();
        let expert_checksum_idx: Vec<u32> = self
            .experts
            .iter()
            .map(|e| {
                let idx = checksums.len() as u32;
                checksums.push(ChecksumEntry {
                    hash_kind: ChecksumEntry::BLAKE3_256,
                    digest: e.digest,
                });
                idx
            })
            .collect();

        let mut extent_table = Vec::with_capacity(self.extents.len() * TENSOR_EXTENT_RECORD_SIZE);
        for (i, e) in self.extents.iter().enumerate() {
            let rec = TensorExtentRecord {
                role_id: e.role_id,
                flags: super::records::encode_layer_flag(e.layer),
                name_string_offset: name_offsets[i],
                file_offset: e.file_offset,
                stored_bytes: e.stored_bytes,
                logical_elements: e.logical_elements,
                rank: e.rank,
                quant_layout_id: e.quant_layout_id,
                dtype_id: e.dtype_id,
                required_alignment: e.required_alignment,
                dims: e.dims,
                checksum_index: extent_checksum_idx[i],
            };
            extent_table.extend_from_slice(&rec.encode());
        }

        // Expert index blob = [ExpertIndexRecord; N] followed by
        // [ExpertTileRecord; M]. The boundary (N) is recovered on read via
        // the `ExpertIndex` section record's `element_count` — the
        // superblock itself has no separate tile-table offset field.
        let mut expert_index_blob = Vec::new();
        let mut tile_blob = Vec::new();
        for (i, ex) in self.experts.iter().enumerate() {
            let tile_first = (tile_blob.len() / EXPERT_TILE_RECORD_SIZE) as u32;
            for t in &ex.tiles {
                tile_blob.extend_from_slice(&t.encode());
            }
            let rec = ExpertIndexRecord {
                layer: ex.layer,
                expert: ex.expert,
                flags: 0,
                layout_id: 0,
                file_offset: ex.file_offset,
                stored_bytes: ex.stored_bytes,
                tile_first,
                tile_count: ex.tiles.len() as u16,
                checksum_index: expert_checksum_idx[i],
            };
            expert_index_blob.extend_from_slice(&rec.encode());
        }
        expert_index_blob.extend_from_slice(&tile_blob);

        let mut checksum_table = Vec::with_capacity(checksums.len() * CHECKSUM_ENTRY_SIZE);
        for c in &checksums {
            checksum_table.extend_from_slice(&c.encode());
        }

        // Lay out the metadata-table blobs after all extent/expert
        // payload data, each 64-byte aligned (spec §120).
        let mut cursor = align_up(self.next_offset, METADATA_TABLE_ALIGNMENT)?;
        let string_table_offset = cursor;
        pad_to(&mut self.file, self.next_offset, string_table_offset)?;
        self.file.write_all(&string_table)?;
        cursor = string_table_offset + string_table.len() as u64;

        cursor = align_up(cursor, METADATA_TABLE_ALIGNMENT)?;
        let extent_table_offset = cursor;
        pad_to(
            &mut self.file,
            string_table_offset + string_table.len() as u64,
            extent_table_offset,
        )?;
        self.file.write_all(&extent_table)?;
        cursor = extent_table_offset + extent_table.len() as u64;

        cursor = align_up(cursor, METADATA_TABLE_ALIGNMENT)?;
        let expert_index_offset = cursor;
        pad_to(
            &mut self.file,
            extent_table_offset + extent_table.len() as u64,
            expert_index_offset,
        )?;
        self.file.write_all(&expert_index_blob)?;
        cursor = expert_index_offset + expert_index_blob.len() as u64;

        cursor = align_up(cursor, METADATA_TABLE_ALIGNMENT)?;
        let checksum_table_offset = cursor;
        pad_to(
            &mut self.file,
            expert_index_offset + expert_index_blob.len() as u64,
            checksum_table_offset,
        )?;
        self.file.write_all(&checksum_table)?;
        cursor = checksum_table_offset + checksum_table.len() as u64;

        cursor = align_up(cursor, METADATA_TABLE_ALIGNMENT)?;
        let section_table_offset = cursor;

        let sections = build_sections(
            &self.extents,
            &self.experts,
            string_table_offset,
            string_table.len() as u64,
            extent_table_offset,
            expert_index_offset,
            expert_index_blob.len() as u64,
            checksum_table_offset,
            checksum_table.len() as u64,
        );
        let mut section_table_bytes = Vec::with_capacity(sections.len() * SECTION_RECORD_SIZE);
        for s in &sections {
            section_table_bytes.extend_from_slice(&s.encode());
        }
        pad_to(
            &mut self.file,
            checksum_table_offset + checksum_table.len() as u64,
            section_table_offset,
        )?;
        self.file.write_all(&section_table_bytes)?;

        let file_length = section_table_offset + section_table_bytes.len() as u64;

        // Metadata root hash over every table in this fixed order (spec
        // §121 reader-validation step 6).
        let mut root_hasher = blake3::Hasher::new();
        root_hasher.update(&section_table_bytes);
        root_hasher.update(&extent_table);
        root_hasher.update(&string_table);
        root_hasher.update(&expert_index_blob);
        root_hasher.update(&checksum_table);
        let metadata_root_blake3 = *root_hasher.finalize().as_bytes();

        let superblock = Superblock {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            backend_id: self.header.backend_id,
            feature_bits: self.header.feature_bits,
            model_family_id: self.header.model_family_id,
            source_sha256: self.header.source_sha256,
            conversion_fingerprint: self.header.conversion_fingerprint,
            file_length,
            section_table_offset,
            section_count: sections.len() as u32,
            section_record_bytes: SECTION_RECORD_SIZE as u32,
            extent_table_offset,
            extent_count: self.extents.len() as u64,
            string_table_offset,
            string_table_bytes: string_table.len() as u64,
            architecture_record_offset: 0,
            tokenizer_record_offset: 0,
            expert_index_offset,
            expert_index_bytes: expert_index_blob.len() as u64,
            checksum_table_offset,
            checksum_table_bytes: checksum_table.len() as u64,
            metadata_root_blake3,
            creation_unix: unix_now(),
            min_reader_capability_bits: 0,
        };

        use std::io::{Seek, SeekFrom};
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&superblock.encode())?;
        self.file.sync_all()?;
        drop(self.file);

        std::fs::rename(&self.partial_path, &self.final_path)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn build_sections(
    extents: &[PendingExtent],
    experts: &[PendingExpert],
    string_table_offset: u64,
    string_table_bytes: u64,
    extent_table_offset: u64,
    expert_index_offset: u64,
    expert_index_bytes: u64,
    checksum_table_offset: u64,
    checksum_table_bytes: u64,
) -> Vec<SectionRecord> {
    let mut sections = Vec::new();

    let mut kinds: Vec<TqfSectionKind> = extents.iter().map(|e| e.section_kind).collect();
    kinds.sort_by_key(|k| *k as u32);
    kinds.dedup_by_key(|k| *k as u32);
    for kind in kinds {
        let group: Vec<&PendingExtent> =
            extents.iter().filter(|e| e.section_kind == kind).collect();
        let start = group.iter().map(|e| e.file_offset).min().unwrap();
        let end = group
            .iter()
            .map(|e| e.file_offset + e.stored_bytes)
            .max()
            .unwrap();
        sections.push(SectionRecord {
            kind: kind as u32,
            flags: 0,
            file_offset: start,
            stored_bytes: end - start,
            logical_bytes: end - start,
            required_alignment: 1,
            checksum_index: 0,
            element_count: group.len() as u64,
            aux_offset: 0,
            aux_count: 0,
        });
    }

    if !experts.is_empty() {
        let start = experts.iter().map(|e| e.file_offset).min().unwrap();
        let end = experts
            .iter()
            .map(|e| e.file_offset + e.stored_bytes as u64)
            .max()
            .unwrap();
        sections.push(SectionRecord {
            kind: TqfSectionKind::RoutedExperts as u32,
            flags: 0,
            file_offset: start,
            stored_bytes: end - start,
            logical_bytes: end - start,
            required_alignment: EXPERT_SUPEREXTENT_ALIGNMENT as u32,
            checksum_index: 0,
            element_count: experts.len() as u64,
            aux_offset: 0,
            aux_count: 0,
        });
        // `element_count` here is load-bearing: it's how the reader
        // recovers the ExpertIndexRecord/ExpertTileRecord boundary inside
        // the combined `expert_index_offset`/`expert_index_bytes` blob.
        sections.push(SectionRecord {
            kind: TqfSectionKind::ExpertIndex as u32,
            flags: 0,
            file_offset: expert_index_offset,
            stored_bytes: expert_index_bytes,
            logical_bytes: expert_index_bytes,
            required_alignment: METADATA_TABLE_ALIGNMENT as u32,
            checksum_index: 0,
            element_count: experts.len() as u64,
            aux_offset: 0,
            aux_count: 0,
        });
    }

    sections.push(SectionRecord {
        kind: TqfSectionKind::StringTable as u32,
        flags: 0,
        file_offset: string_table_offset,
        stored_bytes: string_table_bytes,
        logical_bytes: string_table_bytes,
        required_alignment: METADATA_TABLE_ALIGNMENT as u32,
        checksum_index: 0,
        element_count: 0,
        aux_offset: 0,
        aux_count: 0,
    });
    sections.push(SectionRecord {
        kind: TqfSectionKind::Extents as u32,
        flags: 0,
        file_offset: extent_table_offset,
        stored_bytes: (extents.len() * TENSOR_EXTENT_RECORD_SIZE) as u64,
        logical_bytes: (extents.len() * TENSOR_EXTENT_RECORD_SIZE) as u64,
        required_alignment: METADATA_TABLE_ALIGNMENT as u32,
        checksum_index: 0,
        element_count: extents.len() as u64,
        aux_offset: 0,
        aux_count: 0,
    });
    sections.push(SectionRecord {
        kind: TqfSectionKind::Checksums as u32,
        flags: 0,
        file_offset: checksum_table_offset,
        stored_bytes: checksum_table_bytes,
        logical_bytes: checksum_table_bytes,
        required_alignment: METADATA_TABLE_ALIGNMENT as u32,
        checksum_index: 0,
        element_count: (checksum_table_bytes / CHECKSUM_ENTRY_SIZE as u64),
        aux_offset: 0,
        aux_count: 0,
    });

    sections
}

fn partial_path_for(final_path: &Path) -> PathBuf {
    let mut partial_name = final_path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    partial_name.push(".partial");
    final_path.with_file_name(partial_name)
}

fn pad_to(file: &mut File, current: u64, target: u64) -> Result<()> {
    if target > current {
        let pad = vec![0u8; (target - current) as usize];
        file.write_all(&pad)?;
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment <= 1 {
        return Ok(value);
    }
    let rem = value % alignment;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - rem)
            .ok_or_else(|| ContainerError::IntegerOverflow.into())
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
