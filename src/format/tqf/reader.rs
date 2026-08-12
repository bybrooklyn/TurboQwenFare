//! `TqfReader`: validates and opens a `.tqf` v1 container (spec §278,
//! §121 reader validation order). Steps 1-6 of that order (read 4096-byte
//! superblock; validate magic/size/endian/major version; validate declared
//! file length against `fstat`; checked-bounds every table range; read
//! metadata tables; validate the metadata root hash) all happen in
//! `open_validated` before any extent/expert data is exposed. Step 7
//! ("validate architecture fingerprint before any kernel is constructed")
//! is out of Phase 6 scope — no kernel construction exists yet.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::error::{ContainerError, Result};
use crate::ids::{ExpertId, LayerId};

use super::records::{
    read_string_at, ChecksumEntry, ExpertIndexRecord, ExpertTileRecord, SectionRecord,
    TensorExtentRecord, TqfSectionKind, EXPERT_INDEX_RECORD_SIZE, EXPERT_TILE_RECORD_SIZE,
    SECTION_RECORD_SIZE, TENSOR_EXTENT_RECORD_SIZE,
};
use super::superblock::{Superblock, SUPERBLOCK_SIZE};

#[derive(Debug)]
pub struct TqfReader {
    file: File,
    #[allow(dead_code)]
    path: PathBuf,
    pub superblock: Superblock,
    #[allow(dead_code)]
    sections: Vec<SectionRecord>,
    extents: Vec<TensorExtentRecord>,
    string_table: Vec<u8>,
    experts: Vec<ExpertIndexRecord>,
    tiles: Vec<ExpertTileRecord>,
    checksums: Vec<ChecksumEntry>,
}

impl TqfReader {
    pub fn open_validated(path: &Path) -> Result<Self> {
        let file_len = std::fs::metadata(path)?.len();
        let file = File::open(path)?;

        if file_len < SUPERBLOCK_SIZE as u64 {
            return Err(ContainerError::Truncated {
                needed: SUPERBLOCK_SIZE as u64,
                available: file_len,
            }
            .into());
        }
        let mut sb_buf = [0u8; SUPERBLOCK_SIZE];
        file.read_exact_at(&mut sb_buf, 0)?;
        let superblock = Superblock::decode(&sb_buf)?;

        if superblock.file_length != file_len {
            return Err(ContainerError::FileLengthMismatch {
                declared: superblock.file_length,
                actual: file_len,
            }
            .into());
        }

        let section_table = read_table(
            &file,
            superblock.section_table_offset,
            (superblock.section_count as u64)
                .checked_mul(SECTION_RECORD_SIZE as u64)
                .ok_or(ContainerError::IntegerOverflow)?,
            file_len,
            "section table",
        )?;
        let extent_table = read_table(
            &file,
            superblock.extent_table_offset,
            superblock
                .extent_count
                .checked_mul(TENSOR_EXTENT_RECORD_SIZE as u64)
                .ok_or(ContainerError::IntegerOverflow)?,
            file_len,
            "extent table",
        )?;
        let string_table = read_table(
            &file,
            superblock.string_table_offset,
            superblock.string_table_bytes,
            file_len,
            "string table",
        )?;
        let expert_index_blob = read_table(
            &file,
            superblock.expert_index_offset,
            superblock.expert_index_bytes,
            file_len,
            "expert index",
        )?;
        let checksum_table = read_table(
            &file,
            superblock.checksum_table_offset,
            superblock.checksum_table_bytes,
            file_len,
            "checksum table",
        )?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(&section_table);
        hasher.update(&extent_table);
        hasher.update(&string_table);
        hasher.update(&expert_index_blob);
        hasher.update(&checksum_table);
        let computed = *hasher.finalize().as_bytes();
        if computed != superblock.metadata_root_blake3 {
            return Err(ContainerError::MetadataRootHashMismatch {
                expected: hex(&superblock.metadata_root_blake3),
                computed: hex(&computed),
            }
            .into());
        }

        let mut sections = Vec::new();
        for chunk in section_table.chunks(SECTION_RECORD_SIZE) {
            sections.push(
                SectionRecord::decode(chunk).ok_or(ContainerError::MalformedRecord {
                    table: "section table",
                })?,
            );
        }

        let mut extents = Vec::new();
        for chunk in extent_table.chunks(TENSOR_EXTENT_RECORD_SIZE) {
            let rec = TensorExtentRecord::decode(chunk).ok_or(ContainerError::MalformedRecord {
                table: "extent table",
            })?;
            let end = rec
                .file_offset
                .checked_add(rec.stored_bytes)
                .ok_or(ContainerError::IntegerOverflow)?;
            if end > file_len {
                return Err(ContainerError::TableOutOfBounds {
                    name: "tensor extent",
                    offset: rec.file_offset,
                    len: rec.stored_bytes,
                    file_len,
                }
                .into());
            }
            extents.push(rec);
        }

        let expert_count = sections
            .iter()
            .find(|s| s.kind == TqfSectionKind::ExpertIndex as u32)
            .map(|s| s.element_count)
            .unwrap_or(0);
        let index_bytes: usize = expert_count
            .checked_mul(EXPERT_INDEX_RECORD_SIZE as u64)
            .ok_or(ContainerError::IntegerOverflow)?
            .try_into()
            .map_err(|_| ContainerError::IntegerOverflow)?;
        if index_bytes > expert_index_blob.len() {
            return Err(ContainerError::MalformedRecord {
                table: "expert index",
            }
            .into());
        }
        let (index_part, tile_part) = expert_index_blob.split_at(index_bytes);

        let mut experts = Vec::new();
        for chunk in index_part.chunks(EXPERT_INDEX_RECORD_SIZE) {
            experts.push(ExpertIndexRecord::decode(chunk).ok_or(
                ContainerError::MalformedRecord {
                    table: "expert index",
                },
            )?);
        }
        let mut tiles = Vec::new();
        for chunk in tile_part.chunks(EXPERT_TILE_RECORD_SIZE) {
            tiles.push(
                ExpertTileRecord::decode(chunk).ok_or(ContainerError::MalformedRecord {
                    table: "expert tiles",
                })?,
            );
        }
        for e in &experts {
            let end = e
                .file_offset
                .checked_add(e.stored_bytes as u64)
                .ok_or(ContainerError::IntegerOverflow)?;
            if end > file_len {
                return Err(ContainerError::TableOutOfBounds {
                    name: "expert superextent",
                    offset: e.file_offset,
                    len: e.stored_bytes as u64,
                    file_len,
                }
                .into());
            }
            let tile_end = (e.tile_first as usize)
                .checked_add(e.tile_count as usize)
                .ok_or(ContainerError::IntegerOverflow)?;
            if tile_end > tiles.len() {
                return Err(ContainerError::MalformedRecord {
                    table: "expert tile range",
                }
                .into());
            }
        }

        let mut checksums = Vec::new();
        for chunk in checksum_table.chunks(super::records::CHECKSUM_ENTRY_SIZE) {
            checksums.push(ChecksumEntry::decode(chunk).ok_or(
                ContainerError::MalformedRecord {
                    table: "checksum table",
                },
            )?);
        }

        Ok(Self {
            file,
            path: path.to_path_buf(),
            superblock,
            sections,
            extents,
            string_table,
            experts,
            tiles,
            checksums,
        })
    }

    /// `role_id`/`layer` per the taskbook's `TqfReader::tensor(role/layer)`
    /// API — `layer` disambiguates tensors that repeat per-layer (e.g.
    /// `q_proj` across 40 layers) from layer-independent ones.
    pub fn tensor(&self, role_id: u32, layer: Option<LayerId>) -> Result<&TensorExtentRecord> {
        self.extents
            .iter()
            .find(|e| e.role_id == role_id && e.layer() == layer)
            .ok_or_else(|| {
                ContainerError::TensorNotFound {
                    role_id,
                    layer: layer.map(|l| l.0),
                }
                .into()
            })
    }

    pub fn tensor_name(&self, extent: &TensorExtentRecord) -> Result<String> {
        read_string_at(&self.string_table, extent.name_string_offset)
    }

    pub fn expert(
        &self,
        layer: LayerId,
        expert: ExpertId,
    ) -> Result<(&ExpertIndexRecord, &[ExpertTileRecord])> {
        let idx = self
            .experts
            .iter()
            .find(|e| e.layer == layer && e.expert == expert)
            .ok_or(ContainerError::ExpertNotFound {
                layer: layer.0,
                expert: expert.0,
            })?;
        let start = idx.tile_first as usize;
        let end = start + idx.tile_count as usize;
        Ok((idx, &self.tiles[start..end]))
    }

    /// Reads one tensor's raw bytes and verifies them against the
    /// checksum table before returning them.
    pub fn read_extent_bytes(&self, extent: &TensorExtentRecord) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; extent.stored_bytes as usize];
        self.file.read_exact_at(&mut buf, extent.file_offset)?;
        self.verify_checksum(extent.checksum_index, &buf)?;
        Ok(buf)
    }

    /// Reads one expert's raw (gate_up + down) bytes and verifies them
    /// against the checksum table before returning them.
    pub fn read_expert_bytes(&self, idx: &ExpertIndexRecord) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; idx.stored_bytes as usize];
        self.file.read_exact_at(&mut buf, idx.file_offset)?;
        self.verify_checksum(idx.checksum_index, &buf)?;
        Ok(buf)
    }

    fn verify_checksum(&self, index: u32, data: &[u8]) -> Result<()> {
        let entry = self
            .checksums
            .get(index as usize)
            .ok_or(ContainerError::ChecksumMismatch(format!(
                "checksum_index {index} out of range"
            )))?;
        let digest = blake3::hash(data);
        if digest.as_bytes() != &entry.digest {
            return Err(ContainerError::ChecksumMismatch(format!("checksum_index {index}")).into());
        }
        Ok(())
    }
}

fn read_table(
    file: &File,
    offset: u64,
    len: u64,
    file_len: u64,
    name: &'static str,
) -> Result<Vec<u8>> {
    let end = offset
        .checked_add(len)
        .ok_or(ContainerError::IntegerOverflow)?;
    if end > file_len {
        return Err(ContainerError::TableOutOfBounds {
            name,
            offset,
            len,
            file_len,
        }
        .into());
    }
    let len_usize: usize = len
        .try_into()
        .map_err(|_| ContainerError::IntegerOverflow)?;
    let mut buf = vec![0u8; len_usize];
    file.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
