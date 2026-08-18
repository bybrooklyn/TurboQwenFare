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
use crate::ids::{Bytes, ExpertId, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

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
    _metadata_lease: MemoryLease,
}

impl TqfReader {
    pub fn open_validated_with_broker(path: &Path, broker: &MemoryBroker) -> Result<Self> {
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

        let section_table_bytes = (superblock.section_count as u64)
            .checked_mul(SECTION_RECORD_SIZE as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let extent_table_bytes = superblock
            .extent_count
            .checked_mul(TENSOR_EXTENT_RECORD_SIZE as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        let metadata_raw_bytes = section_table_bytes
            .checked_add(extent_table_bytes)
            .and_then(|total| total.checked_add(superblock.string_table_bytes))
            .and_then(|total| total.checked_add(superblock.expert_index_bytes))
            .and_then(|total| total.checked_add(superblock.checksum_table_bytes))
            .ok_or(ContainerError::IntegerOverflow)?;
        let metadata_envelope = metadata_raw_bytes
            .checked_mul(2)
            .ok_or(ContainerError::IntegerOverflow)?
            .max(1);
        let metadata_lease = broker.reserve(
            MemoryOwner::Core,
            MemoryClass::Fixed,
            Bytes(metadata_envelope),
            64,
        )?;

        let section_table = read_table(
            &file,
            superblock.section_table_offset,
            section_table_bytes,
            file_len,
            "section table",
        )?;
        let extent_table = read_table(
            &file,
            superblock.extent_table_offset,
            extent_table_bytes,
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
            let tile_start = e.tile_first as usize;
            if !super::tiling::partition_is_contiguous(&tiles[tile_start..tile_end], e.stored_bytes)
            {
                return Err(ContainerError::MalformedRecord {
                    table: "expert tile partition",
                }
                .into());
            }
            if e.flags & super::records::EXPERT_INDEX_FLAG_TILE_CHECKSUMS != 0 {
                let checksum_entries = checksum_table.len() / super::records::CHECKSUM_ENTRY_SIZE;
                let required = (e.checksum_index as usize)
                    .checked_add(1)
                    .and_then(|index| index.checked_add(e.tile_count as usize))
                    .ok_or(ContainerError::IntegerOverflow)?;
                if required > checksum_entries {
                    return Err(ContainerError::MalformedRecord {
                        table: "expert per-tile checksum range",
                    }
                    .into());
                }
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
            _metadata_lease: metadata_lease,
        })
    }

    /// Unit-test convenience. Product paths must supply the process-wide
    /// broker through `open_validated_with_broker`.
    #[cfg(test)]
    pub fn open_validated(path: &Path) -> Result<Self> {
        let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
        Self::open_validated_with_broker(path, &broker)
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

    /// Metadata-only iteration for planning/sealing tooling (Phase 24/25
    /// sizing): all tensor extents, no payload reads.
    pub fn extents_iter(&self) -> impl Iterator<Item = &TensorExtentRecord> {
        self.extents.iter()
    }

    /// Metadata-only iteration over the expert index records.
    pub fn experts_iter(&self) -> impl Iterator<Item = &ExpertIndexRecord> {
        self.experts.iter()
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
    #[cfg(test)]
    pub fn read_extent_bytes(&self, extent: &TensorExtentRecord) -> Result<Vec<u8>> {
        let len: usize = extent
            .stored_bytes
            .try_into()
            .map_err(|_| ContainerError::IntegerOverflow)?;
        let mut buf = vec![0u8; len];
        self.read_extent_into(extent, &mut buf)?;
        Ok(buf)
    }

    /// Reads a tensor into caller-owned storage and verifies its checksum.
    /// Runtime callers reserve this destination with the memory broker before
    /// allocating it; this low-level reader never guesses at a memory budget.
    pub fn read_extent_into(
        &self,
        extent: &TensorExtentRecord,
        destination: &mut [u8],
    ) -> Result<()> {
        let expected: usize = extent
            .stored_bytes
            .try_into()
            .map_err(|_| ContainerError::IntegerOverflow)?;
        if destination.len() != expected {
            return Err(ContainerError::MalformedRecord {
                table: "tensor extent destination length",
            }
            .into());
        }
        self.file.read_exact_at(destination, extent.file_offset)?;
        self.verify_checksum(extent.checksum_index, destination)
    }

    /// Reads one expert's raw (gate_up + down) bytes and verifies them
    /// against the checksum table before returning them.
    #[cfg(test)]
    pub fn read_expert_bytes(&self, idx: &ExpertIndexRecord) -> Result<Vec<u8>> {
        let len: usize = idx
            .stored_bytes
            .try_into()
            .map_err(|_| ContainerError::IntegerOverflow)?;
        let mut buf = vec![0u8; len];
        self.read_expert_into(idx, &mut buf)?;
        Ok(buf)
    }

    /// Reads one whole-expert superextent into caller-owned storage and
    /// verifies its checksum. Runtime cache misses must reserve this exact
    /// destination with the memory broker before allocating it.
    pub fn read_expert_into(&self, idx: &ExpertIndexRecord, destination: &mut [u8]) -> Result<()> {
        let expected: usize = idx
            .stored_bytes
            .try_into()
            .map_err(|_| ContainerError::IntegerOverflow)?;
        if destination.len() != expected {
            return Err(ContainerError::MalformedRecord {
                table: "expert superextent destination length",
            }
            .into());
        }
        self.file.read_exact_at(destination, idx.file_offset)?;
        self.verify_checksum(idx.checksum_index, destination)
    }

    /// Phase 22 tile-granular read (spec §294): reads one tile of an
    /// expert's superextent into caller-owned storage. Integrity requires
    /// the converter to have emitted per-tile digests
    /// (`EXPERT_INDEX_FLAG_TILE_CHECKSUMS`); without them this refuses
    /// rather than reading unverifiable bytes - a partially resident
    /// expert is only admissible on a container that can vouch for every
    /// tile independently.
    pub fn read_expert_tile_into(
        &self,
        idx: &ExpertIndexRecord,
        tile_ordinal: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        let tile = self
            .expert_tile(idx, tile_ordinal)
            .ok_or(ContainerError::MalformedRecord {
                table: "expert tile ordinal",
            })?;
        if tile.stored_bytes as usize != destination.len() {
            return Err(ContainerError::MalformedRecord {
                table: "expert tile destination length",
            }
            .into());
        }
        if idx.flags & super::records::EXPERT_INDEX_FLAG_TILE_CHECKSUMS == 0 {
            return Err(ContainerError::MalformedRecord {
                table: "expert per-tile checksums (tile-granular read refused)",
            }
            .into());
        }
        let offset = idx
            .file_offset
            .checked_add(tile.relative_offset as u64)
            .ok_or(ContainerError::IntegerOverflow)?;
        self.file.read_exact_at(destination, offset)?;
        let checksum_index = (idx.checksum_index as usize)
            .checked_add(1)
            .and_then(|index| index.checked_add(tile_ordinal))
            .ok_or(ContainerError::IntegerOverflow)?;
        self.verify_checksum(checksum_index as u32, destination)
    }

    /// Returns the tile record for one ordinal within an expert's tile
    /// table, or `None` if the ordinal is out of range.
    pub fn expert_tile(
        &self,
        idx: &ExpertIndexRecord,
        tile_ordinal: usize,
    ) -> Option<&ExpertTileRecord> {
        let start = idx.tile_first as usize;
        let end = start.checked_add(idx.tile_count as usize)?;
        self.tiles.get(start..end)?.get(tile_ordinal)
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
