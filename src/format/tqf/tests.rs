//! Synthetic-fixture roundtrip and corruption tests for the `.tqf`
//! container (spec §278: "Roundtrip synthetic fixtures and corruption
//! tests"), analogous in spirit to the fuzz-target checklist in spec §246:
//! bad magic, truncated files, metadata-hash mismatch, out-of-bounds
//! tables, tampered payload — every case must be a clean typed error.

use std::path::{Path, PathBuf};

use crate::error::{ContainerError, TqfError};
use crate::ids::{ExpertId, LayerId};

use super::{TqfHeaderInfo, TqfReader, TqfSectionKind, TqfWriter};

const ROLE_EMBEDDING: u32 = 1;
const ROLE_Q_PROJ: u32 = 2;

fn fixture_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tqf-container-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn header() -> TqfHeaderInfo {
    TqfHeaderInfo {
        backend_id: 1,
        feature_bits: 0,
        model_family_id: [0xAA; 16],
        source_sha256: [0xBB; 32],
        conversion_fingerprint: [0xCC; 32],
    }
}

fn embedding_bytes() -> Vec<u8> {
    vec![0xAB; 300]
}

fn q_proj_bytes() -> Vec<u8> {
    vec![0xCD; 150]
}

fn expert_gate_up_bytes() -> Vec<u8> {
    vec![0x11; 64]
}

fn expert_down_bytes() -> Vec<u8> {
    vec![0x22; 32]
}

fn build_sample(path: &Path) {
    let mut writer = TqfWriter::create_partial(path, header()).unwrap();
    writer
        .write_extent(
            ROLE_EMBEDDING,
            "token_embedding",
            None,
            TqfSectionKind::Embeddings,
            &[2048, 248_320],
            12,
            12,
            64,
            &embedding_bytes(),
        )
        .unwrap();
    writer
        .write_extent(
            ROLE_Q_PROJ,
            "layers.0.self_attn.q_proj",
            Some(LayerId(0)),
            TqfSectionKind::ResidentCore,
            &[8192, 2048],
            12,
            12,
            64,
            &q_proj_bytes(),
        )
        .unwrap();
    writer
        .write_expert(
            LayerId(3),
            ExpertId(10),
            12,
            &expert_gate_up_bytes(),
            &expert_down_bytes(),
        )
        .unwrap();
    writer.commit().unwrap();
}

#[test]
fn roundtrip_simple_extents_and_experts() {
    let path = fixture_path("roundtrip.tqf");
    build_sample(&path);

    let reader = TqfReader::open_validated(&path).unwrap();

    let embed = reader.tensor(ROLE_EMBEDDING, None).unwrap();
    assert_eq!(reader.tensor_name(embed).unwrap(), "token_embedding");
    assert_eq!(reader.read_extent_bytes(embed).unwrap(), embedding_bytes());

    let q_proj = reader.tensor(ROLE_Q_PROJ, Some(LayerId(0))).unwrap();
    assert_eq!(
        reader.tensor_name(q_proj).unwrap(),
        "layers.0.self_attn.q_proj"
    );
    assert_eq!(reader.read_extent_bytes(q_proj).unwrap(), q_proj_bytes());

    let (idx, tiles) = reader.expert(LayerId(3), ExpertId(10)).unwrap();
    assert_eq!(tiles.len(), 2);
    let mut expected = expert_gate_up_bytes();
    expected.extend(expert_down_bytes());
    assert_eq!(reader.read_expert_bytes(idx).unwrap(), expected);
}

#[test]
fn duplicate_extent_name_is_rejected() {
    let path = fixture_path("dup-extent.tqf");
    let mut writer = TqfWriter::create_partial(&path, header()).unwrap();
    writer
        .write_extent(
            ROLE_EMBEDDING,
            "dup",
            None,
            TqfSectionKind::Embeddings,
            &[16],
            12,
            12,
            64,
            &[0u8; 16],
        )
        .unwrap();
    let err = writer
        .write_extent(
            ROLE_Q_PROJ,
            "dup",
            None,
            TqfSectionKind::Embeddings,
            &[16],
            12,
            12,
            64,
            &[0u8; 16],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::DuplicateExtent(_)
        ))
    ));
}

#[test]
fn duplicate_expert_is_rejected() {
    let path = fixture_path("dup-expert.tqf");
    let mut writer = TqfWriter::create_partial(&path, header()).unwrap();
    writer
        .write_expert(LayerId(1), ExpertId(1), 12, &[0u8; 16], &[0u8; 16])
        .unwrap();
    let err = writer
        .write_expert(LayerId(1), ExpertId(1), 12, &[0u8; 16], &[0u8; 16])
        .unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::DuplicateExtent(_)
        ))
    ));
}

#[test]
fn rank_over_four_is_rejected() {
    let path = fixture_path("bad-rank.tqf");
    let mut writer = TqfWriter::create_partial(&path, header()).unwrap();
    let err = writer
        .write_extent(
            ROLE_EMBEDDING,
            "too-many-dims",
            None,
            TqfSectionKind::Embeddings,
            &[1, 2, 3, 4, 5],
            12,
            12,
            64,
            &[0u8; 16],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::UnsupportedRank(5)
        ))
    ));
}

#[test]
fn tensor_lookup_miss_is_a_typed_error() {
    let path = fixture_path("tensor-miss.tqf");
    build_sample(&path);
    let reader = TqfReader::open_validated(&path).unwrap();
    let err = reader.tensor(999, None).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::TensorNotFound { .. }
        ))
    ));
}

#[test]
fn expert_lookup_miss_is_a_typed_error() {
    let path = fixture_path("expert-miss.tqf");
    build_sample(&path);
    let reader = TqfReader::open_validated(&path).unwrap();
    let err = reader.expert(LayerId(0), ExpertId(0)).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::ExpertNotFound { .. }
        ))
    ));
}

#[test]
fn corrupted_magic_is_rejected() {
    let path = fixture_path("bad-magic.tqf");
    build_sample(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] = b'X';
    std::fs::write(&path, &bytes).unwrap();

    let err = TqfReader::open_validated(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::BadMagic
        ))
    ));
}

#[test]
fn truncated_file_is_rejected() {
    let path = fixture_path("truncated.tqf");
    build_sample(&path);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..100]).unwrap(); // shorter than the 4096-byte superblock

    let err = TqfReader::open_validated(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::Truncated { .. }
        ))
    ));
}

#[test]
fn file_length_mismatch_is_rejected() {
    let path = fixture_path("length-mismatch.tqf");
    build_sample(&path);
    let bytes = std::fs::read(&path).unwrap();
    // Still >= the 4096-byte superblock, but shorter than the superblock's
    // own declared `file_length` field.
    std::fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();

    let err = TqfReader::open_validated(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::FileLengthMismatch { .. }
        ))
    ));
}

#[test]
fn corrupted_metadata_table_fails_root_hash_check() {
    let path = fixture_path("bad-metadata-hash.tqf");
    build_sample(&path);

    // Learn the real extent-table offset by opening the uncorrupted file
    // once, so the corruption below reliably lands inside a metadata
    // table rather than inside raw tensor payload bytes (which the
    // metadata-root hash does not cover).
    let extent_table_offset = TqfReader::open_validated(&path)
        .unwrap()
        .superblock
        .extent_table_offset;

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[extent_table_offset as usize] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let err = TqfReader::open_validated(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::MetadataRootHashMismatch { .. }
        ))
    ));
}

#[test]
fn tampered_payload_is_caught_by_extent_checksum_not_root_hash() {
    let path = fixture_path("tampered-payload.tqf");
    build_sample(&path);

    // Corrupt a byte inside the first extent's raw payload (well before
    // the metadata tables, which start at `next_offset` after all
    // extent/expert payload data) — the metadata-root hash only covers
    // the section/extent/string/expert-index/checksum tables, not raw
    // tensor bytes, so opening the file must still succeed; only reading
    // the tampered extent's bytes should fail.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[4096] ^= 0xFF; // first byte of the embedding tensor's payload
    std::fs::write(&path, &bytes).unwrap();

    let reader = TqfReader::open_validated(&path).unwrap();
    let embed = reader.tensor(ROLE_EMBEDDING, None).unwrap();
    let err = reader.read_extent_bytes(embed).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Container(
            ContainerError::ChecksumMismatch(_)
        ))
    ));
}
