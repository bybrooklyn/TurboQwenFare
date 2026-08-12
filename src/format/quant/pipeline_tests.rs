//! End-to-end repack + validate pipeline tests (spec §279): a synthetic
//! GGUF fixture goes through `repack::repack_passthrough` and then
//! `validate::validate_tensor`, proving the whole Phase 7 chain — source
//! decode, passthrough pack, independent re-decode — agrees on real bytes,
//! and that corruption is caught with a precise first-mismatch location.

use std::path::PathBuf;

use crate::format::gguf;
use crate::format::quant::{repack, validate, GgmlType};

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Builds a minimal single-tensor GGUF fixture (no metadata beyond what
/// the reader requires) around caller-supplied quant bytes.
fn build_gguf_fixture(name: &str, dims: &[u64], ggml_type_id: u32, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    write_string(&mut out, name);
    out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&ggml_type_id.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // relative_offset

    let header_end = out.len() as u64;
    let alignment = 32u64;
    let data_start = header_end.div_ceil(alignment) * alignment;
    out.resize(data_start as usize, 0);
    out.extend_from_slice(data);
    out
}

fn write_fixture(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tqf-quant-pipeline-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Four Q4_0 blocks (32 elements each) of varied, non-degenerate byte
/// patterns so both nibble halves and a non-trivial scale are exercised.
fn sample_q4_0_blocks(n: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..n {
        // scale (f16 "1.0")
        out.extend_from_slice(&0x3C00u16.to_le_bytes());
        for j in 0..16u8 {
            out.push((i as u8).wrapping_mul(17).wrapping_add(j).wrapping_mul(3));
        }
    }
    out
}

#[test]
fn passthrough_repack_validates_cleanly() {
    let block_bytes = sample_q4_0_blocks(3);
    // GGML wire type id 2 == Q4_0 (see `GgmlType::from_ggml_id`).
    let bytes = build_gguf_fixture("weight", &[96], 2, &block_bytes);
    let path = write_fixture("passthrough.gguf", &bytes);

    let file = gguf::open(&path).unwrap();
    let tensor = file.tensor("weight").unwrap();
    assert_eq!(tensor.ggml_type, GgmlType::Q4_0);

    let mut reader = file.quant_block_reader(tensor).unwrap();
    let repacked = repack::repack_passthrough(&mut reader).unwrap();
    assert_eq!(repacked, block_bytes);
    assert_eq!(
        repack::tqf_quant_layout_id(tensor.ggml_type),
        Some(repack::TQF_QUANT_PASSTHROUGH_Q4_0)
    );

    validate::validate_tensor(&file, tensor, &repacked).unwrap();
}

#[test]
fn corrupted_repack_is_caught_with_precise_location() {
    let block_bytes = sample_q4_0_blocks(3);
    let bytes = build_gguf_fixture("weight", &[96], 2, &block_bytes);
    let path = write_fixture("corrupt.gguf", &bytes);

    let file = gguf::open(&path).unwrap();
    let tensor = file.tensor("weight").unwrap();

    let mut reader = file.quant_block_reader(tensor).unwrap();
    let mut repacked = repack::repack_passthrough(&mut reader).unwrap();
    // Corrupt one nibble inside the second block (block 1, byte offset
    // 18 + 2 within that block's qs region).
    let corrupt_offset = 18 + 2; // block 1's qs[0] (block header is 2 bytes)
    repacked[corrupt_offset] ^= 0x0F;

    let err = validate::validate_tensor(&file, tensor, &repacked).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("block 1"), "expected block 1 in {msg:?}");
    // Block 1 covers elements [32, 64); flipping qs[0]'s low nibble bits
    // changes that block's element 0 (the low-nibble half), i.e. absolute
    // element 32.
    assert!(msg.contains("element 32"), "expected element 32 in {msg:?}");
}

#[test]
fn byte_length_mismatch_is_a_typed_error() {
    let block_bytes = sample_q4_0_blocks(1);
    let bytes = build_gguf_fixture("weight", &[32], 2, &block_bytes);
    let path = write_fixture("length-mismatch.gguf", &bytes);

    let file = gguf::open(&path).unwrap();
    let tensor = file.tensor("weight").unwrap();

    let err = validate::validate_tensor(&file, tensor, &[0u8; 4]).unwrap_err();
    assert!(err.to_string().contains("does not match source byte size"));
}
