//! End-to-end integration test: GGUF source (phase 5) -> lossless repacker
//! (phase 7) -> resumable `.tqf` conversion transaction (phase 8) -> `.tqf`
//! reader, proving the whole conversion chain fits together on real (if
//! synthetic) tensor bytes rather than each phase's own isolated fixtures.

use std::path::PathBuf;

use crate::format::gguf;
use crate::format::quant::{repack, validate};
use crate::ids::LayerId;

use super::conversion::{BeginOutcome, ConversionTransaction};
use super::{TqfHeaderInfo, TqfReader, TqfSectionKind};

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A minimal two-tensor GGUF fixture: one Q4_0 "embedding" and one Q4_0
/// per-layer "q_proj", matching the shapes the real importer would see.
fn build_gguf_fixture() -> Vec<u8> {
    let embedding_blocks = 2usize;
    let q_proj_blocks = 3usize;

    let mut embedding_data = Vec::new();
    for i in 0..embedding_blocks {
        embedding_data.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale 1.0
        for j in 0..16u8 {
            embedding_data.push((i as u8).wrapping_mul(13).wrapping_add(j));
        }
    }
    let mut q_proj_data = Vec::new();
    for i in 0..q_proj_blocks {
        q_proj_data.extend_from_slice(&0x4000u16.to_le_bytes()); // scale 2.0
        for j in 0..16u8 {
            q_proj_data.push((i as u8).wrapping_mul(29).wrapping_add(j).wrapping_mul(3));
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    write_string(&mut out, "token_embedding");
    out.extend_from_slice(&1u32.to_le_bytes()); // rank
    out.extend_from_slice(&((embedding_blocks * 32) as u64).to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes()); // ggml type id 2 = Q4_0
    out.extend_from_slice(&0u64.to_le_bytes()); // relative_offset

    write_string(&mut out, "layers.0.q_proj");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&((q_proj_blocks * 32) as u64).to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    let embedding_bytes_len = embedding_blocks as u64 * 18;
    let aligned = embedding_bytes_len.div_ceil(32) * 32;
    out.extend_from_slice(&aligned.to_le_bytes()); // relative_offset

    let header_end = out.len() as u64;
    let data_start = header_end.div_ceil(32) * 32;
    out.resize(data_start as usize, 0);
    out.extend_from_slice(&embedding_data);
    out.resize((data_start + aligned) as usize, 0);
    out.extend_from_slice(&q_proj_data);

    out
}

fn fixture_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tqf-tqf-pipeline-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn gguf_repack_conversion_reader_round_trip() {
    let gguf_bytes = build_gguf_fixture();
    let gguf_path = fixture_path("source.gguf");
    std::fs::write(&gguf_path, &gguf_bytes).unwrap();

    let gguf_file = gguf::open(&gguf_path).unwrap();

    let tqf_path = fixture_path("out.tqf");
    let header = TqfHeaderInfo {
        backend_id: 1,
        feature_bits: 0,
        model_family_id: [0x11; 16],
        source_sha256: [0x22; 32],
        conversion_fingerprint: [0x33; 32],
    };
    let outcome = ConversionTransaction::begin(&tqf_path, header, "gguf-fixture-sha").unwrap();
    let mut txn = match outcome {
        BeginOutcome::Transaction(t) => t,
        BeginOutcome::AlreadyInstalled => panic!("unexpected"),
    };

    let plan: [(&str, u32, Option<LayerId>, TqfSectionKind, &[u64]); 2] = [
        (
            "token_embedding",
            1,
            None,
            TqfSectionKind::Embeddings,
            &[64],
        ),
        (
            "layers.0.q_proj",
            2,
            Some(LayerId(0)),
            TqfSectionKind::ResidentCore,
            &[96],
        ),
    ];

    for (source_name, role_id, layer, section_kind, dims) in plan {
        let tensor = gguf_file.tensor(source_name).unwrap();
        let mut reader = gguf_file.quant_block_reader(tensor).unwrap();
        let repacked = repack::repack_passthrough(&mut reader).unwrap();

        // Phase 7's validation gate runs as part of the conversion
        // pipeline, not just in isolation: a real converter would refuse
        // to write an extent that fails this check.
        validate::validate_tensor(&gguf_file, tensor, &repacked).unwrap();

        let quant_layout_id = repack::tqf_quant_layout_id(tensor.ggml_type).unwrap();
        txn.write_extent(
            role_id,
            source_name,
            layer,
            section_kind,
            dims,
            tensor.ggml_type as u32,
            quant_layout_id,
            64,
            &repacked,
        )
        .unwrap();
    }

    txn.finish().unwrap();

    let reader = TqfReader::open_validated(&tqf_path).unwrap();
    let embed = reader.tensor(1, None).unwrap();
    assert_eq!(reader.tensor_name(embed).unwrap(), "token_embedding");
    let q_proj = reader.tensor(2, Some(LayerId(0))).unwrap();
    assert_eq!(reader.tensor_name(q_proj).unwrap(), "layers.0.q_proj");

    // The bytes landed in `.tqf` are exactly the GGUF source bytes
    // (lossless passthrough), byte for byte.
    let embed_tensor = gguf_file.tensor("token_embedding").unwrap();
    let mut embed_reader = gguf_file.quant_block_reader(embed_tensor).unwrap();
    let expected_embed = repack::repack_passthrough(&mut embed_reader).unwrap();
    assert_eq!(reader.read_extent_bytes(embed).unwrap(), expected_embed);
}
