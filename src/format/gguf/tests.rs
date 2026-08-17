//! Synthetic-fixture tests for the GGUF reader: a hand-built valid file
//! plus deliberately corrupted variants exercising the fuzz-target
//! checklist named by spec §277 ("Fuzz now") and analogous to §246's list
//! for `.tqf` — bad magic, unsupported version, truncation, integer
//! overflow in offsets/dims, out-of-bounds tensor ranges, unknown quant
//! types, duplicate names. Every case must produce a clean typed error,
//! never a panic or out-of-bounds read.

use std::path::PathBuf;

use crate::error::{GgufError, TqfError};
use crate::format::gguf;

const TYPE_U32: u32 = 4;
const TYPE_STRING: u32 = 8;

struct FixtureTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type_id: u32,
    data: Vec<u8>,
}

#[derive(Default)]
struct GgufBuilder {
    metadata: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<FixtureTensor>,
    alignment: u64,
}

impl GgufBuilder {
    fn new() -> Self {
        Self {
            alignment: 32,
            ..Default::default()
        }
    }

    fn alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
        self
    }

    fn metadata_u32(mut self, key: &str, value: u32) -> Self {
        self.metadata
            .push((key.to_string(), TYPE_U32, value.to_le_bytes().to_vec()));
        self
    }

    fn metadata_string(mut self, key: &str, value: &str) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
        self.metadata.push((key.to_string(), TYPE_STRING, bytes));
        self
    }

    fn tensor(mut self, name: &str, dims: Vec<u64>, ggml_type_id: u32, data: Vec<u8>) -> Self {
        self.tensors.push(FixtureTensor {
            name: name.to_string(),
            dims,
            ggml_type_id,
            data,
        });
        self
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        for (key, type_tag, value_bytes) in &self.metadata {
            write_string(&mut out, key);
            out.extend_from_slice(&type_tag.to_le_bytes());
            out.extend_from_slice(value_bytes);
        }

        let mut relative_offsets = Vec::new();
        let mut cursor = 0u64;
        for t in &self.tensors {
            let aligned = align_up(cursor, self.alignment);
            relative_offsets.push(aligned);
            cursor = aligned + t.data.len() as u64;
        }

        for (t, rel_off) in self.tensors.iter().zip(&relative_offsets) {
            write_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type_id.to_le_bytes());
            out.extend_from_slice(&rel_off.to_le_bytes());
        }

        let header_end = out.len() as u64;
        let data_start = align_up(header_end, self.alignment);
        out.resize(data_start as usize, 0);

        for (t, rel_off) in self.tensors.iter().zip(&relative_offsets) {
            let target_len = (data_start + rel_off) as usize;
            if out.len() < target_len {
                out.resize(target_len, 0);
            }
            out.extend_from_slice(&t.data);
        }

        out
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

fn write_fixture(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tqf-gguf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn q4_0_block_of(fill: u8) -> Vec<u8> {
    vec![fill; 18] // one Q4_0 block: 32 elements, 18 bytes
}

#[test]
fn parses_minimal_valid_file() {
    let block = q4_0_block_of(0xAB);
    let bytes = GgufBuilder::new()
        .metadata_string("general.name", "tqf-test")
        .tensor("weight", vec![32], 2 /* Q4_0 */, block.clone())
        .build();
    let path = write_fixture("minimal.gguf", &bytes);

    let file: gguf::GgufFile = gguf::open(&path).unwrap();
    assert_eq!(file.version, 3);
    assert_eq!(
        file.metadata.get("general.name").and_then(|v| v.as_str()),
        Some("tqf-test")
    );

    let tensor: &gguf::TensorDescriptor = file.tensor("weight").unwrap();
    assert_eq!(tensor.dims, vec![32]);
    assert_eq!(tensor.n_elements, 32);
    assert_eq!(tensor.byte_size, 18);

    let mut reader: gguf::QuantBlockReader = file.quant_block_reader(tensor).unwrap();
    assert_eq!(reader.total_blocks(), 1);
    let batch = reader.next_batch().unwrap().unwrap();
    assert_eq!(batch, block);
    assert!(reader.next_batch().unwrap().is_none());
}

#[test]
fn metadata_probe_requires_broker_admission_before_allocation() {
    let bytes = GgufBuilder::new()
        .tensor("weight", vec![32], 2, q4_0_block_of(0))
        .build();
    let path = write_fixture("metadata-budget.gguf", &bytes);
    let broker = crate::memory::MemoryBroker::new(crate::ids::Bytes(1));

    let error = gguf::open_with_broker(&path, &broker).unwrap_err();
    assert!(matches!(error, crate::error::TqfError::Memory(_)));
    assert_eq!(broker.snapshot().reserved, crate::ids::Bytes(0));
}

#[test]
fn respects_custom_alignment_metadata() {
    let block = q4_0_block_of(0x11);
    let bytes = GgufBuilder::new()
        .alignment(128)
        .metadata_u32("general.alignment", 128)
        .tensor("weight", vec![32], 2, block.clone())
        .build();
    let path = write_fixture("aligned.gguf", &bytes);

    let file = gguf::open(&path).unwrap();
    let tensor = file.tensor("weight").unwrap();
    assert_eq!(tensor.file_offset % 128, 0);
    let mut reader = file.quant_block_reader(tensor).unwrap();
    assert_eq!(reader.next_batch().unwrap().unwrap(), block);
}

#[test]
fn tokenizer_metadata_filters_by_prefix() {
    let bytes = GgufBuilder::new()
        .metadata_string("tokenizer.ggml.model", "qwen")
        .metadata_string("general.name", "not-tokenizer")
        .build();
    let path = write_fixture("tokenizer.gguf", &bytes);

    let file = gguf::open(&path).unwrap();
    let tok_meta = file.tokenizer_metadata();
    assert_eq!(tok_meta.len(), 1);
    assert!(tok_meta.contains_key("tokenizer.ggml.model"));
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = GgufBuilder::new().build();
    bytes[0] = b'X';
    let path = write_fixture("bad-magic.gguf", &bytes);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(GgufError::BadMagic))
    ));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = GgufBuilder::new().build();
    bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    let path = write_fixture("bad-version.gguf", &bytes);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(
            GgufError::UnsupportedVersion(99)
        ))
    ));
}

#[test]
fn rejects_truncated_metadata() {
    let bytes = GgufBuilder::new()
        .metadata_string("general.name", "truncate-me")
        .build();
    // Header preamble (magic + version + tensor_count + kv_count) is
    // exactly 24 bytes; kv_count declares 1 entry but none of its bytes
    // are present, so parsing must fail on the very first metadata read
    // rather than silently treating the declared entry as absent.
    let truncated = &bytes[..24];
    let path = write_fixture("truncated.gguf", truncated);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(GgufError::Truncated { .. }))
    ));
}

#[test]
fn rejects_tensor_range_outside_data_section() {
    let block = q4_0_block_of(0x22);
    let mut bytes = GgufBuilder::new()
        .tensor("weight", vec![32], 2, block)
        .build();
    // Chop off the last byte of the tensor's payload so its declared
    // range extends past the actual file length.
    bytes.truncate(bytes.len() - 1);
    let path = write_fixture("oob-tensor.gguf", &bytes);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(
            GgufError::TensorOutOfBounds { .. }
        ))
    ));
}

#[test]
fn rejects_unknown_quant_type() {
    let bytes = GgufBuilder::new()
        .tensor("weight", vec![32], 9999, vec![0u8; 18])
        .build();
    let path = write_fixture("unknown-type.gguf", &bytes);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(
            GgufError::UnsupportedQuantType(9999)
        ))
    ));
}

#[test]
fn rejects_duplicate_tensor_names() {
    let block = q4_0_block_of(0x33);
    let bytes = GgufBuilder::new()
        .tensor("weight", vec![32], 2, block.clone())
        .tensor("weight", vec![32], 2, block)
        .build();
    let path = write_fixture("dup-name.gguf", &bytes);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(GgufError::DuplicateTensor(
            _
        )))
    ));
}

#[test]
fn rejects_malicious_huge_dims_without_overflow_or_oom() {
    // Two dims whose product overflows u64 — a fuzz-style "malicious huge
    // logical dimensions" case (spec §246 analog). Must be a clean typed
    // error, never an OOM allocation or panic.
    let bytes = GgufBuilder::new()
        .tensor("weight", vec![u64::MAX, 2], 2, vec![0u8; 18])
        .build();
    let path = write_fixture("huge-dims.gguf", &bytes);

    let err = gguf::open(&path).unwrap_err();
    assert!(matches!(
        err,
        TqfError::Format(crate::error::FormatError::Gguf(GgufError::IntegerOverflow))
    ));
}

#[test]
fn quant_block_reader_batches_across_multiple_reads() {
    // 2500 Q4_0 blocks (> the 1024-block default batch size) so this
    // exercises more than one `next_batch()` call.
    let n_blocks = 2500u64;
    let mut data = Vec::with_capacity((n_blocks * 18) as usize);
    for i in 0..n_blocks {
        data.extend_from_slice(&[(i % 256) as u8; 18]);
    }
    let bytes = GgufBuilder::new()
        .tensor("weight", vec![n_blocks * 32], 2, data.clone())
        .build();
    let path = write_fixture("multi-batch.gguf", &bytes);

    let file = gguf::open(&path).unwrap();
    let tensor = file.tensor("weight").unwrap();
    let mut reader = file.quant_block_reader(tensor).unwrap();
    assert_eq!(reader.total_blocks(), n_blocks);

    let mut reconstructed = Vec::new();
    let mut batches = 0;
    while let Some(batch) = reader.next_batch().unwrap() {
        reconstructed.extend_from_slice(&batch);
        batches += 1;
    }
    assert!(batches > 1, "expected more than one batch");
    assert_eq!(reconstructed, data);
}
