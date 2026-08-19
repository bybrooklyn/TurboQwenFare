//! GGUF -> `.tqf` conversion for the vision tower (spec Part XIII, phase
//! 48): reads the real `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` sidecar
//! (`general.architecture = clip`, mixed F32/F16/Q8_0 tensor dtypes —
//! confirmed by direct inspection) and repacks every tensor as F32
//! passthrough, same convention as `helper_model::convert` and
//! `helper_model::gte_reranker::convert`. The vision tower is small
//! (614 MB source) and lazy-loaded only when `--enable-vision` is set
//! and a vision input actually arrives, so a one-shot non-journaled
//! `TqfWriter` pass (not the full resumable expert-streaming path) is
//! sufficient here, matching those two prior helper-model converters.

use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{ModelError, Result};
use crate::format::gguf::GgufFile;
use crate::format::quant::dequant::{dequantize_block, f16_to_f32};
use crate::format::quant::repack::TQF_QUANT_PASSTHROUGH_F32;
use crate::format::quant::GgmlType;
use crate::format::tqf::{TqfHeaderInfo, TqfSectionKind, TqfWriter};
use crate::ids::LayerId;
use crate::memory::MemoryBroker;

use super::geometry::VisionGeometry;
use super::roles::VisionTensorRole;

pub const MODEL_FAMILY_LABEL: &[u8] = b"qwen3.6-vision-clip-qwen3vl-merger";
pub const CONVERSION_FINGERPRINT_LABEL: &[u8] = b"tqf-vision-f32-passthrough-v1";

const F32_DTYPE_ID: u32 = 0;

fn sha256_hex(path: &Path) -> Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

pub fn vision_header(source_sha256: [u8; 32]) -> TqfHeaderInfo {
    let family_hash = blake3::hash(MODEL_FAMILY_LABEL);
    let conversion_hash = blake3::hash(CONVERSION_FINGERPRINT_LABEL);
    let mut model_family_id = [0u8; 16];
    model_family_id.copy_from_slice(&family_hash.as_bytes()[..16]);
    TqfHeaderInfo {
        backend_id: 0,
        feature_bits: 0,
        model_family_id,
        source_sha256,
        conversion_fingerprint: *conversion_hash.as_bytes(),
    }
}

#[derive(Debug, Clone)]
pub struct VisionConversionReport {
    pub extent_count: usize,
    pub verified_output_bytes: u64,
    pub source_sha256: [u8; 32],
}

/// Decodes one GGUF tensor's raw on-disk bytes to `f32`, uniformly
/// across every dtype this checkpoint actually uses (F32 biases/norms/
/// patch/position tables, F16 `ffn_down`, Q8_0 everything else) — the
/// same dtype-dispatch shape as `model::qwen36::weights::decode_values`,
/// duplicated rather than cross-imported (this crate's established
/// per-model-family convention: `helper_model` and `gte_reranker` each
/// own their own decode path too).
fn read_tensor_f32(gguf: &GgufFile, source_file: &std::fs::File, name: &str) -> Result<Vec<f32>> {
    let tensor = gguf
        .tensor(name)
        .ok_or_else(|| ModelError::Unsupported(format!("mmproj tensor {name:?} not found")))?;
    let mut payload = vec![0u8; tensor.byte_size as usize];
    source_file.read_exact_at(&mut payload, tensor.file_offset)?;

    match tensor.ggml_type {
        GgmlType::F32 => Ok(payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4-byte chunk")))
            .collect()),
        GgmlType::F16 => Ok(payload
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().expect("2-byte chunk"))))
            .collect()),
        GgmlType::Bf16 => Ok(payload
            .chunks_exact(2)
            .map(|b| {
                f32::from_bits(
                    (u16::from_le_bytes(b.try_into().expect("2-byte chunk")) as u32) << 16,
                )
            })
            .collect()),
        GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 => {
            let block_bytes = tensor.ggml_type.block_bytes() as usize;
            let mut values = Vec::with_capacity(tensor.n_elements as usize);
            for block in payload.chunks_exact(block_bytes) {
                values.extend(
                    dequantize_block(tensor.ggml_type, block)
                        .expect("canonical quant decoder exists"),
                );
            }
            if values.len() != tensor.n_elements as usize {
                return Err(ModelError::Shape {
                    tensor: "vision tensor decoded elements",
                    expected: tensor.n_elements as usize,
                    actual: values.len(),
                }
                .into());
            }
            Ok(values)
        }
        other => Err(ModelError::Unsupported(format!(
            "vision converter does not support GGML type {}",
            other.ggml_id()
        ))
        .into()),
    }
}

fn write_one(
    writer: &mut TqfWriter,
    gguf: &GgufFile,
    source_file: &std::fs::File,
    role: VisionTensorRole,
    layer: Option<u8>,
) -> Result<u64> {
    let name = role.gguf_name(layer);
    let values = read_tensor_f32(gguf, source_file, &name)?;
    let dims = gguf.tensor(&name).expect("just read").dims.clone();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let stored = bytes.len() as u64;
    writer.write_extent(
        role as u32,
        &name,
        layer.map(LayerId),
        TqfSectionKind::ResidentCore,
        &dims,
        F32_DTYPE_ID,
        TQF_QUANT_PASSTHROUGH_F32,
        64,
        &bytes,
    )?;
    Ok(stored)
}

/// Converts a verified `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` into a `.tqf`
/// container.
pub fn convert_vision_gguf(
    source_path: &Path,
    out_path: &Path,
    broker: &MemoryBroker,
) -> Result<VisionConversionReport> {
    let source_sha256 = sha256_hex(source_path)?;
    let gguf = crate::format::gguf::open_with_broker(source_path, broker)?;
    let source_file = std::fs::File::open(source_path)?;
    let header = vision_header(source_sha256);
    let mut writer = TqfWriter::create_partial(out_path, header)?;

    let mut verified_output_bytes = 0u64;
    for role in [
        VisionTensorRole::PatchEmbedWeight0,
        VisionTensorRole::PatchEmbedWeight1,
        VisionTensorRole::PatchEmbedBias,
        VisionTensorRole::PositionEmbed,
        VisionTensorRole::PostLnWeight,
        VisionTensorRole::PostLnBias,
        VisionTensorRole::MergerFc1Weight,
        VisionTensorRole::MergerFc1Bias,
        VisionTensorRole::MergerFc2Weight,
        VisionTensorRole::MergerFc2Bias,
    ] {
        verified_output_bytes += write_one(&mut writer, &gguf, &source_file, role, None)?;
    }

    let mut extent_count = 10;
    for layer in 0..VisionGeometry::LAYERS as u8 {
        for role in VisionTensorRole::ALL_PER_LAYER {
            verified_output_bytes +=
                write_one(&mut writer, &gguf, &source_file, role, Some(layer))?;
            extent_count += 1;
        }
    }

    writer.sync_payload()?;
    writer.commit()?;

    Ok(VisionConversionReport {
        extent_count,
        verified_output_bytes,
        source_sha256,
    })
}
