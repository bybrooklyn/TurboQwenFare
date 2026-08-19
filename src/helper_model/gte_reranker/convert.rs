//! Safetensors -> `.tqf` conversion for the GTE reranker helper model
//! (spec §43: same "helper `.tqf` conversion" contract as Phase 37's
//! pplx-embed converter, reusing that phase's simpler non-journaled
//! `TqfWriter` path — this checkpoint is ~600 MB, converted once ahead
//! of the transient broker lease that loads it).

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::format::quant::repack::TQF_QUANT_PASSTHROUGH_F32;
use crate::format::tqf::{TqfHeaderInfo, TqfSectionKind, TqfWriter};
use crate::helper_model::safetensors::SafetensorsFile;
use crate::ids::LayerId;

use super::geometry::GteRerankerGeometry;
use super::roles::GteTensorRole;

pub const MODEL_FAMILY_LABEL: &[u8] = b"gte-reranker-modernbert-base";
pub const CONVERSION_FINGERPRINT_LABEL: &[u8] = b"tqf-gte-reranker-f32-passthrough-v1";

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

pub fn gte_reranker_header(source_sha256: [u8; 32]) -> TqfHeaderInfo {
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
pub struct GteConversionReport {
    pub extent_count: usize,
    pub verified_output_bytes: u64,
    pub source_sha256: [u8; 32],
}

fn write_one(
    writer: &mut TqfWriter,
    source: &SafetensorsFile,
    role: GteTensorRole,
    layer: Option<u8>,
) -> Result<u64> {
    let name = role.safetensors_name(layer);
    let values = source.read_f32(&name)?;
    let entry = source.entry(&name).expect("just read");
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let stored = bytes.len() as u64;
    writer.write_extent(
        role as u32,
        &name,
        layer.map(LayerId),
        TqfSectionKind::ResidentCore,
        &entry.shape,
        F32_DTYPE_ID,
        TQF_QUANT_PASSTHROUGH_F32,
        64,
        &bytes,
    )?;
    Ok(stored)
}

/// Converts a verified GTE reranker `model.safetensors` into a `.tqf`
/// container. Layer 0 has no real `attn_norm` tensor (spec's own
/// architecture quirk, confirmed against the real checkpoint) so it is
/// simply not written for layer 0 — `weights.rs` treats its absence as
/// `Identity`, not a missing-tensor error.
pub fn convert_gte_reranker_safetensors(
    source_path: &Path,
    out_path: &Path,
) -> Result<GteConversionReport> {
    let source_sha256 = sha256_hex(source_path)?;
    let source = SafetensorsFile::open(source_path)?;
    let header = gte_reranker_header(source_sha256);
    let mut writer = TqfWriter::create_partial(out_path, header)?;

    let mut verified_output_bytes = 0u64;
    for role in [
        GteTensorRole::TokenEmbedding,
        GteTensorRole::EmbeddingNorm,
        GteTensorRole::FinalNorm,
        GteTensorRole::HeadDense,
        GteTensorRole::HeadNorm,
        GteTensorRole::ClassifierWeight,
        GteTensorRole::ClassifierBias,
    ] {
        verified_output_bytes += write_one(&mut writer, &source, role, None)?;
    }

    let mut extent_count = 7;
    for layer in 0..GteRerankerGeometry::NUM_LAYERS as u8 {
        if layer != 0 {
            verified_output_bytes +=
                write_one(&mut writer, &source, GteTensorRole::AttnNorm, Some(layer))?;
            extent_count += 1;
        }
        for role in GteTensorRole::ALL_PER_LAYER {
            verified_output_bytes += write_one(&mut writer, &source, role, Some(layer))?;
            extent_count += 1;
        }
    }

    writer.sync_payload()?;
    writer.commit()?;

    Ok(GteConversionReport {
        extent_count,
        verified_output_bytes,
        source_sha256,
    })
}
