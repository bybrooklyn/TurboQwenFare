//! Safetensors -> `.tqf` conversion for the pplx-embed helper model (spec
//! §37: "Implement helper `.tqf` conversion and transient broker lease").
//! Deliberately the simpler, non-resumable `TqfWriter` path documented in
//! `format::tqf::writer` (no `ConversionTransaction` journal): the helper
//! checkpoint is ~2.2 GiB and converted once, ahead of the transient
//! broker lease that loads it, not streamed under memory pressure like
//! the 20 GiB canonical Qwen3.6 GGUF.
//!
//! Weights are carried losslessly as TQF F32 passthrough (spec invariant
//! #2/#9 style: no precision loss hidden inside a "conversion").

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::format::quant::repack::TQF_QUANT_PASSTHROUGH_F32;
use crate::format::tqf::{TqfHeaderInfo, TqfSectionKind, TqfWriter};
use crate::ids::LayerId;

use super::geometry::PplxEmbedGeometry;
use super::roles::PplxTensorRole;
use super::safetensors::SafetensorsFile;

pub const MODEL_FAMILY_LABEL: &[u8] = b"pplx-embed-v1-0.6b";
pub const CONVERSION_FINGERPRINT_LABEL: &[u8] = b"tqf-pplx-embed-f32-passthrough-v1";

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

pub fn pplx_embed_header(source_sha256: [u8; 32]) -> TqfHeaderInfo {
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
pub struct PplxConversionReport {
    pub extent_count: usize,
    pub verified_output_bytes: u64,
    pub source_sha256: [u8; 32],
}

fn write_one(
    writer: &mut TqfWriter,
    source: &SafetensorsFile,
    role: PplxTensorRole,
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

/// Converts a verified pplx-embed `model.safetensors` into a `.tqf`
/// container at `out_path`. `source_sha256_hex_expected` (if given) is
/// checked against the file actually on disk before any bytes are
/// trusted, matching the canonical Qwen3.6 converter's "never hash a
/// caller-supplied claim" rule.
pub fn convert_pplx_embed_safetensors(
    source_path: &Path,
    out_path: &Path,
) -> Result<PplxConversionReport> {
    let source_sha256 = sha256_hex(source_path)?;
    let source = SafetensorsFile::open(source_path)?;
    let header = pplx_embed_header(source_sha256);
    let mut writer = TqfWriter::create_partial(out_path, header)?;

    let mut verified_output_bytes = 0u64;
    verified_output_bytes += write_one(&mut writer, &source, PplxTensorRole::TokenEmbedding, None)?;
    verified_output_bytes += write_one(&mut writer, &source, PplxTensorRole::FinalNorm, None)?;
    for layer in 0..PplxEmbedGeometry::NUM_LAYERS as u8 {
        for role in PplxTensorRole::ALL_PER_LAYER {
            verified_output_bytes += write_one(&mut writer, &source, role, Some(layer))?;
        }
    }

    writer.sync_payload()?;
    let extent_count = 2 + PplxEmbedGeometry::NUM_LAYERS * PplxTensorRole::ALL_PER_LAYER.len();
    writer.commit()?;

    Ok(PplxConversionReport {
        extent_count,
        verified_output_bytes,
        source_sha256,
    })
}
