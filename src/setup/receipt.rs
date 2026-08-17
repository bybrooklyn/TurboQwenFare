//! Trusted model receipt (spec Part V section 36). A receipt is written only
//! after a `.tqf` conversion atomically commits and its metadata reopens
//! successfully. It binds the receipt to that concrete container, rather
//! than treating an arbitrary TOML file as proof that a model is installed.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::persisted::atomic_write_toml;
use crate::error::Result;
use crate::format::gguf;
use crate::format::tqf::{
    canonical_header, ConversionReport, TqfReader, FORMAT_MAJOR, FORMAT_MINOR,
};
use crate::memory::MemoryBroker;
use crate::model::qwen36::weights::Qwen36WeightManifest;

pub const RECEIPT_FILE_NAME: &str = "qwen3.6-35b-a3b.receipt.toml";
pub const CANONICAL_MODEL_FAMILY: &str = "qwen3.6-35b-a3b";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelReceipt {
    pub schema_version: u32,
    pub model_family: String,
    pub source_revision: Option<String>,
    pub source_sha256: String,
    pub conversion_fingerprint_blake3: String,
    pub metadata_root_blake3: String,
    pub format_major: u16,
    pub format_minor: u16,
    pub tqf_path: PathBuf,
    /// The verified GGUF remains the authoritative tokenizer metadata source.
    /// TQF v1 stores weight extents losslessly but does not duplicate the
    /// vocab/merge tables, so it cannot be discarded after conversion.
    pub tokenizer_gguf_path: PathBuf,
    /// BLAKE3 of the parsed GGUF header, metadata, and tensor directory. It
    /// covers vocab/merge metadata without rehashing 20+ GiB of immutable
    /// tensor payload that has already been copied into TQF.
    pub tokenizer_header_blake3: String,
    pub tokenizer_source_bytes: u64,
    pub installed_at_unix: u64,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Opens the container named by the receipt and checks all immutable
/// provenance fields against its validated superblock. Payload checksums are
/// deliberately verified lazily by `TqfReader` as each extent is loaded;
/// eagerly reading a multi-gigabyte model at every launch would violate the
/// bounded startup/memory contract.
pub fn validate_trusted_receipt(receipt: &ModelReceipt, broker: &MemoryBroker) -> Result<()> {
    if receipt.schema_version != 4 || receipt.model_family != CANONICAL_MODEL_FAMILY {
        return Err(
            crate::error::ModelError::Unsupported("invalid trusted receipt".to_string()).into(),
        );
    }
    if receipt.source_sha256 == crate::source::pinned::LANGUAGE_CHECKPOINT_SHA256
        && receipt.source_revision.as_deref() != Some(crate::source::pinned::REVISION)
    {
        return Err(crate::error::ModelError::Unsupported(
            "canonical receipt is not bound to the release's immutable source revision".to_string(),
        )
        .into());
    }
    let expected = canonical_header(&receipt.source_sha256)?;
    let reader = TqfReader::open_validated_with_broker(&receipt.tqf_path, broker)?;
    let superblock = &reader.superblock;
    if receipt.format_major != FORMAT_MAJOR
        || receipt.format_minor != FORMAT_MINOR
        || superblock.format_major != receipt.format_major
        || superblock.format_minor != receipt.format_minor
        || superblock.model_family_id != expected.model_family_id
        || superblock.source_sha256 != expected.source_sha256
        || superblock.conversion_fingerprint != expected.conversion_fingerprint
        || receipt.conversion_fingerprint_blake3 != hex(&superblock.conversion_fingerprint)
        || receipt.metadata_root_blake3 != hex(&superblock.metadata_root_blake3)
    {
        return Err(crate::error::ModelError::Unsupported(
            "trusted receipt does not match its TQF container".to_string(),
        )
        .into());
    }
    drop(reader);
    // Provenance alone is not enough to call an installation usable. The
    // fixed Qwen graph must still be present with its canonical roles and
    // dimensions on every later launch, not merely at conversion time.
    Qwen36WeightManifest::open_with_broker(&receipt.tqf_path, broker)?;
    let tokenizer = gguf::open_with_broker(&receipt.tokenizer_gguf_path, broker)?;
    if tokenizer.source_bytes() != receipt.tokenizer_source_bytes
        || hex(&tokenizer.header_blake3()) != receipt.tokenizer_header_blake3
    {
        return Err(crate::error::ModelError::Unsupported(
            "trusted tokenizer GGUF header no longer matches the converted model source"
                .to_string(),
        )
        .into());
    }
    Ok(())
}

/// Writes a receipt only after verifying the just-converted container. The
/// sibling temp-and-rename write means a crash cannot replace a good receipt
/// with a partial one.
pub fn write_trusted_receipt(
    receipts_dir: &Path,
    report: &ConversionReport,
    source_revision: Option<String>,
    tokenizer_gguf_path: PathBuf,
    broker: &MemoryBroker,
) -> Result<ModelReceipt> {
    let tokenizer = gguf::open_with_broker(&tokenizer_gguf_path, broker)?;
    let receipt = ModelReceipt {
        schema_version: 4,
        model_family: CANONICAL_MODEL_FAMILY.to_string(),
        source_revision,
        source_sha256: hex(&report.source_sha256),
        conversion_fingerprint_blake3: hex(
            &canonical_header(&hex(&report.source_sha256))?.conversion_fingerprint
        ),
        metadata_root_blake3: hex(&report.metadata_root_blake3),
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        tqf_path: report.path.clone(),
        tokenizer_gguf_path,
        tokenizer_header_blake3: hex(&tokenizer.header_blake3()),
        tokenizer_source_bytes: tokenizer.source_bytes(),
        installed_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
    };
    drop(tokenizer);
    validate_trusted_receipt(&receipt, broker)?;
    atomic_write_toml(&receipts_dir.join(RECEIPT_FILE_NAME), &receipt)?;
    Ok(receipt)
}

/// Missing, malformed, and provenance-mismatched receipts are all invalid
/// for the section-28 state machine and trigger setup rather than a false
/// successful start.
pub fn load_trusted_receipt(receipts_dir: &Path, broker: &MemoryBroker) -> Option<ModelReceipt> {
    let path = receipts_dir.join(RECEIPT_FILE_NAME);
    let text = std::fs::read_to_string(path).ok()?;
    let receipt: ModelReceipt = toml::from_str(&text).ok()?;
    validate_trusted_receipt(&receipt, broker).ok()?;
    Some(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::tqf::convert_canonical_gguf;
    use crate::ids::Bytes;
    use crate::memory::MemoryBroker;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tqf-receipt-test-{}-{}",
            std::process::id(),
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_receipt_is_none() {
        let dir = fixture_dir();
        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        assert!(load_trusted_receipt(&dir, &broker).is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn receipt_rejects_a_provenanced_but_incomplete_container() {
        let dir = fixture_dir();
        let source = dir.join("fixture.gguf");
        // GGUF with one classified Q4_0 token embedding tensor.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let name = b"token_embd.weight";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&32u64.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend_from_slice(&[0x42; 18]);
        std::fs::write(&source, bytes).unwrap();

        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        let report =
            convert_canonical_gguf(&source, &"cd".repeat(32), &dir.join("model.tqf"), &broker)
                .unwrap();
        // Provenance validation alone succeeds for the tiny container, but
        // it lacks the fixed forty-layer graph and must never become Ready.
        assert!(
            write_trusted_receipt(&dir, &report, Some("pin".to_string()), source, &broker).is_err()
        );
        assert!(load_trusted_receipt(&dir, &broker).is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
