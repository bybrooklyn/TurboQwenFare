//! Trusted model receipt (spec Part V section 36). This is a minimal
//! placeholder schema: the full provenance/checksum contract (source hash,
//! format version, root checksum) lands with the `.tqf` container work
//! (phases 6-7). For now it only has to answer one question honestly —
//! "is a canonical model installed" — for the first-run flow.

use std::path::Path;

use serde::{Deserialize, Serialize};

pub const RECEIPT_FILE_NAME: &str = "qwen3.6-35b-a3b.receipt.toml";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReceipt {
    pub schema_version: u32,
    pub model_family: String,
    pub source_revision: Option<String>,
    pub installed_at_unix: u64,
}

/// Missing and unparseable receipts are both "invalid" for the purposes of
/// the section-28 state machine, which does not distinguish the two before
/// triggering model setup.
pub fn load_trusted_receipt(receipts_dir: &Path) -> Option<ModelReceipt> {
    let path = receipts_dir.join(RECEIPT_FILE_NAME);
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_receipt_is_none() {
        let dir = std::env::temp_dir().join(format!("tqf-receipt-test-{}", std::process::id()));
        assert!(load_trusted_receipt(&dir).is_none());
    }
}
