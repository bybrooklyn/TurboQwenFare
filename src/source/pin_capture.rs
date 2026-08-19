//! Captures the pinned-source constants for an artifact from a local copy
//! of it (spec §13's pinned-source rule; §272's Phase 0 manifest work).
//!
//! Why this exists rather than a live fetch: the canonical Qwen constants
//! in `super::pinned` were resolved against the live HuggingFace API and
//! recorded in `docs/research/canonical-source-manifest.md`. The helper
//! models (spec §37) have no such record, and the environment this was
//! written in cannot reach HuggingFace — its egress policy denies the
//! host, and inventing a plausible-looking SHA-256 is exactly the failure
//! spec §13 exists to prevent ("startup must never silently swap model
//! bytes under an existing benchmark/correctness profile").
//!
//! So the hash is derived from the artifact that will actually be served,
//! which is a stronger guarantee than one copied from a registry listing:
//! it is the file's real content, verified to have the right structure
//! before being recorded.
//!
//! Run it against a local checkpoint with:
//!
//! ```text
//! TQF_PPLX_SAFETENSORS=/path/to/model.safetensors just pin-helper-model
//! ```

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::helper_model::geometry::PplxEmbedGeometry;
use crate::helper_model::roles::PplxTensorRole;
use crate::helper_model::safetensors::SafetensorsFile;

/// What a local artifact turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedPin {
    pub file_name: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub tensor_count: usize,
}

/// Streams the file in chunks rather than reading it whole: these
/// checkpoints run to gigabytes, and the memory broker governs large
/// resident allocations (spec §115 invariant 4). Hashing needs no
/// residency at all.
fn sha256_of(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    let digest = hasher.finalize();
    Ok((
        digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        total,
    ))
}

/// Confirms the file really is a pplx-embed checkpoint before its hash is
/// recorded as the pin.
///
/// Hashing whatever happens to sit at a path would faithfully pin the
/// wrong model. This checks the structure the runtime actually depends on:
/// every per-layer tensor for all 28 layers, plus the two layer-independent
/// ones, each present by the exact name `roles::safetensors_name` derives.
pub fn verify_pplx_embed_structure(file: &SafetensorsFile) -> Result<usize> {
    let mut missing = Vec::new();

    for role in [PplxTensorRole::TokenEmbedding, PplxTensorRole::FinalNorm] {
        let name = role.safetensors_name(None);
        if file.entry(&name).is_none() {
            missing.push(name);
        }
    }
    for layer in 0..PplxEmbedGeometry::NUM_LAYERS as u8 {
        for role in PplxTensorRole::ALL_PER_LAYER {
            let name = role.safetensors_name(Some(layer));
            if file.entry(&name).is_none() {
                missing.push(name);
            }
        }
    }

    if !missing.is_empty() {
        return Err(crate::error::ModelError::Unsupported(format!(
            "this file is not a pplx-embed-v1-0.6b checkpoint: {} expected tensors are absent \
             (first missing: {})",
            missing.len(),
            missing[0]
        ))
        .into());
    }

    Ok(file.names().count())
}

/// Verifies the structure, then hashes the bytes.
pub fn capture(path: &Path) -> Result<CapturedPin> {
    let file = SafetensorsFile::open(path)?;
    let tensor_count = verify_pplx_embed_structure(&file)?;
    let (sha256, size_bytes) = sha256_of(path)?;

    Ok(CapturedPin {
        file_name: path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        size_bytes,
        sha256,
        tensor_count,
    })
}

/// Renders the captured values as the Rust block to paste into
/// `source::pinned`, so the numbers are transcribed by copy rather than
/// by hand.
pub fn render_pin_block(pin: &CapturedPin, revision: &str) -> String {
    format!(
        "/// pplx-embed-v1-0.6b, the helper embedding model (spec §37).\n\
         ///\n\
         /// Captured from a local copy with `just pin-helper-model` on a\n\
         /// machine that has it; the structure was verified against\n\
         /// `helper_model::roles` ({} tensors, all {} layers present)\n\
         /// before the hash was taken.\n\
         pub const EMBED_REPO_ID: &str = \"perplexity-ai/pplx-embed-v1-0.6b\";\n\
         pub const EMBED_REVISION: &str = \"{}\";\n\
         pub const EMBED_FILENAME: &str = \"{}\";\n\
         pub const EMBED_SHA256: &str =\n    \"{}\";\n\
         pub const EMBED_SIZE_BYTES: u64 = {};\n",
        pin.tensor_count,
        PplxEmbedGeometry::NUM_LAYERS,
        revision,
        pin.file_name,
        pin.sha256,
        pin.size_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The structure check must reject a file that parses as safetensors
    /// but is a different model — otherwise the pin faithfully records the
    /// wrong weights.
    #[test]
    fn a_file_without_the_expected_tensors_is_rejected() {
        let dir = std::env::temp_dir().join(format!("tqf-pin-capture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not-pplx.safetensors");

        // A minimal, structurally valid safetensors file holding one
        // tensor that pplx-embed does not have.
        let header = br#"{"some.other.tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let error = capture(&path).expect_err("a non-pplx file must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("not a pplx-embed"),
            "the error must say what the file is not: {message}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Builds a structurally correct (but tiny) pplx-embed checkpoint:
    /// every expected tensor name, one element each.
    fn synthetic_pplx_embed(path: &Path) -> usize {
        synthetic_pplx_embed_omitting(path, None)
    }

    /// `omit` drops one tensor, modelling a truncated or partially
    /// downloaded checkpoint.
    fn synthetic_pplx_embed_omitting(path: &Path, omit: Option<&str>) -> usize {
        let mut names = vec![
            PplxTensorRole::TokenEmbedding.safetensors_name(None),
            PplxTensorRole::FinalNorm.safetensors_name(None),
        ];
        for layer in 0..PplxEmbedGeometry::NUM_LAYERS as u8 {
            for role in PplxTensorRole::ALL_PER_LAYER {
                names.push(role.safetensors_name(Some(layer)));
            }
        }

        if let Some(omit) = omit {
            let before = names.len();
            names.retain(|name| name != omit);
            assert_eq!(before - names.len(), 1, "the omitted tensor must exist");
        }

        let entries: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let start = index * 4;
                format!(
                    "{:?}:{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}",
                    name,
                    start,
                    start + 4
                )
            })
            .collect();
        let header = format!("{{{}}}", entries.join(","));

        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for index in 0..names.len() {
            bytes.extend_from_slice(&(index as f32).to_le_bytes());
        }
        std::fs::write(path, &bytes).unwrap();
        names.len()
    }

    /// The complement of the rejection test: a check that refused every
    /// file would pass that one and still be useless. This proves the
    /// structure check accepts a correct file, and that the tensor count
    /// it reports is the real 2 + 28x11 the runtime needs — which is also
    /// the 310 the Phase 37 qualification record measured against the
    /// real checkpoint.
    #[test]
    fn a_structurally_correct_checkpoint_is_accepted_and_hashed() {
        let dir = std::env::temp_dir().join(format!("tqf-pin-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let expected = synthetic_pplx_embed(&path);
        assert_eq!(expected, 310, "pplx-embed has 2 + 28x11 tensors");

        let pin = capture(&path).expect("a correct checkpoint must be accepted");
        assert_eq!(pin.tensor_count, 310);
        assert_eq!(pin.file_name, "model.safetensors");
        assert_eq!(pin.size_bytes, std::fs::metadata(&path).unwrap().len());
        assert_eq!(pin.sha256.len(), 64);

        // The hash must be the file's real content hash, not a stand-in.
        let expected_hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(std::fs::read(&path).unwrap());
            hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(pin.sha256, expected_hash);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint missing exactly one layer's tensor must still be
    /// rejected — the failure mode is a truncated or partially-downloaded
    /// file, which is precisely what a pin is supposed to catch.
    /// A checkpoint missing exactly one tensor must still be rejected —
    /// the failure mode is a truncated or partially downloaded file, which
    /// is precisely what verifying before pinning is supposed to catch.
    #[test]
    fn a_checkpoint_missing_one_tensor_is_rejected() {
        let dir = std::env::temp_dir().join(format!("tqf-pin-partial-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        let dropped = PplxTensorRole::MlpDownProj
            .safetensors_name(Some(PplxEmbedGeometry::NUM_LAYERS as u8 - 1));
        let remaining = synthetic_pplx_embed_omitting(&path, Some(&dropped));
        assert_eq!(remaining, 309);

        let error = capture(&path).expect_err("a partial checkpoint must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("not a pplx-embed"),
            "expected a structure rejection naming the model, got: {message}"
        );
        assert!(
            message.contains(&dropped),
            "the error must name the missing tensor, got: {message}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_rendered_block_carries_the_captured_values_verbatim() {
        let pin = CapturedPin {
            file_name: "model.safetensors".to_string(),
            size_bytes: 2_271_657_216,
            sha256: "a".repeat(64),
            tensor_count: 310,
        };
        let block = render_pin_block(&pin, "deadbeef");

        assert!(block.contains("\"model.safetensors\""));
        assert!(block.contains("2271657216"));
        assert!(block.contains(&"a".repeat(64)));
        assert!(block.contains("\"deadbeef\""));
        assert!(block.contains("310 tensors"));
    }

    /// The capture tool itself. `#[ignore]`d because it needs the real
    /// 2.2 GiB checkpoint; run it on a machine that has one:
    ///
    /// ```text
    /// TQF_PPLX_SAFETENSORS=/path/to/model.safetensors just pin-helper-model
    /// ```
    #[test]
    #[ignore = "requires a local pplx-embed-v1-0.6b safetensors checkpoint to hash"]
    fn print_pinned_constants_for_the_local_helper_checkpoint() {
        let path = std::env::var("TQF_PPLX_SAFETENSORS")
            .expect("set TQF_PPLX_SAFETENSORS to the local checkpoint");
        // The immutable commit cannot be derived from the file itself, so
        // it is supplied: read it from the model page's "Files and
        // versions" tab, or `git rev-parse HEAD` in a cloned repo.
        let revision = std::env::var("TQF_PPLX_REVISION")
            .unwrap_or_else(|_| "<set TQF_PPLX_REVISION to the immutable commit>".to_string());

        let pin = capture(Path::new(&path)).expect("capture");

        println!("\n=== verified ===");
        println!("file:     {}", pin.file_name);
        println!("bytes:    {}", pin.size_bytes);
        println!("sha256:   {}", pin.sha256);
        println!("tensors:  {}", pin.tensor_count);
        println!("\n=== paste into src/source/pinned.rs ===\n");
        println!("{}", render_pin_block(&pin, &revision));
    }
}
