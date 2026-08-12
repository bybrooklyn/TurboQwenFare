//! Canonical pinned source constants (spec §13, "Canonical checkpoint
//! set"; spec §272, Phase 0 "research harvest and canonical manifest").
//!
//! Resolved against the live `ggml-org/Qwen3.6-35B-A3B-GGUF` repository on
//! 2026-08-11 (see `docs/research/canonical-source-manifest.md` for the
//! full research record, cross-checks, and how to re-verify). The spec
//! itself only publishes the Q4_K_M SHA-256 (§13); the exact commit and
//! filenames below were fetched live, not invented — that fetch is
//! recorded in the research ledger for anyone who wants to re-derive it.
//!
//! Per the spec's pinned-source rule (§13): "The exact source commit/hash
//! used by a TQF release is part of the release contract... startup must
//! never silently swap model bytes under an existing benchmark/correctness
//! profile." `REVISION` below is that immutable commit, not `"main"`.

pub const REPO_ID: &str = "ggml-org/Qwen3.6-35B-A3B-GGUF";

/// Immutable commit on `ggml-org/Qwen3.6-35B-A3B-GGUF`, fetched via the HF
/// API's `sha` field on 2026-08-11 and cross-checked against a second,
/// independent fetch of the same endpoint (both agreed).
pub const REVISION: &str = "baec3ebee244827cda0f4557eafa8b28f7545fa6";

pub const LANGUAGE_CHECKPOINT_FILENAME: &str = "Qwen3.6-35B-A3B-Q4_K_M.gguf";
pub const MTP_FILENAME: &str = "mtp-Qwen3.6-35B-A3B-Q4_0.gguf";
pub const VISION_PROJECTOR_FILENAME: &str = "mmproj-Qwen3.6-35B-A3B-Q8_0.gguf";

/// spec §13: published in the spec text itself and confirmed to match the
/// live repository's LFS metadata exactly.
pub const LANGUAGE_CHECKPOINT_SHA256: &str =
    "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7";
pub const LANGUAGE_CHECKPOINT_SIZE_BYTES: u64 = 20_419_565_568;

/// Not published anywhere in the spec text; fetched from the live
/// repository's LFS metadata on 2026-08-11 (not independently re-verified
/// by a second fetch, unlike `REVISION` and the language checkpoint hash
/// above — treat with slightly less confidence until re-checked).
pub const MTP_SHA256: &str = "606fca331adcbfbdadc107512ce6a7161e84e1646ba0e0018256426f6296877f";
pub const MTP_SIZE_BYTES: u64 = 1_060_038_432;

pub const VISION_PROJECTOR_SHA256: &str =
    "904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d";
pub const VISION_PROJECTOR_SIZE_BYTES: u64 = 614_194_304;

pub const LICENSE_ID: &str = "apache-2.0";

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sha256_hex(s: &str) -> bool {
        s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
    }

    #[test]
    fn all_published_hashes_are_64_hex_chars() {
        assert!(is_sha256_hex(LANGUAGE_CHECKPOINT_SHA256));
        assert!(is_sha256_hex(MTP_SHA256));
        assert!(is_sha256_hex(VISION_PROJECTOR_SHA256));
    }

    #[test]
    fn revision_is_a_40_char_commit_not_a_branch_name() {
        // The pinned-source rule (§13) requires an immutable commit, never
        // a moving ref like "main" — a real git/HF commit sha is exactly
        // 40 lowercase hex characters.
        assert_eq!(REVISION.len(), 40);
        assert!(REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(REVISION, "main");
    }

    #[test]
    fn filenames_are_gguf() {
        assert!(LANGUAGE_CHECKPOINT_FILENAME.ends_with(".gguf"));
        assert!(MTP_FILENAME.ends_with(".gguf"));
        assert!(VISION_PROJECTOR_FILENAME.ends_with(".gguf"));
    }
}
