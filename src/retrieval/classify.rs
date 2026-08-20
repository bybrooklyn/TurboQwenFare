//! Content-first file classification (spec §79-80, §178-179; Phase 35,
//! spec §307). "Path names are almost meaningless as content
//! classifiers... TQF must therefore identify file type from bytes and
//! syntax, using extension/path only as weak priors."
//!
//! The full pipeline (spec §178) calls for trying real Tree-sitter-class
//! parsers on top candidate languages and scoring `parser_quality` from
//! real AST coverage/error-node ratio. That is out of scope here — no
//! grammar-parsing dependency is added in this phase. Instead, "parse top
//! candidate grammars" is replaced by a cheaper, real, syntax-*fingerprint*
//! density score (keyword/construct pattern matching, no AST), and the
//! reference scoring weights are renormalized onto that substitute signal.
//! This is recorded honestly rather than claimed as the spec's real
//! parser-quality metric — see the qualification doc's "Status" section.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    Code,
    Configuration,
    StructuredData,
    Documentation,
    PlainText,
    Binary,
    Asset,
    UnknownText,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub kind: ContentKind,
    pub language: Option<&'static str>,
    /// Calibrated [0,1] confidence in `language` (0 when `language` is
    /// `None`).
    pub confidence: f32,
    pub generated_probability: f32,
    pub vendor_probability: f32,
}

const SAMPLE_BYTES: usize = 64 * 1024;

/// Spec §178's byte-sniff stage. Checked before any language fingerprint
/// work — a confidently-binary file never reaches syntax scoring.
struct ByteSniff {
    is_valid_utf8: bool,
    has_nul: bool,
    control_ratio: f32,
    entropy_bits: f32,
}

fn sniff(sample: &[u8]) -> ByteSniff {
    let has_nul = sample.contains(&0);
    let is_valid_utf8 = std::str::from_utf8(sample).is_ok();
    let control_count = sample
        .iter()
        .filter(|&&b| b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r')
        .count();
    let control_ratio = if sample.is_empty() {
        0.0
    } else {
        control_count as f32 / sample.len() as f32
    };

    let mut histogram = [0u32; 256];
    for &b in sample {
        histogram[b as usize] += 1;
    }
    let len = sample.len().max(1) as f32;
    let entropy_bits = histogram
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f32 / len;
            -p * p.log2()
        })
        .sum();

    ByteSniff {
        is_valid_utf8,
        has_nul,
        control_ratio,
        entropy_bits,
    }
}

/// Known binary magic numbers (spec §178 "check known binary magic").
const MAGIC_SIGNATURES: &[(&[u8], &str)] = &[
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"%PDF-", "PDF"),
    (b"\x7fELF", "ELF"),
    (b"PK\x03\x04", "ZIP"),
    (b"\x1f\x8b", "GZIP"),
    (b"\x00asm", "WASM"),
    (b"\xca\xfe\xba\xbe", "Java class"),
    (b"MZ", "PE/EXE"),
    (b"BM", "BMP"),
    (b"RIFF", "RIFF (WAV/AVI/WEBP)"),
];

fn magic_match(sample: &[u8]) -> Option<&'static str> {
    MAGIC_SIGNATURES
        .iter()
        .find(|(magic, _)| sample.starts_with(magic))
        .map(|(_, name)| *name)
}

/// One candidate language's cheap syntax fingerprint (spec §178 "generate
/// candidate languages from syntax-token fingerprints"): substrings that,
/// when present, are strong-ish evidence for the language, each with a
/// weight. Not exhaustive — a REFERENCE BASELINE covering the spec §81
/// "initial broad set" languages most likely to appear in a real repo.
struct LanguageFingerprint {
    name: &'static str,
    kind: ContentKind,
    extensions: &'static [&'static str],
    shebang_hints: &'static [&'static str],
    markers: &'static [(&'static str, f32)],
}

const FINGERPRINTS: &[LanguageFingerprint] = &[
    LanguageFingerprint {
        name: "Rust",
        kind: ContentKind::Code,
        extensions: &["rs"],
        shebang_hints: &[],
        markers: &[
            ("fn ", 1.0),
            ("let mut ", 1.2),
            ("impl ", 1.2),
            ("pub struct ", 1.4),
            ("pub enum ", 1.3),
            ("use crate::", 1.5),
            ("#[derive(", 1.5),
            // `//!`/`///` doc comments and `#[...]` attributes are
            // essentially unmistakable Rust syntax (no other language
            // here uses either), so they anchor classification for
            // real, doc-comment-heavy Rust files whose *code* markers
            // are sparse (thin `mod.rs` re-export files, files that are
            // mostly a module-level doc comment) — exactly the
            // false-negative case a real scan of this crate surfaced.
            ("//! ", 1.4),
            ("/// ", 1.1),
            ("#[cfg(", 1.3),
            ("pub mod ", 1.2),
            ("pub(crate)", 1.3),
            ("->", 0.3),
            ("::", 0.4),
        ],
    },
    LanguageFingerprint {
        name: "Python",
        kind: ContentKind::Code,
        extensions: &["py"],
        shebang_hints: &["python"],
        markers: &[
            ("def ", 1.2),
            ("import ", 0.8),
            ("elif ", 1.4),
            ("self.", 0.8),
            ("__init__", 1.4),
            ("    return ", 0.6),
        ],
    },
    LanguageFingerprint {
        name: "TypeScript",
        kind: ContentKind::Code,
        extensions: &["ts", "tsx"],
        shebang_hints: &[],
        markers: &[
            ("interface ", 1.3),
            ("export ", 0.7),
            (": string", 1.0),
            (": number", 1.0),
            ("import {", 0.9),
            ("=>", 0.3),
        ],
    },
    LanguageFingerprint {
        name: "JavaScript",
        kind: ContentKind::Code,
        extensions: &["js", "jsx", "mjs"],
        shebang_hints: &["node"],
        markers: &[
            ("function ", 0.9),
            ("const ", 0.6),
            ("require(", 1.0),
            ("module.exports", 1.3),
            ("=>", 0.4),
        ],
    },
    LanguageFingerprint {
        name: "Go",
        kind: ContentKind::Code,
        extensions: &["go"],
        shebang_hints: &[],
        markers: &[
            ("package ", 1.4),
            ("func ", 1.0),
            (":= ", 1.2),
            ("import (", 1.1),
            ("fmt.", 0.9),
        ],
    },
    LanguageFingerprint {
        name: "C/C++",
        kind: ContentKind::Code,
        extensions: &["c", "h", "cpp", "cc", "hpp"],
        shebang_hints: &[],
        markers: &[
            ("#include", 1.4),
            ("int main(", 1.3),
            ("std::", 1.0),
            ("void ", 0.6),
            ("->", 0.2),
        ],
    },
    LanguageFingerprint {
        name: "Shell",
        kind: ContentKind::Code,
        extensions: &["sh", "bash"],
        shebang_hints: &["bash", "sh", "zsh"],
        markers: &[
            ("echo ", 0.7),
            ("if [ ", 1.1),
            ("fi\n", 0.9),
            ("export ", 0.5),
            ("$(", 0.4),
        ],
    },
    LanguageFingerprint {
        name: "Markdown",
        kind: ContentKind::Documentation,
        extensions: &["md", "markdown"],
        shebang_hints: &[],
        markers: &[
            ("\n# ", 1.0),
            ("\n## ", 1.0),
            ("```", 1.2),
            ("](", 0.9),
            ("**", 0.4),
        ],
    },
    LanguageFingerprint {
        name: "JSON",
        kind: ContentKind::StructuredData,
        extensions: &["json"],
        shebang_hints: &[],
        markers: &[("{\"", 1.2), ("\": ", 0.8), ("[]", 0.3), ("null", 0.4)],
    },
    LanguageFingerprint {
        name: "TOML",
        kind: ContentKind::Configuration,
        extensions: &["toml"],
        shebang_hints: &[],
        markers: &[("[dependencies]", 1.5), ("= \"", 0.6), ("[package]", 1.3)],
    },
    LanguageFingerprint {
        name: "YAML",
        kind: ContentKind::Configuration,
        extensions: &["yaml", "yml"],
        shebang_hints: &[],
        markers: &[("---\n", 1.0), (":\n", 0.3), ("- name:", 1.2)],
    },
];

/// Special basenames recognized independent of extension (spec §178
/// "known special basenames (Makefile/Dockerfile/etc.)").
const KNOWN_BASENAMES: &[(&str, &str, ContentKind)] = &[
    ("Makefile", "Makefile", ContentKind::Configuration),
    ("Dockerfile", "Dockerfile", ContentKind::Configuration),
    ("CMakeLists.txt", "CMake", ContentKind::Configuration),
    ("Cargo.toml", "TOML", ContentKind::Configuration),
    ("Cargo.lock", "TOML", ContentKind::StructuredData),
];

fn basename(path_hint: &str) -> &str {
    path_hint.rsplit('/').next().unwrap_or(path_hint)
}

fn extension(path_hint: &str) -> Option<&str> {
    let base = basename(path_hint);
    base.rsplit_once('.')
        .map(|(_, ext)| ext)
        .filter(|ext| !ext.is_empty() && *ext != base)
}

fn shebang_hint(text: &str) -> Option<&'static str> {
    let first_line = text.lines().next()?;
    if !first_line.starts_with("#!") {
        return None;
    }
    for fp in FINGERPRINTS {
        for hint in fp.shebang_hints {
            if first_line.contains(hint) {
                return Some(fp.name);
            }
        }
    }
    None
}

/// Scores every candidate language's marker density against `text`,
/// returning `(language, kind, fingerprint_score in [0,1])` for the
/// best match, or `None` if nothing scores above a noise floor.
/// Below this raw (pre-normalization) score, a "match" is indistinguishable
/// from an incidental collision with ordinary English prose — e.g. a doc
/// comment saying "the GGUF *import* reader" trivially contains Python's
/// `"import "` marker. One solid single-marker hit (the smallest markers
/// are weight ~0.8, but combined with the `1+ln(count)` bonus from even a
/// couple of repeats, or one higher-weight marker, clears this) is required
/// before a language is trusted over "no fingerprint signal" (which falls
/// through to the extension/shebang priors instead — see `classify`).
const MIN_RAW_SCORE: f32 = 1.0;

/// Every marker across every fingerprint, flattened once, plus the
/// automaton that finds them all in a single pass.
///
/// The obvious loop — `text.matches(marker)` for each marker of each
/// language — rescans the sample once per marker. With twelve languages
/// and a 64 KiB sample that is several megabytes of scanning per file,
/// and it measured as 94% of the entire scan phase (155 ms of 165 ms over
/// this repository's 259 files). One pass finds the same occurrences.
struct MarkerSet {
    /// `(fingerprint index, weight)` parallel to the automaton's pattern
    /// ids.
    owners: Vec<(usize, f32)>,
    automaton: aho_corasick::AhoCorasick,
}

fn marker_set() -> &'static MarkerSet {
    static SET: std::sync::OnceLock<MarkerSet> = std::sync::OnceLock::new();
    SET.get_or_init(|| {
        let mut patterns = Vec::new();
        let mut owners = Vec::new();
        for (index, fp) in FINGERPRINTS.iter().enumerate() {
            for (marker, weight) in fp.markers {
                patterns.push(*marker);
                owners.push((index, *weight));
            }
        }
        MarkerSet {
            automaton: aho_corasick::AhoCorasick::new(&patterns)
                .expect("the marker table is a fixed, valid pattern set"),
            owners,
        }
    })
}

fn best_fingerprint(text: &str) -> Option<(&'static str, ContentKind, f32)> {
    let set = marker_set();

    // `str::matches` is leftmost and non-overlapping *per pattern*: two
    // different markers may cover the same bytes, but one marker's own
    // matches never overlap each other. Overlapping search plus a
    // per-pattern end cursor reproduces that exactly — plain
    // `find_iter` would not, because a match of one pattern would
    // suppress a different pattern's match on the same bytes.
    let mut counts = vec![0u32; set.owners.len()];
    let mut last_end = vec![0usize; set.owners.len()];
    let mut started = vec![false; set.owners.len()];
    for m in set.automaton.find_overlapping_iter(text) {
        let id = m.pattern().as_usize();
        if !started[id] || m.start() >= last_end[id] {
            counts[id] += 1;
            last_end[id] = m.end();
            started[id] = true;
        }
    }

    let mut raw_scores = vec![0f32; FINGERPRINTS.len()];
    for (id, count) in counts.iter().enumerate() {
        if *count > 0 {
            let (fingerprint, weight) = set.owners[id];
            // Diminishing returns per repeated marker so one very
            // common short token (e.g. "->") can't dominate alone.
            raw_scores[fingerprint] += weight * (1.0 + (*count as f32).ln());
        }
    }

    let mut scores: HashMap<&'static str, (ContentKind, f32)> = HashMap::new();
    for (index, fp) in FINGERPRINTS.iter().enumerate() {
        scores.insert(fp.name, (fp.kind, raw_scores[index]));
    }
    // The winner is decided by raw score, not a per-language-normalized
    // one: normalizing by each language's own total marker weight would
    // otherwise penalize languages with a larger, more thorough marker
    // set relative to ones with only 2-3 markers, which is backwards (a
    // real Rust file that hits 5 of Rust's 14 markers should still beat
    // a false-positive TOML hit on one generic marker, even though 5/14
    // is a smaller *fraction* than 1/3).
    let winner = scores
        .iter()
        .filter(|(_, (_, raw_score))| *raw_score >= MIN_RAW_SCORE)
        .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
        .map(|(&name, &(kind, raw_score))| (name, kind, raw_score))?;
    // Confidence is still normalized against the winner's own total
    // marker weight, purely for a bounded [0,1] reported number.
    let own_total: f32 = FINGERPRINTS
        .iter()
        .find(|fp| fp.name == winner.0)
        .map(|fp| fp.markers.iter().map(|(_, w)| w).sum::<f32>())
        .unwrap_or(1.0)
        .max(1.0);
    Some((winner.0, winner.1, (winner.2 / own_total).min(1.0)))
}

/// Spec §178's reference weighting, substituting the (unavailable)
/// real parser-quality signal with the fingerprint density score — see
/// the module doc's scope note.
const WEIGHT_SYNTAX_FINGERPRINT: f32 = 0.85;
const WEIGHT_METADATA_PRIOR: f32 = 0.10;
const WEIGHT_SHEBANG: f32 = 0.05;

/// Full pipeline (spec §178). `path_hint` may be empty for content with no
/// known path; extension/basename then contribute nothing (matching the
/// spec's "path only as weak prior" stance — classification still works
/// content-first without one).
pub fn classify(bytes: &[u8], path_hint: &str) -> Classification {
    let sample = &bytes[..bytes.len().min(SAMPLE_BYTES)];

    if let Some(base_kind) = KNOWN_BASENAMES
        .iter()
        .find(|(name, ..)| basename(path_hint) == *name)
    {
        return Classification {
            kind: base_kind.2,
            language: Some(base_kind.1),
            confidence: 1.0,
            generated_probability: 0.0,
            vendor_probability: 0.0,
        };
    }

    if let Some(magic) = magic_match(sample) {
        return Classification {
            kind: ContentKind::Asset,
            language: Some(magic),
            confidence: 1.0,
            generated_probability: 0.0,
            vendor_probability: 0.0,
        };
    }

    let sniff = sniff(sample);
    // Git's own heuristic: any NUL in the sample is a confident binary
    // signal, independent of UTF-8 validity (some binary formats are
    // technically valid-if-unlikely UTF-8 byte sequences).
    let confidently_binary = sniff.has_nul
        || !sniff.is_valid_utf8
        || sniff.control_ratio > 0.30
        || sniff.entropy_bits > 7.6;
    if confidently_binary {
        return Classification {
            kind: ContentKind::Binary,
            language: None,
            confidence: 0.0,
            generated_probability: 0.0,
            vendor_probability: vendor_probability(path_hint),
        };
    }

    let text = std::str::from_utf8(sample).unwrap_or("");
    let shebang = shebang_hint(text);
    let fingerprint = best_fingerprint(text);
    let ext_hint = extension(path_hint).and_then(|ext| {
        FINGERPRINTS
            .iter()
            .find(|fp| fp.extensions.contains(&ext))
            .map(|fp| (fp.name, fp.kind))
    });

    let (language, kind, confidence) = match (fingerprint, shebang, ext_hint) {
        (Some((lang, kind, score)), shebang, ext) => {
            let shebang_bonus = if shebang == Some(lang) {
                WEIGHT_SHEBANG
            } else {
                0.0
            };
            let metadata_bonus = if ext.map(|(name, _)| name) == Some(lang) {
                WEIGHT_METADATA_PRIOR
            } else {
                0.0
            };
            let confidence =
                (WEIGHT_SYNTAX_FINGERPRINT * score + metadata_bonus + shebang_bonus).min(1.0);
            (Some(lang), kind, confidence)
        }
        (None, Some(lang), _) => (
            Some(lang),
            ContentKind::Code,
            WEIGHT_SHEBANG + WEIGHT_METADATA_PRIOR,
        ),
        (None, None, Some((lang, kind))) => (Some(lang), kind, WEIGHT_METADATA_PRIOR),
        (None, None, None) => (None, ContentKind::UnknownText, 0.0),
    };

    let kind = if language.is_none() && text.trim().is_empty() {
        ContentKind::PlainText
    } else {
        kind
    };

    Classification {
        kind,
        language,
        confidence,
        generated_probability: generated_probability(text),
        vendor_probability: vendor_probability(path_hint),
    }
}

/// Cheap, independent generated-file detector (spec §80): lock files and
/// common "do not edit" headers.
fn generated_probability(text: &str) -> f32 {
    let head = &text[..text.len().min(400)];
    let head_lower = head.to_ascii_lowercase();
    if head_lower.contains("do not edit")
        || head_lower.contains("autogenerated")
        || head_lower.contains("auto-generated")
        || head_lower.contains("@generated")
        || head_lower.contains("code generated by")
    {
        0.95
    } else {
        0.0
    }
}

/// Cheap, independent vendor-path detector (spec §80). Path-based only
/// (content rarely signals "vendored" on its own); overridable by
/// `.tqfignore` re-inclusion at the scan layer, not here.
fn vendor_probability(path_hint: &str) -> f32 {
    const VENDOR_SEGMENTS: &[&str] = &[
        "vendor",
        "node_modules",
        "third_party",
        "third-party",
        ".venv",
    ];
    if path_hint
        .split('/')
        .any(|segment| VENDOR_SEGMENTS.contains(&segment))
    {
        0.9
    } else {
        0.0
    }
}

#[cfg(test)]
mod fingerprint_equivalence {
    use super::*;

    /// The implementation `best_fingerprint` replaced, kept verbatim as
    /// the oracle. The one-pass version is an optimization, so the bar is
    /// identical output, not merely plausible output.
    fn best_fingerprint_by_rescanning(text: &str) -> Option<(&'static str, ContentKind, f32)> {
        let mut scores: HashMap<&'static str, (ContentKind, f32)> = HashMap::new();
        for fp in FINGERPRINTS {
            let mut raw_score = 0f32;
            for (marker, weight) in fp.markers {
                let count = text.matches(marker).count();
                if count > 0 {
                    raw_score += weight * (1.0 + (count as f32).ln());
                }
            }
            scores.insert(fp.name, (fp.kind, raw_score));
        }
        let winner = scores
            .iter()
            .filter(|(_, (_, raw_score))| *raw_score >= MIN_RAW_SCORE)
            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
            .map(|(&name, &(kind, raw_score))| (name, kind, raw_score))?;
        let own_total: f32 = FINGERPRINTS
            .iter()
            .find(|fp| fp.name == winner.0)
            .map(|fp| fp.markers.iter().map(|(_, w)| w).sum::<f32>())
            .unwrap_or(1.0)
            .max(1.0);
        Some((winner.0, winner.1, (winner.2 / own_total).min(1.0)))
    }

    /// Run against this crate's own tree — real files of several
    /// languages, not fixtures chosen to agree.
    #[test]
    fn one_pass_marker_search_matches_rescanning_on_this_repository() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let report = crate::retrieval::scan::scan_root(root).unwrap();

        let mut compared = 0usize;
        for file in &report.files {
            let Ok(bytes) = std::fs::read(root.join(&file.relative_path)) else {
                continue;
            };
            let sample = &bytes[..bytes.len().min(SAMPLE_BYTES)];
            let Ok(text) = std::str::from_utf8(sample) else {
                continue;
            };
            compared += 1;

            let fast = best_fingerprint(text);
            let slow = best_fingerprint_by_rescanning(text);
            match (fast, slow) {
                (None, None) => {}
                (Some(fast), Some(slow)) => {
                    assert_eq!(
                        (fast.0, fast.1),
                        (slow.0, slow.1),
                        "language/kind differ for {}",
                        file.relative_path
                    );
                    assert!(
                        (fast.2 - slow.2).abs() < 1e-6,
                        "confidence differs for {}: {} vs {}",
                        file.relative_path,
                        fast.2,
                        slow.2
                    );
                }
                (fast, slow) => panic!(
                    "one returned a language and the other did not for {}: {fast:?} vs {slow:?}",
                    file.relative_path
                ),
            }
        }
        assert!(
            compared > 100,
            "expected a real corpus, compared {compared}"
        );
    }

    /// The overlap rule is the part that is easy to get wrong: two
    /// different markers may cover the same bytes, but one marker's own
    /// matches never overlap each other. A plain non-overlapping search
    /// across the whole pattern set would silently drop counts.
    #[test]
    fn markers_that_share_bytes_are_counted_the_way_str_matches_does() {
        for text in [
            // `const ` (JS) sits inside no other marker, but `fn ` and
            // `impl ` co-occur densely in real Rust.
            "fn a() {} fn b() {} impl X for Y {} let mut z = 1;",
            // Self-overlapping candidates and repeated markers.
            "aaaa fn fn fn  ::  ::  :: std::std::std::",
            "",
            "import os\nimport sys\ndef f():\n    elif_not_really = 1\n",
        ] {
            assert_eq!(
                best_fingerprint(text).map(|f| (f.0, f.1)),
                best_fingerprint_by_rescanning(text).map(|f| (f.0, f.1)),
                "{text:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal Phase 35 exit gate: content wins over a misleading
    /// extension or path, in both directions.
    #[test]
    fn rust_extension_with_png_bytes_classifies_as_binary_not_rust() {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&[0u8; 32]);
        let result = classify(&bytes, "src/evil.rs");
        assert_eq!(result.kind, ContentKind::Asset);
        assert_eq!(result.language, Some("PNG"));
    }

    #[test]
    fn real_rust_content_classifies_as_rust_even_with_no_extension_at_all() {
        let source = r#"
use crate::error::Result;
pub struct Widget {
    name: String,
}
impl Widget {
    pub fn new() -> Self {
        let mut value = 0;
        Self { name: String::new() }
    }
}
"#;
        let result = classify(source.as_bytes(), "src/widget_no_ext");
        assert_eq!(result.language, Some("Rust"));
        assert_eq!(result.kind, ContentKind::Code);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn real_rust_content_under_a_misleading_txt_extension_still_classifies_as_rust() {
        let source = r#"
pub enum Shape { Circle, Square }
impl Shape {
    pub fn area(&self) -> f64 {
        match self {
            Shape::Circle => 3.14,
            Shape::Square => 1.0,
        }
    }
}
"#;
        let result = classify(source.as_bytes(), "notes/thoughts.txt");
        assert_eq!(result.language, Some("Rust"));
    }

    #[test]
    fn misleading_directory_name_does_not_affect_binary_detection() {
        let mut bytes = b"\xff\xd8\xff".to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        let result = classify(&bytes, "~/gaysex/meridian/photo.rs");
        assert_eq!(result.kind, ContentKind::Asset);
        assert_eq!(result.language, Some("JPEG"));
    }

    #[test]
    fn python_source_is_distinguished_from_similar_looking_shell() {
        let python = "def main():\n    import sys\n    self.value = 1\n    if x:\n        return x\n    elif y:\n        return y\n";
        let result = classify(python.as_bytes(), "");
        assert_eq!(result.language, Some("Python"));
    }

    #[test]
    fn shebang_hints_a_language_even_without_matching_extension() {
        let script = "#!/usr/bin/env python3\nimport os\ndef run():\n    self.x = 1\n";
        let result = classify(script.as_bytes(), "run");
        assert_eq!(result.language, Some("Python"));
    }

    #[test]
    fn known_basename_is_recognized_independent_of_extension_or_content() {
        let result = classify(b"FROM rust:latest\nRUN cargo build\n", "Dockerfile");
        assert_eq!(result.language, Some("Dockerfile"));
        assert_eq!(result.kind, ContentKind::Configuration);
    }

    #[test]
    fn generated_header_is_detected_independent_of_language() {
        let source =
            "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage main\nfunc main() {}\n";
        let result = classify(source.as_bytes(), "generated.go");
        assert!(result.generated_probability > 0.5);
        assert_eq!(result.language, Some("Go"));
    }

    #[test]
    fn vendor_path_segment_is_flagged_independent_of_content() {
        let result = classify(b"pub fn helper() {}\n", "third_party/lib/src/helper.rs");
        assert!(result.vendor_probability > 0.5);
    }

    #[test]
    fn empty_text_file_is_plain_text_not_unknown() {
        let result = classify(b"   \n\n  ", "notes.txt");
        assert_eq!(result.kind, ContentKind::PlainText);
    }

    #[test]
    fn json_content_is_distinguished_from_javascript() {
        let json = r#"{"name": "widget", "version": "1.0", "deps": [], "meta": null}"#;
        let result = classify(json.as_bytes(), "data");
        assert_eq!(result.language, Some("JSON"));
    }
}
