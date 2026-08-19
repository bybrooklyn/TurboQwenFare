# Phase 35: file catalog and content-first classifier

Spec Phase 35 deliverable (spec §307, §79-81, §178-179; exit gate row 35:
"Misleading extensions/paths test suite passes").

## Scope decision: no grammar-parsing dependency

Spec §178's full pipeline scores `parser_quality` from real Tree-sitter-
class AST parsing (coverage, error-node ratio, tree shape) as the
dominant 0.60-weight signal. Adding a multi-language grammar-parsing
dependency is out of scope for this phase (comparable in scope to
standing up a second, independent language-tooling subsystem). Instead,
"parse top candidate grammars" is replaced by a cheaper, still-real
syntax-*fingerprint* density score (weighted keyword/construct substring
matching, no AST), and spec §178's reference weights are renormalized onto
that substitute signal (0.85 fingerprint / 0.10 metadata prior / 0.05
shebang, replacing 0.60 parser-quality / 0.25 fingerprint / 0.10 metadata
/ 0.05 shebang). This is recorded honestly as a substitute, not claimed as
the spec's real parser-quality metric.

## What was built

- `retrieval::ignore`: a real (scoped-down, not full-gitignore-spec)
  glob matcher — `*`, `**`, `?`, leading-`/` anchoring, trailing-`/`
  directory-only patterns, `!` negation with git's real "last matching
  pattern wins" precedence, and per-directory `.gitignore`/`.tqfignore`
  scoping.
- `retrieval::classify`: the spec §178 pipeline — bounded 64 KiB byte
  sample, known binary magic-number table (PNG/JPEG/GIF/PDF/ELF/ZIP/GZIP/
  WASM/Java-class/PE/BMP/RIFF), UTF-8/NUL/control-ratio/Shannon-entropy
  binary detection, a language fingerprint table covering spec §81's
  initial broad-set languages most likely in a real repo (Rust, Python,
  TypeScript, JavaScript, Go, C/C++, Shell, Markdown, JSON, TOML, YAML),
  shebang hints, known special basenames (Makefile/Dockerfile/CMakeLists/
  Cargo.toml/Cargo.lock), and independent generated/vendor probability
  detectors.
- `retrieval::scan`: real filesystem traversal — symlinked directories are
  resolved and checked against the canonical scan root (escaping symlinks
  skipped), a visited-canonical-path set detects and skips cycles,
  `.gitignore`/`.tqfignore` are discovered and scoped per directory as the
  walk descends, `.git` is unconditionally skipped, and per-file read/
  metadata errors are collected rather than aborting the whole scan.

## Measured evidence

**The literal exit gate, and its fix.** `classify::tests` includes the
gate directly: a `.rs`-named file containing real PNG magic bytes
classifies as `Asset`/`PNG`, not Rust; a file with **no extension at all**
containing real Rust source classifies as Rust; the same content under a
misleading `.txt` extension still classifies as Rust; a misleading
*directory* name (`~/gaysex/meridian/photo.rs`, spec §79's own example)
does not affect binary detection of real JPEG bytes.

**Real-repository validation, not just synthetic fixtures.**
`scans_the_real_tqf_repository_and_classifies_its_own_source` scans this
actual crate's live source tree (150 files) and asserts every one of its
112 real `.rs` files classifies as Rust. This caught two real,
independent bugs before they could ship as false confidence:

1. **A scanner bug**: `std::fs::DirEntry::metadata()` does not follow
   symlinks (it returns the symlink's own metadata, never `is_dir() ==
   true` for a directory symlink) — a directory reached through a symlink
   silently fell through to a failed file read instead of being walked,
   which meant the symlink-cycle and escaping-root checks the synthetic
   unit tests (`symlinked_directory_cycle_is_detected_and_skipped`,
   `symlinked_directory_escaping_the_root_is_skipped`) were written
   against were **never actually exercised** until those tests were run
   and failed. Fixed by resolving `is_dir` through `fs::metadata` (which
   follows symlinks) instead.
2. **A classifier weighting bug**: real, idiomatic Rust files with heavy
   `//!` module-doc-comment prose and thin/no code bodies (`mod.rs`
   re-export files, `main.rs`) were misclassified — English prose in doc
   comments trivially contains other languages' generic keyword markers
   (`"the GGUF *import* reader"` matches Python's `"import "` marker), and
   per-language-normalized scoring penalized Rust's own broader marker set
   relative to a spuriously-matching language's narrower one. Fixed by
   (a) adding `//!`/`///`/`#[cfg(`/`pub mod `/`pub(crate)` as strong,
   effectively Rust-exclusive markers, (b) deciding the winning language
   by raw score rather than per-language-normalized score, and (c)
   tightening Markdown's markers away from generic single characters
   (`"["`, `"- "`) that collide with ordinary code syntax (array indexing,
   subtraction) toward more specific ones (`"]("`, `"\n# "`).

Neither bug would have been caught by synthetic fixtures alone — both
needed a real, large, heterogeneous, already-correct-by-construction
corpus (this crate's own source) to surface.

```
phase35_real_scan files=150 rust_files=112 errors=0 ignored=7 misclassified_rs_files=[]
```

## Status and remaining work

- No real AST parsing exists; `parser_quality` (spec §179's dominant
  0.60-weight signal) has no counterpart here. The fingerprint substitute
  is real and now validated against this crate's own source, but it is a
  narrower, more collision-prone signal than genuine parse-coverage
  scoring, and only 11 languages are covered against spec §81's broader
  target list (Swift, Java, C#, Lua, Ruby/PHP, WGSL/GLSL, HTML/CSS are not
  yet fingerprinted).
- The misleading-extension test suite covers the specific cases spec §79
  names by example (wrong extension containing real other-format bytes,
  misleading directory name); it is not an exhaustive fuzzing corpus.
- Structural chunking, symbol records, and the program graph (spec §82-84,
  §180-184) all assume real AST output and are explicitly Phase 36+
  territory — this phase produces classification only, not chunks or
  symbols.
