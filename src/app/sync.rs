//! `tqf sync` / `tqf unsync` (spec §3, §173).
//!
//! A real scan, a real index build, and a real `.tqi` written to
//! `<root>/.tqf/index.tqi` — so the work survives the process that did
//! it, which is the whole point of syncing a root.
//!
//! What is still absent is stated in the report rather than left for the
//! user to discover: the semantic lane needs the helper model, and the
//! walk admits Rust only because no other language has a real parser in
//! this build (spec §307).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::error::Result;
use crate::retrieval::scan::{scan_root, ScanReport};
use crate::retrieval::sync::{full_correctness_walk, FileTable, SyncEngine};
use crate::retrieval::tqi::{self, codec::IndexContents, registry, segments::FileRecord};

/// What one `tqf sync` run observed.
#[derive(Debug, Default)]
pub struct SyncReport {
    pub root: String,
    pub files_scanned: usize,
    pub bytes_scanned: u64,
    pub ignored: u64,
    pub by_language: BTreeMap<String, usize>,
    pub indexable: usize,
    pub indexed_chunks: usize,
    pub lexical_terms: usize,
    pub errors: Vec<String>,
    pub symlink_cycles_skipped: u64,
    pub elapsed_ms: u128,
    /// Where the index was written, and which generation it became.
    pub index_path: Option<String>,
    pub generation: u64,
    pub index_bytes: u64,
}

pub fn run_sync(path: &Path) -> Result<()> {
    let report = index(path)?;
    print!("{}", render(&report));
    Ok(())
}

pub fn run_unsync(path: &Path) -> Result<()> {
    let display = path.display();
    if !path.exists() {
        println!("tqf unsync: {display} does not exist.");
        return Ok(());
    }
    let root = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let index_path = tqi::index_path(&root);
    if !index_path.exists() {
        println!(
            "tqf unsync: {display} is not synced (no index at {}).",
            index_path.display()
        );
        return Ok(());
    }

    // Remove the index, and the `.tqf` directory too when it holds
    // nothing else — leaving an empty directory behind would make an
    // unsynced root still look synced to anything checking for it.
    std::fs::remove_file(&index_path)?;
    let _ = std::fs::remove_file(registry::project_file_path(&root));
    registry::deregister(&root)?;
    if let Some(parent) = index_path.parent() {
        let empty = std::fs::read_dir(parent)
            .map(|mut d| d.next().is_none())
            .unwrap_or(false);
        if empty {
            let _ = std::fs::remove_dir(parent);
        }
    }
    println!("tqf unsync: removed {}.", index_path.display());
    Ok(())
}

fn index(path: &Path) -> Result<SyncReport> {
    let started = std::time::Instant::now();
    let root = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    let scan: ScanReport = scan_root(&root)?;
    let mut report = SyncReport {
        root: root.display().to_string(),
        files_scanned: scan.files.len(),
        bytes_scanned: scan.files.iter().map(|f| f.size_bytes).sum(),
        ignored: scan.ignored_count,
        symlink_cycles_skipped: scan.symlink_cycles_skipped,
        errors: scan
            .errors
            .iter()
            .map(|e| format!("{}: {}", e.relative_path, e.message))
            .collect(),
        ..SyncReport::default()
    };
    for file in &scan.files {
        let language = file.classification.language.unwrap_or("(unclassified)");
        *report.by_language.entry(language.to_string()).or_insert(0) += 1;
    }

    // The real walk and the real lexical/exact index build.
    let table = FileTable::default();
    let (plan, contents) = full_correctness_walk(&root, &table)?;
    report.indexable = contents.len();

    let mut engine = SyncEngine::empty();
    engine.apply_structural_lexical(&plan, &contents);
    report.indexed_chunks = engine.lexical.document_count();
    report.lexical_terms = engine.lexical.term_count();

    // Persist it (spec §173's project-local committed file). Written
    // atomically, so an interrupted sync leaves the previous index intact
    // rather than a truncated one that would still parse.
    let index_path = tqi::index_path(&root);
    let previous = tqi::codec::read(&index_path).ok();
    let persisted = build_contents(&engine, &root, &contents);
    tqi::codec::write(
        &index_path,
        &root,
        &persisted,
        previous.as_ref().map(|loaded| &loaded.superblock),
    )?;

    let loaded = tqi::codec::read(&index_path)?;

    // Register the root so a server started anywhere can find it, and
    // write the project file that carries the index identity with the
    // project (spec §218).
    registry::write_project_file(
        &root,
        &registry::ProjectFile {
            index_uuid: loaded
                .superblock
                .index_uuid
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            root: root.display().to_string(),
            last_synced_unix: loaded.superblock.last_compacted_unix,
            generation: loaded.superblock.latest_generation,
        },
    )?;
    registry::register(&root)?;

    report.generation = loaded.superblock.latest_generation;
    report.index_bytes = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
    report.index_path = Some(index_path.display().to_string());

    report.elapsed_ms = started.elapsed().as_millis();
    Ok(report)
}

/// Turns the in-memory engine into the persistable form.
///
/// Each file record carries the same BLAKE3 content hash the walk uses
/// for change detection, so a later sync can tell changed files from
/// unchanged ones without re-reading them (spec §177's `content_hash` is
/// identity evidence; §176's `FileId` is the stable key).
fn build_contents(
    engine: &SyncEngine,
    root: &Path,
    file_contents: &std::collections::HashMap<String, String>,
) -> IndexContents {
    let (postings, exact, chunk_lengths, paths) = engine.lexical.export();

    let mut contents = IndexContents {
        postings,
        exact,
        chunk_lengths,
        ..IndexContents::default()
    };

    for (index, path) in paths.iter().enumerate() {
        let id = index as u64;
        let text = file_contents.get(path);
        let mtime_ns = std::fs::metadata(root.join(path))
            .ok()
            .and_then(|meta| meta.modified().ok())
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|since| since.as_nanos() as u64)
            .unwrap_or(0);

        contents.files.push(FileRecord {
            id,
            // Assigned by the writer when it interns the path.
            path: 0,
            byte_len: text.map(|t| t.len() as u64).unwrap_or(0),
            mtime_ns,
            content_hash: text
                .map(|t| *blake3::hash(t.as_bytes()).as_bytes())
                .unwrap_or([0u8; 32]),
            language: 1,
            first_chunk: id,
            chunk_count: 1,
        });
        contents.paths.insert(id, path.clone());
    }

    contents.next_file_id = contents.files.len() as u64;
    contents.next_chunk_id = contents.chunk_lengths.len() as u64;
    contents
}

fn render(report: &SyncReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "scanned {}", report.root);
    let _ = writeln!(
        out,
        "  {} files, {:.1} MiB, {} ignored, in {} ms",
        report.files_scanned,
        report.bytes_scanned as f64 / (1024.0 * 1024.0),
        report.ignored,
        report.elapsed_ms
    );
    if report.symlink_cycles_skipped > 0 {
        let _ = writeln!(
            out,
            "  {} symlink cycles skipped",
            report.symlink_cycles_skipped
        );
    }

    if !report.by_language.is_empty() {
        let _ = writeln!(out, "\nclassified:");
        // Most-common first, since that is what a user scans for.
        let mut langs: Vec<_> = report.by_language.iter().collect();
        langs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (language, count) in langs.iter().take(12) {
            let _ = writeln!(out, "  {count:>6}  {language}");
        }
    }

    let _ = writeln!(out, "\nindexed:");
    let _ = writeln!(
        out,
        "  {} files into {} chunks, {} distinct lexical terms",
        report.indexable, report.indexed_chunks, report.lexical_terms
    );

    if report.indexable < report.files_scanned {
        let _ = writeln!(
            out,
            "  {} of {} scanned files were not indexed: the walk currently admits Rust\n\
             \x20 sources only, because no other language has a real parser in this build\n\
             \x20 (spec §307's scope decision).",
            report.files_scanned - report.indexable,
            report.files_scanned
        );
    }

    if !report.errors.is_empty() {
        let _ = writeln!(out, "\nunreadable ({}):", report.errors.len());
        for error in report.errors.iter().take(5) {
            let _ = writeln!(out, "  {error}");
        }
        if report.errors.len() > 5 {
            let _ = writeln!(out, "  ... and {} more", report.errors.len() - 5);
        }
    }

    match &report.index_path {
        Some(path) => {
            let _ = writeln!(
                out,
                "\nwritten:\n  {path}\n  generation {}, {:.1} KiB",
                report.generation,
                report.index_bytes as f64 / 1024.0
            );
            let _ = writeln!(
                out,
                "\nThe lexical and exact lanes are retained and reload without re-reading any\n\
                 source file. The semantic lane is not in this index: it needs the helper\n\
                 embedding model, which is not installed (see `just pin-helper-model`)."
            );
        }
        None => {
            let _ = writeln!(out, "\nThe index was not written.");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SyncReport {
        SyncReport {
            root: "/repo".to_string(),
            files_scanned: 150,
            bytes_scanned: 3 * 1024 * 1024,
            ignored: 20,
            by_language: [("Rust".to_string(), 112), ("Markdown".to_string(), 30)]
                .into_iter()
                .collect(),
            indexable: 112,
            indexed_chunks: 112,
            lexical_terms: 9000,
            elapsed_ms: 42,
            ..SyncReport::default()
        }
    }

    /// The load-bearing property, in both directions: a user must not be
    /// left believing an index was saved when it was not, nor told it was
    /// discarded when it is on disk.
    #[test]
    fn the_report_names_where_the_index_was_written() {
        let mut report = sample();
        report.index_path = Some("/repo/.tqf/index.tqi".to_string());
        report.generation = 3;
        report.index_bytes = 2048;

        let rendered = render(&report);
        assert!(rendered.contains("/repo/.tqf/index.tqi"), "{rendered}");
        assert!(rendered.contains("generation 3"), "{rendered}");
        assert!(
            !rendered.contains("NOT retained"),
            "the index is retained now: {rendered}"
        );
    }

    /// The semantic lane genuinely is absent, so the report still has to
    /// say so — persisting the lexical lanes must not read as "retrieval
    /// is complete".
    #[test]
    fn the_report_still_states_what_the_index_does_not_contain() {
        let mut report = sample();
        report.index_path = Some("/repo/.tqf/index.tqi".to_string());

        let rendered = render(&report);
        assert!(
            rendered.contains("semantic lane is not in this index"),
            "{rendered}"
        );
        assert!(rendered.contains("pin-helper-model"), "{rendered}");
    }

    /// If the write never happened, the report must not imply it did.
    #[test]
    fn a_report_with_no_index_path_does_not_claim_one_was_written() {
        let rendered = render(&sample());
        assert!(rendered.contains("was not written"), "{rendered}");
        assert!(!rendered.contains("generation "), "{rendered}");
    }

    #[test]
    fn the_report_carries_the_real_counts_it_measured() {
        let rendered = render(&sample());
        assert!(rendered.contains("150 files"), "{rendered}");
        assert!(rendered.contains("20 ignored"), "{rendered}");
        assert!(rendered.contains("112 files into 112 chunks"), "{rendered}");
        assert!(
            rendered.contains("9000 distinct lexical terms"),
            "{rendered}"
        );
    }

    /// A user seeing "150 scanned, 112 indexed" deserves to know why the
    /// other 38 were skipped rather than assuming a bug.
    #[test]
    fn a_gap_between_scanned_and_indexed_is_explained() {
        let rendered = render(&sample());
        assert!(rendered.contains("38 of 150"), "{rendered}");
        assert!(rendered.contains("Rust"), "{rendered}");
    }

    #[test]
    fn languages_are_listed_most_common_first() {
        let rendered = render(&sample());
        let rust = rendered.find("Rust").expect("Rust must appear");
        let markdown = rendered.find("Markdown").expect("Markdown must appear");
        assert!(rust < markdown, "more common language must come first");
    }

    /// A real scan of this crate's own source tree, not a fixture: the
    /// same validation approach Phases 35/36 used.
    #[test]
    fn indexing_this_crates_own_source_tree_finds_its_real_rust_files() {
        let report = index(Path::new("src")).expect("scanning src/ must succeed");
        assert!(
            report.files_scanned > 100,
            "expected this crate's real files, got {}",
            report.files_scanned
        );
        assert!(
            report.indexable > 100,
            "expected the Rust sources to be indexable, got {}",
            report.indexable
        );
        assert!(
            report.lexical_terms > 1000,
            "a real index over 100+ files should have many terms, got {}",
            report.lexical_terms
        );
        assert_eq!(
            report.by_language.get("Rust").copied().unwrap_or(0),
            report.indexable,
            "every indexed file should be a classified Rust source"
        );
    }
}
