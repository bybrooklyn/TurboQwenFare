//! `tqf sync` / `tqf unsync` (spec §3).
//!
//! These do a real scan and a real index build, and then say plainly that
//! the result is not retained.
//!
//! That honesty is the point. `retrieval::sync` builds its index entirely
//! in memory — there is no `.tqi` writer, no registry of synced roots
//! (§218), and no journal (§198). A `tqf sync .` that quietly built an
//! index and discarded it at exit would look like it worked and leave the
//! user wondering why nothing was ever searchable. Reporting exactly what
//! was scanned, what is indexable today, and what is missing is more
//! useful than either a fake success or a bare "not implemented".

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::error::Result;
use crate::retrieval::scan::{scan_root, ScanReport};
use crate::retrieval::sync::{full_correctness_walk, FileTable, SyncEngine};

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
    // Nothing persists a registration, so there is nothing to remove.
    // Saying so beats reporting a success that removed nothing.
    println!(
        "tqf unsync: {display} is not registered.\n\
         \n\
         No root can be registered yet: `tqf sync` builds its index in memory for the\n\
         duration of the process, and index persistence (spec §218's project registry\n\
         and the `.tqi` container) is not implemented. There is nothing on disk to remove."
    );
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

    report.elapsed_ms = started.elapsed().as_millis();
    Ok(report)
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

    let _ = writeln!(
        out,
        "\nThis index was NOT retained. Index persistence — spec §218's project registry\n\
         and the `.tqi` container that would hold this on disk — is not implemented, so\n\
         the work above lives only for the duration of this command.\n\
         \n\
         Retrieval is reachable in-process (the lexical, exact, and semantic lanes are\n\
         implemented and measured; see docs/research/qualification/), but nothing yet\n\
         reloads an index at startup or serves it over HTTP."
    );
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

    /// The load-bearing property: a user must not be left believing an
    /// index was saved when it was not.
    #[test]
    fn the_report_says_the_index_was_not_retained() {
        let rendered = render(&sample());
        assert!(rendered.contains("NOT retained"), "{rendered}");
        assert!(rendered.contains("not implemented"), "{rendered}");
        assert!(
            rendered.contains("§218"),
            "must cite what is missing: {rendered}"
        );
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
