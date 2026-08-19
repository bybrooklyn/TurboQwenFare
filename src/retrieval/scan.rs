//! Full repository scan (spec §81, §307; Phase 35): symlink-safe
//! traversal honoring `.gitignore`/`.tqfignore`, feeding each surviving
//! file through `classify::classify`. "Directory symlinks are followed
//! only when the resolved target stays within the indexed root; cycle
//! detection is mandatory."

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::retrieval::classify::{self, Classification};
use crate::retrieval::ignore::IgnoreSet;

const SAMPLE_BYTES: usize = 64 * 1024;
const IGNORE_FILE_NAMES: &[&str] = &[".gitignore", ".tqfignore"];

#[derive(Debug, Clone)]
pub struct ScannedFile {
    /// `/`-separated, relative to the scan root.
    pub relative_path: String,
    pub size_bytes: u64,
    pub classification: Classification,
}

#[derive(Debug, Clone)]
pub struct ScanError {
    pub relative_path: String,
    pub message: String,
}

#[derive(Debug, Default)]
pub struct ScanReport {
    pub files: Vec<ScannedFile>,
    pub errors: Vec<ScanError>,
    pub ignored_count: u64,
    pub symlink_cycles_skipped: u64,
    pub symlinks_escaping_root_skipped: u64,
}

/// Scans `root` for real, indexable files. Best-effort: an unreadable
/// individual file is recorded in `ScanReport::errors` rather than
/// aborting the whole walk (matching a real repository scan, where one
/// permission-denied file must not lose every other result).
pub fn scan_root(root: &Path) -> std::io::Result<ScanReport> {
    let canonical_root = fs::canonicalize(root)?;
    let mut report = ScanReport::default();
    let mut visited_dirs: HashSet<PathBuf> = HashSet::new();
    visited_dirs.insert(canonical_root.clone());
    walk(root, &canonical_root, "", IgnoreSet::new(), &mut visited_dirs, &mut report)?;
    Ok(report)
}

fn walk(
    dir: &Path,
    canonical_root: &Path,
    rel_prefix: &str,
    mut ignores: IgnoreSet,
    visited_dirs: &mut HashSet<PathBuf>,
    report: &mut ScanReport,
) -> std::io::Result<()> {
    for ignore_name in IGNORE_FILE_NAMES {
        let candidate = dir.join(ignore_name);
        if let Ok(contents) = fs::read_to_string(&candidate) {
            ignores.add_file(rel_prefix, &contents);
        }
    }

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            report.errors.push(ScanError {
                relative_path: rel_prefix.to_string(),
                message: error.to_string(),
            });
            return Ok(());
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.errors.push(ScanError {
                    relative_path: rel_prefix.to_string(),
                    message: error.to_string(),
                });
                continue;
            }
        };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // `.git` is never indexable content (VCS internals, not a
        // repository's actual files) — skipped unconditionally, the same
        // way every real code-search tool treats it, rather than relying
        // on the target repo's own `.gitignore` to say so.
        if name == ".git" {
            continue;
        }

        let rel_path = if rel_prefix.is_empty() {
            name.to_string()
        } else {
            format!("{rel_prefix}/{name}")
        };

        let entry_path = entry.path();
        // `DirEntry::metadata` does not follow symlinks (it returns the
        // symlink's own metadata, never `is_dir() == true`); a directory
        // reached through a symlink must still be recognized as a
        // directory so it goes through the escape/cycle checks below
        // rather than silently falling through to a failed file read.
        let is_dir = match fs::metadata(&entry_path) {
            Ok(metadata) => metadata.is_dir(),
            Err(error) => {
                // A broken symlink (or a file that vanished mid-scan)
                // resolves to a read error here — record and skip.
                report.errors.push(ScanError {
                    relative_path: rel_path,
                    message: error.to_string(),
                });
                continue;
            }
        };

        if ignores.is_ignored(&rel_path, is_dir) {
            report.ignored_count += 1;
            continue;
        }

        if is_dir {
            let (target, was_symlink) = match resolve_directory_target(&entry_path) {
                Ok(pair) => pair,
                Err(error) => {
                    report.errors.push(ScanError {
                        relative_path: rel_path,
                        message: error.to_string(),
                    });
                    continue;
                }
            };
            if was_symlink && !target.starts_with(canonical_root) {
                report.symlinks_escaping_root_skipped += 1;
                continue;
            }
            if visited_dirs.contains(&target) {
                report.symlink_cycles_skipped += 1;
                continue;
            }
            visited_dirs.insert(target.clone());
            walk(&entry_path, canonical_root, &rel_path, ignores.clone(), visited_dirs, report)?;
            continue;
        }

        match fs::read(&entry_path) {
            Ok(bytes) => {
                let size_bytes = bytes.len() as u64;
                let sample = &bytes[..bytes.len().min(SAMPLE_BYTES)];
                let classification = classify::classify(sample, &rel_path);
                report.files.push(ScannedFile {
                    relative_path: rel_path,
                    size_bytes,
                    classification,
                });
            }
            Err(error) => {
                report.errors.push(ScanError {
                    relative_path: rel_path,
                    message: error.to_string(),
                });
            }
        }
    }
    Ok(())
}

/// Resolves a directory entry's real, canonical path and whether reaching
/// it required following a symlink (spec §81's escape-the-root check only
/// applies to symlinked directories, not the ordinary case).
fn resolve_directory_target(path: &Path) -> std::io::Result<(PathBuf, bool)> {
    let symlink_metadata = fs::symlink_metadata(path)?;
    let was_symlink = symlink_metadata.file_type().is_symlink();
    let canonical = fs::canonicalize(path)?;
    Ok((canonical, was_symlink))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tqf-scan-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(path: &Path, contents: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(contents).unwrap();
    }

    /// Real-repository sanity check: scans this actual crate's source
    /// tree (not a synthetic fixture) and confirms the classifier gets
    /// its own `.rs` files right at scale, real `.gitignore` handling
    /// excludes `target/`, and the walk terminates and produces plausible
    /// counts on genuine, large, heterogeneous input.
    #[test]
    fn scans_the_real_tqf_repository_and_classifies_its_own_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let report = scan_root(root).unwrap();
        assert!(report.files.len() > 50, "expected a substantial real repo scan");
        assert!(
            report.files.iter().all(|f| !f.relative_path.starts_with("target/")),
            "target/ should have been excluded by the real .gitignore"
        );
        let rust_files = report
            .files
            .iter()
            .filter(|f| f.classification.language == Some("Rust"))
            .count();
        assert!(rust_files > 50, "expected most of this crate's own .rs files to classify as Rust");
        let misclassified_rust: Vec<&str> = report
            .files
            .iter()
            .filter(|f| f.relative_path.ends_with(".rs") && f.classification.language != Some("Rust"))
            .map(|f| f.relative_path.as_str())
            .collect();
        println!(
            "phase35_real_scan files={} rust_files={} errors={} ignored={} misclassified_rs_files={misclassified_rust:?}",
            report.files.len(),
            rust_files,
            report.errors.len(),
            report.ignored_count,
        );
        assert!(
            misclassified_rust.is_empty(),
            "real .rs source files misclassified: {misclassified_rust:?}"
        );
    }

    #[test]
    fn scans_real_files_and_classifies_them() {
        let root = temp_dir("basic");
        write_file(&root.join("src/main.rs"), b"fn main() {\n    let mut x = 1;\n}\n");
        write_file(&root.join("README.md"), b"# Title\n\nSome text.\n");

        let report = scan_root(&root).unwrap();
        assert_eq!(report.files.len(), 2);
        let rust_file = report
            .files
            .iter()
            .find(|f| f.relative_path == "src/main.rs")
            .unwrap();
        assert_eq!(rust_file.classification.language, Some("Rust"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn gitignore_excludes_matching_files_and_directories() {
        let root = temp_dir("ignore");
        write_file(&root.join(".gitignore"), b"target/\n*.log\n");
        write_file(&root.join("target/debug/build.o"), b"binary junk");
        write_file(&root.join("app.log"), b"log contents");
        write_file(&root.join("src/lib.rs"), b"pub fn f() {}\n");

        let report = scan_root(&root).unwrap();
        let paths: Vec<&str> = report.files.iter().map(|f| f.relative_path.as_str()).collect();
        assert!(paths.contains(&"src/lib.rs"));
        assert!(!paths.contains(&"app.log"));
        assert!(!paths.iter().any(|p| p.starts_with("target/")));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tqfignore_can_re_include_content_gitignore_excluded() {
        let root = temp_dir("reinclude");
        write_file(&root.join(".gitignore"), b"*.generated\n");
        write_file(&root.join(".tqfignore"), b"!important.generated\n");
        write_file(&root.join("important.generated"), b"pub fn f() {}\n");
        write_file(&root.join("other.generated"), b"pub fn g() {}\n");

        let report = scan_root(&root).unwrap();
        let paths: Vec<&str> = report.files.iter().map(|f| f.relative_path.as_str()).collect();
        assert!(paths.contains(&"important.generated"));
        assert!(!paths.contains(&"other.generated"));
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_directory_cycle_is_detected_and_skipped() {
        let root = temp_dir("cycle");
        fs::create_dir_all(root.join("real")).unwrap();
        write_file(&root.join("real/file.rs"), b"fn f() {}\n");
        std::os::unix::fs::symlink(&root, root.join("real/loop")).unwrap();

        let report = scan_root(&root).unwrap();
        assert!(report.symlink_cycles_skipped > 0);
        // The real file is still found exactly once, not looped forever.
        assert_eq!(
            report.files.iter().filter(|f| f.relative_path == "real/file.rs").count(),
            1
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_directory_escaping_the_root_is_skipped() {
        let root = temp_dir("escape-root");
        let outside = temp_dir("escape-outside");
        write_file(&outside.join("secret.rs"), b"fn secret() {}\n");
        std::os::unix::fs::symlink(&outside, root.join("escaped")).unwrap();

        let report = scan_root(&root).unwrap();
        assert!(report.symlinks_escaping_root_skipped > 0);
        assert!(report.files.iter().all(|f| f.relative_path != "escaped/secret.rs"));
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside);
    }
}
