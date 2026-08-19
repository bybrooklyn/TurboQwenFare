//! `.gitignore`/`.tqfignore` pattern matching (spec §81: "`.gitignore` is
//! a strong default. `.tqfignore` can extend or explicitly re-include
//! appropriate content."). A real, if intentionally scoped-down, glob
//! matcher rather than a full gitignore-spec implementation: `*`, `**`,
//! `?`, a leading `/` anchor to the ignore file's own directory, a
//! trailing `/` for directory-only patterns, and `!` negation with
//! "last matching pattern wins" (real git semantics) are supported.
//! Character classes (`[abc]`) and `\`-escapes are not.

/// One compiled pattern from a `.gitignore`/`.tqfignore` file.
#[derive(Debug, Clone)]
pub struct IgnorePattern {
    negated: bool,
    dir_only: bool,
    /// Anchored to the ignore file's directory (pattern contained a `/`
    /// before its final character) vs matchable against any path
    /// component (no `/`, git's "match the basename anywhere" rule).
    anchored: bool,
    /// Glob source, `/`-separated, without a leading `/` or trailing `/`.
    glob: String,
    /// Directory (relative to the scan root) the pattern's `.gitignore`
    /// lives in; `""` for the scan root itself.
    base_dir: String,
}

impl IgnorePattern {
    fn parse(line: &str, base_dir: &str) -> Option<Self> {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let mut rest = line;
        let negated = if let Some(stripped) = rest.strip_prefix('!') {
            rest = stripped;
            true
        } else {
            false
        };
        let dir_only = rest.ends_with('/') && !rest.ends_with("\\/");
        let rest = rest.strip_suffix('/').unwrap_or(rest);
        let anchored = rest.contains('/');
        let glob = rest.strip_prefix('/').unwrap_or(rest).to_string();
        Some(Self {
            negated,
            dir_only,
            anchored,
            glob,
            base_dir: base_dir.to_string(),
        })
    }

    /// `rel_path` is `/`-separated, relative to the scan root, no leading
    /// `/`. `is_dir` gates directory-only patterns.
    fn matches(&self, rel_path: &str, is_dir: bool) -> bool {
        if self.dir_only && !is_dir {
            return false;
        }
        let scoped = match rel_path.strip_prefix(&self.base_dir) {
            Some(rest) if self.base_dir.is_empty() => rest,
            Some(rest) => rest.strip_prefix('/').unwrap_or(rest),
            None => return false,
        };
        if self.anchored {
            glob_match(&self.glob, scoped)
        } else {
            // Unanchored: match against any path component (git's
            // "basename anywhere in the tree" rule for patterns without a
            // slash), or the full remaining relative path for `**` forms.
            scoped
                .split('/')
                .any(|component| glob_match(&self.glob, component))
                || glob_match(&self.glob, scoped)
        }
    }
}

/// Minimal glob matcher: `*` (any run excluding `/`), `**` (any run
/// including `/`), `?` (one non-`/` char), literal otherwise.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_match_rec(&p, &t)
}

fn glob_match_rec(p: &[char], t: &[char]) -> bool {
    if p.is_empty() {
        return t.is_empty();
    }
    match p[0] {
        '*' if p.len() >= 2 && p[1] == '*' => {
            // `**`: match zero or more path segments (including `/`). A
            // following `/` is elided in the zero-segment case (`**/*.rs`
            // must match `main.rs`, not just `dir/main.rs`).
            let rest = if p.len() >= 3 && p[2] == '/' { &p[3..] } else { &p[2..] };
            for split in 0..=t.len() {
                if glob_match_rec(rest, &t[split..]) {
                    return true;
                }
            }
            false
        }
        '*' => {
            let rest = &p[1..];
            for split in 0..=t.len() {
                if t[..split].contains(&'/') {
                    break;
                }
                if glob_match_rec(rest, &t[split..]) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if t.is_empty() || t[0] == '/' {
                false
            } else {
                glob_match_rec(&p[1..], &t[1..])
            }
        }
        c => {
            if t.first() == Some(&c) {
                glob_match_rec(&p[1..], &t[1..])
            } else {
                false
            }
        }
    }
}

/// The accumulated ignore rules for one scan, from every `.gitignore`/
/// `.tqfignore` discovered so far (applied in discovery order, so a
/// parent directory's rules are overridden by a child's later ones,
/// matching git's precedence).
#[derive(Debug, Clone, Default)]
pub struct IgnoreSet {
    patterns: Vec<IgnorePattern>,
}

impl IgnoreSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses one ignore file's contents (`.gitignore` or `.tqfignore`
    /// syntax is identical) found at `base_dir` (relative to the scan
    /// root, `""` for the root) and appends its patterns.
    pub fn add_file(&mut self, base_dir: &str, contents: &str) {
        for line in contents.lines() {
            if let Some(pattern) = IgnorePattern::parse(line, base_dir) {
                self.patterns.push(pattern);
            }
        }
    }

    /// Real git precedence: the *last* matching pattern (across every
    /// ignore file added so far, in discovery order) decides; a match
    /// with `negated` re-includes.
    pub fn is_ignored(&self, rel_path: &str, is_dir: bool) -> bool {
        let mut ignored = false;
        for pattern in &self.patterns {
            if pattern.matches(rel_path, is_dir) {
                ignored = !pattern.negated;
            }
        }
        ignored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_matches_within_one_path_segment_only() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "src/main.rs"));
    }

    #[test]
    fn double_star_matches_across_segments() {
        assert!(glob_match("**/*.rs", "src/nested/main.rs"));
        assert!(glob_match("**/*.rs", "main.rs"));
    }

    #[test]
    fn unanchored_pattern_matches_any_directory_component() {
        let mut set = IgnoreSet::new();
        set.add_file("", "target\n");
        assert!(set.is_ignored("target", true));
        assert!(set.is_ignored("crate/target", true));
        assert!(set.is_ignored("crate/target/deep/file.rs", false));
    }

    #[test]
    fn anchored_pattern_only_matches_from_its_own_directory() {
        let mut set = IgnoreSet::new();
        set.add_file("", "/build\n");
        assert!(set.is_ignored("build", true));
        assert!(!set.is_ignored("crate/build", true));
    }

    #[test]
    fn dir_only_pattern_does_not_match_a_file_of_the_same_name() {
        let mut set = IgnoreSet::new();
        set.add_file("", "vendor/\n");
        assert!(set.is_ignored("vendor", true));
        assert!(!set.is_ignored("vendor", false));
    }

    #[test]
    fn negation_re_includes_after_a_broader_ignore() {
        let mut set = IgnoreSet::new();
        set.add_file("", "*.log\n!important.log\n");
        assert!(set.is_ignored("debug.log", false));
        assert!(!set.is_ignored("important.log", false));
    }

    #[test]
    fn nested_ignore_file_scopes_to_its_own_directory() {
        let mut set = IgnoreSet::new();
        set.add_file("", "*.tmp\n");
        set.add_file("sub", "*.cache\n");
        assert!(set.is_ignored("sub/x.tmp", false));
        assert!(set.is_ignored("sub/x.cache", false));
        assert!(!set.is_ignored("other/x.cache", false));
    }

    #[test]
    fn later_pattern_overrides_an_earlier_one_matching_git_precedence() {
        let mut set = IgnoreSet::new();
        set.add_file("", "*.rs\n");
        set.add_file("", "!keep.rs\n");
        assert!(!set.is_ignored("keep.rs", false));
        assert!(set.is_ignored("other.rs", false));
    }
}
