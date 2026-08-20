//! Incremental live sync (spec §42, §91, §198-199, §314). "Connect
//! watcher to incremental generation transactions. Stress editor save
//! storms and watcher overflow. Search remains usable during deferred
//! semantic updates."
//!
//! Scope decision: spec §198's full transaction diagram ends in a
//! durable `index.journal`/superblock-generation-pointer commit — that
//! needs the persisted `.tqi` index storage format every retrieval
//! phase since Phase 36 has explicitly deferred ("this phase proves the
//! scoring/tokenization logic works on real data before that storage
//! engineering is warranted"). This phase keeps that same scope
//! boundary: the *semantics* of incremental sync — content-hash change
//! detection, embedding reuse, `semantic_pending` deferral, watcher
//! debounce/coalesce, overflow fallback to a full walk — are built and
//! genuinely stress-tested; durable journal/fsync/generation-pointer
//! commit is not, because there is no on-disk index yet to commit to.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use notify::Watcher as _;

use crate::error::Result;
use crate::helper_model::PplxEmbedRuntime;

use super::flat::l2_normalize;
use super::lexical::LexicalIndex;
use super::scan::{scan_root, ScanReport};

#[derive(Debug, Clone)]
pub struct FileRecord {
    pub content_hash: [u8; 32],
    pub generation: u64,
}

/// spec §91/§198's `FileTable`: the correctness anchor a full walk
/// diffs against. Deliberately in-memory only this phase (see module
/// doc) — a real system would persist this alongside the index.
#[derive(Debug, Default)]
pub struct FileTable {
    pub entries: HashMap<String, FileRecord>,
    pub generation: u64,
}

#[derive(Debug, Default, Clone)]
pub struct SyncPlan {
    pub new: Vec<String>,
    pub changed: Vec<String>,
    pub deleted: Vec<String>,
    pub unchanged: Vec<String>,
}

/// spec §198's first three steps: "full root walk → compare
/// path/stat/quick hash against FileTable → for changed/new candidates
/// compute full content hash." Uses BLAKE3 directly as the "full
/// content hash" (no separate quick-hash prefilter this phase — the
/// corpus sizes this session validates against don't need one).
/// Returns the diff plus every current file's contents, so callers
/// don't need a second disk pass.
pub fn full_correctness_walk(
    root: &Path,
    table: &FileTable,
) -> Result<(SyncPlan, HashMap<String, String>)> {
    let report = scan_root(root)?;
    full_correctness_walk_of(root, table, &report)
}

/// The same walk against a scan a caller already has.
///
/// `scan_root` reads every file in the tree to classify it by content, so
/// a caller that scans for its own reasons and then calls
/// `full_correctness_walk` pays for the whole tree twice. `tqf sync` did
/// exactly that.
pub fn full_correctness_walk_of(
    root: &Path,
    table: &FileTable,
    report: &ScanReport,
) -> Result<(SyncPlan, HashMap<String, String>)> {
    let mut seen = HashSet::new();
    let mut plan = SyncPlan::default();
    let mut contents = HashMap::new();

    for file in &report.files {
        if file.classification.language != Some("Rust") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(root.join(&file.relative_path)) else {
            continue;
        };
        let hash = *blake3::hash(text.as_bytes()).as_bytes();
        seen.insert(file.relative_path.clone());
        match table.entries.get(&file.relative_path) {
            None => plan.new.push(file.relative_path.clone()),
            Some(record) if record.content_hash != hash => {
                plan.changed.push(file.relative_path.clone())
            }
            Some(_) => plan.unchanged.push(file.relative_path.clone()),
        }
        contents.insert(file.relative_path.clone(), text);
    }
    for path in table.entries.keys() {
        if !seen.contains(path) {
            plan.deleted.push(path.clone());
        }
    }
    Ok((plan, contents))
}

/// Ties the file table, the (cheap, always-fresh) lexical/exact lane,
/// and the (expensive, deferrable) semantic lane together.
pub struct SyncEngine {
    pub table: FileTable,
    pub lexical: LexicalIndex,
    /// Paths whose semantic vector is missing or stale and awaiting
    /// (re-)embedding (spec §198: `semantic_pending` flags).
    pub semantic_pending: HashSet<String>,
    /// Committed semantic vectors. A path in `semantic_pending` may
    /// still have an entry here — its *old* vector, kept so semantic
    /// search stays usable (spec §199's "search remains usable during
    /// deferred semantic updates") instead of going blind for that file
    /// until re-embedding completes.
    pub semantic_vectors: HashMap<String, Vec<f32>>,
}

impl SyncEngine {
    pub fn empty() -> Self {
        Self {
            table: FileTable::default(),
            lexical: LexicalIndex::build(&[]),
            semantic_pending: HashSet::new(),
            semantic_vectors: HashMap::new(),
        }
    }

    /// spec §198's structural/lexical-commit step: rebuilds the cheap
    /// lexical/exact lane over every currently-live file immediately,
    /// and marks every new/changed file `semantic_pending` without
    /// touching the (expensive) semantic lane at all. This is the
    /// mechanism behind "structural/lexical changes can commit first...
    /// a later semantic delta fills them without making search
    /// unavailable."
    pub fn apply_structural_lexical(
        &mut self,
        plan: &SyncPlan,
        contents: &HashMap<String, String>,
    ) {
        let mut documents: Vec<(String, String)> = Vec::new();
        for path in plan
            .new
            .iter()
            .chain(plan.changed.iter())
            .chain(plan.unchanged.iter())
        {
            if let Some(text) = contents.get(path) {
                documents.push((path.clone(), text.clone()));
            }
        }
        self.lexical = LexicalIndex::build(&documents);

        self.table.generation += 1;
        let generation = self.table.generation;
        for path in plan.new.iter().chain(plan.changed.iter()) {
            if let Some(text) = contents.get(path) {
                self.table.entries.insert(
                    path.clone(),
                    FileRecord {
                        content_hash: *blake3::hash(text.as_bytes()).as_bytes(),
                        generation,
                    },
                );
            }
            self.semantic_pending.insert(path.clone());
        }
        for path in &plan.deleted {
            self.table.entries.remove(path);
            self.semantic_pending.remove(path);
            self.semantic_vectors.remove(path);
        }
    }

    /// spec §199: "pause/deprioritize semantic embedding under
    /// inference pressure." Processes at most `budget` pending paths
    /// per call — the caller (a real decode loop) decides how much
    /// embedding work to allow between higher-priority requests, rather
    /// than this engine draining the whole backlog unconditionally.
    /// Returns how many paths were actually processed.
    pub fn process_pending_semantic(
        &mut self,
        runtime: &PplxEmbedRuntime,
        contents: &HashMap<String, String>,
        max_input_tokens: usize,
        budget: usize,
    ) -> Result<usize> {
        let to_process: Vec<String> = self.semantic_pending.iter().take(budget).cloned().collect();
        for path in &to_process {
            let Some(text) = contents.get(path) else {
                self.semantic_pending.remove(path);
                continue;
            };
            let embedding = runtime.embed_with_input_budget(text, None, Some(max_input_tokens))?;
            let mut vector = embedding.fp32;
            l2_normalize(&mut vector);
            self.semantic_vectors.insert(path.clone(), vector);
            self.semantic_pending.remove(path);
        }
        Ok(to_process.len())
    }
}

/// spec §199's watcher debounce/coalesce logic, kept as pure
/// synchronous state so it can be stress-tested deterministically (no
/// dependency on real OS event timing): "debounce bursts in RAM;
/// coalesce repeated writes/renames." Every `record()` for the same
/// path just refreshes its last-seen timestamp; `drain_ready` emits
/// each path *once*, however many times it was recorded, once its
/// debounce window has elapsed.
pub struct DebouncedEventQueue {
    pending: HashMap<String, Instant>,
    window: Duration,
}

impl DebouncedEventQueue {
    pub fn new(window: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            window,
        }
    }

    pub fn record(&mut self, path: String, at: Instant) {
        self.pending.insert(path, at);
    }

    /// Returns (and removes) every path whose debounce window has
    /// elapsed as of `now`, coalesced to one entry each.
    pub fn drain_ready(&mut self, now: Instant) -> Vec<String> {
        let ready: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, &last_seen)| now.duration_since(last_seen) >= self.window)
            .map(|(path, _)| path.clone())
            .collect();
        for path in &ready {
            self.pending.remove(path);
        }
        ready
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// spec §199: "on overflow/lost events, schedule a full correctness
/// walk." A fixed-capacity sink for raw watcher paths: once full, every
/// further event is *dropped* (not queued, not overwriting an existing
/// slot) and `overflowed` latches permanently until explicitly reset —
/// this is deliberate data loss, matching "watcher events are hints,
/// not the source of truth." A caller that later observes `overflowed`
/// must not trust whatever partial path list is in the sink and must
/// fall back to `full_correctness_walk` instead.
pub struct BoundedEventSink {
    capacity: usize,
    buffer: std::sync::Mutex<Vec<String>>,
    overflowed: std::sync::atomic::AtomicBool,
}

impl BoundedEventSink {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: std::sync::Mutex::new(Vec::new()),
            overflowed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn push(&self, path: String) {
        let mut buffer = self.buffer.lock().expect("event sink mutex poisoned");
        if buffer.len() >= self.capacity {
            self.overflowed
                .store(true, std::sync::atomic::Ordering::SeqCst);
            return;
        }
        buffer.push(path);
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn drain(&self) -> Vec<String> {
        std::mem::take(&mut *self.buffer.lock().expect("event sink mutex poisoned"))
    }
}

/// Thin real wiring over `notify`'s cross-platform OS watcher (FSEvents
/// on macOS, inotify on Linux — spec §199's watcher, not a hand-rolled
/// kqueue/inotify binding). Every raw event's paths are pushed into a
/// `BoundedEventSink`; the watcher handle must be kept alive for the
/// duration of watching (dropping it stops delivery), which is why this
/// wraps it rather than returning the raw `notify::RecommendedWatcher`.
pub struct LiveWatcher {
    _watcher: notify::RecommendedWatcher,
}

impl LiveWatcher {
    pub fn watch(root: &Path, sink: std::sync::Arc<BoundedEventSink>) -> notify::Result<Self> {
        // FSEvents (macOS) and other backends report *canonicalized*
        // paths (symlinks resolved — e.g. `/var/folders/...` becomes
        // `/private/var/folders/...` on macOS, since `TMPDIR` is itself
        // a symlink), not necessarily whatever form the caller passed
        // in. Comparing against the raw `root` here silently dropped
        // every event (a real bug this test caught): canonicalize once
        // up front so the prefix strip actually matches.
        let root_owned = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let mut watcher =
            notify::recommended_watcher(move |result: notify::Result<notify::Event>| {
                if let Ok(event) = result {
                    for path in event.paths {
                        let canonical = std::fs::canonicalize(&path).unwrap_or(path);
                        if let Ok(relative) = canonical.strip_prefix(&root_owned) {
                            sink.push(relative.to_string_lossy().replace('\\', "/"));
                        }
                    }
                }
            })?;
        watcher.watch(root, notify::RecursiveMode::Recursive)?;
        Ok(Self { _watcher: watcher })
    }
}

#[cfg(test)]
mod tests {
    /// `tqf sync` scanned the tree, then called `full_correctness_walk`,
    /// which scanned it again — and `scan_root` reads every file to
    /// classify it by content, so the whole repository was read twice.
    /// Taking the scan as a parameter fixes that, and this pins the two
    /// entry points to the same answer so the caller that passes its own
    /// scan cannot drift from the one that makes its own.
    #[test]
    fn passing_a_scan_in_gives_the_same_walk_as_making_one() {
        // Same isolation idiom the other tests here use: a counter as
        // well as the pid, because `cargo test` runs these in parallel
        // and a pid-only name collided.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "tqf-walk-equivalence-{}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/broker.rs"),
            "pub struct MemoryBroker { budget: u64 }\nimpl MemoryBroker { fn reserve() {} }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/cache.rs"),
            "pub struct WholeExpertLfuCache;\nfn evict_least_frequently_used() {}\n",
        )
        .unwrap();
        // A non-Rust file, so the language filter is exercised rather
        // than trivially satisfied.
        std::fs::write(root.join("README.md"), "# notes\n").unwrap();

        let table = super::FileTable::default();
        let (own_plan, own_contents) = super::full_correctness_walk(&root, &table).unwrap();

        let scan = super::scan_root(&root).unwrap();
        let (given_plan, given_contents) =
            super::full_correctness_walk_of(&root, &table, &scan).unwrap();

        assert_eq!(own_plan.new, given_plan.new);
        assert_eq!(own_plan.changed, given_plan.changed);
        assert_eq!(own_plan.unchanged, given_plan.unchanged);
        assert_eq!(own_plan.deleted, given_plan.deleted);
        assert_eq!(own_contents, given_contents);
        // And it really found the Rust files, so an equality of two
        // empty walks cannot pass this.
        assert_eq!(own_plan.new.len(), 2, "{:?}", own_plan.new);

        std::fs::remove_dir_all(&root).ok();
    }

    use super::*;

    #[test]
    fn debounce_queue_coalesces_a_repeated_burst_into_one_entry_per_path() {
        let mut queue = DebouncedEventQueue::new(Duration::from_millis(50));
        let start = Instant::now();
        // "Editor save storm": 500 rapid writes across only 5 distinct
        // paths, all within the debounce window.
        for i in 0..500 {
            let path = format!("src/file_{}.rs", i % 5);
            queue.record(path, start);
        }
        assert_eq!(
            queue.pending_count(),
            5,
            "500 bursty events on 5 paths should coalesce to 5 pending entries"
        );

        // Before the window elapses, nothing is ready.
        assert!(queue.drain_ready(start).is_empty());

        // After the window elapses, exactly the 5 distinct paths drain,
        // each exactly once.
        let ready = queue.drain_ready(start + Duration::from_millis(60));
        assert_eq!(ready.len(), 5);
        let unique: HashSet<String> = ready.into_iter().collect();
        assert_eq!(unique.len(), 5);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn debounce_queue_extends_the_window_on_repeated_writes_to_the_same_path() {
        let mut queue = DebouncedEventQueue::new(Duration::from_millis(50));
        let start = Instant::now();
        queue.record("src/hot.rs".to_string(), start);
        // A second write 30ms later refreshes the debounce timer.
        queue.record("src/hot.rs".to_string(), start + Duration::from_millis(30));
        // At 60ms since the *first* write (but only 30ms since the
        // second), the path must not be ready yet.
        assert!(queue
            .drain_ready(start + Duration::from_millis(60))
            .is_empty());
        // 50ms after the *second* write, it is ready.
        assert_eq!(
            queue.drain_ready(start + Duration::from_millis(81)),
            vec!["src/hot.rs".to_string()]
        );
    }

    fn write_file(path: &Path, contents: &str) {
        std::fs::write(path, contents).unwrap();
    }

    /// Builds a private, exclusively-owned temp directory seeded with
    /// real content copied from a few of this crate's own real source
    /// files (not synthetic text) under a `src/` subdirectory, so
    /// `scan_root`'s language classification behaves exactly as it
    /// would on the real repo. Every scratch-mutating test below uses
    /// its own isolated snapshot rather than mutating the live
    /// repository's `src/` tree directly — `cargo test` runs tests in
    /// parallel, and two tests concurrently creating/deleting files
    /// under the *same* real `src/` directory raced each other's
    /// `full_correctness_walk` results in an earlier version of this
    /// test (a real, found-and-fixed test-isolation bug, not a logic
    /// bug: each test's own walk correctly reflected whatever the
    /// filesystem actually looked like at that instant — the fix is
    /// giving each test exclusive filesystem ownership, not changing
    /// the walk logic).
    fn isolated_real_snapshot(label: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let temp_root = std::env::temp_dir().join(format!(
            "tqf-phase42-{label}-{}-{}-{unique}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = temp_root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        for (source, dest_name) in [
            ("src/memory/mod.rs", "memory_mod.rs"),
            ("src/experts/policy.rs", "experts_policy.rs"),
            ("src/retrieval/ignore.rs", "retrieval_ignore.rs"),
        ] {
            let contents = std::fs::read_to_string(crate_root.join(source)).unwrap();
            std::fs::write(src_dir.join(dest_name), contents).unwrap();
        }
        temp_root
    }

    /// Real end-to-end test on an isolated snapshot of this crate's own
    /// real source content (see `isolated_real_snapshot`): a full
    /// correctness walk correctly classifies a modified file as
    /// `changed`, a brand-new file as `new`, and leaves every untouched
    /// real file `unchanged` — then further walks after modifying/
    /// removing those files correctly detect `changed`/`deleted`.
    #[test]
    fn real_repo_incremental_walk_detects_change_new_and_delete() {
        let root = isolated_real_snapshot("walk");
        let table = FileTable::default();
        let (plan1, contents1) = full_correctness_walk(&root, &table).unwrap();
        assert!(
            plan1.unchanged.is_empty(),
            "an empty FileTable should classify everything as new"
        );
        assert_eq!(plan1.new.len(), 3, "expected the three real seeded files");

        let mut engine = SyncEngine::empty();
        engine.apply_structural_lexical(&plan1, &contents1);
        assert_eq!(engine.semantic_pending.len(), plan1.new.len());
        assert_eq!(engine.table.entries.len(), plan1.new.len());

        // Second walk with no real changes: everything should now be
        // `unchanged`.
        let (plan2, _contents2) = full_correctness_walk(&root, &engine.table).unwrap();
        assert!(plan2.new.is_empty());
        assert!(plan2.changed.is_empty());
        assert!(plan2.deleted.is_empty());
        assert_eq!(plan2.unchanged.len(), plan1.new.len());

        // A real new file, exercised through the real walker.
        let new_file = root.join("src").join("new_module.rs");
        write_file(&new_file, "pub fn phase42_probe() {}\n");
        let (plan3, contents3) = full_correctness_walk(&root, &engine.table).unwrap();
        let relative_new = "src/new_module.rs";
        assert!(
            plan3.new.iter().any(|p| p == relative_new),
            "expected the freshly created file to be detected as new: {:?}",
            plan3.new
        );
        engine.apply_structural_lexical(&plan3, &contents3);
        assert!(engine.semantic_pending.contains(relative_new));

        // Modify it: must be detected as `changed`, not `new` or
        // `unchanged`.
        write_file(&new_file, "pub fn phase42_probe() { let _ = 1 + 1; }\n");
        let (plan4, contents4) = full_correctness_walk(&root, &engine.table).unwrap();
        assert!(
            plan4.changed.iter().any(|p| p == relative_new),
            "expected the edited file to be detected as changed: {:?}",
            plan4.changed
        );
        engine.apply_structural_lexical(&plan4, &contents4);

        // Delete it: must be detected as `deleted`, and removed from
        // the sync engine's live state.
        std::fs::remove_file(&new_file).unwrap();
        let (plan5, contents5) = full_correctness_walk(&root, &engine.table).unwrap();
        assert!(
            plan5.deleted.iter().any(|p| p == relative_new),
            "expected the removed file to be detected as deleted: {:?}",
            plan5.deleted
        );
        engine.apply_structural_lexical(&plan5, &contents5);
        assert!(!engine.table.entries.contains_key(relative_new));
        assert!(!engine.semantic_pending.contains(relative_new));

        std::fs::remove_dir_all(&root).ok();
    }

    /// Proves "search remains usable during deferred semantic updates"
    /// directly: after a structural/lexical commit with a pending
    /// semantic file, lexical search over the *new* content works
    /// immediately, while the semantic lane either has no vector yet
    /// (a genuinely new file) or still serves its last-committed
    /// (stale) vector (a changed file) rather than going blind.
    #[test]
    fn lexical_search_stays_usable_while_semantic_is_pending() {
        let mut engine = SyncEngine::empty();
        let mut contents = HashMap::new();
        contents.insert(
            "src/example.rs".to_string(),
            "pub struct PendingSemanticProbe;".to_string(),
        );
        let plan = SyncPlan {
            new: vec!["src/example.rs".to_string()],
            ..Default::default()
        };
        engine.apply_structural_lexical(&plan, &contents);

        // Lexical/exact search is immediately usable over the new
        // content, with no semantic embedding having run at all.
        let hits = engine.lexical.exact_lookup("PendingSemanticProbe");
        assert_eq!(hits, vec!["src/example.rs"]);
        assert!(engine.semantic_pending.contains("src/example.rs"));
        assert!(!engine.semantic_vectors.contains_key("src/example.rs"));

        // Simulate a prior committed semantic vector, then a real edit:
        // the old vector must remain servable (stale-but-available)
        // rather than being deleted the moment the file changes.
        engine
            .semantic_vectors
            .insert("src/example.rs".to_string(), vec![1.0, 0.0]);
        engine.semantic_pending.remove("src/example.rs");
        contents.insert(
            "src/example.rs".to_string(),
            "pub struct PendingSemanticProbeV2;".to_string(),
        );
        let plan2 = SyncPlan {
            changed: vec!["src/example.rs".to_string()],
            ..Default::default()
        };
        engine.apply_structural_lexical(&plan2, &contents);
        assert!(engine.semantic_pending.contains("src/example.rs"));
        assert_eq!(
            engine.semantic_vectors.get("src/example.rs"),
            Some(&vec![1.0, 0.0]),
            "the stale semantic vector must still be servable while re-embedding is pending"
        );
        let hits = engine.lexical.exact_lookup("PendingSemanticProbeV2");
        assert_eq!(hits, vec!["src/example.rs"]);
    }

    /// Deterministic overflow stress test (spec §199: "on overflow/lost
    /// events, schedule a full correctness walk"): pushes far more
    /// events than the sink's capacity, proves the sink both latches
    /// `overflowed` and genuinely drops the excess (not silently
    /// growing), then proves the *recovery* path — a full correctness
    /// walk detects every real change correctly regardless of how much
    /// of the raw event stream was lost, exactly the "watcher events are
    /// hints, not the source of truth" guarantee.
    #[test]
    fn overflowing_event_sink_triggers_a_correct_full_walk_fallback() {
        let sink = BoundedEventSink::new(5);
        for i in 0..50 {
            sink.push(format!("src/burst_{i}.rs"));
        }
        assert!(
            sink.overflowed(),
            "50 events into a 5-capacity sink must overflow"
        );
        let drained = sink.drain();
        assert_eq!(
            drained.len(),
            5,
            "the sink must drop events past capacity, not grow unbounded"
        );

        // Recovery: even though the sink lost 45 of 50 real changes,
        // a full correctness walk (which never consults the sink at
        // all) still detects every real change against real files.
        let root = isolated_real_snapshot("overflow");
        let src_dir = root.join("src");
        let extra_paths: Vec<_> = (0..8)
            .map(|i| src_dir.join(format!("extra_{i}.rs")))
            .collect();
        for p in &extra_paths {
            write_file(p, "pub fn f() {}\n");
        }
        let table = FileTable::default();
        let (plan, _contents) = full_correctness_walk(&root, &table).unwrap();
        let detected = plan
            .new
            .iter()
            .filter(|p| p.starts_with("src/extra_"))
            .count();
        assert_eq!(
            detected, 8,
            "the full walk must find all 8 real new files even though the watcher sink only kept 5 of 50 hints"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// Real OS integration smoke test: wires a genuine `notify` watcher
    /// (FSEvents on macOS) to a real temp directory, performs a real
    /// file write, and confirms at least one real OS event actually
    /// reaches `LiveWatcher`'s sink. `#[ignore]` because real filesystem
    /// event delivery timing is environment-dependent (sandboxes/CI
    /// containers can restrict or delay FSEvents/inotify) — this proves
    /// the wiring is genuinely connected, not a substitute for the
    /// deterministic logic tests above.
    #[test]
    #[ignore = "depends on real OS filesystem event delivery timing"]
    fn real_os_watcher_delivers_a_real_file_write_event() {
        let dir = std::env::temp_dir().join(format!("tqf-phase42-watch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sink = std::sync::Arc::new(BoundedEventSink::new(64));
        let _watcher = LiveWatcher::watch(&dir, sink.clone()).expect("start real OS watcher");

        write_file(&dir.join("probe.rs"), "pub fn probe() {}\n");

        let deadline = Instant::now() + Duration::from_secs(3);
        let mut observed = Vec::new();
        while Instant::now() < deadline {
            observed = sink.drain();
            if !observed.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(
            !observed.is_empty(),
            "expected at least one real OS filesystem event within 3s of a real file write"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
