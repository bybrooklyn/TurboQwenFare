//! Writing and reading `.tqi` (spec §173-§177).
//!
//! **Commit model.** Spec §175 describes appending immutable segments per
//! sync and compacting later. This baseline writes one whole generation
//! per sync and replaces the file atomically — which is §173's compaction
//! path used as the ordinary path. It is the honest REFERENCE BASELINE:
//! correct, simple to validate, and cheap at the corpus sizes measured so
//! far (Phase 41 found flat search competitive at this scale). Appending
//! deltas is the optimization, and it needs the tombstone and overlay
//! machinery §175 describes; it is not written here rather than written
//! badly.
//!
//! **Durability.** Invariant #9: a persisted write that would be
//! expensive to recompute uses temp + fsync + atomic rename. An index is
//! exactly that — rebuilding it means re-walking and re-tokenizing the
//! whole tree — and a torn `.tqi` that still parses would be worse than
//! none, because it would be trusted.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use crate::error::{Result, RetrievalError};

use super::segments::{FileRecord, GenerationRecord, SegmentKind, SegmentRecord, FILE_RECORD_SIZE};
use super::superblock::{TqiSuperblock, FORMAT_MAJOR, FORMAT_MINOR, SUPERBLOCK_SIZE};

/// Everything one generation persists.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct IndexContents {
    pub files: Vec<FileRecord>,
    /// `term -> [(chunk_id, term_frequency)]`, sorted by chunk id per
    /// spec §185.
    pub postings: BTreeMap<String, Vec<(u32, u32)>>,
    /// Case-preserved exact identifier lane (spec §83, §182).
    pub exact: BTreeMap<String, Vec<u32>>,
    /// Token count per chunk, which BM25 needs for length normalization.
    pub chunk_lengths: Vec<u32>,
    /// `FileId -> path`. Paths live in the string table on disk; this is
    /// the resolved form both sides work with.
    pub paths: BTreeMap<u64, String>,
    pub next_file_id: u64,
    pub next_chunk_id: u64,
}

/// Interns strings once so paths and terms are not duplicated per record.
#[derive(Default)]
struct StringTable {
    bytes: Vec<u8>,
    offsets: BTreeMap<String, u32>,
}

impl StringTable {
    fn intern(&mut self, value: &str) -> u32 {
        if let Some(id) = self.offsets.get(value) {
            return *id;
        }
        let id = self.bytes.len() as u32;
        self.bytes
            .extend_from_slice(&(value.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
        self.offsets.insert(value.to_string(), id);
        id
    }

    fn read(bytes: &[u8], id: u32) -> Result<String> {
        let start = id as usize;
        let end = start
            .checked_add(4)
            .ok_or(RetrievalError::IndexIntegerOverflow)?;
        if end > bytes.len() {
            return Err(RetrievalError::IndexOutOfBounds {
                name: "string table entry",
                offset: id as u64,
                len: 4,
                file_len: bytes.len() as u64,
            }
            .into());
        }
        let len = u32::from_le_bytes(bytes[start..end].try_into().unwrap()) as usize;
        let text_end = end
            .checked_add(len)
            .ok_or(RetrievalError::IndexIntegerOverflow)?;
        if text_end > bytes.len() {
            return Err(RetrievalError::IndexOutOfBounds {
                name: "string table text",
                offset: end as u64,
                len: len as u64,
                file_len: bytes.len() as u64,
            }
            .into());
        }
        String::from_utf8(bytes[end..text_end].to_vec())
            .map_err(|_| RetrievalError::IndexMalformed("string table is not UTF-8").into())
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Serializes `contents` and replaces `path` atomically.
pub fn write(
    path: &Path,
    root: &Path,
    contents: &IndexContents,
    previous: Option<&TqiSuperblock>,
) -> Result<()> {
    let mut strings = StringTable::default();

    // Paths first, so a file record's `path` id is valid by construction.
    let mut file_bytes = Vec::with_capacity(contents.files.len() * FILE_RECORD_SIZE);
    for file in &contents.files {
        let mut record = file.clone();
        // A file with no recorded path would silently become
        // "<unknown>" and stay unsearchable, so refuse to write one.
        let path = contents.paths.get(&file.id).ok_or_else(|| {
            RetrievalError::Failed(format!("file id {} has no recorded path", file.id))
        })?;
        record.path = strings.intern(path);
        record.encode(&mut file_bytes);
    }

    let mut postings_bytes = Vec::new();
    postings_bytes.extend_from_slice(&(contents.postings.len() as u64).to_le_bytes());
    for (term, list) in &contents.postings {
        postings_bytes.extend_from_slice(&strings.intern(term).to_le_bytes());
        postings_bytes.extend_from_slice(&(list.len() as u32).to_le_bytes());
        for (chunk, frequency) in list {
            postings_bytes.extend_from_slice(&chunk.to_le_bytes());
            postings_bytes.extend_from_slice(&frequency.to_le_bytes());
        }
    }

    let mut exact_bytes = Vec::new();
    exact_bytes.extend_from_slice(&(contents.exact.len() as u64).to_le_bytes());
    for (identifier, list) in &contents.exact {
        exact_bytes.extend_from_slice(&strings.intern(identifier).to_le_bytes());
        exact_bytes.extend_from_slice(&(list.len() as u32).to_le_bytes());
        for chunk in list {
            exact_bytes.extend_from_slice(&chunk.to_le_bytes());
        }
    }

    let mut chunk_bytes = Vec::new();
    chunk_bytes.extend_from_slice(&(contents.chunk_lengths.len() as u64).to_le_bytes());
    for length in &contents.chunk_lengths {
        chunk_bytes.extend_from_slice(&length.to_le_bytes());
    }

    let mut statistics = Vec::new();
    statistics.extend_from_slice(&(contents.files.len() as u64).to_le_bytes());
    statistics.extend_from_slice(&(contents.postings.len() as u64).to_le_bytes());
    statistics.extend_from_slice(&(contents.exact.len() as u64).to_le_bytes());

    // The string table is written last because interning happens while
    // building the others, but it is placed first on disk so a reader can
    // resolve ids while streaming the rest.
    let ordered: Vec<(SegmentKind, Vec<u8>)> = vec![
        (SegmentKind::StringTable, std::mem::take(&mut strings.bytes)),
        (SegmentKind::FileTable, file_bytes),
        (SegmentKind::ChunkTable, chunk_bytes),
        (SegmentKind::LexicalPostings, postings_bytes),
        (SegmentKind::ExactIdentifiers, exact_bytes),
        (SegmentKind::Statistics, statistics),
    ];

    let mut body = Vec::new();
    let mut records = Vec::with_capacity(ordered.len());
    for (kind, bytes) in ordered {
        let offset = (SUPERBLOCK_SIZE + body.len()) as u64;
        records.push(SegmentRecord {
            kind,
            offset,
            bytes: bytes.len() as u64,
            blake3: *blake3::hash(&bytes).as_bytes(),
        });
        body.extend_from_slice(&bytes);
    }

    let generation = GenerationRecord {
        generation: previous.map(|s| s.latest_generation + 1).unwrap_or(1),
        committed_unix: unix_now(),
        next_file_id: contents.next_file_id,
        next_chunk_id: contents.next_chunk_id,
        segments: records,
    };
    let generation_bytes = generation.encode();
    let generation_offset = (SUPERBLOCK_SIZE + body.len()) as u64;

    let index_uuid = match previous {
        Some(previous) => previous.index_uuid,
        None => super::superblock::new_index_uuid()?,
    };
    let superblock = TqiSuperblock {
        format_major: FORMAT_MAJOR,
        format_minor: FORMAT_MINOR,
        index_uuid,
        root_identity: super::superblock::root_identity(root, &index_uuid)?,
        latest_generation: generation.generation,
        generation_table_offset: generation_offset,
        generation_table_bytes: generation_bytes.len() as u64,
        generation_table_hash: *blake3::hash(&generation_bytes).as_bytes(),
        created_unix: previous.map(|s| s.created_unix).unwrap_or_else(unix_now),
        last_compacted_unix: unix_now(),
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Temp + fsync + rename, so a crash mid-write leaves the previous
    // index intact rather than a truncated one that still parses.
    let temp = path.with_extension("tqi.tmp");
    {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(&superblock.encode())?;
        file.write_all(&body)?;
        file.write_all(&generation_bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temp, path)?;
    Ok(())
}

/// A validated, loaded index.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedIndex {
    pub superblock: TqiSuperblock,
    pub generation: GenerationRecord,
    pub contents: IndexContents,
}

fn segment_slice<'a>(
    raw: &'a [u8],
    record: &SegmentRecord,
    name: &'static str,
) -> Result<&'a [u8]> {
    let end = record
        .offset
        .checked_add(record.bytes)
        .ok_or(RetrievalError::IndexIntegerOverflow)?;
    if end > raw.len() as u64 {
        return Err(RetrievalError::IndexOutOfBounds {
            name,
            offset: record.offset,
            len: record.bytes,
            file_len: raw.len() as u64,
        }
        .into());
    }
    let bytes = &raw[record.offset as usize..end as usize];

    // Validate before parsing, not after: a corrupt segment that happens
    // to decode into plausible records is the case a checksum exists for.
    let computed = blake3::hash(bytes);
    if computed.as_bytes() != &record.blake3 {
        return Err(RetrievalError::IndexChecksumMismatch {
            segment: name,
            expected: hex(&record.blake3),
            computed: hex(computed.as_bytes()),
        }
        .into());
    }
    Ok(bytes)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn read(path: &Path) -> Result<LoadedIndex> {
    let raw = std::fs::read(path)?;
    let superblock = TqiSuperblock::decode(&raw)?;

    let table_end = superblock
        .generation_table_offset
        .checked_add(superblock.generation_table_bytes)
        .ok_or(RetrievalError::IndexIntegerOverflow)?;
    if table_end > raw.len() as u64 {
        return Err(RetrievalError::IndexOutOfBounds {
            name: "generation table",
            offset: superblock.generation_table_offset,
            len: superblock.generation_table_bytes,
            file_len: raw.len() as u64,
        }
        .into());
    }
    let table_bytes = &raw[superblock.generation_table_offset as usize..table_end as usize];
    let computed = blake3::hash(table_bytes);
    if computed.as_bytes() != &superblock.generation_table_hash {
        return Err(RetrievalError::IndexChecksumMismatch {
            segment: "generation table",
            expected: hex(&superblock.generation_table_hash),
            computed: hex(computed.as_bytes()),
        }
        .into());
    }
    let generation = GenerationRecord::decode(table_bytes)?;

    let strings = match generation.segment(SegmentKind::StringTable) {
        Some(record) => segment_slice(&raw, record, "string table")?.to_vec(),
        None => Vec::new(),
    };

    let mut contents = IndexContents {
        next_file_id: generation.next_file_id,
        next_chunk_id: generation.next_chunk_id,
        ..IndexContents::default()
    };

    if let Some(record) = generation.segment(SegmentKind::FileTable) {
        let bytes = segment_slice(&raw, record, "file table")?;
        if !bytes.len().is_multiple_of(FILE_RECORD_SIZE) {
            return Err(RetrievalError::IndexMalformed("file table length").into());
        }
        for chunk in bytes.chunks_exact(FILE_RECORD_SIZE) {
            let file = FileRecord::decode(chunk)?;
            contents
                .paths
                .insert(file.id, StringTable::read(&strings, file.path)?);
            contents.files.push(file);
        }
    }

    if let Some(record) = generation.segment(SegmentKind::ChunkTable) {
        let bytes = segment_slice(&raw, record, "chunk table")?;
        let mut cursor = Cursor::new(bytes);
        let count = cursor.u64()? as usize;
        contents.chunk_lengths.reserve(count.min(1 << 20));
        for _ in 0..count {
            contents.chunk_lengths.push(cursor.u32()?);
        }
    }

    if let Some(record) = generation.segment(SegmentKind::LexicalPostings) {
        let bytes = segment_slice(&raw, record, "lexical postings")?;
        let mut cursor = Cursor::new(bytes);
        let terms = cursor.u64()?;
        for _ in 0..terms {
            let term = StringTable::read(&strings, cursor.u32()?)?;
            let count = cursor.u32()? as usize;
            let mut list = Vec::with_capacity(count.min(1 << 20));
            for _ in 0..count {
                list.push((cursor.u32()?, cursor.u32()?));
            }
            contents.postings.insert(term, list);
        }
    }

    if let Some(record) = generation.segment(SegmentKind::ExactIdentifiers) {
        let bytes = segment_slice(&raw, record, "exact identifiers")?;
        let mut cursor = Cursor::new(bytes);
        let count = cursor.u64()?;
        for _ in 0..count {
            let identifier = StringTable::read(&strings, cursor.u32()?)?;
            let chunks = cursor.u32()? as usize;
            let mut list = Vec::with_capacity(chunks.min(1 << 20));
            for _ in 0..chunks {
                list.push(cursor.u32()?);
            }
            contents.exact.insert(identifier, list);
        }
    }

    Ok(LoadedIndex {
        superblock,
        generation,
        contents,
    })
}

/// Bounds-checked sequential reader. Every read is validated against the
/// remaining length, so a truncated segment errors rather than panicking
/// on a slice index.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .ok_or(RetrievalError::IndexIntegerOverflow)?;
        if end > self.bytes.len() {
            return Err(RetrievalError::IndexTruncated {
                what: "segment",
                expected: end as u64,
                actual: self.bytes.len() as u64,
            }
            .into());
        }
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tqf-tqi-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample() -> IndexContents {
        let mut contents = IndexContents {
            next_file_id: 3,
            next_chunk_id: 3,
            chunk_lengths: vec![12, 7, 40],
            ..IndexContents::default()
        };
        for (id, path) in [(0u64, "src/main.rs"), (1, "src/lib.rs"), (2, "README.md")] {
            contents.files.push(FileRecord {
                id,
                path: 0,
                byte_len: 100 + id,
                mtime_ns: 1_700_000_000_000_000_000 + id,
                content_hash: [id as u8; 32],
                language: 1,
                first_chunk: id,
                chunk_count: 1,
            });
            contents.paths.insert(id, path.to_string());
        }
        contents
            .postings
            .insert("broker".to_string(), vec![(0, 3), (2, 1)]);
        contents
            .postings
            .insert("memory".to_string(), vec![(0, 5), (1, 2)]);
        contents
            .exact
            .insert("MemoryBroker".to_string(), vec![0, 1]);
        contents
    }

    /// The property the whole format exists for: what a sync writes is
    /// what a later process reads.
    #[test]
    fn an_index_round_trips_through_the_filesystem() {
        let dir = scratch("roundtrip");
        let path = dir.join("index.tqi");
        let contents = sample();

        write(&path, &dir, &contents, None).unwrap();
        let loaded = read(&path).unwrap();

        assert_eq!(loaded.contents.files, contents.files_with_interned_paths());
        assert_eq!(loaded.contents.paths, contents.paths);
        assert_eq!(loaded.contents.postings, contents.postings);
        assert_eq!(loaded.contents.exact, contents.exact);
        assert_eq!(loaded.contents.chunk_lengths, contents.chunk_lengths);
        assert_eq!(loaded.contents.next_file_id, 3);
        assert_eq!(loaded.superblock.latest_generation, 1);
        assert_eq!(loaded.generation.segments.len(), 6);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Spec §176: IDs are persisted and monotonic, so a second sync
    /// continues the sequence rather than reissuing an ID that other
    /// records already reference.
    #[test]
    fn a_second_generation_advances_and_keeps_the_index_identity() {
        let dir = scratch("generations");
        let path = dir.join("index.tqi");

        write(&path, &dir, &sample(), None).unwrap();
        let first = read(&path).unwrap();

        let mut next = sample();
        next.next_file_id = 9;
        write(&path, &dir, &next, Some(&first.superblock)).unwrap();
        let second = read(&path).unwrap();

        assert_eq!(second.superblock.latest_generation, 2);
        assert_eq!(second.contents.next_file_id, 9);
        // Compaction rewrites the file but must not change its identity,
        // nor forget when it was created.
        assert_eq!(second.superblock.index_uuid, first.superblock.index_uuid);
        assert_eq!(
            second.superblock.created_unix,
            first.superblock.created_unix
        );
        assert_eq!(
            second.superblock.root_identity,
            first.superblock.root_identity
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_that_is_not_an_index_is_rejected_by_magic() {
        let dir = scratch("magic");
        let path = dir.join("index.tqi");
        std::fs::write(&path, vec![0u8; SUPERBLOCK_SIZE + 64]).unwrap();
        let error = read(&path).unwrap_err().to_string();
        assert!(error.contains("bad magic"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A newer major version may have moved any field, so an old reader
    /// must refuse rather than misread confidently.
    #[test]
    fn a_future_major_version_is_refused() {
        let dir = scratch("version");
        let path = dir.join("index.tqi");
        write(&path, &dir, &sample(), None).unwrap();

        let mut raw = std::fs::read(&path).unwrap();
        raw[0x008..0x00A].copy_from_slice(&(FORMAT_MAJOR + 1).to_le_bytes());
        std::fs::write(&path, &raw).unwrap();

        let error = read(&path).unwrap_err().to_string();
        assert!(error.contains("unsupported"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The case checksums exist for: bytes that still parse into
    /// plausible records but are not what was written.
    #[test]
    fn a_flipped_byte_inside_a_segment_is_caught_before_it_is_parsed() {
        let dir = scratch("corrupt");
        let path = dir.join("index.tqi");
        write(&path, &dir, &sample(), None).unwrap();

        let mut raw = std::fs::read(&path).unwrap();
        // Somewhere inside the segment body, past the superblock.
        let victim = SUPERBLOCK_SIZE + 8;
        raw[victim] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        let error = read(&path).unwrap_err().to_string();
        assert!(
            error.contains("checksum mismatch"),
            "corruption must be caught by checksum, got: {error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_tampered_generation_table_is_caught() {
        let dir = scratch("table");
        let path = dir.join("index.tqi");
        write(&path, &dir, &sample(), None).unwrap();

        let loaded = read(&path).unwrap();
        let mut raw = std::fs::read(&path).unwrap();
        let at = loaded.superblock.generation_table_offset as usize;
        raw[at] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        let error = read(&path).unwrap_err().to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A truncated file must error rather than panic on a slice index.
    #[test]
    fn truncation_at_any_point_errors_rather_than_panicking() {
        let dir = scratch("truncate");
        let path = dir.join("index.tqi");
        write(&path, &dir, &sample(), None).unwrap();
        let full = std::fs::read(&path).unwrap();

        for cut in [
            0,
            SUPERBLOCK_SIZE / 2,
            SUPERBLOCK_SIZE,
            SUPERBLOCK_SIZE + 16,
            full.len() - 1,
        ] {
            std::fs::write(&path, &full[..cut]).unwrap();
            assert!(
                read(&path).is_err(),
                "a {cut}-byte prefix must be rejected, not parsed"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Invariant #9: a crash mid-write must leave the previous index
    /// intact. The temp file is what makes that true, so it must not be
    /// the destination and must not survive a successful write.
    #[test]
    fn writing_replaces_atomically_and_leaves_no_temp_behind() {
        let dir = scratch("atomic");
        let path = dir.join("index.tqi");

        write(&path, &dir, &sample(), None).unwrap();
        let first = std::fs::read(&path).unwrap();

        let mut second_contents = sample();
        second_contents
            .postings
            .insert("added".into(), vec![(1, 1)]);
        let first_superblock = read(&path).unwrap().superblock;
        write(&path, &dir, &second_contents, Some(&first_superblock)).unwrap();

        assert_ne!(std::fs::read(&path).unwrap(), first, "the file must change");
        assert!(
            !path.with_extension("tqi.tmp").exists(),
            "the temp file must not survive a successful write"
        );
        assert!(read(&path).unwrap().contents.postings.contains_key("added"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_with_no_recorded_path_is_refused_rather_than_written_unknown() {
        let dir = scratch("nopath");
        let path = dir.join("index.tqi");
        let mut contents = sample();
        contents.paths.remove(&1);

        let error = write(&path, &dir, &contents, None).unwrap_err().to_string();
        assert!(error.contains("no recorded path"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_index_round_trips() {
        let dir = scratch("empty");
        let path = dir.join("index.tqi");
        write(&path, &dir, &IndexContents::default(), None).unwrap();

        let loaded = read(&path).unwrap();
        assert!(loaded.contents.files.is_empty());
        assert!(loaded.contents.postings.is_empty());
        assert_eq!(loaded.superblock.latest_generation, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    impl IndexContents {
        /// The written records carry string-table ids in `path`, so a
        /// round-trip comparison has to expect the ids the writer
        /// assigned rather than the caller's placeholder.
        fn files_with_interned_paths(&self) -> Vec<FileRecord> {
            let mut strings = StringTable::default();
            self.files
                .iter()
                .map(|file| {
                    let mut record = file.clone();
                    record.path = strings.intern(&self.paths[&file.id]);
                    record
                })
                .collect()
        }
    }
}
