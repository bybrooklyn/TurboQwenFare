//! Generation and segment records (spec §175, §176, §177).
//!
//! A sync produces one logical generation made of immutable segments. The
//! commit record — here the generation table — references each segment's
//! offset, length, and hash, so a reader validates what it is about to
//! parse before parsing it.
//!
//! **Which segment kinds exist here, and why the rest do not.** Spec §175
//! lists nine. Four have real producers in this build and are written:
//! the file table, the chunk table, the lexical postings, and the exact
//! identifier map, plus a string table they share and a statistics
//! record. The other five are deliberately absent rather than written
//! empty:
//!
//! - `SymbolTableDelta` and `GraphEdgeDelta` need real AST output, which
//!   Phase 35/36 scoped out; there is no symbol or edge to record.
//! - `VectorDelta` and `PartitionDelta` need embeddings from the helper
//!   model, which is not installed (see `source::pin_capture`).
//! - `Tombstones` are meaningful only once generations are appended
//!   incrementally; this baseline commits one whole generation per sync
//!   (see `writer`), so nothing is ever superseded within a file.
//!
//! An empty segment would claim the capability exists and produced no
//! rows, which is a different and false statement (spec §335).

use crate::error::{Result, RetrievalError};

/// Discriminants are persisted, so they are assigned explicitly and never
/// reordered — the reserved values keep space for §175's remaining kinds
/// without renumbering these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SegmentKind {
    StringTable = 1,
    FileTable = 2,
    ChunkTable = 3,
    LexicalPostings = 4,
    ExactIdentifiers = 5,
    Statistics = 6,
    // 7 SymbolTable, 8 GraphEdge, 9 Vector, 10 Partition, 11 Tombstones
    // are reserved for the kinds listed in §175 that have no producer yet.
}

impl SegmentKind {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            1 => Self::StringTable,
            2 => Self::FileTable,
            3 => Self::ChunkTable,
            4 => Self::LexicalPostings,
            5 => Self::ExactIdentifiers,
            6 => Self::Statistics,
            _ => return None,
        })
    }
}

pub const SEGMENT_RECORD_SIZE: usize = 56;

/// Where one immutable segment lives and what it should hash to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentRecord {
    pub kind: SegmentKind,
    pub offset: u64,
    pub bytes: u64,
    pub blake3: [u8; 32],
}

impl SegmentRecord {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(self.kind as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // padding, keeps u64s aligned
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.bytes.to_le_bytes());
        buf.extend_from_slice(&self.blake3);
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SEGMENT_RECORD_SIZE {
            return Err(RetrievalError::IndexTruncated {
                what: "record",
                expected: SEGMENT_RECORD_SIZE as u64,
                actual: bytes.len() as u64,
            }
            .into());
        }
        let raw = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let kind = SegmentKind::from_u32(raw)
            .ok_or(RetrievalError::IndexMalformed("generation record"))?;
        Ok(Self {
            kind,
            offset: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            bytes: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            blake3: bytes[24..56].try_into().unwrap(),
        })
    }
}

/// One committed generation (spec §175).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRecord {
    pub generation: u64,
    pub committed_unix: u64,
    /// Highest `FileId` assigned so far, so the next sync continues the
    /// monotonic sequence rather than reusing an ID (spec §176: persisted
    /// monotonic IDs, not hashes, as primary keys).
    pub next_file_id: u64,
    pub next_chunk_id: u64,
    pub segments: Vec<SegmentRecord>,
}

impl GenerationRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.generation.to_le_bytes());
        buf.extend_from_slice(&self.committed_unix.to_le_bytes());
        buf.extend_from_slice(&self.next_file_id.to_le_bytes());
        buf.extend_from_slice(&self.next_chunk_id.to_le_bytes());
        buf.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for segment in &self.segments {
            segment.encode(&mut buf);
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        const HEADER: usize = 40;
        if bytes.len() < HEADER {
            return Err(RetrievalError::IndexTruncated {
                what: "record",
                expected: HEADER as u64,
                actual: bytes.len() as u64,
            }
            .into());
        }
        let count = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
        // Validate the declared count against the bytes actually present
        // before allocating for it (invariant #3: convert to `usize` only
        // after a checked bounds check).
        let needed = HEADER
            .checked_add(
                count
                    .checked_mul(SEGMENT_RECORD_SIZE)
                    .ok_or(RetrievalError::IndexMalformed("generation record"))?,
            )
            .ok_or(RetrievalError::IndexMalformed("generation record"))?;
        if bytes.len() < needed {
            return Err(RetrievalError::IndexTruncated {
                what: "record",
                expected: needed as u64,
                actual: bytes.len() as u64,
            }
            .into());
        }

        let mut segments = Vec::with_capacity(count);
        for index in 0..count {
            let start = HEADER + index * SEGMENT_RECORD_SIZE;
            segments.push(SegmentRecord::decode(
                &bytes[start..start + SEGMENT_RECORD_SIZE],
            )?);
        }

        Ok(Self {
            generation: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            committed_unix: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            next_file_id: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            next_chunk_id: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            segments,
        })
    }

    pub fn segment(&self, kind: SegmentKind) -> Option<&SegmentRecord> {
        self.segments.iter().find(|segment| segment.kind == kind)
    }
}

/// Spec §177's file record, carrying the fields this build actually
/// derives. `first_chunk`/`chunk_count` describe whole-file chunking —
/// Phase 36's scope boundary — so every file has exactly one chunk today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRecord {
    pub id: u64,
    pub path: u32,
    pub byte_len: u64,
    pub mtime_ns: u64,
    pub content_hash: [u8; 32],
    pub language: u32,
    pub first_chunk: u64,
    pub chunk_count: u32,
}

pub const FILE_RECORD_SIZE: usize = 80;

impl FileRecord {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.extend_from_slice(&self.path.to_le_bytes());
        buf.extend_from_slice(&self.language.to_le_bytes());
        buf.extend_from_slice(&self.byte_len.to_le_bytes());
        buf.extend_from_slice(&self.mtime_ns.to_le_bytes());
        buf.extend_from_slice(&self.content_hash);
        buf.extend_from_slice(&self.first_chunk.to_le_bytes());
        buf.extend_from_slice(&self.chunk_count.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < FILE_RECORD_SIZE {
            return Err(RetrievalError::IndexTruncated {
                what: "record",
                expected: FILE_RECORD_SIZE as u64,
                actual: bytes.len() as u64,
            }
            .into());
        }
        Ok(Self {
            id: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            path: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            language: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            byte_len: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            mtime_ns: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            content_hash: bytes[32..64].try_into().unwrap(),
            first_chunk: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            chunk_count: u32::from_le_bytes(bytes[72..76].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every persisted record must survive a round trip with every field
    /// intact. Written first because the encoder and decoder are two
    /// independent transcriptions of the same layout, and the failure
    /// mode is a field silently reading as another field's bytes.
    #[test]
    fn a_file_record_round_trips_every_field() {
        let record = FileRecord {
            id: 0x0102_0304_0506_0708,
            path: 0x1112_1314,
            byte_len: 0x2122_2324_2526_2728,
            mtime_ns: 0x3132_3334_3536_3738,
            content_hash: [0xAB; 32],
            language: 0x4142_4344,
            first_chunk: 0x5152_5354_5556_5758,
            chunk_count: 0x6162_6364,
        };

        let mut buf = Vec::new();
        record.encode(&mut buf);
        assert_eq!(
            buf.len(),
            FILE_RECORD_SIZE,
            "the declared record size must match what encode writes"
        );
        assert_eq!(FileRecord::decode(&buf).unwrap(), record);
    }

    #[test]
    fn a_segment_record_round_trips_and_its_size_is_declared_correctly() {
        let record = SegmentRecord {
            kind: SegmentKind::LexicalPostings,
            offset: 0x1122_3344_5566_7788,
            bytes: 0x99AA_BBCC_DDEE_FF00,
            blake3: [0x5A; 32],
        };
        let mut buf = Vec::new();
        record.encode(&mut buf);
        assert_eq!(buf.len(), SEGMENT_RECORD_SIZE);
        assert_eq!(SegmentRecord::decode(&buf).unwrap(), record);
    }

    #[test]
    fn a_generation_round_trips_with_its_segments_and_id_counters() {
        let generation = GenerationRecord {
            generation: 7,
            committed_unix: 1_760_000_000,
            next_file_id: 512,
            next_chunk_id: 512,
            segments: vec![
                SegmentRecord {
                    kind: SegmentKind::StringTable,
                    offset: 4096,
                    bytes: 128,
                    blake3: [1; 32],
                },
                SegmentRecord {
                    kind: SegmentKind::FileTable,
                    offset: 4224,
                    bytes: 240,
                    blake3: [2; 32],
                },
            ],
        };
        let encoded = generation.encode();
        assert_eq!(GenerationRecord::decode(&encoded).unwrap(), generation);
        assert_eq!(
            generation.segment(SegmentKind::FileTable).unwrap().offset,
            4224
        );
        assert!(generation.segment(SegmentKind::Statistics).is_none());
    }

    /// A truncated generation table must be rejected rather than parsed
    /// into whatever bytes happen to follow.
    #[test]
    fn a_truncated_generation_is_rejected() {
        let generation = GenerationRecord {
            generation: 1,
            committed_unix: 0,
            next_file_id: 0,
            next_chunk_id: 0,
            segments: vec![SegmentRecord {
                kind: SegmentKind::FileTable,
                offset: 4096,
                bytes: 16,
                blake3: [0; 32],
            }],
        };
        let encoded = generation.encode();
        for cut in [0, 8, 39, encoded.len() - 1] {
            assert!(
                GenerationRecord::decode(&encoded[..cut]).is_err(),
                "a {cut}-byte prefix must not decode"
            );
        }
    }

    /// A declared segment count that would overflow, or that the bytes
    /// cannot possibly contain, must be refused before it is used to size
    /// an allocation (invariant #3).
    #[test]
    fn an_implausible_segment_count_is_refused_without_allocating() {
        let mut bytes = vec![0u8; 40];
        bytes[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(GenerationRecord::decode(&bytes).is_err());
    }

    /// Persisted discriminants must never be reordered; this pins them.
    #[test]
    fn segment_kind_discriminants_are_stable() {
        for (kind, value) in [
            (SegmentKind::StringTable, 1),
            (SegmentKind::FileTable, 2),
            (SegmentKind::ChunkTable, 3),
            (SegmentKind::LexicalPostings, 4),
            (SegmentKind::ExactIdentifiers, 5),
            (SegmentKind::Statistics, 6),
        ] {
            assert_eq!(kind as u32, value);
            assert_eq!(SegmentKind::from_u32(value), Some(kind));
        }
        // Reserved for §175's kinds that have no producer yet.
        assert_eq!(SegmentKind::from_u32(7), None);
        assert_eq!(SegmentKind::from_u32(0), None);
    }
}
