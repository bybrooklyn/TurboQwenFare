//! `.tqi` superblock v1 (spec §174), byte-for-byte per the spec's table.
//!
//! Mirrors `format::tqf::superblock`'s conventions deliberately: a
//! fixed 4096-byte header, little-endian throughout (invariant #2), `u64`
//! offsets and lengths on disk (invariant #3), and a major-version
//! rejection so a future format cannot be silently misread by an older
//! reader.

use crate::error::{Result, RetrievalError};

pub const SUPERBLOCK_SIZE: usize = 4096;
pub const MAGIC: &[u8; 8] = b"TQFINDEX";
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;

/// The `.tqi` header. Field order and offsets are fixed by spec §174.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TqiSuperblock {
    pub format_major: u16,
    pub format_minor: u16,
    /// Identifies this index across compactions, which rewrite the file
    /// but must not change its identity.
    pub index_uuid: [u8; 16],
    /// Normalized root device/path identity plus the index UUID. Spec
    /// §174 is explicit that the human directory name is not semantic
    /// evidence, so the name is deliberately absent from this hash — a
    /// renamed directory is the same root, and two clones with the same
    /// name are not.
    pub root_identity: [u8; 32],
    pub latest_generation: u64,
    pub generation_table_offset: u64,
    pub generation_table_bytes: u64,
    pub generation_table_hash: [u8; 32],
    pub created_unix: u64,
    pub last_compacted_unix: u64,
}

impl TqiSuperblock {
    pub fn encode(&self) -> [u8; SUPERBLOCK_SIZE] {
        let mut buf = [0u8; SUPERBLOCK_SIZE];
        buf[0x000..0x008].copy_from_slice(MAGIC);
        buf[0x008..0x00A].copy_from_slice(&self.format_major.to_le_bytes());
        buf[0x00A..0x00C].copy_from_slice(&self.format_minor.to_le_bytes());
        buf[0x00C..0x010].copy_from_slice(&(SUPERBLOCK_SIZE as u32).to_le_bytes());
        buf[0x010..0x020].copy_from_slice(&self.index_uuid);
        buf[0x020..0x040].copy_from_slice(&self.root_identity);
        buf[0x040..0x048].copy_from_slice(&self.latest_generation.to_le_bytes());
        buf[0x048..0x050].copy_from_slice(&self.generation_table_offset.to_le_bytes());
        buf[0x050..0x058].copy_from_slice(&self.generation_table_bytes.to_le_bytes());
        buf[0x058..0x078].copy_from_slice(&self.generation_table_hash);
        buf[0x078..0x080].copy_from_slice(&self.created_unix.to_le_bytes());
        buf[0x080..0x088].copy_from_slice(&self.last_compacted_unix.to_le_bytes());
        // 0x088.. stays reserved zero.
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SUPERBLOCK_SIZE {
            return Err(RetrievalError::IndexTruncated {
                what: "record",
                expected: SUPERBLOCK_SIZE as u64,
                actual: bytes.len() as u64,
            }
            .into());
        }
        if &bytes[0x000..0x008] != MAGIC {
            return Err(RetrievalError::IndexBadMagic.into());
        }

        let format_major = u16::from_le_bytes(bytes[0x008..0x00A].try_into().unwrap());
        // Reject rather than guess: a newer major version may have moved
        // any field below, so parsing on would read the wrong bytes and
        // report them confidently.
        if format_major != FORMAT_MAJOR {
            return Err(RetrievalError::IndexUnsupportedMajorVersion(format_major).into());
        }
        let declared = u32::from_le_bytes(bytes[0x00C..0x010].try_into().unwrap());
        if declared as usize != SUPERBLOCK_SIZE {
            return Err(RetrievalError::IndexTruncated {
                what: "record",
                expected: SUPERBLOCK_SIZE as u64,
                actual: declared as u64,
            }
            .into());
        }

        Ok(Self {
            format_major,
            format_minor: u16::from_le_bytes(bytes[0x00A..0x00C].try_into().unwrap()),
            index_uuid: bytes[0x010..0x020].try_into().unwrap(),
            root_identity: bytes[0x020..0x040].try_into().unwrap(),
            latest_generation: u64::from_le_bytes(bytes[0x040..0x048].try_into().unwrap()),
            generation_table_offset: u64::from_le_bytes(bytes[0x048..0x050].try_into().unwrap()),
            generation_table_bytes: u64::from_le_bytes(bytes[0x050..0x058].try_into().unwrap()),
            generation_table_hash: bytes[0x058..0x078].try_into().unwrap(),
            created_unix: u64::from_le_bytes(bytes[0x078..0x080].try_into().unwrap()),
            last_compacted_unix: u64::from_le_bytes(bytes[0x080..0x088].try_into().unwrap()),
        })
    }
}

/// Root identity per spec §174: device and inode of the root directory,
/// plus the index UUID. Uses the filesystem's own identity rather than the
/// path string, so a renamed or symlinked root is still recognized as the
/// same root — and two same-named clones are not confused for each other.
pub fn root_identity(root: &std::path::Path, index_uuid: &[u8; 16]) -> Result<[u8; 32]> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::metadata(root)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"tqi-root-identity-v1");
    hasher.update(&metadata.dev().to_le_bytes());
    hasher.update(&metadata.ino().to_le_bytes());
    hasher.update(index_uuid);
    Ok(*hasher.finalize().as_bytes())
}

/// 16 random bytes for a new index's UUID.
pub fn new_index_uuid() -> Result<[u8; 16]> {
    use std::io::Read;
    let mut file = std::fs::File::open("/dev/urandom")?;
    let mut buf = [0u8; 16];
    file.read_exact(&mut buf)?;
    Ok(buf)
}
