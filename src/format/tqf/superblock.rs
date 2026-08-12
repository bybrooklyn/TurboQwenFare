//! `.tqf` superblock v1 (spec §121, REFERENCE BASELINE): the first 4096
//! bytes of every `.tqf` file, byte-for-byte per the spec's table.

use crate::error::{ContainerError, Result};

pub const SUPERBLOCK_SIZE: usize = 4096;
pub const MAGIC: &[u8; 8] = b"TQFMODEL";
pub const ENDIAN_MARKER: u32 = 0x0102_0304;
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;

#[derive(Debug, Clone)]
pub struct Superblock {
    pub format_major: u16,
    pub format_minor: u16,
    pub backend_id: u32,
    pub feature_bits: u64,
    pub model_family_id: [u8; 16],
    pub source_sha256: [u8; 32],
    pub conversion_fingerprint: [u8; 32],
    pub file_length: u64,
    pub section_table_offset: u64,
    pub section_count: u32,
    pub section_record_bytes: u32,
    pub extent_table_offset: u64,
    pub extent_count: u64,
    pub string_table_offset: u64,
    pub string_table_bytes: u64,
    pub architecture_record_offset: u64,
    pub tokenizer_record_offset: u64,
    pub expert_index_offset: u64,
    pub expert_index_bytes: u64,
    pub checksum_table_offset: u64,
    pub checksum_table_bytes: u64,
    pub metadata_root_blake3: [u8; 32],
    pub creation_unix: u64,
    pub min_reader_capability_bits: u64,
}

impl Superblock {
    pub fn encode(&self) -> [u8; SUPERBLOCK_SIZE] {
        let mut buf = [0u8; SUPERBLOCK_SIZE];
        buf[0x000..0x008].copy_from_slice(MAGIC);
        buf[0x008..0x00A].copy_from_slice(&self.format_major.to_le_bytes());
        buf[0x00A..0x00C].copy_from_slice(&self.format_minor.to_le_bytes());
        buf[0x00C..0x010].copy_from_slice(&(SUPERBLOCK_SIZE as u32).to_le_bytes());
        buf[0x010..0x014].copy_from_slice(&ENDIAN_MARKER.to_le_bytes());
        buf[0x014..0x018].copy_from_slice(&self.backend_id.to_le_bytes());
        buf[0x018..0x020].copy_from_slice(&self.feature_bits.to_le_bytes());
        buf[0x020..0x030].copy_from_slice(&self.model_family_id);
        buf[0x030..0x050].copy_from_slice(&self.source_sha256);
        buf[0x050..0x070].copy_from_slice(&self.conversion_fingerprint);
        buf[0x070..0x078].copy_from_slice(&self.file_length.to_le_bytes());
        buf[0x078..0x080].copy_from_slice(&self.section_table_offset.to_le_bytes());
        buf[0x080..0x084].copy_from_slice(&self.section_count.to_le_bytes());
        buf[0x084..0x088].copy_from_slice(&self.section_record_bytes.to_le_bytes());
        buf[0x088..0x090].copy_from_slice(&self.extent_table_offset.to_le_bytes());
        buf[0x090..0x098].copy_from_slice(&self.extent_count.to_le_bytes());
        buf[0x098..0x0A0].copy_from_slice(&self.string_table_offset.to_le_bytes());
        buf[0x0A0..0x0A8].copy_from_slice(&self.string_table_bytes.to_le_bytes());
        buf[0x0A8..0x0B0].copy_from_slice(&self.architecture_record_offset.to_le_bytes());
        buf[0x0B0..0x0B8].copy_from_slice(&self.tokenizer_record_offset.to_le_bytes());
        buf[0x0B8..0x0C0].copy_from_slice(&self.expert_index_offset.to_le_bytes());
        buf[0x0C0..0x0C8].copy_from_slice(&self.expert_index_bytes.to_le_bytes());
        buf[0x0C8..0x0D0].copy_from_slice(&self.checksum_table_offset.to_le_bytes());
        buf[0x0D0..0x0D8].copy_from_slice(&self.checksum_table_bytes.to_le_bytes());
        buf[0x0D8..0x0F8].copy_from_slice(&self.metadata_root_blake3);
        buf[0x0F8..0x100].copy_from_slice(&self.creation_unix.to_le_bytes());
        buf[0x100..0x108].copy_from_slice(&self.min_reader_capability_bits.to_le_bytes());
        // 0x108..0x1000 (3832 bytes) reserved, left zero.
        buf
    }

    /// Steps 1-3 of the spec §121 reader validation order: magic,
    /// superblock size, endian marker, and format major version. Table
    /// bounds checking against the real file length (step 4) happens in
    /// `TqfReader::open_validated`, which is the only place that knows the
    /// real `fstat`-derived file length.
    pub fn decode(buf: &[u8; SUPERBLOCK_SIZE]) -> Result<Self> {
        if &buf[0x000..0x008] != MAGIC {
            return Err(ContainerError::BadMagic.into());
        }
        let format_major = u16::from_le_bytes(buf[0x008..0x00A].try_into().unwrap());
        let format_minor = u16::from_le_bytes(buf[0x00A..0x00C].try_into().unwrap());
        let superblock_bytes = u32::from_le_bytes(buf[0x00C..0x010].try_into().unwrap());
        if superblock_bytes != SUPERBLOCK_SIZE as u32 {
            return Err(ContainerError::BadSuperblockSize(superblock_bytes).into());
        }
        let endian_marker = u32::from_le_bytes(buf[0x010..0x014].try_into().unwrap());
        if endian_marker != ENDIAN_MARKER {
            return Err(ContainerError::BadEndianMarker(endian_marker).into());
        }
        if format_major != FORMAT_MAJOR {
            return Err(ContainerError::UnsupportedMajorVersion(format_major).into());
        }

        Ok(Self {
            format_major,
            format_minor,
            backend_id: u32::from_le_bytes(buf[0x014..0x018].try_into().unwrap()),
            feature_bits: u64::from_le_bytes(buf[0x018..0x020].try_into().unwrap()),
            model_family_id: buf[0x020..0x030].try_into().unwrap(),
            source_sha256: buf[0x030..0x050].try_into().unwrap(),
            conversion_fingerprint: buf[0x050..0x070].try_into().unwrap(),
            file_length: u64::from_le_bytes(buf[0x070..0x078].try_into().unwrap()),
            section_table_offset: u64::from_le_bytes(buf[0x078..0x080].try_into().unwrap()),
            section_count: u32::from_le_bytes(buf[0x080..0x084].try_into().unwrap()),
            section_record_bytes: u32::from_le_bytes(buf[0x084..0x088].try_into().unwrap()),
            extent_table_offset: u64::from_le_bytes(buf[0x088..0x090].try_into().unwrap()),
            extent_count: u64::from_le_bytes(buf[0x090..0x098].try_into().unwrap()),
            string_table_offset: u64::from_le_bytes(buf[0x098..0x0A0].try_into().unwrap()),
            string_table_bytes: u64::from_le_bytes(buf[0x0A0..0x0A8].try_into().unwrap()),
            architecture_record_offset: u64::from_le_bytes(buf[0x0A8..0x0B0].try_into().unwrap()),
            tokenizer_record_offset: u64::from_le_bytes(buf[0x0B0..0x0B8].try_into().unwrap()),
            expert_index_offset: u64::from_le_bytes(buf[0x0B8..0x0C0].try_into().unwrap()),
            expert_index_bytes: u64::from_le_bytes(buf[0x0C0..0x0C8].try_into().unwrap()),
            checksum_table_offset: u64::from_le_bytes(buf[0x0C8..0x0D0].try_into().unwrap()),
            checksum_table_bytes: u64::from_le_bytes(buf[0x0D0..0x0D8].try_into().unwrap()),
            metadata_root_blake3: buf[0x0D8..0x0F8].try_into().unwrap(),
            creation_unix: u64::from_le_bytes(buf[0x0F8..0x100].try_into().unwrap()),
            min_reader_capability_bits: u64::from_le_bytes(buf[0x100..0x108].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Superblock {
        Superblock {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            backend_id: 1,
            feature_bits: 0,
            model_family_id: [7u8; 16],
            source_sha256: [1u8; 32],
            conversion_fingerprint: [2u8; 32],
            file_length: 123_456,
            section_table_offset: 4096,
            section_count: 3,
            section_record_bytes: 64,
            extent_table_offset: 8192,
            extent_count: 5,
            string_table_offset: 16384,
            string_table_bytes: 256,
            architecture_record_offset: 0,
            tokenizer_record_offset: 0,
            expert_index_offset: 0,
            expert_index_bytes: 0,
            checksum_table_offset: 20000,
            checksum_table_bytes: 400,
            metadata_root_blake3: [9u8; 32],
            creation_unix: 1_700_000_000,
            min_reader_capability_bits: 0,
        }
    }

    #[test]
    fn round_trips() {
        let sb = sample();
        let encoded = sb.encode();
        assert_eq!(encoded.len(), SUPERBLOCK_SIZE);
        let decoded = Superblock::decode(&encoded).unwrap();
        assert_eq!(decoded.file_length, 123_456);
        assert_eq!(decoded.extent_count, 5);
        assert_eq!(decoded.source_sha256, [1u8; 32]);
    }

    #[test]
    fn reserved_tail_is_zero() {
        let encoded = sample().encode();
        assert!(encoded[0x108..].iter().all(|&b| b == 0));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut encoded = sample().encode();
        encoded[0] = b'X';
        assert!(Superblock::decode(&encoded).is_err());
    }

    #[test]
    fn rejects_bad_endian_marker() {
        let mut encoded = sample().encode();
        encoded[0x010..0x014].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes());
        assert!(Superblock::decode(&encoded).is_err());
    }

    #[test]
    fn rejects_unsupported_major_version() {
        let mut encoded = sample().encode();
        encoded[0x008..0x00A].copy_from_slice(&99u16.to_le_bytes());
        assert!(Superblock::decode(&encoded).is_err());
    }
}
