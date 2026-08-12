//! Bounds-checked little-endian byte reading, shared by the GGUF importer
//! and the `.tqf` container reader (spec §115 invariant #2: all persisted
//! integers in these formats are little-endian, readers reject anything
//! else rather than guessing; invariant #3: offsets/lengths are `u64`,
//! converted to `usize` only after checked bounds validation). Every read
//! here returns `None` on short input instead of panicking or reading out
//! of bounds — callers convert `None` into a format-specific typed error.

pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        if end > self.buf.len() {
            return None;
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Some(slice)
    }

    pub fn read_u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }

    pub fn read_bool(&mut self) -> Option<bool> {
        self.read_u8().map(|b| b != 0)
    }

    pub fn read_u16(&mut self) -> Option<u16> {
        self.take(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_i16(&mut self) -> Option<i16> {
        self.take(2)
            .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_i32(&mut self) -> Option<i32> {
        self.take(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_f32(&mut self) -> Option<f32> {
        self.take(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_u64(&mut self) -> Option<u64> {
        self.take(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_i64(&mut self) -> Option<i64> {
        self.take(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn read_f64(&mut self) -> Option<f64> {
        self.take(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
    }
}

pub fn read_u32_at(buf: &[u8], offset: usize) -> Option<u32> {
    ByteReader::new(buf.get(offset..)?).read_u32()
}

pub fn read_u64_at(buf: &[u8], offset: usize) -> Option<u64> {
    ByteReader::new(buf.get(offset..)?).read_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian_scalars_in_sequence() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x01u8.to_le_bytes());
        bytes.extend_from_slice(&0x0203u16.to_le_bytes());
        bytes.extend_from_slice(&0x04050607u32.to_le_bytes());
        bytes.extend_from_slice(&0x08090a0b0c0d0e0fu64.to_le_bytes());

        let mut reader = ByteReader::new(&bytes);
        assert_eq!(reader.read_u8(), Some(0x01));
        assert_eq!(reader.read_u16(), Some(0x0203));
        assert_eq!(reader.read_u32(), Some(0x0405_0607));
        assert_eq!(reader.read_u64(), Some(0x0809_0a0b_0c0d_0e0f));
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn short_reads_return_none_not_panic() {
        let bytes = [0u8; 3];
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(reader.read_u32(), None);
        // A failed read must not advance the cursor or corrupt state.
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn take_rejects_offset_overflow() {
        let bytes = [0u8; 4];
        let mut reader = ByteReader::new(&bytes);
        reader.pos = usize::MAX - 1;
        assert_eq!(reader.take(4), None);
    }
}
