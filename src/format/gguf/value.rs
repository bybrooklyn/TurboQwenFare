//! GGUF metadata key/value parsing (external wire format — llama.cpp's
//! GGUF metadata value types; the TQF spec does not redefine these, spec
//! §326: "GGUF importer does not imply generic GGUF runtime," so only what
//! the pinned checkpoints actually use is implemented).

use crate::error::GgufError;
use crate::format::byte_reader::ByteReader;

/// A metadata string longer than this is treated as corrupt/hostile input
/// rather than an unusually large legitimate value — no GGUF metadata
/// string in the canonical checkpoints approaches this size.
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
/// Same reasoning, for array element counts.
const MAX_ARRAY_ELEMENTS: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::U64(v) => Some(v),
            _ => None,
        }
    }

    /// Widens any integer variant (signed or unsigned) to `i64` — used for
    /// fields like `tokenizer.ggml.token_type`, which llama.cpp's GGUF
    /// convention stores as signed `INT32` even though every value it
    /// actually uses is small and non-negative.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U8(v) => Some(v as i64),
            Self::U16(v) => Some(v as i64),
            Self::U32(v) => Some(v as i64),
            Self::U64(v) => Some(v as i64),
            Self::I8(v) => Some(v as i64),
            Self::I16(v) => Some(v as i64),
            Self::I32(v) => Some(v as i64),
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// GGUF metadata value type tags (external format constants).
const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;

pub fn read_gguf_string(reader: &mut ByteReader) -> Result<String, GgufError> {
    let len = reader.read_u64().ok_or(GgufError::Truncated {
        offset: reader.position() as u64,
        needed: 8,
        available: reader.remaining() as u64,
    })?;
    if len > MAX_STRING_BYTES {
        return Err(GgufError::StringTooLong(len));
    }
    let len = len as usize; // checked above against a sane bound before cast
    let bytes = reader.take(len).ok_or(GgufError::Truncated {
        offset: reader.position() as u64,
        needed: len as u64,
        available: reader.remaining() as u64,
    })?;
    String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
}

pub fn read_gguf_value(reader: &mut ByteReader, value_type: u32) -> Result<GgufValue, GgufError> {
    let need = |n: u64| -> Result<(), GgufError> {
        if (reader.remaining() as u64) < n {
            Err(GgufError::Truncated {
                offset: reader.position() as u64,
                needed: n,
                available: reader.remaining() as u64,
            })
        } else {
            Ok(())
        }
    };

    Ok(match value_type {
        TYPE_U8 => {
            need(1)?;
            GgufValue::U8(reader.read_u8().unwrap())
        }
        TYPE_I8 => {
            need(1)?;
            GgufValue::I8(reader.read_u8().unwrap() as i8)
        }
        TYPE_U16 => {
            need(2)?;
            GgufValue::U16(reader.read_u16().unwrap())
        }
        TYPE_I16 => {
            need(2)?;
            GgufValue::I16(reader.read_i16().unwrap())
        }
        TYPE_U32 => {
            need(4)?;
            GgufValue::U32(reader.read_u32().unwrap())
        }
        TYPE_I32 => {
            need(4)?;
            GgufValue::I32(reader.read_i32().unwrap())
        }
        TYPE_F32 => {
            need(4)?;
            GgufValue::F32(reader.read_f32().unwrap())
        }
        TYPE_BOOL => {
            need(1)?;
            GgufValue::Bool(reader.read_bool().unwrap())
        }
        TYPE_STRING => GgufValue::String(read_gguf_string(reader)?),
        TYPE_U64 => {
            need(8)?;
            GgufValue::U64(reader.read_u64().unwrap())
        }
        TYPE_I64 => {
            need(8)?;
            GgufValue::I64(reader.read_i64().unwrap())
        }
        TYPE_F64 => {
            need(8)?;
            GgufValue::F64(reader.read_f64().unwrap())
        }
        TYPE_ARRAY => {
            let element_type = reader.read_u32().ok_or(GgufError::Truncated {
                offset: reader.position() as u64,
                needed: 4,
                available: reader.remaining() as u64,
            })?;
            if element_type == TYPE_ARRAY {
                // GGUF arrays are homogeneous scalar/string; nested arrays
                // are not part of the format this importer accepts.
                return Err(GgufError::UnsupportedValueType(TYPE_ARRAY));
            }
            let count = reader.read_u64().ok_or(GgufError::Truncated {
                offset: reader.position() as u64,
                needed: 8,
                available: reader.remaining() as u64,
            })?;
            if count > MAX_ARRAY_ELEMENTS {
                return Err(GgufError::IntegerOverflow);
            }
            let mut values = Vec::with_capacity(count.min(1024) as usize);
            for _ in 0..count {
                values.push(read_gguf_value(reader, element_type)?);
            }
            GgufValue::Array(values)
        }
        other => return Err(GgufError::UnsupportedValueType(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scalar_types() {
        let bytes = 42u32.to_le_bytes();
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(
            read_gguf_value(&mut reader, TYPE_U32).unwrap(),
            GgufValue::U32(42)
        );
    }

    #[test]
    fn reads_string() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5u64.to_le_bytes());
        bytes.extend_from_slice(b"hello");
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(
            read_gguf_value(&mut reader, TYPE_STRING).unwrap(),
            GgufValue::String("hello".to_string())
        );
    }

    #[test]
    fn reads_array_of_u32() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&TYPE_U32.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        let mut reader = ByteReader::new(&bytes);
        assert_eq!(
            read_gguf_value(&mut reader, TYPE_ARRAY).unwrap(),
            GgufValue::Array(vec![
                GgufValue::U32(1),
                GgufValue::U32(2),
                GgufValue::U32(3)
            ])
        );
    }

    #[test]
    fn rejects_nested_arrays() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&TYPE_ARRAY.to_le_bytes());
        let mut reader = ByteReader::new(&bytes);
        assert!(matches!(
            read_gguf_value(&mut reader, TYPE_ARRAY),
            Err(GgufError::UnsupportedValueType(_))
        ));
    }

    #[test]
    fn rejects_oversized_declared_string_length_without_allocating() {
        let bytes = u64::MAX.to_le_bytes();
        let mut reader = ByteReader::new(&bytes);
        assert!(matches!(
            read_gguf_string(&mut reader),
            Err(GgufError::StringTooLong(_))
        ));
    }

    #[test]
    fn rejects_unknown_type_tag() {
        let bytes = [0u8; 8];
        let mut reader = ByteReader::new(&bytes);
        assert!(matches!(
            read_gguf_value(&mut reader, 999),
            Err(GgufError::UnsupportedValueType(999))
        ));
    }

    #[test]
    fn truncated_value_is_an_error_not_a_panic() {
        let bytes = [0u8; 2]; // needs 4 for a u32
        let mut reader = ByteReader::new(&bytes);
        assert!(matches!(
            read_gguf_value(&mut reader, TYPE_U32),
            Err(GgufError::Truncated { .. })
        ));
    }
}
