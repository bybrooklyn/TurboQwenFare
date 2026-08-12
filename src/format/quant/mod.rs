//! Shared Q4 quant-block schema used by both the GGUF reader and the `.tqf`
//! writer (spec Part V, section 30; Part XIV quant-layout records).

use serde::{Deserialize, Serialize};

use crate::error::GgufError;

pub mod dequant;
#[cfg(test)]
mod pipeline_tests;
pub mod repack;
pub mod validate;

/// GGML tensor element types as used by the GGUF wire format (external
/// prior art — llama.cpp/ggml's type IDs and block layouts, not defined by
/// the TQF spec itself). Only the subset actually needed to import the
/// pinned canonical checkpoints (Q4_K_M language, Q4_0 MTP, Q8_0 vision)
/// plus their common companion full-precision/K-quant types is
/// implemented — an unrecognized type ID is a hard parse error, never a
/// guess (spec §115 invariant #2: "readers must reject unsupported ...
/// rather than guessing").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
}

impl GgmlType {
    pub fn from_ggml_id(value: u32) -> Result<Self, GgufError> {
        Ok(match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::Bf16,
            other => return Err(GgufError::UnsupportedQuantType(other)),
        })
    }

    /// Elements per quantization block (1 for plain float/int types).
    pub const fn block_size(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64
            | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Bytes per quantization block.
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,
            Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Total byte size for `n_elements` elements of this type, checked
    /// against overflow so a malicious/corrupt huge element count never
    /// causes an unchecked multiply (spec §115 invariant #3).
    pub fn byte_size(self, n_elements: u64) -> Result<u64, GgufError> {
        let block_size = self.block_size();
        let n_blocks = n_elements
            .checked_add(block_size - 1)
            .and_then(|v| v.checked_div(block_size))
            .ok_or(GgufError::IntegerOverflow)?;
        n_blocks
            .checked_mul(self.block_bytes())
            .ok_or(GgufError::IntegerOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_known_ids() {
        assert_eq!(GgmlType::from_ggml_id(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_ggml_id(12).unwrap(), GgmlType::Q4K);
        assert_eq!(GgmlType::from_ggml_id(8).unwrap(), GgmlType::Q8_0);
    }

    #[test]
    fn unknown_id_is_a_typed_error() {
        assert!(matches!(
            GgmlType::from_ggml_id(255),
            Err(GgufError::UnsupportedQuantType(255))
        ));
    }

    #[test]
    fn byte_size_rounds_up_to_whole_blocks() {
        // Q4_0: 32 elements/block, 18 bytes/block.
        assert_eq!(GgmlType::Q4_0.byte_size(32).unwrap(), 18);
        assert_eq!(GgmlType::Q4_0.byte_size(33).unwrap(), 36); // rounds up to 2 blocks
        assert_eq!(GgmlType::F32.byte_size(10).unwrap(), 40);
    }

    #[test]
    fn byte_size_rejects_overflow() {
        assert!(matches!(
            GgmlType::Q8_0.byte_size(u64::MAX),
            Err(GgufError::IntegerOverflow)
        ));
    }
}
