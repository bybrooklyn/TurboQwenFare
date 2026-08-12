//! On-disk formats: the native `.tqf` container, the GGUF import reader,
//! and shared Q4 quant-schema types (spec Part V).

pub mod byte_reader;
pub mod gguf;
pub mod quant;
pub mod tqf;
