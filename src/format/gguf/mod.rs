//! GGUF import reader scoped to canonical Qwen3.6 checkpoints (spec Part V,
//! phase 5). Strict bounds-checked parsing; not a general GGUF library.

mod reader;
#[cfg(test)]
mod tests;
mod value;

pub use reader::{open, GgufFile, QuantBlockReader, TensorDescriptor};
pub use value::GgufValue;
