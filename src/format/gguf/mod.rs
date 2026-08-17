//! GGUF import reader scoped to canonical Qwen3.6 checkpoints (spec Part V,
//! phase 5). Strict bounds-checked parsing; not a general GGUF library.

mod reader;
#[cfg(test)]
mod tests;
mod value;

#[cfg(test)]
pub use reader::open;
pub use reader::{open_with_broker, GgufFile, QuantBlockReader, TensorDescriptor};
pub use value::GgufValue;
