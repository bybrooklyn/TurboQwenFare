//! The `.tqf` execution container: superblock, extent tables, expert tiles,
//! trusted receipts (spec Part V sections 31-36; Part XIV sections 120-126).

pub mod conversion;
pub mod importer;
#[cfg(test)]
mod pipeline_tests;
mod reader;
mod records;
mod superblock;
#[cfg(test)]
mod tests;
mod writer;

pub use importer::{canonical_header, convert_canonical_gguf, ConversionReport};
pub use reader::TqfReader;
pub use records::{
    ExpertIndexRecord, ExpertMatrix, ExpertTileRecord, SectionRecord, TensorExtentRecord,
    TqfSectionKind,
};
pub use superblock::{Superblock, FORMAT_MAJOR, FORMAT_MINOR};
pub use writer::{RecoveredExpert, RecoveredExtent, RecoveredTile, TqfHeaderInfo, TqfWriter};
