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
pub mod tiling;
mod writer;

// Module facade. `tqf` is a bin-only crate (spec §23: one crate, one
// binary, no `[lib]` target), so rustc reachability-analyses every
// `pub use` from `main` and reports the ones the product surface does not
// yet consume. These re-exports are the module's real interface — keeping
// them is deliberate; the allows go away as each is wired up.
#[allow(unused_imports)]
pub use importer::{canonical_header, convert_canonical_gguf, ConversionReport};
#[allow(unused_imports)]
pub use reader::TqfReader;
#[allow(unused_imports)]
pub use records::{
    ExpertIndexRecord, ExpertMatrix, ExpertTileRecord, SectionRecord, TensorExtentRecord,
    TqfSectionKind, EXPERT_INDEX_FLAG_TILE_CHECKSUMS,
};
#[allow(unused_imports)]
pub use superblock::{Superblock, FORMAT_MAJOR, FORMAT_MINOR};
#[allow(unused_imports)]
pub use writer::{RecoveredExpert, RecoveredExtent, RecoveredTile, TqfHeaderInfo, TqfWriter};
