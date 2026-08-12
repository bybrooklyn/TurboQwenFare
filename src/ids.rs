//! Foundational newtypes for layer/expert/page IDs and byte/token counts
//! (spec Part XIV section 116, REFERENCE BASELINE) — used throughout the
//! crate instead of passing around bare `u64`/`usize` so these domains can
//! never be silently mixed up (e.g. a byte count where a token count was
//! meant).

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LayerId(pub u8); // 0..39

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ExpertId(pub u16); // 0..255

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TileId(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ContextPageId(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Bytes(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Tokens(pub u32);

/// The canonical layer-kind table is compiled from the official
/// 3-linear/1-full pattern and also verified against the installed model
/// manifest. A mismatch is a fatal architecture error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    GatedDeltaNet,
    FullAttention,
}
