//! Session scheduler and per-token decode loop; owns Qwen execution
//! (spec Part IV section 22; Part XIV sections 133-136). The scheduler and
//! decode loop themselves arrive with the model core (phases 12-15); this
//! phase only establishes the request/session shapes they'll operate on.

pub mod decode;
pub mod generation;
pub mod mtp;
pub mod request;
pub mod session;

pub use decode::{
    DecodeDiagnostics, DecodeTimings, DecodeToken, LayerHash,
    LogitCandidate, RouterTrace,
};
pub use generation::{
    GeneratedOutput, Qwen36Generator, Qwen36ResidentReferenceGenerator,
};
pub use request::{
    Message, MessageToolCall, NormalizedRequest, ProtocolFlavor, Role, ToolDefinition,
};
pub use session::{GenerationSlot, Session};
