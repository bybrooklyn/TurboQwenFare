//! Session scheduler and per-token decode loop; owns Qwen execution
//! (spec Part IV section 22; Part XIV sections 133-136). The scheduler and
//! decode loop themselves arrive with the model core (phases 12-15); this
//! phase only establishes the request/session shapes they'll operate on.

pub mod decode;
pub mod generation;
pub mod request;
pub mod session;

pub use decode::{
    decode_greedy, DecodeDiagnostics, DecodeTimings, DecodeToken, LayerHash, LayerStep,
    LogitCandidate, RouterTrace,
};
pub use generation::{
    GeneratedOutput, GeneratedToolCall, Qwen36Generator, Qwen36ResidentReferenceGenerator,
};
pub use request::{
    Message, MessageToolCall, NormalizedRequest, ProtocolFlavor, RetrievalPolicy, Role,
    SamplingParams, ToolDefinition, VisionInput,
};
pub use session::{GenerationSlot, Session, SessionId};
