//! Session scheduler and per-token decode loop; owns Qwen execution
//! (spec Part IV section 22; Part XIV sections 133-136). The scheduler and
//! decode loop themselves arrive with the model core (phases 12-15); this
//! phase only establishes the request/session shapes they'll operate on.

pub mod decode;
pub mod generation;
pub mod mtp;
pub mod request;
pub mod session;
pub mod stream_decoder;

// Module facade. `tqf` is a bin-only crate (spec §23: one crate, one
// binary, no `[lib]` target), so rustc reachability-analyses every
// `pub use` from `main` and reports the ones the product surface does not
// yet consume. These re-exports are the module's real interface — keeping
// them is deliberate; the allows go away as each is wired up.
#[allow(unused_imports)]
pub use decode::{
    decode_greedy, DecodeDiagnostics, DecodeTimings, DecodeToken, LayerHash, LayerStep,
    LogitCandidate, RouterTrace,
};
#[allow(unused_imports)]
pub use generation::{
    GeneratedOutput, GeneratedToolCall, Qwen36Generator, Qwen36ResidentReferenceGenerator,
};
#[allow(unused_imports)]
pub use request::{
    Message, MessageToolCall, NormalizedRequest, ProtocolFlavor, RetrievalPolicy, Role,
    SamplingParams, ToolDefinition, VisionInput,
};
#[allow(unused_imports)]
pub use session::{GenerationSlot, Session, SessionId};
