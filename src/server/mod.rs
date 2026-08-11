//! Protocol servers. All flavors normalize into one internal request/event
//! representation before reaching the runtime (spec Part IV section 26;
//! Part IX).

pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod tqf_api;
