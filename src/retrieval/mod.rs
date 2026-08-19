//! TQIndex retrieval facade: optional RAG augmentation the model core does
//! not depend on (spec Part X). Must remain removable without affecting
//! core inference validity.

pub mod adaptive;
pub mod classify;
pub mod flat;
pub mod hybrid;
pub mod ignore;
pub mod lexical;
pub mod scan;
pub mod sync;
pub mod tqvec;
