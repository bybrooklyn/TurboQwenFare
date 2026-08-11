//! Transactional first-run model setup (spec Part V sections 28-30). Real
//! download/verify/repack land in phases 4-8; this phase implements the
//! state machine's decision point (hardware detect, receipt validation,
//! the install prompt) honestly, without pretending install exists yet.

pub mod flow;
pub mod hardware;
pub mod receipt;
