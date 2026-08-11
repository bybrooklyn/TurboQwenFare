//! Crate-wide error taxonomy (spec Part XIV, section 119, REFERENCE BASELINE).
//! Structure is fixed by the spec; most subsystem variants are placeholders
//! until the owning subsystem exists.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, TqfError>;

#[derive(Debug, Error)]
pub enum TqfError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Setup(#[from] SetupError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error("cancelled")]
    Cancelled,
    #[error(transparent)]
    Internal(#[from] InternalError),
}

impl From<std::io::Error> for TqfError {
    fn from(e: std::io::Error) -> Self {
        TqfError::Io(IoError::from(e))
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid size value {0:?}: expected a byte/token count, optionally suffixed with K, M, or G")]
    InvalidSize(String),
    #[error("invalid --host value {0:?}: expected an IP address")]
    InvalidHost(String),
    #[error("environment error: {0}")]
    Environment(String),
    #[error("failed to serialize config: {0}")]
    Serialize(String),
}

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("model setup declined")]
    Declined,
    #[error("no model installed and no interactive terminal to confirm setup (use --yes)")]
    NonInteractiveConfirmationRequired,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model: {0}")]
    Unsupported(String),
}

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("corrupt or incompatible .tqf container: {0}")]
    Corrupt(String),
}

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error(
        "memory budget exceeded for {owner}: requested {requested} bytes, \
         available {available} bytes (try: {suggestion})"
    )]
    BudgetExceeded {
        requested: u64,
        available: u64,
        owner: String,
        suggestion: String,
    },
}

#[derive(Debug, Error)]
pub enum IoError {
    #[error(transparent)]
    Std(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("GPU backend failure: {0}")]
    Gpu(String),
}

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("context/session error: {0}")]
    Invalid(String),
}

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("retrieval error: {0}")]
    Failed(String),
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("protocol error: {0}")]
    Invalid(String),
}

/// Indicates a violated TQF invariant rather than user/environment error.
/// Always carries an incident id so it can be correlated with logs.
#[derive(Debug, Error)]
#[error("internal error (incident {incident_id}): {message}")]
pub struct InternalError {
    pub incident_id: String,
    pub message: String,
}
