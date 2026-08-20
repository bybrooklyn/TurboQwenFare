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
    #[error(transparent)]
    Source(#[from] SourceError),
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

// `#[from]` on `FormatError::Gguf`/`FormatError::Container` only gives
// `From<GgufError> for FormatError` (one level) — these fill in the second
// hop so parsing code can freely use `?` in functions returning the
// crate-wide `Result` without an intermediate `.map_err(FormatError::from)`.
impl From<GgufError> for TqfError {
    fn from(e: GgufError) -> Self {
        TqfError::Format(FormatError::Gguf(e))
    }
}

impl From<ContainerError> for TqfError {
    fn from(e: ContainerError) -> Self {
        TqfError::Format(FormatError::Container(e))
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid size value {0:?}: expected a byte/token count, optionally suffixed with K, M, or G")]
    InvalidSize(String),
    #[error("invalid --host value {0:?}: expected an IP address")]
    InvalidHost(String),
    #[error("--model {0:?} does not exist")]
    ModelPathMissing(String),
    #[error("--model {0:?} is a directory; point it at a .gguf checkpoint file")]
    ModelPathNotAFile(String),
    #[error(
        "--memory {given} is below the {floor} experimental floor; \
         use 2G for the experimental profile or 4G for the supported default (spec §4, §40)"
    )]
    MemoryBudgetTooSmall { given: String, floor: String },
    #[error("environment error: {0}")]
    Environment(String),
    #[error("failed to serialize config: {0}")]
    Serialize(String),
    #[error("invalid --open value {0:?}: expected one of opencode, claude, codex")]
    InvalidClient(String),
}

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("model setup declined")]
    Declined,
    #[error("no model installed and no interactive terminal to confirm setup (use --yes)")]
    NonInteractiveConfirmationRequired,
    /// A coding client could not be prepared or launched (spec §99-100).
    /// Distinct from `ConfigError::InvalidClient`, which is a bad flag
    /// value rather than a failure to run a valid one.
    #[error("could not launch the coding client: {0}")]
    ClientLaunch(String),
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported model: {0}")]
    Unsupported(String),
    #[error("tokenizer metadata key {0:?} is missing from the source checkpoint")]
    TokenizerMetadataMissing(&'static str),
    #[error(
        "unsupported tokenizer.ggml.model {0:?}: only byte-level BPE (\"gpt2\") is implemented"
    )]
    UnsupportedTokenizerModel(String),
    #[error("failed to build tokenizer: {0}")]
    TokenizerBuild(String),
    #[error("invalid {tensor} shape: expected {expected} elements, got {actual}")]
    Shape {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("full-attention layer {layer} reached configured context capacity {capacity}")]
    ContextCapacity { layer: u8, capacity: usize },
}

/// On-disk format errors: GGUF import (spec Part XVI phase 5) and the
/// native `.tqf` container (phase 6). Bounds/structure violations must
/// always resolve to one of these typed variants rather than a panic or an
/// out-of-bounds read (spec §119: "A corrupted `.tqf` or index never
/// triggers unsafe reads; readers bounds-check before mapping/dispatch.").
#[derive(Debug, Error)]
pub enum FormatError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Container(#[from] ContainerError),
    #[error(transparent)]
    Safetensors(#[from] SafetensorsError),
}

/// Errors from the minimal safetensors reader (`helper_model::safetensors`,
/// spec §37) used to ingest the pplx-embed helper-model checkpoint. Not a
/// general safetensors library — only the subset needed to read one known
/// model's F32 tensors.
#[derive(Debug, Error)]
pub enum SafetensorsError {
    #[error("safetensors header length {0} is missing or exceeds the sane bound")]
    HeaderLengthInvalid(u64),
    #[error("safetensors header is not valid JSON")]
    InvalidHeader,
    #[error("safetensors tensor {0:?} not found")]
    TensorNotFound(String),
    #[error("safetensors tensor {name:?} has unsupported dtype {dtype:?} (only F32 is read)")]
    UnsupportedDtype { name: String, dtype: String },
}

/// Errors from the strict GGUF import reader (spec §277, §115 invariants
/// #2/#3: little-endian only, offsets validated as `u64` before any
/// `usize` cast). Not a general GGUF library — only the subset needed to
/// enumerate the pinned canonical checkpoints.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("not a GGUF file: bad magic")]
    BadMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("integer overflow validating GGUF structure")]
    IntegerOverflow,
    #[error(
        "GGUF file truncated: needed {needed} bytes at offset {offset}, file is {available} bytes"
    )]
    Truncated {
        offset: u64,
        needed: u64,
        available: u64,
    },
    #[error("GGUF string length {0} exceeds sane bound")]
    StringTooLong(u64),
    #[error("invalid UTF-8 in GGUF string")]
    InvalidUtf8,
    #[error("unsupported GGUF metadata value type tag {0}")]
    UnsupportedValueType(u32),
    #[error("unsupported quant/ggml type id {0}")]
    UnsupportedQuantType(u32),
    #[error("tensor {name:?} has unsupported rank {rank} (max 4)")]
    UnsupportedRank { name: String, rank: u32 },
    #[error(
        "tensor {name:?} byte range [{offset}, {offset}+{len}) is outside the tensor-data \
         section (size {data_len})"
    )]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        len: u64,
        data_len: u64,
    },
    #[error("duplicate tensor name {0:?}")]
    DuplicateTensor(String),
    #[error("duplicate metadata key {0:?}")]
    DuplicateMetadataKey(String),
    #[error("declared tensor count {0} exceeds sane bound")]
    TooManyTensors(u64),
    #[error("declared metadata entry count {0} exceeds sane bound")]
    TooManyMetadataEntries(u64),
    #[error("invalid GGUF alignment value {0} (must be a nonzero power of two)")]
    InvalidAlignment(u64),
    #[error(
        "tensor {0:?} could not be classified into a known logical role (spec §118: unknown \
         production-language tensors are fatal, not silently skipped)"
    )]
    UnclassifiedTensor(String),
}

/// Errors from the native `.tqf` container reader/writer (spec Part XIV
/// sections 120-126, phase 6). Mirrors the fuzz-target checklist in spec
/// §246: bad magic, integer overflow in offsets/counts, truncated files,
/// out-of-bounds tables, unsupported quant layouts, corrupted metadata
/// hash.
#[derive(Debug, Error)]
pub enum ContainerError {
    #[error("not a .tqf container: bad magic")]
    BadMagic,
    #[error("unsupported .tqf superblock size {0} (expected 4096)")]
    BadSuperblockSize(u32),
    #[error("unsupported .tqf endian marker {0:#x}")]
    BadEndianMarker(u32),
    #[error("unsupported .tqf format major version {0}")]
    UnsupportedMajorVersion(u16),
    #[error("declared file length {declared} does not match actual file length {actual}")]
    FileLengthMismatch { declared: u64, actual: u64 },
    #[error("integer overflow validating .tqf table bounds")]
    IntegerOverflow,
    #[error("table {name} range [{offset}, {offset}+{len}) exceeds file length {file_len}")]
    TableOutOfBounds {
        name: &'static str,
        offset: u64,
        len: u64,
        file_len: u64,
    },
    #[error("metadata root hash mismatch: expected {expected}, computed {computed}")]
    MetadataRootHashMismatch { expected: String, computed: String },
    #[error("tensor with role {role_id} layer {layer:?} not found")]
    TensorNotFound { role_id: u32, layer: Option<u8> },
    #[error("expert (layer {layer}, expert {expert}) not found")]
    ExpertNotFound { layer: u8, expert: u16 },
    #[error("checksum mismatch for extent {0:?}")]
    ChecksumMismatch(String),
    #[error("duplicate extent name {0:?}")]
    DuplicateExtent(String),
    #[error("string table offset {offset} + length {len} exceeds string table size {table_len}")]
    StringTableOutOfBounds {
        offset: u64,
        len: u64,
        table_len: u64,
    },
    #[error(".tqf file truncated: needed {needed} bytes, file is {available} bytes")]
    Truncated { needed: u64, available: u64 },
    #[error("unknown .tqf section kind {0}")]
    UnknownSectionKind(u32),
    #[error("tensor extent has unsupported rank {0} (max 4)")]
    UnsupportedRank(u32),
    #[error("malformed .tqf record in {table}")]
    MalformedRecord { table: &'static str },
    #[error("{0}")]
    QuantMismatch(crate::format::quant::validate::QuantMismatchReport),
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

    // `.tqi` container faults (spec §174-§177). Distinct from
    // `ContainerError`, whose messages all name `.tqf`: reporting "not a
    // .tqf container" for an index file would send a reader looking at
    // the wrong format entirely.
    #[error("not a .tqi index: bad magic")]
    IndexBadMagic,
    #[error("`tqf sync {0:?}`: no such directory")]
    SyncPathMissing(String),
    #[error("`tqf sync {0:?}`: not a directory — sync indexes a project directory, not a file")]
    SyncPathNotADirectory(String),
    #[error("unsupported .tqi format major version {0}")]
    IndexUnsupportedMajorVersion(u16),
    #[error("malformed .tqi {what}: expected at least {expected} bytes, found {actual}")]
    IndexTruncated {
        what: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("malformed .tqi {0}")]
    IndexMalformed(&'static str),
    #[error("integer overflow validating .tqi table bounds")]
    IndexIntegerOverflow,
    #[error(".tqi {name} range [{offset}, {offset}+{len}) exceeds file length {file_len}")]
    IndexOutOfBounds {
        name: &'static str,
        offset: u64,
        len: u64,
        file_len: u64,
    },
    #[error(".tqi {segment} checksum mismatch: expected {expected}, computed {computed}")]
    IndexChecksumMismatch {
        segment: &'static str,
        expected: String,
        computed: String,
    },
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("protocol error: {0}")]
    Invalid(String),
}

/// Errors from the source resolver/downloader (spec Part V section 29-30,
/// Part XVI phase 4). Kept distinct from `SetupError` (terminal UX/policy
/// outcomes like "user declined") and `ModelError` (model *compatibility*):
/// a future `tqf doctor` needs to tell "network/verification failed" apart
/// from those.
#[derive(Debug, Error)]
pub enum SourceError {
    #[error("network request failed: {0}")]
    Network(#[from] reqwest::Error),
    #[error("http status {status} fetching {url}")]
    HttpStatus { status: u16, url: String },
    #[error("server does not honor byte-range requests for {url}")]
    RangeNotSupported { url: String },
    #[error("checksum mismatch for {artifact}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        artifact: String,
        expected: String,
        actual: String,
    },
    #[error("source revision/ETag changed mid-download for {artifact}: expected {expected}, got {actual}")]
    RevisionChanged {
        artifact: String,
        expected: String,
        actual: String,
    },
    #[error("resume journal is corrupt or inconsistent: {0}")]
    JournalCorrupt(String),
    #[error("local source unavailable: {0}")]
    LocalSourceUnavailable(std::io::Error),
    #[error("source metadata for {artifact} has no known size; cannot plan a resumable download")]
    UnknownSize { artifact: String },
    #[error(
        "short read for {artifact} at offset {offset}: expected {expected} bytes, got {actual}"
    )]
    ShortRead {
        artifact: String,
        offset: u64,
        expected: u64,
        actual: u64,
    },
}

/// Indicates a violated TQF invariant rather than user/environment error.
/// Always carries an incident id so it can be correlated with logs.
#[derive(Debug, Error)]
#[error("internal error (incident {incident_id}): {message}")]
pub struct InternalError {
    pub incident_id: String,
    pub message: String,
}
