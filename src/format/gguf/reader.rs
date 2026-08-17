//! Strict GGUF reader (spec §277, §326: "GGUF importer does not imply
//! generic GGUF runtime" — only the subset needed to enumerate the pinned
//! canonical checkpoints). Parses the header/metadata/tensor-info section
//! (small, bounded) into memory, then reads tensor payload bytes directly
//! from disk on demand via `QuantBlockReader` — a 20+ GB checkpoint is
//! never loaded whole.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::error::{GgufError, Result};
use crate::format::byte_reader::ByteReader;
use crate::format::gguf::value::{read_gguf_string, read_gguf_value, GgufValue};
use crate::format::quant::GgmlType;
use crate::ids::Bytes;
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: u32 = 4;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
/// The pinned checkpoints' header regions (magic + version + counts + KV
/// metadata + tensor-info array) are well under a megabyte even with a
/// large vocab; this bound keeps `open()` from ever reading anywhere near
/// the multi-gigabyte tensor-data section just to parse metadata.
const MAX_HEADER_PROBE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    /// Dimensions in GGUF file order (fastest-varying first).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub n_elements: u64,
    pub byte_size: u64,
    /// Absolute, bounds-validated offset of this tensor's bytes in the
    /// file.
    pub file_offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    source_bytes: u64,
    header_blake3: [u8; 32],
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorDescriptor>,
    _metadata_lease: MemoryLease,
}

impl GgufFile {
    /// Digest of the parsed GGUF header/metadata/tensor-directory region,
    /// excluding alignment padding and tensor payload bytes. This makes
    /// tokenizer provenance cheap to revalidate at startup.
    pub fn header_blake3(&self) -> [u8; 32] {
        self.header_blake3
    }

    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Metadata keys under the `tokenizer.` prefix — the "tokenizer source
    /// data" output named in spec §277.
    pub fn tokenizer_metadata(&self) -> HashMap<&str, &GgufValue> {
        self.metadata
            .iter()
            .filter(|(k, _)| k.starts_with("tokenizer."))
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// Bounded-memory reader over one tensor's quantization blocks — the
    /// "source quant block iterator" named in spec §277. Never loads the
    /// whole (potentially multi-gigabyte) tensor into memory at once.
    pub fn quant_block_reader(&self, tensor: &TensorDescriptor) -> Result<QuantBlockReader> {
        Ok(QuantBlockReader::open(&self.path, tensor.clone())?)
    }
}

pub fn open_with_broker(path: &Path, broker: &MemoryBroker) -> Result<GgufFile> {
    let file_len = std::fs::metadata(path)?.len();

    let probe_len = file_len.min(MAX_HEADER_PROBE_BYTES);
    // Parsing duplicates strings and descriptor data while the raw probe is
    // still live. Reserve both the temporary probe and a conservative 4x
    // retained-metadata/tokenizer envelope before either allocation can
    // occur; the extra headroom covers HashMap/string allocator overhead.
    let metadata_envelope = probe_len
        .checked_mul(4)
        .ok_or(GgufError::IntegerOverflow)?
        .max(1);
    let metadata_lease = broker.reserve(
        MemoryOwner::Core,
        MemoryClass::Fixed,
        Bytes(metadata_envelope),
        64,
    )?;
    let probe_lease = broker.reserve(
        MemoryOwner::IoStaging,
        MemoryClass::Transient,
        Bytes(probe_len.max(1)),
        64,
    )?;
    let probe_capacity: usize = probe_len
        .try_into()
        .map_err(|_| GgufError::IntegerOverflow)?;
    let mut probe = Vec::with_capacity(probe_capacity);
    File::open(path)?.take(probe_len).read_to_end(&mut probe)?;

    let parsed = parse_header_and_metadata(&probe, file_len)?;
    let header_len: usize = parsed
        .header_end
        .try_into()
        .map_err(|_| GgufError::IntegerOverflow)?;
    let header = probe.get(..header_len).ok_or(GgufError::Truncated {
        offset: 0,
        needed: parsed.header_end,
        available: probe.len() as u64,
    })?;
    let header_blake3 = *blake3::hash(header).as_bytes();
    drop(probe_lease);
    Ok(GgufFile {
        path: path.to_path_buf(),
        source_bytes: file_len,
        header_blake3,
        version: parsed.version,
        metadata: parsed.metadata,
        tensors: parsed.tensors,
        _metadata_lease: metadata_lease,
    })
}

/// Unit-test convenience. Product paths must supply the process-wide broker
/// through `open_with_broker` so metadata cannot escape `--memory`.
#[cfg(test)]
pub fn open(path: &Path) -> Result<GgufFile> {
    let broker = MemoryBroker::new(Bytes(5 * MAX_HEADER_PROBE_BYTES));
    open_with_broker(path, &broker)
}

struct ParsedHeader {
    version: u32,
    header_end: u64,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorDescriptor>,
}

fn trunc(reader: &ByteReader, needed: u64) -> GgufError {
    GgufError::Truncated {
        offset: reader.position() as u64,
        needed,
        available: reader.remaining() as u64,
    }
}

fn parse_header_and_metadata(
    probe: &[u8],
    file_len: u64,
) -> std::result::Result<ParsedHeader, GgufError> {
    let mut reader = ByteReader::new(probe);

    let magic = reader.take(4).ok_or_else(|| trunc(&reader, 4))?;
    if magic != MAGIC {
        return Err(GgufError::BadMagic);
    }

    let version = reader.read_u32().ok_or_else(|| trunc(&reader, 4))?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = reader.read_u64().ok_or_else(|| trunc(&reader, 8))?;
    if tensor_count > MAX_TENSORS {
        return Err(GgufError::TooManyTensors(tensor_count));
    }
    let metadata_kv_count = reader.read_u64().ok_or_else(|| trunc(&reader, 8))?;
    if metadata_kv_count > MAX_METADATA_ENTRIES {
        return Err(GgufError::TooManyMetadataEntries(metadata_kv_count));
    }

    let mut metadata = HashMap::with_capacity(metadata_kv_count.min(4096) as usize);
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value_type = reader.read_u32().ok_or_else(|| trunc(&reader, 4))?;
        let value = read_gguf_value(&mut reader, value_type)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufError::DuplicateMetadataKey(key));
        }
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment(alignment));
    }

    // First pass: parse every tensor-info record so the reader cursor
    // lands exactly at the end of the tensor-info array (needed to compute
    // `data_start` below) before any bounds validation against file
    // length.
    let mut raw_tensors = Vec::with_capacity(tensor_count.min(65_536) as usize);
    let mut seen_names = HashSet::with_capacity(tensor_count.min(65_536) as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)?;
        if !seen_names.insert(name.clone()) {
            return Err(GgufError::DuplicateTensor(name));
        }
        let n_dims = reader.read_u32().ok_or_else(|| trunc(&reader, 4))?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(GgufError::UnsupportedRank { name, rank: n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(reader.read_u64().ok_or_else(|| trunc(&reader, 8))?);
        }
        let ggml_type_id = reader.read_u32().ok_or_else(|| trunc(&reader, 4))?;
        let ggml_type = GgmlType::from_ggml_id(ggml_type_id)?;
        let relative_offset = reader.read_u64().ok_or_else(|| trunc(&reader, 8))?;

        let n_elements = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or(GgufError::IntegerOverflow)?;
        let byte_size = ggml_type.byte_size(n_elements)?;

        raw_tensors.push((
            name,
            dims,
            ggml_type,
            n_elements,
            byte_size,
            relative_offset,
        ));
    }

    let header_end = reader.position() as u64;
    let data_start = align_up(header_end, alignment)?;
    let data_section_len = file_len.saturating_sub(data_start);

    let mut tensors = Vec::with_capacity(raw_tensors.len());
    for (name, dims, ggml_type, n_elements, byte_size, relative_offset) in raw_tensors {
        let end_relative = relative_offset
            .checked_add(byte_size)
            .ok_or(GgufError::IntegerOverflow)?;
        if end_relative > data_section_len {
            return Err(GgufError::TensorOutOfBounds {
                name,
                offset: relative_offset,
                len: byte_size,
                data_len: data_section_len,
            });
        }
        let file_offset = data_start
            .checked_add(relative_offset)
            .ok_or(GgufError::IntegerOverflow)?;
        tensors.push(TensorDescriptor {
            name,
            dims,
            ggml_type,
            n_elements,
            byte_size,
            file_offset,
        });
    }

    Ok(ParsedHeader {
        version,
        header_end,
        metadata,
        tensors,
    })
}

fn align_up(value: u64, alignment: u64) -> std::result::Result<u64, GgufError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(GgufError::IntegerOverflow)
}

/// Reads one tensor's quantization blocks directly from disk in bounded
/// batches — the "source quant block iterator" named in spec §277.
pub struct QuantBlockReader {
    file: File,
    ggml_type: GgmlType,
    start_offset: u64,
    block_bytes: u64,
    total_blocks: u64,
    next_block: u64,
    batch_blocks: u64,
}

/// ~1024 blocks per batch keeps peak memory small (at most ~1024 * 292
/// bytes ≈ 300 KiB for the widest K-quant block type) while avoiding a
/// syscall per individual block on a 20+ GB tensor.
const DEFAULT_BATCH_BLOCKS: u64 = 1024;

impl QuantBlockReader {
    fn open(path: &Path, tensor: TensorDescriptor) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let block_bytes = tensor.ggml_type.block_bytes();
        Ok(Self {
            file,
            ggml_type: tensor.ggml_type,
            start_offset: tensor.file_offset,
            block_bytes,
            // `byte_size` is always an exact whole-block multiple (built
            // via `GgmlType::byte_size`), so this division is exact.
            total_blocks: tensor.byte_size / block_bytes,
            next_block: 0,
            batch_blocks: DEFAULT_BATCH_BLOCKS,
        })
    }

    pub fn ggml_type(&self) -> GgmlType {
        self.ggml_type
    }

    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_blocks * self.block_bytes
    }

    pub fn max_batch_bytes(&self) -> u64 {
        self.total_blocks.min(self.batch_blocks) * self.block_bytes
    }

    /// Reads the next batch of blocks (up to the configured batch size) as
    /// one contiguous byte buffer, or `None` once the tensor is exhausted.
    pub fn next_batch(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        if self.next_block >= self.total_blocks {
            return Ok(None);
        }
        let blocks_remaining = self.total_blocks - self.next_block;
        let batch = blocks_remaining.min(self.batch_blocks);
        let byte_offset = self.start_offset + self.next_block * self.block_bytes;
        let byte_len = (batch * self.block_bytes) as usize;

        let mut buf = vec![0u8; byte_len];
        self.file.read_exact_at(&mut buf, byte_offset)?;
        self.next_block += batch;
        Ok(Some(buf))
    }
}
