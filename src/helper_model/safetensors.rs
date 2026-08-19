//! Minimal read-only safetensors parser (spec §37's "helper `.tqf`
//! conversion"): the pplx-embed source checkpoint ships as a single
//! `model.safetensors` file, not GGUF, so `format::gguf` doesn't apply.
//! Only the subset needed to enumerate a known checkpoint's F32 tensors
//! is implemented — this is not a general safetensors library, matching
//! `format::gguf`'s own "only the subset needed" scope note.
//!
//! Wire format: an 8-byte little-endian header length, then that many
//! bytes of UTF-8 JSON (a map of tensor name to `{dtype, shape,
//! data_offsets: [start, end]}`, plus an optional `__metadata__` key),
//! then the raw tensor bytes at those offsets relative to the end of the
//! header.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;

use serde::Deserialize;

use crate::error::{FormatError, Result, SafetensorsError};

#[derive(Debug, Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Clone)]
pub struct SafetensorsEntry {
    pub dtype: String,
    pub shape: Vec<u64>,
    start: u64,
    end: u64,
}

pub struct SafetensorsFile {
    file: File,
    data_base: u64,
    entries: HashMap<String, SafetensorsEntry>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf);
        // Sane bound: the pplx-embed header is a few KB; refuse anything
        // that looks like a corrupt/adversarial length rather than
        // allocating an unbounded buffer (spec §115 invariant #2/#3 style:
        // never trust an on-disk length blindly).
        const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(
                FormatError::Safetensors(SafetensorsError::HeaderLengthInvalid(header_len)).into(),
            );
        }
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact(&mut header_buf)?;
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)
            .map_err(|_| FormatError::Safetensors(SafetensorsError::InvalidHeader))?;

        let mut entries = HashMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let entry: RawEntry = serde_json::from_value(value)
                .map_err(|_| FormatError::Safetensors(SafetensorsError::InvalidHeader))?;
            entries.insert(
                name,
                SafetensorsEntry {
                    dtype: entry.dtype,
                    shape: entry.shape,
                    start: entry.data_offsets[0],
                    end: entry.data_offsets[1],
                },
            );
        }

        let data_base = 8 + header_len;
        Ok(Self {
            file,
            data_base,
            entries,
        })
    }

    pub fn entry(&self, name: &str) -> Option<&SafetensorsEntry> {
        self.entries.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Reads one tensor's raw little-endian F32 bytes and decodes them.
    /// Rejects any dtype other than `F32` rather than silently
    /// reinterpreting bytes (spec §115 invariant #2: readers reject
    /// unsupported encodings rather than guessing).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let entry = self.entries.get(name).ok_or_else(|| {
            FormatError::Safetensors(SafetensorsError::TensorNotFound(name.to_string()))
        })?;
        if entry.dtype != "F32" {
            return Err(
                FormatError::Safetensors(SafetensorsError::UnsupportedDtype {
                    name: name.to_string(),
                    dtype: entry.dtype.clone(),
                })
                .into(),
            );
        }
        let byte_len = (entry.end - entry.start) as usize;
        let mut buf = vec![0u8; byte_len];
        self.file
            .read_exact_at(&mut buf, self.data_base + entry.start)?;
        let mut out = Vec::with_capacity(byte_len / 4);
        for chunk in buf.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")));
        }
        Ok(out)
    }
}
