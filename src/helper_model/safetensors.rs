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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tqf-safetensors-test-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Audit finding H-02 (2026-08-20). `open` validates the header
    /// length but never the extent directory: offsets are not checked for
    /// ordering, file bounds, or agreement with `shape * dtype size`.
    /// `read_f32` then computes `entry.end - entry.start` unchecked,
    /// which panics in debug and wraps to a ~16 EiB allocation request in
    /// release. This is a model-file trust boundary (spec §246/§268:
    /// hostile input must produce a clean typed error).
    ///
    /// Asserts a typed error, not a panic. Currently fails.
    #[test]
    fn reversed_data_offsets_are_a_typed_error() {
        let dir = scratch("reversed-offsets");
        let path = dir.join("reversed.safetensors");

        // end < start: no real safetensors file can say this.
        let header = br#"{"weight":{"dtype":"F32","shape":[4],"data_offsets":[64,0]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header);
        file.extend_from_slice(&[0u8; 64]);
        std::fs::write(&path, &file).unwrap();

        let outcome = std::panic::catch_unwind(|| {
            let opened = SafetensorsFile::open(&path)?;
            opened.read_f32("weight")
        });

        match outcome {
            Ok(Err(_)) => { /* correct: rejected with a typed error */ }
            Ok(Ok(values)) => panic!(
                "a reversed extent must not decode; got {} values",
                values.len()
            ),
            Err(_) => panic!("a reversed extent must not panic the parser"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Audit finding H-02, second half: the declared shape and the byte
    /// range must agree. `[4]` F32 values are 16 bytes; this file claims
    /// 64. `read_f32` currently returns 16 values for a 4-element tensor.
    ///
    /// Currently fails.
    #[test]
    fn shape_must_agree_with_the_byte_range() {
        let dir = scratch("shape-disagreement");
        let path = dir.join("mismatched.safetensors");

        let header = br#"{"weight":{"dtype":"F32","shape":[4],"data_offsets":[0,64]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header);
        file.extend_from_slice(&[0u8; 64]);
        std::fs::write(&path, &file).unwrap();

        let opened = SafetensorsFile::open(&path);
        match opened.and_then(|file| file.read_f32("weight")) {
            Ok(values) => panic!(
                "shape [4] declares 16 bytes, the extent claims 64; got {} values",
                values.len()
            ),
            Err(_) => { /* correct */ }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
