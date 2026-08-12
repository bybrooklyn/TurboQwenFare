//! Streaming SHA-256, used both per-chunk during download and once over the
//! whole `.part` file at finalization (spec Part V section 13 publishes an
//! exact SHA-256 to match; Part XIV section 126 requires the final gate to
//! be a real hash comparison, not "bytes exist").

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;

pub fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Hashes a file in bounded memory (fixed-size read blocks), never loading
/// the whole file into RAM — required for multi-gigabyte model artifacts.
pub fn hex_digest_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Minimal hex encoding so this module doesn't need a `hex` crate dependency
/// for what is otherwise a one-line operation.
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        let mut out = String::with_capacity(bytes.as_ref().len() * 2);
        for b in bytes.as_ref() {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_digest_matches_known_vector() {
        // SHA-256("abc")
        assert_eq!(
            hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hex_digest_file_matches_in_memory_digest() {
        let dir = std::env::temp_dir().join(format!("tqf-checksum-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.bin");
        let data = vec![0x5au8; 3 * 1024 * 1024 + 17]; // spans multiple 1MB blocks
        std::fs::write(&path, &data).unwrap();

        assert_eq!(hex_digest_file(&path).unwrap(), hex_digest(&data));

        std::fs::remove_dir_all(&dir).ok();
    }
}
