//! Trivial `ModelSource` over an already-local file — the backend behind
//! `tqf --model ./compatible-qwen36-q4.gguf` (spec §3/§29 experimental
//! import path). No revision concept, no journal, no download: the file is
//! read (and, via `fetch_verified`'s `local_path` short-circuit, hashed) in
//! place and never copied — TQF must never delete a user-pointed source
//! (spec §29/§127).

use std::path::{Path, PathBuf};

use crate::error::{Result, SourceError};
use crate::source::{ModelSource, SourceMetadata};

pub struct LocalFileSource {
    path: PathBuf,
    metadata: SourceMetadata,
}

impl LocalFileSource {
    /// Hash-agnostic for Phase 4: measures and records the file's hash but
    /// doesn't compare it against anything (no `--model`-supplied expected
    /// hash exists yet).
    pub fn open(path: PathBuf) -> Result<Self> {
        let file_meta = std::fs::metadata(&path).map_err(SourceError::LocalSourceUnavailable)?;
        let artifact_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        Ok(Self {
            metadata: SourceMetadata {
                artifact_name,
                size_bytes: Some(file_meta.len()),
                revision: None,
                expected_sha256: None,
                source_id: "local".to_string(),
            },
            path,
        })
    }
}

#[async_trait::async_trait]
impl ModelSource for LocalFileSource {
    fn metadata(&self) -> &SourceMetadata {
        &self.metadata
    }

    async fn read_range(
        &self,
        offset: u64,
        len: u64,
    ) -> std::result::Result<bytes::Bytes, SourceError> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || -> std::result::Result<bytes::Bytes, SourceError> {
            use std::os::unix::fs::FileExt;
            let file = std::fs::File::open(&path).map_err(SourceError::LocalSourceUnavailable)?;
            let mut buf = vec![0u8; len as usize];
            file.read_exact_at(&mut buf, offset)
                .map_err(SourceError::LocalSourceUnavailable)?;
            Ok(bytes::Bytes::from(buf))
        })
        .await
        .map_err(|join_err| {
            SourceError::LocalSourceUnavailable(std::io::Error::other(join_err.to_string()))
        })?
    }

    fn local_path(&self) -> Option<&Path> {
        Some(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp(name: &str, contents: &[u8]) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tqf-local-source-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[tokio::test]
    async fn reads_exact_slice() {
        let path = write_temp("slice.bin", b"0123456789");
        let source = LocalFileSource::open(path).unwrap();

        let bytes = source.read_range(2, 4).await.unwrap();
        assert_eq!(&bytes[..], b"2345");
    }

    #[tokio::test]
    async fn out_of_bounds_range_errors() {
        let path = write_temp("short.bin", b"abc");
        let source = LocalFileSource::open(path).unwrap();

        assert!(source.read_range(0, 100).await.is_err());
    }

    #[test]
    fn reports_local_path_and_size() {
        let path = write_temp("meta.bin", b"hello world");
        let source = LocalFileSource::open(path.clone()).unwrap();

        assert_eq!(source.local_path(), Some(path.as_path()));
        assert_eq!(source.metadata().size_bytes, Some(11));
    }
}
