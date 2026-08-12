//! Pinned Hugging Face HTTP range source (spec §276, §13, §29 pinned-source
//! rule: revision is always a pinned commit hash, never a moving "main").
//! Sequential, single-connection only — parallel multi-range fetching is
//! later-phase territory (spec §112 row 19).

use std::path::Path;
use std::time::Duration;

use reqwest::header::{CONTENT_RANGE, ETAG, RANGE};
use reqwest::{Client, StatusCode};

use crate::error::SourceError;
use crate::source::{ModelSource, SourceMetadata};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const HF_BASE_URL: &str = "https://huggingface.co";

pub struct HfRangeSource {
    client: Client,
    url: String,
    metadata: SourceMetadata,
}

impl HfRangeSource {
    /// Probes the source with a `Range: bytes=0-0` request (more reliable
    /// against a CDN-fronted resolve URL than `HEAD`): captures total size
    /// from `Content-Range` and the `ETag` as this download's revision
    /// fingerprint. A `200` response instead of `206` means the server
    /// ignored the range header — fails immediately rather than silently
    /// falling back to a non-resumable full download.
    pub async fn resolve(
        repo_id: &str,
        revision: &str,
        filename: &str,
        expected_sha256: Option<String>,
    ) -> std::result::Result<Self, SourceError> {
        Self::resolve_with_base_url(HF_BASE_URL, repo_id, revision, filename, expected_sha256).await
    }

    /// Test seam: points at an arbitrary base URL instead of the real HF
    /// host, so the resumable-download/retry/verification logic can be
    /// exercised against a real local HTTP server rather than a mock
    /// (matches this crate's existing "real bind, real HTTP, no mocks"
    /// test precedent in `src/server/tests.rs`).
    pub(crate) async fn resolve_with_base_url(
        base_url: &str,
        repo_id: &str,
        revision: &str,
        filename: &str,
        expected_sha256: Option<String>,
    ) -> std::result::Result<Self, SourceError> {
        let client = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(SourceError::Network)?;
        let url = format!("{base_url}/{repo_id}/resolve/{revision}/{filename}");

        let response = client
            .get(&url)
            .header(RANGE, "bytes=0-0")
            .send()
            .await
            .map_err(SourceError::Network)?;

        let status = response.status();
        if status == StatusCode::OK {
            // Server accepted the request but ignored Range entirely.
            return Err(SourceError::RangeNotSupported { url });
        }
        if status != StatusCode::PARTIAL_CONTENT {
            return Err(SourceError::HttpStatus {
                status: status.as_u16(),
                url,
            });
        }

        let total_size = parse_content_range_total(response.headers()).ok_or_else(|| {
            SourceError::HttpStatus {
                status: status.as_u16(),
                url: url.clone(),
            }
        })?;
        let etag = response
            .headers()
            .get(ETAG)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        Ok(Self {
            client,
            url,
            metadata: SourceMetadata {
                artifact_name: filename.to_string(),
                size_bytes: Some(total_size),
                revision: etag,
                expected_sha256,
                source_id: repo_id.to_string(),
            },
        })
    }
}

#[async_trait::async_trait]
impl ModelSource for HfRangeSource {
    fn metadata(&self) -> &SourceMetadata {
        &self.metadata
    }

    async fn read_range(
        &self,
        offset: u64,
        len: u64,
    ) -> std::result::Result<bytes::Bytes, SourceError> {
        let range_header = format!("bytes={offset}-{}", offset + len - 1);
        let response = self
            .client
            .get(&self.url)
            .header(RANGE, range_header)
            .send()
            .await
            .map_err(SourceError::Network)?;

        let status = response.status();
        if status == StatusCode::OK {
            // Server accepted the request but ignored Range entirely.
            return Err(SourceError::RangeNotSupported {
                url: self.url.clone(),
            });
        }
        if status != StatusCode::PARTIAL_CONTENT {
            return Err(SourceError::HttpStatus {
                status: status.as_u16(),
                url: self.url.clone(),
            });
        }

        // Defense-in-depth, fail-fast signal only: the real correctness
        // gate is the whole-file SHA-256 check in `fetch_verified`. CDN
        // ETags aren't always present on every response, so only compare
        // when one is actually returned.
        if let Some(expected_etag) = &self.metadata.revision {
            if let Some(actual_etag) = response.headers().get(ETAG).and_then(|v| v.to_str().ok()) {
                if actual_etag != expected_etag {
                    return Err(SourceError::RevisionChanged {
                        artifact: self.metadata.artifact_name.clone(),
                        expected: expected_etag.clone(),
                        actual: actual_etag.to_string(),
                    });
                }
            }
        }

        response.bytes().await.map_err(SourceError::Network)
    }

    fn local_path(&self) -> Option<&Path> {
        None
    }
}

fn parse_content_range_total(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    // Format: "bytes 0-0/12345678".
    let value = headers.get(CONTENT_RANGE)?.to_str().ok()?;
    value.rsplit('/').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_total_size_from_content_range() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_RANGE, "bytes 0-0/20400000000".parse().unwrap());
        assert_eq!(parse_content_range_total(&headers), Some(20_400_000_000));
    }

    #[test]
    fn missing_content_range_is_none() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_content_range_total(&headers), None);
    }
}
