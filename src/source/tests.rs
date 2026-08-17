//! Integration tests for the source resolver/downloader: a real local HTTP
//! server (axum, already a direct dependency) driven by the real `reqwest`
//! client — no mocks, matching the "real bind, real HTTP" precedent in
//! `src/server/tests.rs`. `HfRangeSource::resolve_with_base_url` is the
//! test seam that points the client at this local server instead of the
//! real Hugging Face host.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;

use crate::error::{SourceError, TqfError};
use crate::ids::Bytes;
use crate::memory::MemoryBroker;
use crate::source::hf::HfRangeSource;
use crate::source::local::LocalFileSource;
use crate::source::manifest::FetchedArtifact;
use crate::source::retry::RetryPolicy;
use crate::source::{
    checksum, fetch_verified as fetch_verified_with_broker, FetchOptions, ModelSource,
    SourceOwnership,
};

const TEST_REPO: &str = "repo";
const TEST_REVISION: &str = "rev";
const TEST_FILENAME: &str = "model.bin";

struct TestServer {
    body: Vec<u8>,
    etag: String,
    request_log: Mutex<Vec<(u64, u64)>>,
    ignore_range: AtomicBool,
    fail_from_offset: AtomicU64,
    corrupt_at_offset: AtomicU64,
    etag_change_after_offset: AtomicU64,
    flaky_remaining: AtomicU32,
    not_found_chunks: AtomicBool,
}

impl TestServer {
    fn new(body: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            body,
            etag: "\"original-etag\"".to_string(),
            request_log: Mutex::new(Vec::new()),
            ignore_range: AtomicBool::new(false),
            fail_from_offset: AtomicU64::new(u64::MAX),
            corrupt_at_offset: AtomicU64::new(u64::MAX),
            etag_change_after_offset: AtomicU64::new(u64::MAX),
            flaky_remaining: AtomicU32::new(0),
            not_found_chunks: AtomicBool::new(false),
        })
    }

    fn request_count(&self) -> usize {
        self.request_log.lock().unwrap().len()
    }
}

fn parse_range(value: &str, body_len: u64) -> Option<(u64, u64)> {
    let spec = value.strip_prefix("bytes=")?;
    let (start, end) = spec.split_once('-')?;
    let start: u64 = start.parse().ok()?;
    let end: u64 = if end.is_empty() {
        body_len.saturating_sub(1)
    } else {
        end.parse().ok()?
    };
    Some((start, end))
}

async fn handler(State(state): State<Arc<TestServer>>, headers: HeaderMap) -> Response {
    if state.ignore_range.load(Ordering::SeqCst) {
        state
            .request_log
            .lock()
            .unwrap()
            .push((0, state.body.len() as u64));
        return (StatusCode::OK, state.body.clone()).into_response();
    }

    let Some(range_value) = headers
        .get(axum::http::header::RANGE)
        .and_then(|v| v.to_str().ok())
    else {
        return (StatusCode::BAD_REQUEST, "range required").into_response();
    };
    let Some((start, end)) = parse_range(range_value, state.body.len() as u64) else {
        return (StatusCode::RANGE_NOT_SATISFIABLE, "bad range").into_response();
    };
    let len = end - start + 1;
    state.request_log.lock().unwrap().push((start, len));

    // The initial resolve() probe always requests exactly one byte
    // (`bytes=0-0`); real chunk fetches ask for more. Behaviors that
    // should only affect chunk fetches (not the metadata probe) key off
    // this distinction.
    let is_probe = len == 1 && start == 0;

    if !is_probe && state.not_found_chunks.load(Ordering::SeqCst) {
        return (StatusCode::NOT_FOUND, "chunk missing").into_response();
    }
    if start >= state.fail_from_offset.load(Ordering::SeqCst) {
        return (StatusCode::SERVICE_UNAVAILABLE, "try again").into_response();
    }
    if !is_probe && state.flaky_remaining.load(Ordering::SeqCst) > 0 {
        state.flaky_remaining.fetch_sub(1, Ordering::SeqCst);
        return (StatusCode::SERVICE_UNAVAILABLE, "try again").into_response();
    }

    let etag = if start >= state.etag_change_after_offset.load(Ordering::SeqCst) {
        "\"changed-etag\"".to_string()
    } else {
        state.etag.clone()
    };

    let mut body_slice = state.body[start as usize..=(end as usize)].to_vec();
    if start == state.corrupt_at_offset.load(Ordering::SeqCst) {
        body_slice[0] ^= 0xFF;
    }

    let mut resp_headers = HeaderMap::new();
    resp_headers.insert(
        axum::http::header::ETAG,
        HeaderValue::from_str(&etag).unwrap(),
    );
    resp_headers.insert(
        axum::http::header::CONTENT_RANGE,
        HeaderValue::from_str(&format!("bytes {start}-{end}/{}", state.body.len())).unwrap(),
    );

    (StatusCode::PARTIAL_CONTENT, resp_headers, body_slice).into_response()
}

async fn spawn_test_server(server: Arc<TestServer>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let route = format!("/{TEST_REPO}/resolve/{TEST_REVISION}/{TEST_FILENAME}");
    let router = Router::new().route(&route, get(handler)).with_state(server);
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    tokio::task::yield_now().await;
    format!("http://{addr}")
}

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tqf-source-it-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fast_policy() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 3,
        base_delay: Duration::from_millis(2),
        max_delay: Duration::from_millis(10),
    }
}

fn test_body() -> Vec<u8> {
    (0..64u8).collect() // 64 bytes -> 4 chunks of 16 bytes
}

/// Keep the integration-test call sites focused on source semantics while
/// still exercising the production brokered API. The hash pass needs 1 MiB.
async fn fetch_verified(
    source: &dyn ModelSource,
    ownership: SourceOwnership,
    dest_dir: &std::path::Path,
    options: FetchOptions,
) -> crate::error::Result<FetchedArtifact> {
    let broker = MemoryBroker::new(Bytes(2 * 1024 * 1024));
    let result = fetch_verified_with_broker(source, ownership, dest_dir, options, &broker).await;
    assert_eq!(broker.snapshot().reserved, Bytes(0));
    result
}

#[tokio::test]
async fn happy_path_downloads_and_verifies() {
    let body = test_body();
    let expected_sha256 = checksum::hex_digest(&body);
    let server = TestServer::new(body.clone());
    let base_url = spawn_test_server(server).await;
    let dest_dir = test_dir("happy-path");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        Some(expected_sha256.clone()),
    )
    .await
    .unwrap();

    let artifact = fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        FetchOptions {
            chunk_size_bytes: 16,
            retry_policy: fast_policy(),
        },
    )
    .await
    .unwrap();

    assert_eq!(artifact.sha256, expected_sha256);
    let final_bytes = std::fs::read(dest_dir.join(TEST_FILENAME)).unwrap();
    assert_eq!(final_bytes, body);
    assert!(!dest_dir.join(format!("{TEST_FILENAME}.journal")).exists());
    assert!(FetchedArtifact::load(&dest_dir).unwrap().is_some());
}

#[tokio::test]
async fn chunk_budget_is_reserved_before_the_http_body_is_requested() {
    let body = test_body();
    let server = TestServer::new(body);
    let base_url = spawn_test_server(server.clone()).await;
    let dest_dir = test_dir("chunk-budget");
    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        None,
    )
    .await
    .unwrap();
    assert_eq!(server.request_count(), 1, "metadata probe only");

    let broker = MemoryBroker::new(Bytes(8));
    let result = fetch_verified_with_broker(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        FetchOptions {
            chunk_size_bytes: 16,
            retry_policy: fast_policy(),
        },
        &broker,
    )
    .await;

    assert!(matches!(
        result,
        Err(TqfError::Memory(
            crate::error::MemoryError::BudgetExceeded {
                requested: 16,
                available: 8,
                ..
            }
        ))
    ));
    assert_eq!(
        server.request_count(),
        1,
        "no chunk request before admission"
    );
    assert_eq!(broker.snapshot().reserved, Bytes(0));
}

#[tokio::test]
async fn interrupted_transfer_resumes_without_refetching_verified_chunks() {
    let body = test_body();
    let expected_sha256 = checksum::hex_digest(&body);
    let server = TestServer::new(body.clone());
    server.fail_from_offset.store(32, Ordering::SeqCst); // chunks at offset 32/48 fail
    let base_url = spawn_test_server(server.clone()).await;
    let dest_dir = test_dir("resume");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        Some(expected_sha256.clone()),
    )
    .await
    .unwrap();

    let options = FetchOptions {
        chunk_size_bytes: 16,
        retry_policy: RetryPolicy {
            max_attempts: 2,
            base_delay: Duration::from_millis(2),
            max_delay: Duration::from_millis(5),
        },
    };

    let first_attempt = fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        options.clone(),
    )
    .await;
    assert!(first_attempt.is_err(), "expected the first attempt to fail");
    let requests_after_first_attempt = server.request_count();

    server.fail_from_offset.store(u64::MAX, Ordering::SeqCst); // "connection recovers"

    let second_attempt =
        fetch_verified(&source, SourceOwnership::TqfManaged, &dest_dir, options).await;
    let artifact = second_attempt.expect("resumed fetch should succeed");
    assert_eq!(artifact.sha256, expected_sha256);
    assert_eq!(std::fs::read(dest_dir.join(TEST_FILENAME)).unwrap(), body);

    // Chunks already verified before the failure (offsets 0 and 16) must
    // never be re-requested during the resumed attempt.
    let log = server.request_log.lock().unwrap();
    let re_requested_verified_chunks = log[requests_after_first_attempt..]
        .iter()
        .any(|&(start, _)| start == 0 || start == 16);
    assert!(
        !re_requested_verified_chunks,
        "resume re-fetched an already-verified chunk: {:?}",
        &log[requests_after_first_attempt..]
    );
}

#[tokio::test]
async fn corrupt_chunk_content_is_rejected_by_whole_file_hash() {
    let body = test_body();
    let expected_sha256 = checksum::hex_digest(&body);
    let server = TestServer::new(body.clone());
    server.corrupt_at_offset.store(16, Ordering::SeqCst);
    let base_url = spawn_test_server(server).await;
    let dest_dir = test_dir("corrupt");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        Some(expected_sha256),
    )
    .await
    .unwrap();

    let result = fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        FetchOptions {
            chunk_size_bytes: 16,
            retry_policy: fast_policy(),
        },
    )
    .await;

    match result {
        Err(TqfError::Source(SourceError::ChecksumMismatch { .. })) => {}
        other => panic!("expected ChecksumMismatch, got {other:?}"),
    }
    assert!(!dest_dir.join(TEST_FILENAME).exists());
    assert!(dest_dir.join(format!("{TEST_FILENAME}.part")).exists());
    assert!(dest_dir.join(format!("{TEST_FILENAME}.journal")).exists());
}

#[tokio::test]
async fn server_ignoring_range_header_is_rejected() {
    let body = test_body();
    let server = TestServer::new(body);
    server.ignore_range.store(true, Ordering::SeqCst);
    let base_url = spawn_test_server(server).await;

    let result = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        None,
    )
    .await;

    assert!(matches!(result, Err(SourceError::RangeNotSupported { .. })));
}

#[tokio::test]
async fn etag_change_mid_download_is_rejected() {
    let body = test_body();
    let expected_sha256 = checksum::hex_digest(&body);
    let server = TestServer::new(body);
    server.etag_change_after_offset.store(16, Ordering::SeqCst);
    let base_url = spawn_test_server(server).await;
    let dest_dir = test_dir("etag-change");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        Some(expected_sha256),
    )
    .await
    .unwrap();

    let result = fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        FetchOptions {
            chunk_size_bytes: 16,
            retry_policy: fast_policy(),
        },
    )
    .await;

    match result {
        Err(TqfError::Source(SourceError::RevisionChanged { .. })) => {}
        other => panic!("expected RevisionChanged, got {other:?}"),
    }
    assert!(!dest_dir.join(TEST_FILENAME).exists());
}

#[tokio::test]
async fn retries_transient_5xx_then_succeeds() {
    let body: Vec<u8> = (0..16u8).collect(); // one chunk exactly
    let expected_sha256 = checksum::hex_digest(&body);
    let server = TestServer::new(body.clone());
    server.flaky_remaining.store(2, Ordering::SeqCst); // fail twice, then succeed
    let base_url = spawn_test_server(server.clone()).await;
    let dest_dir = test_dir("flaky-retry");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        Some(expected_sha256.clone()),
    )
    .await
    .unwrap();

    let artifact = fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        FetchOptions {
            chunk_size_bytes: 16,
            retry_policy: fast_policy(),
        },
    )
    .await
    .unwrap();

    assert_eq!(artifact.sha256, expected_sha256);
    // 1 resolve probe + 2 failed chunk attempts + 1 successful chunk attempt.
    assert_eq!(server.request_count(), 4);
}

#[tokio::test]
async fn not_found_chunk_fails_without_retry() {
    let body = test_body();
    let server = TestServer::new(body);
    let base_url = spawn_test_server(server.clone()).await;
    let dest_dir = test_dir("not-found");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        None,
    )
    .await
    .unwrap();

    server.not_found_chunks.store(true, Ordering::SeqCst);

    let result = fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        FetchOptions {
            chunk_size_bytes: 16,
            retry_policy: RetryPolicy {
                max_attempts: 5,
                base_delay: Duration::from_millis(2),
                max_delay: Duration::from_millis(5),
            },
        },
    )
    .await;

    assert!(matches!(
        result,
        Err(TqfError::Source(SourceError::HttpStatus {
            status: 404,
            ..
        }))
    ));
    // 1 resolve probe + exactly 1 chunk attempt: a 404 is terminal, never
    // retried, regardless of the 5-attempt policy above.
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn idempotent_rerun_does_not_refetch_a_finalized_artifact() {
    let body = test_body();
    let expected_sha256 = checksum::hex_digest(&body);
    let server = TestServer::new(body.clone());
    let base_url = spawn_test_server(server.clone()).await;
    let dest_dir = test_dir("idempotent");

    let source = HfRangeSource::resolve_with_base_url(
        &base_url,
        TEST_REPO,
        TEST_REVISION,
        TEST_FILENAME,
        Some(expected_sha256.clone()),
    )
    .await
    .unwrap();

    let options = FetchOptions {
        chunk_size_bytes: 16,
        retry_policy: fast_policy(),
    };
    fetch_verified(
        &source,
        SourceOwnership::TqfManaged,
        &dest_dir,
        options.clone(),
    )
    .await
    .unwrap();
    let requests_after_first_fetch = server.request_count();

    let artifact = fetch_verified(&source, SourceOwnership::TqfManaged, &dest_dir, options)
        .await
        .unwrap();

    assert_eq!(artifact.sha256, expected_sha256);
    assert_eq!(
        server.request_count(),
        requests_after_first_fetch,
        "idempotent re-run should not issue any new requests"
    );
}

#[tokio::test]
async fn local_file_source_records_user_owned_and_never_copies() {
    let source_dir = test_dir("local-source-file");
    let source_path = source_dir.join("imported.gguf");
    std::fs::write(&source_path, b"local model bytes").unwrap();

    let source = LocalFileSource::open(source_path.clone()).unwrap();
    let dest_dir = test_dir("local-fetch-dest");

    let artifact = fetch_verified(
        &source,
        SourceOwnership::UserOwned,
        &dest_dir,
        FetchOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(artifact.ownership, SourceOwnership::UserOwned);
    assert_eq!(artifact.local_path, source_path.display().to_string());
    assert!(
        !dest_dir.join(&artifact.artifact_name).exists(),
        "local source must never be copied into dest_dir"
    );
    assert!(FetchedArtifact::load(&dest_dir).unwrap().is_some());
}
