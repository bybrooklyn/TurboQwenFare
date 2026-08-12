//! Generic exponential-backoff-with-jitter retry combinator (spec §276:
//! "Downloader has retry/backoff"; no numeric values are specified in the
//! spec, so the defaults here are an implementation choice, not derived).
//! Not download-specific — any fallible async op can use it.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 6,
            base_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(30),
        }
    }
}

/// Distinguishes errors worth retrying (network blip, 5xx, truncated
/// stream) from ones that won't be fixed by trying again (404, checksum
/// mismatch, a server that ignores range requests) — retrying the latter
/// would just waste time and mask a real problem.
pub enum RetryOutcome<E> {
    Retryable(E),
    Terminal(E),
}

/// Runs `op` until it succeeds, exhausts `policy.max_attempts`, or returns a
/// `Terminal` error (which is never retried). On exhaustion, returns the
/// last error observed.
pub async fn retry_with_backoff<T, E, F, Fut>(
    policy: &RetryPolicy,
    op_name: &str,
    mut op: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, RetryOutcome<E>>>,
{
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match op().await {
            Ok(value) => return Ok(value),
            Err(RetryOutcome::Terminal(err)) => return Err(err),
            Err(RetryOutcome::Retryable(err)) => {
                if attempt >= policy.max_attempts {
                    return Err(err);
                }
                let delay = jittered_delay(policy.base_delay, policy.max_delay, attempt);
                tracing::warn!(op_name, attempt, ?delay, "retrying after transient error");
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// `min(max_delay, base_delay * 2^(attempt-1))`, scaled by a `[0, 1)`
/// pseudo-random factor ("full jitter") to avoid synchronized retry storms.
/// The randomness only needs to be non-degenerate across attempts, not
/// cryptographically strong, so this avoids pulling in a `rand` dependency.
fn jittered_delay(base_delay: Duration, max_delay: Duration, attempt: u32) -> Duration {
    let shift = attempt.saturating_sub(1).min(31);
    let exponential = base_delay.saturating_mul(1u32 << shift);
    let capped = exponential.min(max_delay);
    capped.mul_f64(pseudo_random_unit(attempt))
}

fn pseudo_random_unit(seed_extra: u32) -> f64 {
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u32(seed_extra);
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    (hasher.finish() as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test(start_paused = true)]
    async fn retries_retryable_errors_until_success() {
        let calls = AtomicU32::new(0);
        let policy = RetryPolicy::default();

        let result: Result<&str, &str> = retry_with_backoff(&policy, "test-op", || {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 2 {
                    Err(RetryOutcome::Retryable("transient"))
                } else {
                    Ok("done")
                }
            }
        })
        .await;

        assert_eq!(result, Ok("done"));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn terminal_errors_short_circuit_with_zero_retries() {
        let calls = AtomicU32::new(0);
        let policy = RetryPolicy::default();

        let result: Result<&str, &str> = retry_with_backoff(&policy, "test-op", || {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Err(RetryOutcome::Terminal("fatal")) }
        })
        .await;

        assert_eq!(result, Err("fatal"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn exhausts_max_attempts_and_returns_last_error() {
        let calls = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_attempts: 3,
            ..RetryPolicy::default()
        };

        let result: Result<&str, u32> = retry_with_backoff(&policy, "test-op", || {
            let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
            async move { Err(RetryOutcome::Retryable(n)) }
        })
        .await;

        assert_eq!(result, Err(3));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}
