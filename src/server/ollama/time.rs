//! RFC 3339 UTC timestamps for Ollama's `created_at` / `modified_at`.
//!
//! Hand-rolled rather than adding `chrono` or `time`: the crate keeps its
//! dependency surface deliberately small (spec §114), and this is one
//! well-understood calendar conversion with a published algorithm and a
//! test against known epoch values. Adding a date-time crate for a single
//! output format would be the larger change.

use std::time::{SystemTime, UNIX_EPOCH};

/// The current time as `YYYY-MM-DDTHH:MM:SS.fffffffffZ`, the shape real
/// Ollama emits (nanosecond precision, always UTC, always `Z`).
pub fn rfc3339_utc_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339_utc(now.as_secs(), now.subsec_nanos())
}

/// RFC 3339 for a specific instant, so the formatting is testable without
/// depending on wall-clock time.
pub fn rfc3339_utc(epoch_seconds: u64, nanos: u32) -> String {
    let days = (epoch_seconds / 86_400) as i64;
    let seconds_of_day = epoch_seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{nanos:09}Z",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60,
    )
}

/// Days since the Unix epoch to a civil (proleptic Gregorian) date, via
/// Howard Hinnant's `civil_from_days` — integer-only, no lookup tables,
/// and correct across leap years and century rules.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the epoch to 0000-03-01 so leap days land at the end of the
    // 400-year era, which is what removes every special case below.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64; // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153; // March-based month, [0, 11]
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known epoch values, including a leap day and a year boundary —
    /// the two places a hand-rolled calendar conversion goes wrong.
    #[test]
    fn formats_known_epoch_instants() {
        for (epoch, nanos, expected) in [
            (0u64, 0u32, "1970-01-01T00:00:00.000000000Z"),
            (1_000_000_000, 0, "2001-09-09T01:46:40.000000000Z"),
            // 2024-02-29: a leap day in a leap century-rule year.
            (1_709_164_800, 0, "2024-02-29T00:00:00.000000000Z"),
            // 1999-12-31T23:59:59, the last second before a year rollover.
            (946_684_799, 999_999_999, "1999-12-31T23:59:59.999999999Z"),
            (946_684_800, 0, "2000-01-01T00:00:00.000000000Z"),
            // 1900 is NOT a leap year (the century rule); 2100 is not either.
            (4_107_542_400, 0, "2100-03-01T00:00:00.000000000Z"),
        ] {
            assert_eq!(rfc3339_utc(epoch, nanos), expected, "epoch {epoch}");
        }
    }

    #[test]
    fn the_current_time_is_well_formed_and_plausible() {
        let now = rfc3339_utc_now();
        assert!(now.ends_with('Z'), "{now}");
        assert_eq!(now.len(), "1970-01-01T00:00:00.000000000Z".len(), "{now}");
        let year: i32 = now[..4].parse().expect("a numeric year");
        assert!((2024..2200).contains(&year), "implausible year in {now}");
    }
}
