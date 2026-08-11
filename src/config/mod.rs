//! Process-level config (spec Part IX, sections 76-77). Persistent `~/.tqf`
//! state and hardware-profile detection land in phase 3; for now this only
//! turns CLI flags into a validated in-memory `Config`.

use crate::error::ConfigError;

#[derive(Debug, Default, Clone)]
pub struct Config {
    pub memory_budget_bytes: Option<u64>,
    pub context_limit_tokens: Option<u64>,
    pub enable_vision: bool,
    pub host: Option<String>,
}

/// Parses a size like "4G", "512M", "128K", or a bare integer, into a count
/// (bytes for `--memory`, tokens for `--context`). Suffixes are
/// case-insensitive base-1024 multipliers.
pub fn parse_human_quantity(input: &str) -> Result<u64, ConfigError> {
    let trimmed = input.trim();
    let Some(last) = trimmed.chars().last() else {
        return Err(ConfigError::InvalidSize(input.to_string()));
    };

    let (digits, multiplier) = if last.eq_ignore_ascii_case(&'g') {
        (&trimmed[..trimmed.len() - 1], 1024u64.pow(3))
    } else if last.eq_ignore_ascii_case(&'m') {
        (&trimmed[..trimmed.len() - 1], 1024u64.pow(2))
    } else if last.eq_ignore_ascii_case(&'k') {
        (&trimmed[..trimmed.len() - 1], 1024u64)
    } else {
        (trimmed, 1u64)
    };

    let value: u64 = digits
        .trim()
        .parse()
        .map_err(|_| ConfigError::InvalidSize(input.to_string()))?;

    value
        .checked_mul(multiplier)
        .ok_or_else(|| ConfigError::InvalidSize(input.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_suffixed_sizes() {
        assert_eq!(parse_human_quantity("4G").unwrap(), 4 * 1024 * 1024 * 1024);
        assert_eq!(parse_human_quantity("128k").unwrap(), 128 * 1024);
        assert_eq!(parse_human_quantity("1M").unwrap(), 1024 * 1024);
        assert_eq!(parse_human_quantity("512").unwrap(), 512);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_human_quantity("").is_err());
        assert!(parse_human_quantity("4GB").is_err());
        assert!(parse_human_quantity("many").is_err());
    }
}
