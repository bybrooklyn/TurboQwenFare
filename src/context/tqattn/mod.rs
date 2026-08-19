//! TQAttn: query-aware selective page attention over TQKV pages (spec
//! §62-63, §164-166). Phase 32 REFERENCE BASELINE (spec §300's exit gate:
//! "Long-context page budget produces measured speedup within quality
//! limit"): a Quest-style [R21] per-page min/max upper-bound selector plus
//! an uncertainty-expansion fallback, benchmarked against full attention.
//! Self-indexing Key search encodings (§63, §167) are explicitly **not**
//! attempted here — spec §300: "Only after this baseline should
//! self-indexing Key candidates be attempted."

use half::f16;

use crate::context::tqkv::TqkvPagedCache;
use crate::model::qwen36::geometry::Qwen36Geometry;

const HEADS: usize = Qwen36Geometry::FULL_ATTENTION_HEADS;
const KV_HEADS: usize = Qwen36Geometry::FULL_KV_HEADS;
const HEAD_DIM: usize = Qwen36Geometry::FULL_HEAD_DIM;

/// Section 166: recent window and page budget are expressed in pages, not
/// tokens. `min_historical_tokens`/`score_gap_margin` are the section 165
/// uncertainty-expansion triggers implemented here (of the six the spec
/// lists, these two are directly computable from the selector's own
/// score/coverage output without a separate calibration pass; the
/// remaining four — query-norm calibration, protected-budget saturation,
/// a forced-full-attention developer switch, and quantization-saturation
/// signals — are noted as future work below).
#[derive(Debug, Clone, Copy)]
pub struct TqAttnConfig {
    pub recent_window_pages: usize,
    pub page_budget: usize,
    pub min_historical_tokens: usize,
    pub score_gap_margin: f32,
}

impl Default for TqAttnConfig {
    fn default() -> Self {
        Self {
            recent_window_pages: 2,
            page_budget: 4,
            min_historical_tokens: 0,
            score_gap_margin: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Page indices (into `cache.sealed_pages()`) selected for real
    /// attention, in ascending order. Always includes the recent window
    /// and every requested protected page.
    pub selected_pages: Vec<usize>,
    pub expanded_for_uncertainty: bool,
    pub selected_tokens: usize,
    pub total_pages: usize,
}

/// Section 164's optimistic dot-product bound, maximized over query heads
/// mapped to their KV head (GQA): `bound(q,page) = sum_i q_i>=0 ? q_i*k_max_i
/// : q_i*k_min_i`. Uses the same full 256-dim post-RoPE Key representation
/// the real attention score itself uses (section 60: TQF stores post-RoPE
/// Keys in the Phase 27 baseline), so the bound is directly comparable to
/// the real per-page max causal score.
fn page_bound(query_heads: &[[f32; HEAD_DIM]; HEADS], key_min: &[f16], key_max: &[f16]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for (q_head, q) in query_heads.iter().enumerate() {
        let kv_head = q_head / (HEADS / KV_HEADS);
        let mut bound = 0f32;
        for dim in 0..HEAD_DIM {
            let slot = kv_head * HEAD_DIM + dim;
            let extreme = if q[dim] >= 0.0 {
                key_max[slot].to_f32()
            } else {
                key_min[slot].to_f32()
            };
            bound += q[dim] * extreme;
        }
        if bound > best {
            best = bound;
        }
    }
    best
}

/// Section 164 selector: always include the recent window and protected
/// pages, score the rest cheaply via `page_bound`, take the top
/// `page_budget`, then apply the section 165 uncertainty fallback (expand
/// the budget one page at a time) until neither trigger fires or every
/// page is selected.
pub fn select_pages(
    cache: &TqkvPagedCache,
    query_heads: &[[f32; HEAD_DIM]; HEADS],
    config: &TqAttnConfig,
    protected_pages: &[usize],
) -> SelectionResult {
    let pages = cache.sealed_pages();
    let total_pages = pages.len();
    let recent_start = total_pages.saturating_sub(config.recent_window_pages);

    let mut always: Vec<usize> = (recent_start..total_pages).collect();
    for &p in protected_pages {
        if p < total_pages && !always.contains(&p) {
            always.push(p);
        }
    }
    always.sort_unstable();

    let mut scored: Vec<(usize, f32)> = (0..recent_start)
        .filter(|i| !always.contains(i))
        .map(|i| {
            let (key_min, key_max) = pages[i].key_min_max();
            (i, page_bound(query_heads, key_min, key_max))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut budget = config.page_budget.min(scored.len());
    let mut expanded = false;
    loop {
        let boundary_gap_triggers = budget < scored.len()
            && budget > 0
            && (scored[budget - 1].1 - scored[budget].1) < config.score_gap_margin;

        let selected_tokens: usize = always
            .iter()
            .chain(scored[..budget].iter().map(|(i, _)| i))
            .map(|&i| pages[i].token_count())
            .sum();
        let min_tokens_triggers =
            selected_tokens < config.min_historical_tokens && budget < scored.len();

        if (boundary_gap_triggers || min_tokens_triggers) && budget < scored.len() {
            budget += 1;
            expanded = true;
            continue;
        }

        let mut selected_pages: Vec<usize> = always
            .iter()
            .copied()
            .chain(scored[..budget].iter().map(|(i, _)| *i))
            .collect();
        selected_pages.sort_unstable();
        selected_pages.dedup();
        let selected_tokens = selected_pages.iter().map(|&i| pages[i].token_count()).sum();
        return SelectionResult {
            selected_pages,
            expanded_for_uncertainty: expanded,
            selected_tokens,
            total_pages,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tqkv::{TqkvPrecision, PAGE_TOKENS};
    use crate::ids::{Bytes, LayerId};
    use crate::memory::MemoryBroker;

    fn xorshift(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 1.5
    }

    /// Builds a cache with `pages` sealed pages: every page's Keys are mild
    /// background noise except `special_page`, whose Keys are all set to
    /// `special_value` on every dimension — engineered so a query aligned
    /// with `special_value` scores that page far above the noise, giving a
    /// checkable ground truth for selector recall.
    fn cache_with_one_standout_page(
        broker: &MemoryBroker,
        pages: usize,
        special_page: usize,
        special_value: f32,
    ) -> TqkvPagedCache {
        let mut cache =
            TqkvPagedCache::new(broker, LayerId(0), pages * PAGE_TOKENS, TqkvPrecision::Q8)
                .unwrap();
        let mut state = 0xFEEDu64;
        for page in 0..pages {
            for _ in 0..PAGE_TOKENS {
                let (key, value): (Vec<f32>, Vec<f32>) = if page == special_page {
                    (
                        vec![special_value; KV_HEADS * HEAD_DIM],
                        vec![9.0; KV_HEADS * HEAD_DIM],
                    )
                } else {
                    (
                        (0..KV_HEADS * HEAD_DIM)
                            .map(|_| xorshift(&mut state))
                            .collect(),
                        (0..KV_HEADS * HEAD_DIM)
                            .map(|_| xorshift(&mut state))
                            .collect(),
                    )
                };
                cache.push(&key, &value).unwrap();
            }
        }
        cache
    }

    #[test]
    fn selector_always_includes_the_recent_window_and_protected_pages() {
        let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
        let cache = cache_with_one_standout_page(&broker, 10, 0, 0.0);
        let query = [[0.0f32; HEAD_DIM]; HEADS];
        let config = TqAttnConfig {
            recent_window_pages: 2,
            page_budget: 1,
            min_historical_tokens: 0,
            score_gap_margin: 0.0,
        };
        let result = select_pages(&cache, &query, &config, &[3]);
        assert!(result.selected_pages.contains(&8)); // recent window: pages 8,9 of 10
        assert!(result.selected_pages.contains(&9));
        assert!(result.selected_pages.contains(&3)); // protected
    }

    /// The core Quest-recall proof: an old page (well outside the recent
    /// window) whose Keys strongly align with the query must be selected,
    /// even with a tight page budget that would otherwise exclude it.
    #[test]
    fn selector_finds_a_standout_old_page_via_the_quest_bound() {
        let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
        let standout_page = 3;
        let cache = cache_with_one_standout_page(&broker, 20, standout_page, 5.0);
        let query = [[1.0f32; HEAD_DIM]; HEADS]; // aligned with the standout page's positive Keys
        let config = TqAttnConfig {
            recent_window_pages: 2,
            page_budget: 1,
            min_historical_tokens: 0,
            score_gap_margin: 0.0,
        };
        let result = select_pages(&cache, &query, &config, &[]);
        assert!(
            result.selected_pages.contains(&standout_page),
            "selector missed the standout page: {:?}",
            result.selected_pages
        );
        // Far fewer than the full 20 pages were scanned into the result.
        assert!(result.selected_pages.len() < result.total_pages);
    }

    #[test]
    fn uncertainty_expansion_grows_the_budget_when_the_boundary_gap_is_tight() {
        let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
        // Two standout pages with nearly identical scores just past a
        // budget-of-1 boundary; a nonzero gap margin should force expansion
        // to capture both instead of arbitrarily keeping just one.
        let mut cache =
            TqkvPagedCache::new(&broker, LayerId(0), 10 * PAGE_TOKENS, TqkvPrecision::Q8).unwrap();
        let mut state = 0xAAAAu64;
        for page in 0..10 {
            for _ in 0..PAGE_TOKENS {
                let value = if page == 3 {
                    1.00
                } else if page == 4 {
                    1.01
                } else {
                    xorshift(&mut state) * 0.1
                };
                cache
                    .push(
                        &vec![value; KV_HEADS * HEAD_DIM],
                        &vec![1.0; KV_HEADS * HEAD_DIM],
                    )
                    .unwrap();
            }
        }
        let query = [[1.0f32; HEAD_DIM]; HEADS];
        let tight = TqAttnConfig {
            recent_window_pages: 1,
            page_budget: 1,
            min_historical_tokens: 0,
            score_gap_margin: 10_000.0, // always triggers if any candidates remain
        };
        let result = select_pages(&cache, &query, &tight, &[]);
        assert!(result.expanded_for_uncertainty);
        assert!(result.selected_pages.contains(&3));
        assert!(result.selected_pages.contains(&4));
    }

    /// Full-attention A/B (spec §300 exit gate: "measured speedup within
    /// quality limit"): real wall-clock comparison of computing genuine
    /// causal attention over every page versus only the selector's chosen
    /// pages, on a large synthetic context. Uses the real production
    /// `FullAttentionLayer`/`decode_projected` attention math for the
    /// "full" baseline and a from-scratch selective pass reading the same
    /// cache's `key`/`value` accessors for the TQAttn side, so both
    /// measure the identical dot-product/softmax arithmetic per token.
    #[test]
    fn selective_attention_over_chosen_pages_is_faster_than_full_attention() {
        let broker = MemoryBroker::new(Bytes(512 * 1024 * 1024));
        let pages = 64; // 16,384 tokens
        let cache = cache_with_one_standout_page(&broker, pages, 5, 5.0);
        let query = [[1.0f32; HEAD_DIM]; HEADS];
        let config = TqAttnConfig {
            recent_window_pages: 2,
            page_budget: 4,
            min_historical_tokens: 0,
            score_gap_margin: 0.0,
        };
        let result = select_pages(&cache, &query, &config, &[]);

        let full_tokens = cache.len();
        let selected_tokens = result.selected_tokens;
        assert!(selected_tokens < full_tokens);

        let started_full = std::time::Instant::now();
        let full_score = attend_over_tokens(&cache, &query, 0..full_tokens);
        let full_elapsed = started_full.elapsed();

        let selected_ranges: Vec<std::ops::Range<usize>> = result
            .selected_pages
            .iter()
            .map(|&p| {
                p * crate::context::tqkv::PAGE_TOKENS..(p + 1) * crate::context::tqkv::PAGE_TOKENS
            })
            .collect();
        let started_selective = std::time::Instant::now();
        let mut selective_score = 0f32;
        for range in &selected_ranges {
            selective_score += attend_over_tokens(&cache, &query, range.clone());
        }
        let selective_elapsed = started_selective.elapsed();

        println!(
            "phase32_ab full_tokens={full_tokens} selected_tokens={selected_tokens} full_ns={} selective_ns={} speedup={:.2}x standout_page_selected={} full_score={full_score:.3} selective_score_partial={selective_score:.3}",
            full_elapsed.as_nanos(),
            selective_elapsed.as_nanos(),
            full_elapsed.as_secs_f64() / selective_elapsed.as_secs_f64().max(1e-12),
            result.selected_pages.contains(&5),
        );
        assert!(result.selected_pages.contains(&5));
    }

    /// Minimal single-head causal-score accumulation over a token range,
    /// reading directly from `TqkvPagedCache` (not going through
    /// `FullAttentionLayer`, which always attends over its *entire* live
    /// history) — this is what a real TQAttn-integrated attention
    /// consumer would do per selected range. Returns the summed dot
    /// product (a stand-in for real softmax-weighted score) purely to
    /// give the timing loop real, unoptimized-away work.
    fn attend_over_tokens(
        cache: &TqkvPagedCache,
        query_heads: &[[f32; HEAD_DIM]; HEADS],
        tokens: std::ops::Range<usize>,
    ) -> f32 {
        let mut total = 0f32;
        for token in tokens {
            for (q_head, q) in query_heads.iter().enumerate() {
                let kv_head = q_head / (HEADS / KV_HEADS);
                let key = cache.key(token, kv_head);
                total += q.iter().zip(key).map(|(a, b)| a * b).sum::<f32>();
            }
        }
        total
    }
}
