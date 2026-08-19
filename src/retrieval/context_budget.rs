//! Dynamic context budget (spec §44, §94): "The system estimates
//! whether external local context is useful and chooses a dynamic
//! injection budget. A simple symbol question may need a few hundred
//! or thousand tokens; a cross-module architecture question may need a
//! much larger set plus graph expansion."
//!
//! Builds directly on Phase 40's `QueryIntent` classification — the
//! router already tells us how identifier-like vs. how open-ended a
//! query is, which is exactly the signal spec §94 wants a budget
//! decision to be based on.

use super::hybrid::{classify_query, QueryIntent};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextBudgetDecision {
    /// Whether retrieval is worth running at all for this query (spec
    /// §85: "Retrieval should be skipped entirely when it is not
    /// useful").
    pub use_retrieval: bool,
    /// Token budget for injected evidence if `use_retrieval` is true.
    pub token_budget: usize,
    /// How many fused candidates to pull evidence from.
    pub candidate_count: usize,
}

const NARROW_TOKEN_BUDGET: usize = 400;
const MODERATE_TOKEN_BUDGET: usize = 1500;
const BROAD_TOKEN_BUDGET: usize = 6000;

/// spec §94's estimator. A pure function of the router's intent
/// confidences — no model call, no I/O — so a caller decides *before*
/// paying for any lane's actual work.
pub fn estimate_budget(intents: &[(QueryIntent, f32)]) -> ContextBudgetDecision {
    if intents.is_empty() {
        return ContextBudgetDecision {
            use_retrieval: false,
            token_budget: 0,
            candidate_count: 0,
        };
    }

    let exact_confidence = intents
        .iter()
        .filter(|(intent, _)| matches!(intent, QueryIntent::ExactSymbol | QueryIntent::ExactPath))
        .map(|(_, c)| *c)
        .fold(0.0f32, f32::max);
    let semantic_confidence = intents
        .iter()
        .find(|(intent, _)| *intent == QueryIntent::SemanticQuestion)
        .map(|(_, c)| *c)
        .unwrap_or(0.0);
    let mixed_confidence = intents
        .iter()
        .find(|(intent, _)| *intent == QueryIntent::Mixed)
        .map(|(_, c)| *c)
        .unwrap_or(0.0);

    // A narrow, unambiguous identifier/path lookup: a handful of exact
    // hits answer it, so a large injection budget would just waste
    // context window on irrelevant material.
    if exact_confidence > 0.0 && semantic_confidence == 0.0 && mixed_confidence == 0.0 {
        return ContextBudgetDecision {
            use_retrieval: true,
            token_budget: NARROW_TOKEN_BUDGET,
            candidate_count: 5,
        };
    }

    // Genuinely mixed (both exact and open-ended signals fired): needs
    // more room than a pure lookup but doesn't need the full
    // cross-module budget.
    if mixed_confidence > 0.0 {
        return ContextBudgetDecision {
            use_retrieval: true,
            token_budget: MODERATE_TOKEN_BUDGET,
            candidate_count: 12,
        };
    }

    // A broad, open-ended natural-language question: scales with how
    // confident the router is that this is genuinely a semantic
    // question (a weakly-scored NL-shaped query gets a moderate
    // budget, not the full cross-module allotment).
    if semantic_confidence > 0.0 {
        let token_budget = if semantic_confidence >= 0.7 {
            BROAD_TOKEN_BUDGET
        } else {
            MODERATE_TOKEN_BUDGET
        };
        return ContextBudgetDecision {
            use_retrieval: true,
            token_budget,
            candidate_count: 16,
        };
    }

    // Some other signal fired (e.g. ErrorLiteral alone) with no
    // exact/semantic/mixed confidence: a moderate default rather than
    // skipping retrieval outright.
    ContextBudgetDecision {
        use_retrieval: true,
        token_budget: MODERATE_TOKEN_BUDGET,
        candidate_count: 8,
    }
}

/// Convenience wrapper: classifies `query` and estimates its budget in
/// one call.
pub fn estimate_budget_for_query(query: &str) -> ContextBudgetDecision {
    estimate_budget(&classify_query(query))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_queries_get_a_narrow_budget() {
        let decision = estimate_budget_for_query("MemoryBroker");
        assert!(decision.use_retrieval);
        assert_eq!(decision.token_budget, NARROW_TOKEN_BUDGET);
    }

    #[test]
    fn open_ended_questions_get_a_broad_budget() {
        let decision = estimate_budget_for_query(
            "how does the memory broker interact with the expert cache across modules",
        );
        assert!(decision.use_retrieval);
        assert!(decision.token_budget >= MODERATE_TOKEN_BUDGET);
    }

    #[test]
    fn empty_signal_skips_retrieval() {
        let decision = estimate_budget(&[]);
        assert!(!decision.use_retrieval);
        assert_eq!(decision.token_budget, 0);
    }

    #[test]
    fn narrow_budget_is_smaller_than_broad_budget() {
        let narrow = estimate_budget_for_query("MemoryBroker");
        let broad = estimate_budget_for_query(
            "how does the memory broker interact with the expert cache across modules and what triggers eviction",
        );
        assert!(narrow.token_budget < broad.token_budget);
        assert!(narrow.candidate_count < broad.candidate_count);
    }
}
