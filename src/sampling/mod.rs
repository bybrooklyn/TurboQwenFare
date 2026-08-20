//! Token sampling over LM-head logits (spec Part XIII §153).
//!
//! The one invariant that governs this module: **greedy decode must stay
//! bit-identical.** Every oracle-parity record in
//! `docs/research/qualification/` — and the 512-token divergence
//! investigation in particular — compares TQF's greedy token sequence
//! against an independent runtime. If introducing sampling perturbed
//! greedy selection by even one ULP, every one of those records would
//! silently stop meaning what it says.
//!
//! That is why [`Sampler::Greedy`] is a distinct variant that returns the
//! argmax the decode loop already computed, rather than "temperature → 0
//! falls out of the softmax." **No floating-point operation touches the
//! logits on the greedy path.** `Sampler::new` selects it on an exact
//! `temperature == 0.0` comparison, and `SamplingParams`'s own default
//! temperature is `0.0`, so an adapter that forgets to set sampling gets
//! greedy rather than silently starting to sample.
//!
//! The stochastic path applies filters in the order llama.cpp and Ollama
//! use, because client-tuned parameter sets assume that order:
//! repetition/frequency/presence penalties, `top_k`, temperature softmax,
//! `top_p` nucleus, `min_p`, renormalize, inverse-CDF draw.

use crate::runtime::request::SamplingParams;

/// Selects one token per decode step.
///
/// Constructed once per generation (not per step) so the RNG stream is
/// continuous and a `seed` reproduces a whole sequence, not just a token.
#[derive(Debug)]
pub enum Sampler {
    /// Exact argmax. Structurally separate from the stochastic path so no
    /// change to sampling can perturb qualified greedy parity.
    Greedy,
    Stochastic {
        rng: Xoshiro256PlusPlus,
        params: SamplingParams,
    },
}

impl Sampler {
    /// `temperature == 0.0` means greedy. The comparison is exact on
    /// purpose: a "close enough to zero" epsilon would make which path
    /// runs depend on float formatting of a client-supplied value.
    pub fn new(params: &SamplingParams) -> Self {
        if params.temperature == 0.0 {
            return Self::Greedy;
        }
        Self::Stochastic {
            rng: Xoshiro256PlusPlus::seeded(params.seed),
            params: params.clone(),
        }
    }

    pub fn is_greedy(&self) -> bool {
        matches!(self, Self::Greedy)
    }

    /// Picks the next token.
    ///
    /// `greedy_token` is the argmax the caller already computed for its
    /// diagnostics (`top_logit_candidates(..)[0].token`). The greedy path
    /// returns it verbatim — that identity is what keeps qualified parity
    /// intact, so it must not be recomputed here.
    ///
    /// `history` is the tokens generated so far in this request, used by
    /// the repetition penalties.
    pub fn select(&mut self, logits: &[f32], history: &[u32], greedy_token: u32) -> u32 {
        match self {
            Self::Greedy => greedy_token,
            Self::Stochastic { rng, params } => sample(logits, history, params, rng),
        }
    }
}

/// A candidate surviving the filter chain. Kept as an explicit pair so
/// filtering can reorder freely without losing the token id.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    token: u32,
    value: f32,
}

fn sample(
    logits: &[f32],
    history: &[u32],
    params: &SamplingParams,
    rng: &mut Xoshiro256PlusPlus,
) -> u32 {
    let mut candidates: Vec<Candidate> = logits
        .iter()
        .enumerate()
        .map(|(token, &value)| Candidate {
            token: token as u32,
            value,
        })
        .collect();

    apply_penalties(&mut candidates, history, params);
    apply_top_k(&mut candidates, params.top_k);
    let mut probabilities = softmax(&candidates, params.temperature);
    apply_top_p(&mut probabilities, params.top_p);
    apply_min_p(&mut probabilities, params.min_p);

    draw(&probabilities, rng)
}

/// llama.cpp's repetition penalty divides positive logits and multiplies
/// negative ones, so the penalty always moves a logit *down* regardless of
/// sign. A flat subtraction (the other common formulation) would make a
/// strongly-negative logit more attractive, which is the opposite of the
/// intent. Frequency and presence penalties follow OpenAI's definitions:
/// frequency scales with the repeat count, presence is binary.
fn apply_penalties(candidates: &mut [Candidate], history: &[u32], params: &SamplingParams) {
    let repeat = params.repeat_penalty;
    let frequency = params.frequency_penalty;
    let presence = params.presence_penalty;
    if repeat == 1.0 && frequency == 0.0 && presence == 0.0 {
        return;
    }

    let window = params.repeat_last_n.min(history.len());
    let recent = &history[history.len() - window..];
    if recent.is_empty() {
        return;
    }

    // Vocabulary is ~250k and the window is small, so counting the window
    // once beats scanning it per candidate.
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &token in recent {
        *counts.entry(token).or_insert(0) += 1;
    }

    for candidate in candidates.iter_mut() {
        let Some(&count) = counts.get(&candidate.token) else {
            continue;
        };
        if repeat != 1.0 {
            candidate.value = if candidate.value > 0.0 {
                candidate.value / repeat
            } else {
                candidate.value * repeat
            };
        }
        candidate.value -= frequency * count as f32 + presence;
    }
}

/// Keeps the `k` highest logits. `select_nth_unstable_by` partitions in
/// O(n) rather than sorting all ~250k candidates, which matters because
/// this runs on every decode step.
fn apply_top_k(candidates: &mut Vec<Candidate>, top_k: Option<u32>) {
    let Some(k) = top_k else { return };
    let k = k as usize;
    if k == 0 || k >= candidates.len() {
        return;
    }
    candidates.select_nth_unstable_by(k - 1, |a, b| compare_desc(a.value, b.value));
    candidates.truncate(k);
}

/// Softmax with temperature, shifted by the maximum for numerical
/// stability. Returns candidates sorted by descending probability, which
/// `top_p` then relies on.
fn softmax(candidates: &[Candidate], temperature: f32) -> Vec<Candidate> {
    let max = candidates
        .iter()
        .map(|c| c.value)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut scaled: Vec<Candidate> = candidates
        .iter()
        .map(|c| Candidate {
            token: c.token,
            value: ((c.value - max) / temperature).exp(),
        })
        .collect();

    let total: f32 = scaled.iter().map(|c| c.value).sum();
    if total > 0.0 {
        for candidate in scaled.iter_mut() {
            candidate.value /= total;
        }
    }
    // Ties broken by ascending token id so a given logit vector always
    // produces the same ordering, and therefore the same seeded draw.
    scaled.sort_unstable_by(|a, b| compare_desc(a.value, b.value).then(a.token.cmp(&b.token)));
    scaled
}

/// Nucleus sampling: the shortest descending prefix whose probabilities
/// reach `top_p`. Always keeps at least one candidate, so a degenerate
/// `top_p` cannot empty the distribution.
fn apply_top_p(probabilities: &mut Vec<Candidate>, top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let mut cumulative = 0.0;
    let mut keep = 0;
    for candidate in probabilities.iter() {
        cumulative += candidate.value;
        keep += 1;
        if cumulative >= top_p {
            break;
        }
    }
    probabilities.truncate(keep.max(1));
}

/// Drops candidates below `min_p` of the most likely one — a relative
/// floor, so it adapts to how peaked the distribution already is.
fn apply_min_p(probabilities: &mut Vec<Candidate>, min_p: Option<f32>) {
    let Some(min_p) = min_p else { return };
    if min_p <= 0.0 || probabilities.is_empty() {
        return;
    }
    let floor = min_p * probabilities[0].value;
    let keep = probabilities
        .iter()
        .take_while(|c| c.value >= floor)
        .count();
    probabilities.truncate(keep.max(1));
}

/// Inverse-CDF draw over the (possibly truncated, so renormalized) set.
fn draw(probabilities: &[Candidate], rng: &mut Xoshiro256PlusPlus) -> u32 {
    if probabilities.is_empty() {
        return 0;
    }
    // A non-finite or non-positive total means the distribution collapsed
    // (all-NaN or all-zero logits). Fall back to the most likely candidate
    // rather than scaling a random draw by garbage.
    let total: f32 = probabilities.iter().map(|c| c.value).sum();
    if !total.is_finite() || total <= 0.0 {
        return probabilities[0].token;
    }
    let target = rng.next_f32() * total;
    let mut cumulative = 0.0;
    for candidate in probabilities {
        cumulative += candidate.value;
        if cumulative >= target {
            return candidate.token;
        }
    }
    // Float accumulation can leave `target` just past the final sum.
    probabilities[probabilities.len() - 1].token
}

/// Descending order with NaN sorted last, so a NaN logit can never win a
/// comparison and be selected.
///
/// Mapping an incomparable pair to `Equal` is not a valid total order, and
/// `sort_unstable_by` may place NaN anywhere under an inconsistent
/// comparator — at real vocabulary sizes it reliably lands at index 0,
/// where `draw`'s non-finite-total fallback then returns it. NaN is
/// therefore ordered explicitly rather than left to `unwrap_or`.
fn compare_desc(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        // `a` is NaN: it sorts after `b` in a descending order.
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.partial_cmp(&a).expect("neither value is NaN"),
    }
}

/// xoshiro256++ — small, fast, and with a long enough period that a
/// generation cannot exhaust it. Implemented here rather than pulling in
/// `rand`: the crate deliberately keeps its dependency surface small
/// (spec §114), and this is ~25 lines with a published reference vector to
/// test against.
#[derive(Debug, Clone)]
pub struct Xoshiro256PlusPlus {
    state: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Seeds all four words from one `u64` via SplitMix64, as the
    /// reference implementation prescribes — seeding the state directly
    /// from a small integer leaves it near-zero and produces poor early
    /// output.
    pub fn from_seed(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }

    /// An absent seed still has to produce *different* streams across
    /// requests, so it mixes the wall clock. Callers wanting
    /// reproducibility pass an explicit seed (spec §204 permits `seed`
    /// "where deterministic sampling implementation permits" — this
    /// implementation permits it).
    fn seeded(seed: Option<u64>) -> Self {
        Self::from_seed(seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5DEE_CE66_D1CE_1234)
        }))
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`. Takes the top 24 bits, which is the full
    /// precision an `f32` mantissa can represent without bias.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::decode::top_logit_candidates;

    fn greedy_params() -> SamplingParams {
        SamplingParams::default()
    }

    fn stochastic(temperature: f32, seed: u64) -> SamplingParams {
        SamplingParams {
            temperature,
            seed: Some(seed),
            ..SamplingParams::default()
        }
    }

    /// Deterministic pseudo-random logits, so failures reproduce exactly.
    fn logits(count: usize, seed: u64) -> Vec<f32> {
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        (0..count).map(|_| (rng.next_f32() * 20.0) - 10.0).collect()
    }

    /// The load-bearing test for this whole module. Every greedy-parity
    /// qualification record depends on sampling never perturbing argmax
    /// selection, so this asserts the identity across many random logit
    /// vectors rather than one convenient case.
    #[test]
    fn greedy_returns_exactly_the_argmax_the_decoder_already_computed() {
        for seed in 0..500u64 {
            let values = logits(1024, seed);
            let expected = top_logit_candidates(&values)[0].token;

            let mut sampler = Sampler::new(&greedy_params());
            assert!(sampler.is_greedy());
            assert_eq!(
                sampler.select(&values, &[], expected),
                expected,
                "greedy selection diverged on seed {seed}"
            );
        }
    }

    /// A default-constructed `SamplingParams` must be greedy: an adapter
    /// that forgets to set sampling should decode deterministically, not
    /// quietly start sampling and invalidate parity records.
    #[test]
    fn default_sampling_params_are_greedy() {
        assert_eq!(SamplingParams::default().temperature, 0.0);
        assert!(Sampler::new(&SamplingParams::default()).is_greedy());
    }

    /// Temperature 0 wins over every other knob. Without this, a client
    /// sending `temperature: 0` alongside `top_p`/`top_k` would get
    /// nondeterministic output while believing it asked for greedy.
    #[test]
    fn temperature_zero_stays_greedy_regardless_of_other_knobs() {
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 0.5,
            top_k: Some(3),
            min_p: Some(0.1),
            seed: Some(7),
            repeat_penalty: 1.5,
            frequency_penalty: 0.9,
            presence_penalty: 0.9,
            ..SamplingParams::default()
        };
        let values = logits(512, 99);
        let expected = top_logit_candidates(&values)[0].token;
        let mut sampler = Sampler::new(&params);
        assert!(sampler.is_greedy());
        assert_eq!(sampler.select(&values, &[1, 1, 1], expected), expected);
    }

    /// `top_logit_candidates` uses a strict `>`, so on equal logits the
    /// lower token index wins. Codified here so a future refactor to `>=`
    /// cannot silently change which token every greedy run produces.
    #[test]
    fn greedy_tie_break_prefers_the_lower_token_index() {
        let values = vec![1.0, 5.0, 5.0, 5.0, 2.0];
        assert_eq!(top_logit_candidates(&values)[0].token, 1);
    }

    /// A NaN logit must never be selected. `compare_desc` sorts NaN last
    /// and `top_logit_candidates`' `>` comparison is false against NaN, so
    /// both paths agree.
    /// Regression: the original comparator mapped an incomparable pair to
    /// `Ordering::Equal`, which is not a valid total order. `sort_unstable_by`
    /// is free to place NaN anywhere under an inconsistent comparator, and at
    /// realistic vocabulary sizes it lands at index 0 — where `draw`'s
    /// non-finite-total fallback then returns it. The four-element test below
    /// passed only because four elements are too few to trigger it.
    #[test]
    fn a_nan_logit_never_wins_at_realistic_vocabulary_size() {
        let mut values: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();
        values[2000] = f32::NAN;
        let expected = top_logit_candidates(&values)[0].token;

        let mut sampler = Sampler::new(&stochastic(1.0, 17));
        for _ in 0..200 {
            let token = sampler.select(&values, &[], expected);
            assert!(
                !values[token as usize].is_nan(),
                "selected the NaN token {token}"
            );
        }
    }

    #[test]
    fn a_nan_logit_is_never_selected() {
        let values = vec![1.0, f32::NAN, 3.0, f32::NAN];
        assert_eq!(top_logit_candidates(&values)[0].token, 2);

        let mut sampler = Sampler::new(&stochastic(1.0, 5));
        for _ in 0..50 {
            let token = sampler.select(&values, &[], 2);
            assert!(token == 0 || token == 2, "selected a NaN token: {token}");
        }
    }

    #[test]
    fn the_same_seed_reproduces_the_same_sequence_and_different_seeds_diverge() {
        let values = logits(256, 3);
        let run = |seed: u64| {
            let mut sampler = Sampler::new(&stochastic(1.0, seed));
            (0..40)
                .map(|_| sampler.select(&values, &[], 0))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(42), run(42), "a seeded run must be reproducible");
        assert_ne!(
            run(42),
            run(43),
            "different seeds must produce different runs"
        );
    }

    /// "Sampling is real" needs evidence, not assertion: draws from a
    /// known distribution must track its probabilities. Uses a sharply
    /// separated 4-way distribution so the tolerance can be tight without
    /// being flaky.
    #[test]
    fn draw_frequencies_track_the_softmax_distribution() {
        // Softmax over these at temperature 1 is ~[0.6439, 0.2369, 0.0871, 0.0321].
        let values = vec![3.0f32, 2.0, 1.0, 0.0];
        let expected = {
            let total: f32 = values.iter().map(|v| v.exp()).sum();
            values.iter().map(|v| v.exp() / total).collect::<Vec<_>>()
        };

        const DRAWS: usize = 200_000;
        let mut counts = [0usize; 4];
        let mut sampler = Sampler::new(&stochastic(1.0, 20260819));
        for _ in 0..DRAWS {
            counts[sampler.select(&values, &[], 0) as usize] += 1;
        }

        for (token, &count) in counts.iter().enumerate() {
            let observed = count as f32 / DRAWS as f32;
            assert!(
                (observed - expected[token]).abs() < 0.01,
                "token {token}: observed {observed:.4}, expected {:.4}",
                expected[token]
            );
        }
    }

    /// Low temperature concentrates mass on the argmax; high temperature
    /// spreads it. This is the property clients actually tune for.
    #[test]
    fn temperature_controls_how_concentrated_the_draw_is() {
        let values = vec![3.0f32, 2.0, 1.0, 0.0];
        let argmax_share = |temperature: f32| {
            let mut sampler = Sampler::new(&stochastic(temperature, 11));
            let hits = (0..20_000)
                .filter(|_| sampler.select(&values, &[], 0) == 0)
                .count();
            hits as f32 / 20_000.0
        };
        let cold = argmax_share(0.1);
        let hot = argmax_share(5.0);
        assert!(
            cold > 0.99,
            "temperature 0.1 should be near-deterministic: {cold}"
        );
        assert!(hot < 0.45, "temperature 5.0 should spread mass: {hot}");
        assert!(cold > hot);
    }

    #[test]
    fn top_k_restricts_selection_to_the_k_highest_logits() {
        let values = vec![0.0f32, 10.0, 9.5, 1.0, 0.5];
        let params = SamplingParams {
            temperature: 2.0,
            top_k: Some(2),
            seed: Some(1),
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(&params);
        for _ in 0..2_000 {
            let token = sampler.select(&values, &[], 1);
            assert!(token == 1 || token == 2, "top_k=2 leaked token {token}");
        }
    }

    #[test]
    fn top_p_keeps_the_smallest_prefix_reaching_the_threshold() {
        // Softmax ~[0.6439, 0.2369, 0.0871, 0.0321]: top_p 0.8 keeps the
        // first two (0.6439, then 0.8808 >= 0.8).
        let values = vec![3.0f32, 2.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.8,
            seed: Some(2),
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(&params);
        for _ in 0..5_000 {
            let token = sampler.select(&values, &[], 0);
            assert!(token < 2, "top_p=0.8 leaked token {token}");
        }
    }

    #[test]
    fn min_p_drops_candidates_far_below_the_most_likely_one() {
        // min_p 0.5 keeps only candidates at >= 50% of the top
        // probability, which here is the argmax alone.
        let values = vec![5.0f32, 2.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            min_p: Some(0.5),
            seed: Some(3),
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(&params);
        for _ in 0..2_000 {
            assert_eq!(sampler.select(&values, &[], 0), 0);
        }
    }

    /// The penalty must lower a repeated token's logit whether it started
    /// positive or negative — the reason for the divide/multiply pair
    /// rather than a flat subtraction.
    #[test]
    fn repetition_penalty_lowers_repeated_tokens_of_either_sign() {
        let params = SamplingParams {
            temperature: 1.0,
            repeat_penalty: 2.0,
            ..SamplingParams::default()
        };
        let mut candidates = [
            Candidate {
                token: 0,
                value: 4.0,
            },
            Candidate {
                token: 1,
                value: -4.0,
            },
            Candidate {
                token: 2,
                value: 4.0,
            },
        ];
        apply_penalties(&mut candidates, &[0, 1], &params);

        assert_eq!(candidates[0].value, 2.0, "positive logit must be divided");
        assert_eq!(
            candidates[1].value, -8.0,
            "negative logit must be multiplied"
        );
        assert_eq!(
            candidates[2].value, 4.0,
            "unrepeated token must be untouched"
        );
    }

    #[test]
    fn the_repetition_window_ignores_tokens_older_than_repeat_last_n() {
        let params = SamplingParams {
            temperature: 1.0,
            repeat_penalty: 2.0,
            repeat_last_n: 2,
            ..SamplingParams::default()
        };
        let mut candidates = [Candidate {
            token: 9,
            value: 4.0,
        }];
        // Token 9 appears only outside the two-token window.
        apply_penalties(&mut candidates, &[9, 1, 2], &params);
        assert_eq!(candidates[0].value, 4.0);
    }

    #[test]
    fn frequency_penalty_scales_with_count_while_presence_is_binary() {
        let frequency = SamplingParams {
            temperature: 1.0,
            frequency_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut candidates = [Candidate {
            token: 0,
            value: 10.0,
        }];
        apply_penalties(&mut candidates, &[0, 0, 0], &frequency);
        assert_eq!(candidates[0].value, 7.0, "three repeats, penalty 1.0 each");

        let presence = SamplingParams {
            temperature: 1.0,
            presence_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut candidates = [Candidate {
            token: 0,
            value: 10.0,
        }];
        apply_penalties(&mut candidates, &[0, 0, 0], &presence);
        assert_eq!(
            candidates[0].value, 9.0,
            "presence applies once regardless of count"
        );
    }

    /// Reference output of xoshiro256++ for the all-ones state, from the
    /// algorithm's published definition. Pins the generator so a
    /// refactor cannot silently change every seeded stream.
    #[test]
    fn xoshiro256pp_matches_its_reference_definition() {
        let mut rng = Xoshiro256PlusPlus {
            state: [1, 1, 1, 1],
        };
        // s0 + s3 = 2, rotl(2, 23) = 16777216, + s0 = 16777217.
        assert_eq!(rng.next_u64(), 16_777_217);
        assert_ne!(rng.next_u64(), 16_777_217, "state must advance");
    }

    #[test]
    fn next_f32_stays_in_the_unit_interval() {
        let mut rng = Xoshiro256PlusPlus::from_seed(12345);
        for _ in 0..100_000 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value), "out of range: {value}");
        }
    }

    /// Seeding must not leave the state near-zero, which is what makes
    /// SplitMix64 expansion necessary rather than decorative.
    #[test]
    fn seeding_from_zero_still_produces_a_well_mixed_stream() {
        let mut rng = Xoshiro256PlusPlus::from_seed(0);
        let first: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
        assert!(first.iter().all(|&v| v != 0));
        assert_eq!(
            first.iter().collect::<std::collections::HashSet<_>>().len(),
            8,
            "seeded stream repeated a value in its first 8 outputs"
        );
    }
}
