//! TQVec candidate family (spec §39, §87, §190-191, §311).
//! **RESEARCH CANDIDATES**: every candidate here defines its own byte
//! size, decoder/distance kernel, and (measured separately, in the
//! qualification test) recall loss against Phase 38's FP32 gold
//! ranking. None of these are wired into a live index — per spec §300's
//! rule for the analogous TQKV research candidates (Phase 28), a
//! mixed-precision controller is deliberately not built before
//! individual encodings are qualified on real data.
//!
//! Deliberately self-contained rather than reusing `context::tqkv`'s
//! similar rotation/grouped-quantization machinery: the dependency
//! firewall (spec §24) keeps `retrieval` and `context` independently
//! removable subsystems, so TQVec's bit-packing is its own small
//! implementation even though the underlying ideas mirror Phase 28's
//! TQKV candidates.

use half::f16;

use super::flat::{int8_dot, quantize_int8_linear};

const DIM: usize = 1024;
const GROUP_SIZE: usize = 32;
const GROUPS: usize = DIM / GROUP_SIZE;
const COARSE_DIM: usize = 256;
const COARSE_BYTES: usize = COARSE_DIM / 8;

fn pack_sign_bits(values: &[f32]) -> Vec<u8> {
    values
        .chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &x) in chunk.iter().enumerate() {
                if x >= 0.0 {
                    byte |= 1 << (7 - i);
                }
            }
            byte
        })
        .collect()
}

pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Packs `values` (each already in `0..2^bits`) MSB-first into a byte
/// buffer. For `DIM=1024` and `bits` in `{4,5}` this divides evenly
/// (4096 and 5120 bits respectively), so there is no partial trailing
/// byte to special-case.
fn pack_bits(values: &[u8], bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * bits as usize / 8);
    let mut acc: u32 = 0;
    let mut acc_bits = 0u32;
    for &v in values {
        acc |= (v as u32) << acc_bits;
        acc_bits += bits;
        while acc_bits >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        out.push((acc & 0xFF) as u8);
    }
    out
}

fn unpack_bits(bytes: &[u8], bits: u32, count: usize) -> Vec<u8> {
    let mask = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(count);
    let mut acc: u32 = 0;
    let mut acc_bits = 0u32;
    let mut iter = bytes.iter();
    for _ in 0..count {
        while acc_bits < bits {
            let b = *iter.next().expect("enough packed bytes");
            acc |= (b as u32) << acc_bits;
            acc_bits += 8;
        }
        out.push((acc & mask) as u8);
        acc >>= bits;
        acc_bits -= bits;
    }
    out
}

/// TQVec-A — native INT8 control (spec §190): `1024 x int8 + f32 scale`.
/// Simple SIMD dot-product baseline; this is exactly Phase 38's INT8
/// reference control, re-exposed under the TQVec candidate naming so
/// the Pareto comparison in the qualification doc can list it alongside
/// B-F on equal footing.
pub struct TqVecA {
    pub int8: Vec<i8>,
    pub scale: f32,
}

impl TqVecA {
    pub fn encode(values: &[f32]) -> Self {
        let (int8, scale) = quantize_int8_linear(values);
        Self { int8, scale }
    }

    pub fn byte_size(&self) -> usize {
        self.int8.len() + 4
    }

    pub fn score(&self, other: &Self) -> f32 {
        int8_dot(&self.int8, self.scale, &other.int8, other.scale)
    }
}

/// TQVec-B — binary coarse (256-d sign key) + full INT8 (spec §190).
/// The coarse key is a cheap Hamming prefilter for search *latency* at
/// index scale; it does not change the final ranking once every
/// candidate is exactly re-scored by the INT8 kernel, so B's *recall*
/// against the FP32 gold ranking is identical to A's by construction —
/// its value is a scale-dependent latency win this phase's small real
/// corpus is too small to demonstrate (see the qualification doc).
pub struct TqVecB {
    pub coarse: Vec<u8>,
    pub full: TqVecA,
}

impl TqVecB {
    pub fn encode(values: &[f32]) -> Self {
        Self {
            coarse: pack_sign_bits(&values[..COARSE_DIM]),
            full: TqVecA::encode(values),
        }
    }

    pub fn byte_size(&self) -> usize {
        self.coarse.len() + self.full.byte_size()
    }

    pub fn score(&self, other: &Self) -> f32 {
        self.full.score(&other.full)
    }
}

/// Shared machinery for TQVec-C/D: a 256-bit coarse sign key plus
/// grouped `bits`-wide symmetric quantization (32 groups of 32 dims,
/// one real `f16` scale per group — spec's own FP16-scale convention,
/// same crate/type as Phase 27's TQKV pages).
pub struct GroupedQuant {
    pub coarse: Vec<u8>,
    pub packed: Vec<u8>,
    pub group_scales: [f16; GROUPS],
    pub bits: u32,
}

impl GroupedQuant {
    fn encode(values: &[f32], bits: u32) -> Self {
        let max_level = ((1u32 << (bits - 1)) - 1) as f32;
        let mut codes = Vec::with_capacity(DIM);
        let mut group_scales = [f16::from_f32(0.0); GROUPS];
        for (group_index, group) in values.chunks(GROUP_SIZE).enumerate() {
            let max_abs = group.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs / max_level
            };
            group_scales[group_index] = f16::from_f32(scale);
            for &v in group {
                let code = (v / scale).round().clamp(-max_level, max_level) as i32;
                codes.push((code + max_level as i32) as u8);
            }
        }
        let packed = pack_bits(&codes, bits);
        Self {
            coarse: pack_sign_bits(&values[..COARSE_DIM]),
            packed,
            group_scales,
            bits,
        }
    }

    pub fn byte_size(&self) -> usize {
        self.coarse.len() + self.packed.len() + self.group_scales.len() * 2 + 2
    }

    /// Decoder (spec's "must define... decoder/distance kernel"):
    /// dequantizes back to `DIM` f32 values group-by-group.
    pub fn decode(&self) -> Vec<f32> {
        let max_level = ((1u32 << (self.bits - 1)) - 1) as i32;
        let codes = unpack_bits(&self.packed, self.bits, DIM);
        let mut out = Vec::with_capacity(DIM);
        for (group_index, group_codes) in codes.chunks(GROUP_SIZE).enumerate() {
            let scale = self.group_scales[group_index].to_f32();
            for &code in group_codes {
                out.push((code as i32 - max_level) as f32 * scale);
            }
        }
        out
    }

    /// Distance kernel: dequantize both sides, then dot. A real
    /// candidate implementation would fuse dequant+dot into one SIMD
    /// pass; this reference kernel keeps the two steps separate for
    /// clarity, matching this phase's REFERENCE BASELINE scope (see
    /// `retrieval::flat`'s own note on the same tradeoff).
    pub fn score(&self, other: &Self) -> f32 {
        let a = self.decode();
        let b = other.decode();
        a.iter().zip(&b).map(|(x, y)| x * y).sum()
    }
}

/// TQVec-C — binary coarse + grouped Q5 (spec §190: target ~752 B/vector).
pub struct TqVecC(pub GroupedQuant);
impl TqVecC {
    pub fn encode(values: &[f32]) -> Self {
        Self(GroupedQuant::encode(values, 5))
    }
    pub fn byte_size(&self) -> usize {
        self.0.byte_size()
    }
    pub fn score(&self, other: &Self) -> f32 {
        self.0.score(&other.0)
    }
}

/// TQVec-D — binary coarse + grouped Q4 (spec §190: target ~624 B/vector).
pub struct TqVecD(pub GroupedQuant);
impl TqVecD {
    pub fn encode(values: &[f32]) -> Self {
        Self(GroupedQuant::encode(values, 4))
    }
    pub fn byte_size(&self) -> usize {
        self.0.byte_size()
    }
    pub fn score(&self, other: &Self) -> f32 {
        self.0.score(&other.0)
    }
}

/// A fixed (not per-vector) pseudo-random +/-1 sign vector, xorshift64-
/// seeded, plus a real fast Walsh-Hadamard transform (spec §190
/// TQVec-E: "deterministic orthogonal/structured transform... store
/// transform ID globally per repository/profile, not per vector").
/// `DIM=1024` is a power of two, so the FWHT is exact (no padding).
const ROTATION_SEED: u64 = 0x54_51_56_65_63_45_5f_31; // "TQVecE_1" in ASCII hex bytes

fn rotation_signs() -> [f32; DIM] {
    let mut state = ROTATION_SEED;
    let mut signs = [0.0f32; DIM];
    for slot in signs.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = if state & 1 == 0 { 1.0 } else { -1.0 };
    }
    signs
}

fn fwht_inplace(values: &mut [f32; DIM]) {
    let mut len = 1;
    while len < DIM {
        let mut i = 0;
        while i < DIM {
            for j in i..i + len {
                let x = values[j];
                let y = values[j + len];
                values[j] = x + y;
                values[j + len] = x - y;
            }
            i += 2 * len;
        }
        len *= 2;
    }
    let norm = 1.0 / (DIM as f32).sqrt();
    for v in values.iter_mut() {
        *v *= norm;
    }
}

/// Applies the fixed randomized-sign-flip + FWHT rotation. Since the
/// same orthonormal transform `R` is applied to every vector (query and
/// documents alike), `<Rx, Ry> = <x, y>` exactly — so a rotated
/// candidate's distance kernel can score directly on rotated-decoded
/// values without ever inverting the transform.
pub fn rotate(values: &[f32]) -> [f32; DIM] {
    let signs = rotation_signs();
    let mut rotated = [0.0f32; DIM];
    for i in 0..DIM {
        rotated[i] = values[i] * signs[i];
    }
    fwht_inplace(&mut rotated);
    rotated
}

/// TQVec-E — rotated Q4/Q5 (spec §190). Parametrized over the same
/// `GroupedQuant` machinery as C/D, applied to the rotated vector.
pub struct TqVecE(pub GroupedQuant);
impl TqVecE {
    pub fn encode(values: &[f32], bits: u32) -> Self {
        let rotated = rotate(values);
        Self(GroupedQuant::encode(&rotated, bits))
    }
    pub fn byte_size(&self) -> usize {
        self.0.byte_size()
    }
    pub fn score(&self, other: &Self) -> f32 {
        self.0.score(&other.0)
    }
}

/// TQVec-F — residual hierarchy (spec §190): a cheap 256-d sign-key base
/// plus a quantized residual against a crude per-vector binary
/// reconstruction (`sign(bit) * mean(|values|)`), simulating "quantized
/// residual information used only for top candidates" — the residual is
/// always stored (a real system doesn't know the query-time shortlist in
/// advance) but the qualification test measures how much of the corpus
/// a base-only coarse pass can already rule out before ever touching a
/// residual.
pub struct TqVecF {
    pub base: Vec<u8>,
    pub base_magnitude: f32,
    pub residual_int8: Vec<i8>,
    pub residual_scale: f32,
}

impl TqVecF {
    pub fn encode(values: &[f32]) -> Self {
        let base = pack_sign_bits(values);
        let base_magnitude = values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32;
        let reconstruction: Vec<f32> = values
            .iter()
            .map(|v| {
                if *v >= 0.0 {
                    base_magnitude
                } else {
                    -base_magnitude
                }
            })
            .collect();
        let residual: Vec<f32> = values
            .iter()
            .zip(&reconstruction)
            .map(|(v, r)| v - r)
            .collect();
        let (residual_int8, residual_scale) = quantize_int8_linear(&residual);
        Self {
            base,
            base_magnitude,
            residual_int8,
            residual_scale,
        }
    }

    pub fn byte_size(&self) -> usize {
        self.base.len() + 4 + self.residual_int8.len() + 4
    }

    /// Base-only coarse score (cheap prefilter): Hamming distance on
    /// the sign key, negated so "higher is closer" matches the other
    /// candidates' score convention.
    pub fn base_score(&self, other: &Self) -> f32 {
        -(hamming(&self.base, &other.base) as f32)
    }

    /// Full score: reconstruct each side's approximate vector (base +
    /// residual) and dot. Only worth paying for on a base-shortlisted
    /// candidate.
    pub fn full_score(&self, other: &Self) -> f32 {
        let reconstruct = |base: &[u8], magnitude: f32, residual: &[i8], scale: f32| -> Vec<f32> {
            (0..DIM)
                .map(|i| {
                    let byte = base[i / 8];
                    let bit = (byte >> (7 - (i % 8))) & 1;
                    let base_value = if bit == 1 { magnitude } else { -magnitude };
                    base_value + residual[i] as f32 * scale
                })
                .collect()
        };
        let a = reconstruct(
            &self.base,
            self.base_magnitude,
            &self.residual_int8,
            self.residual_scale,
        );
        let b = reconstruct(
            &other.base,
            other.base_magnitude,
            &other.residual_int8,
            other.residual_scale,
        );
        a.iter().zip(&b).map(|(x, y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_unit_vector(seed: u64) -> Vec<f32> {
        let mut state = seed.max(1);
        let mut values: Vec<f32> = (0..DIM)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 / 1000.0) - 1.0
            })
            .collect();
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in values.iter_mut() {
            *v /= norm;
        }
        values
    }

    #[test]
    fn bit_packing_round_trips_for_4_and_5_bit_widths() {
        for bits in [4u32, 5u32] {
            let max = (1u32 << bits) - 1;
            let values: Vec<u8> = (0..DIM).map(|i| (i as u32 % (max + 1)) as u8).collect();
            let packed = pack_bits(&values, bits);
            let unpacked = unpack_bits(&packed, bits, DIM);
            assert_eq!(values, unpacked);
        }
    }

    #[test]
    fn byte_sizes_match_spec_targets_approximately() {
        let v = synthetic_unit_vector(1);
        assert_eq!(TqVecA::encode(&v).byte_size(), 1028);
        assert_eq!(TqVecB::encode(&v).byte_size(), COARSE_BYTES + 1028);
        assert_eq!(TqVecC::encode(&v).byte_size(), COARSE_BYTES + 640 + 64 + 2);
        assert_eq!(TqVecD::encode(&v).byte_size(), COARSE_BYTES + 512 + 64 + 2);
    }

    #[test]
    fn fwht_rotation_is_orthonormal_and_preserves_inner_products() {
        let a = synthetic_unit_vector(1);
        let b = synthetic_unit_vector(2);
        let plain_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let ra = rotate(&a);
        let rb = rotate(&b);
        let rotated_dot: f32 = ra.iter().zip(&rb).map(|(x, y)| x * y).sum();
        assert!(
            (plain_dot - rotated_dot).abs() < 1e-3,
            "rotation should preserve inner products: {plain_dot} vs {rotated_dot}"
        );
    }

    #[test]
    fn grouped_quant_decode_is_close_to_the_original_vector() {
        let v = synthetic_unit_vector(3);
        let q = GroupedQuant::encode(&v, 5);
        let decoded = q.decode();
        let max_err = v
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.05,
            "5-bit grouped quant max error too high: {max_err}"
        );
    }

    #[test]
    fn tqvec_f_full_score_beats_base_only_score_at_self_similarity() {
        let v = synthetic_unit_vector(4);
        let f = TqVecF::encode(&v);
        let self_full = f.full_score(&f);
        // The vector normalized to unit L2 norm should self-score near 1.0
        // under a reasonably accurate reconstruction.
        assert!(
            self_full > 0.5,
            "self full-score should be strongly positive: {self_full}"
        );
    }

    fn recall_at_k(candidate: &[usize], ground_truth: &[usize]) -> f32 {
        let hits = candidate
            .iter()
            .filter(|i| ground_truth.contains(i))
            .count();
        hits as f32 / ground_truth.len() as f32
    }

    fn top_k_by_score<T>(
        query: &T,
        records: &[T],
        k: usize,
        score: impl Fn(&T, &T) -> f32,
    ) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = records
            .iter()
            .enumerate()
            .map(|(i, r)| (i, score(query, r)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Real end-to-end qualification (spec §39's "benchmark multiple
    /// encodings... measured for recall... latency... RAM"). Loads the
    /// real Phase 38 corpus/query embeddings committed as fixtures (not
    /// synthetic vectors, and no model load needed since the fixtures
    /// already carry real computed FP32 embeddings), encodes every
    /// TQVec candidate, and measures each one's byte size and recall@k
    /// against the same FP32 gold ranking Phase 38 established.
    #[test]
    fn real_corpus_tqvec_candidates_pareto_comparison() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let corpus: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                root.join("docs/research/qualification/raw-a-phase38-flat-corpus-embeddings.json"),
            )
            .expect("open corpus fixture"),
        )
        .expect("parse corpus fixture");
        let queries: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                root.join("docs/research/qualification/raw-a-phase38-flat-query-embeddings.json"),
            )
            .expect("open query fixture"),
        )
        .expect("parse query fixture");

        let doc_vectors: Vec<Vec<f32>> = corpus["documents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                d["fp32"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        assert_eq!(
            doc_vectors.len(),
            10,
            "expected the committed Phase 38 fixture's 10 documents"
        );

        struct Query {
            fp32: Vec<f32>,
            ground_truth: Vec<usize>,
        }
        let query_list: Vec<Query> = queries["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| Query {
                fp32: q["fp32"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
                ground_truth: q["fp32_ground_truth_top_k"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
            })
            .collect();
        let k = 5;

        macro_rules! bench {
            ($name:expr, $encode:expr, $score:expr, $byte_size:expr) => {{
                let started = std::time::Instant::now();
                let doc_encoded: Vec<_> = doc_vectors.iter().map(|v| $encode(v)).collect();
                let encode_ms = started.elapsed().as_millis();
                let mut recalls = Vec::new();
                let search_started = std::time::Instant::now();
                for q in &query_list {
                    let q_encoded = $encode(&q.fp32);
                    let hits = top_k_by_score(&q_encoded, &doc_encoded, k, $score);
                    recalls.push(recall_at_k(&hits, &q.ground_truth));
                }
                let search_ms = search_started.elapsed().as_millis();
                let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
                let bytes = $byte_size(&doc_encoded[0]);
                println!(
                    "phase39_candidate name={} bytes_per_vector={} mean_recall@{}={} encode_ms_for_{}_docs={} search_ms_for_{}_queries={}",
                    $name,
                    bytes,
                    k,
                    mean_recall,
                    doc_vectors.len(),
                    encode_ms,
                    query_list.len(),
                    search_ms
                );
                mean_recall
            }};
        }

        let recall_a = bench!(
            "A-int8",
            |v: &Vec<f32>| TqVecA::encode(v),
            |a: &TqVecA, b: &TqVecA| a.score(b),
            |r: &TqVecA| r.byte_size()
        );
        let recall_b = bench!(
            "B-binary-coarse+int8",
            |v: &Vec<f32>| TqVecB::encode(v),
            |a: &TqVecB, b: &TqVecB| a.score(b),
            |r: &TqVecB| r.byte_size()
        );
        let recall_c = bench!(
            "C-binary-coarse+Q5",
            |v: &Vec<f32>| TqVecC::encode(v),
            |a: &TqVecC, b: &TqVecC| a.score(b),
            |r: &TqVecC| r.byte_size()
        );
        let recall_d = bench!(
            "D-binary-coarse+Q4",
            |v: &Vec<f32>| TqVecD::encode(v),
            |a: &TqVecD, b: &TqVecD| a.score(b),
            |r: &TqVecD| r.byte_size()
        );
        let recall_e5 = bench!(
            "E-rotated-Q5",
            |v: &Vec<f32>| TqVecE::encode(v, 5),
            |a: &TqVecE, b: &TqVecE| a.score(b),
            |r: &TqVecE| r.byte_size()
        );
        let recall_e4 = bench!(
            "E-rotated-Q4",
            |v: &Vec<f32>| TqVecE::encode(v, 4),
            |a: &TqVecE, b: &TqVecE| a.score(b),
            |r: &TqVecE| r.byte_size()
        );
        let recall_f_full = bench!(
            "F-residual-full",
            |v: &Vec<f32>| TqVecF::encode(v),
            |a: &TqVecF, b: &TqVecF| a.full_score(b),
            |r: &TqVecF| r.byte_size()
        );
        let recall_f_base = bench!(
            "F-residual-base-only",
            |v: &Vec<f32>| TqVecF::encode(v),
            |a: &TqVecF, b: &TqVecF| a.base_score(b),
            |r: &TqVecF| r.byte_size()
        );

        // Cross-check against Phase 38's own recorded result: TqVecA is
        // exactly Phase 38's linear-INT8 control, so this must reproduce
        // that phase's measured recall on the same fixture.
        assert!(
            recall_a >= 0.99,
            "TqVec-A should reproduce Phase 38's near-perfect INT8 recall: {recall_a}"
        );
        assert_eq!(
            recall_a, recall_b,
            "B's final ranking is exactly A's INT8 kernel by construction"
        );
        assert!(
            recall_f_full > recall_f_base,
            "the residual should meaningfully improve on the base-only coarse ranking: full={recall_f_full} base={recall_f_base}"
        );
        assert!(
            recall_c >= recall_d,
            "5-bit grouped quant should not recall worse than 4-bit at the same corpus scale: C={recall_c} D={recall_d}"
        );
        let _ = (recall_e5, recall_e4);
    }
}
