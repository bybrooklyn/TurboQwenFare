//! Adaptive ANN research (spec §41, §89-90, §313). "Only now implement
//! custom semantic partitions. Baselines are already available,
//! preventing an unmeasured bespoke index." Builds directly on Phase
//! 38's flat gold baseline and measures against it, per spec §89's own
//! sequencing rule.
//!
//! Candidate development sequence (spec §313) has five steps; only the
//! first two are attempted this phase — see the module's qualification
//! doc for why steps 3-5 (hot/cold residency, split/merge, workload-
//! adaptive routing) need live query/update traffic over time that a
//! small offline corpus cannot meaningfully exercise:
//!
//! 1. static balanced semantic partitions (`SemanticPartitionIndex`);
//! 2. repo-hierarchy overlay (`HierarchyOverlay`, derived from path
//!    structure alone — repository/module/file, spec §90's first three
//!    hierarchy levels — since type/function levels need real AST,
//!    which Phase 35/36 already scoped out).

use std::collections::HashMap;

use super::flat::l2_normalize;

/// One static semantic partition: a centroid plus the corpus indices
/// assigned to it.
#[derive(Debug, Clone)]
pub struct SemanticPartition {
    pub centroid: Vec<f32>,
    pub member_indices: Vec<usize>,
}

/// A simple balanced k-means (Lloyd's algorithm) over L2-normalized
/// vectors, using cosine similarity (dot product, since inputs are
/// unit-norm) as the assignment metric. Deterministic given
/// `seed` — no reliance on external randomness, so a benchmark run is
/// reproducible.
pub struct SemanticPartitionIndex {
    pub partitions: Vec<SemanticPartition>,
}

fn xorshift_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl SemanticPartitionIndex {
    /// Builds `k` partitions over `vectors` (assumed already
    /// L2-normalized, as Phase 38's stored records are) with a fixed
    /// number of Lloyd iterations. `seed` picks the deterministic
    /// initial centroids (a simple reservoir-style pick, not k-means++,
    /// since this corpus is small enough that init quality is not the
    /// bottleneck).
    pub fn build(vectors: &[Vec<f32>], k: usize, iterations: usize, seed: u64) -> Self {
        let k = k.min(vectors.len()).max(1);
        let mut state = seed.max(1);
        let mut chosen = Vec::with_capacity(k);
        let mut available: Vec<usize> = (0..vectors.len()).collect();
        for _ in 0..k {
            let pick = (xorshift_next(&mut state) as usize) % available.len();
            chosen.push(available.remove(pick));
        }
        let mut centroids: Vec<Vec<f32>> = chosen.iter().map(|&i| vectors[i].clone()).collect();

        let mut assignment = vec![0usize; vectors.len()];
        for _ in 0..iterations {
            for (i, v) in vectors.iter().enumerate() {
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let score = dot(v, centroid);
                    if score > best_score {
                        best_score = score;
                        best = c;
                    }
                }
                assignment[i] = best;
            }
            let dim = vectors[0].len();
            let mut sums = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, v) in vectors.iter().enumerate() {
                let c = assignment[i];
                counts[c] += 1;
                for (s, value) in sums[c].iter_mut().zip(v) {
                    *s += value;
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    continue;
                }
                let mut mean = sums[c].clone();
                for value in mean.iter_mut() {
                    *value /= counts[c] as f32;
                }
                l2_normalize(&mut mean);
                centroids[c] = mean;
            }
        }

        let mut partitions: Vec<SemanticPartition> = centroids
            .into_iter()
            .map(|centroid| SemanticPartition {
                centroid,
                member_indices: Vec::new(),
            })
            .collect();
        for (i, &c) in assignment.iter().enumerate() {
            partitions[c].member_indices.push(i);
        }
        Self { partitions }
    }

    /// Searches only the `nprobe` partitions whose centroid is closest
    /// to `query`, doing an exact dot-product scan within each. Returns
    /// `(candidate_index, score)` sorted descending, top `k`, plus how
    /// many of the corpus's vectors were actually scanned (the whole
    /// point of partitioning: scan less than the full corpus).
    pub fn search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        nprobe: usize,
        k: usize,
    ) -> (Vec<(usize, f32)>, usize) {
        let mut partition_order: Vec<(usize, f32)> = self
            .partitions
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dot(query, &p.centroid)))
            .collect();
        partition_order.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut scanned = 0usize;
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for &(partition_index, _) in partition_order.iter().take(nprobe.max(1)) {
            for &member in &self.partitions[partition_index].member_indices {
                scored.push((member, dot(query, &vectors[member])));
                scanned += 1;
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        (scored, scanned)
    }
}

/// Repo-hierarchy overlay (spec §90's first three levels —
/// repository/module/file; type/function need real AST, out of scope
/// per Phase 35/36's own decision). Derived purely from path structure:
/// `module` is the first path component after `src/`.
pub struct HierarchyOverlay {
    /// module name -> corpus indices belonging to it
    pub modules: HashMap<String, Vec<usize>>,
}

/// The first path component after `src/` (or the whole id if there's no
/// `src/` prefix) — spec §90's coarsest hierarchy level, derivable
/// without any AST.
pub fn module_of(chunk_id: &str) -> String {
    chunk_id
        .strip_prefix("src/")
        .unwrap_or(chunk_id)
        .split('/')
        .next()
        .unwrap_or(chunk_id)
        .to_string()
}

impl HierarchyOverlay {
    pub fn build(chunk_ids: &[String]) -> Self {
        let mut modules: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, id) in chunk_ids.iter().enumerate() {
            modules.entry(module_of(id)).or_default().push(i);
        }
        Self { modules }
    }

    /// Spec §90: "The query router may enter from either side." A
    /// same-module-as-an-active-file bonus a hybrid fusion step could
    /// add alongside RRF (spec §194's "active-file/module proximity
    /// bonus").
    pub fn same_module_bonus(
        &self,
        chunk_ids: &[String],
        active_file: &str,
        candidate: usize,
    ) -> f32 {
        let active_module = module_of(active_file);
        if module_of(&chunk_ids[candidate]) == active_module {
            1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_of_extracts_the_first_path_component_after_src() {
        assert_eq!(module_of("src/retrieval/ignore.rs"), "retrieval");
        assert_eq!(module_of("src/memory/mod.rs"), "memory");
    }

    #[test]
    fn hierarchy_overlay_groups_by_module() {
        let ids = vec![
            "src/memory/mod.rs".to_string(),
            "src/retrieval/ignore.rs".to_string(),
            "src/retrieval/classify.rs".to_string(),
        ];
        let overlay = HierarchyOverlay::build(&ids);
        assert_eq!(overlay.modules["retrieval"].len(), 2);
        assert_eq!(overlay.modules["memory"].len(), 1);
        assert_eq!(
            overlay.same_module_bonus(&ids, "src/retrieval/scan.rs", 1),
            1.0
        );
        assert_eq!(
            overlay.same_module_bonus(&ids, "src/retrieval/scan.rs", 0),
            0.0
        );
    }

    #[test]
    fn partitioned_search_never_exceeds_full_corpus_scan() {
        let mut state = 7u64;
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|_| {
                let mut v: Vec<f32> = (0..16)
                    .map(|_| (xorshift_next(&mut state) % 2000) as f32 / 1000.0 - 1.0)
                    .collect();
                l2_normalize(&mut v);
                v
            })
            .collect();
        let index = SemanticPartitionIndex::build(&vectors, 4, 5, 42);
        let (results, scanned) = index.search(&vectors[0], &vectors, 1, 5);
        assert!(!results.is_empty());
        assert!(scanned <= vectors.len());
        assert!(
            scanned < vectors.len(),
            "nprobe=1 of 4 partitions should scan less than the whole corpus"
        );
    }

    fn recall_at_k(candidate: &[usize], ground_truth: &[usize]) -> f32 {
        let hits = candidate
            .iter()
            .filter(|i| ground_truth.contains(i))
            .count();
        hits as f32 / ground_truth.len() as f32
    }

    /// Real end-to-end measurement (spec §89: "A custom ANN feature
    /// survives only if it improves a defined Pareto frontier"). Reuses
    /// Phase 38's committed real corpus/query fixtures (no model load
    /// needed) to measure static semantic partitioning's actual
    /// recall-vs-scan-fraction tradeoff against Phase 38's own FP32 gold
    /// ranking, and the hierarchy overlay's real module grouping.
    #[test]
    fn real_corpus_static_partitions_and_hierarchy_overlay() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let corpus: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                root.join("docs/research/qualification/raw-a-phase38-flat-corpus-embeddings.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let chunk_ids: Vec<String> = corpus["documents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["id"].as_str().unwrap().to_string())
            .collect();
        let vectors: Vec<Vec<f32>> = corpus["documents"]
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
        assert_eq!(vectors.len(), 10);

        let queries: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(
                root.join("docs/research/qualification/raw-a-phase38-flat-query-embeddings.json"),
            )
            .unwrap(),
        )
        .unwrap();

        let overlay = HierarchyOverlay::build(&chunk_ids);
        println!(
            "phase41_hierarchy modules={:?}",
            overlay
                .modules
                .iter()
                .map(|(m, ids)| (m.clone(), ids.len()))
                .collect::<std::collections::BTreeMap<_, _>>()
        );
        assert!(
            overlay.modules.len() >= 5,
            "expected the real corpus to span at least five distinct top-level modules: {:?}",
            overlay.modules.keys().collect::<Vec<_>>()
        );

        let k = 5;
        for num_partitions in [2usize, 3usize] {
            let index = SemanticPartitionIndex::build(&vectors, num_partitions, 10, 1234);
            let mut recalls_nprobe1 = Vec::new();
            let mut scan_fractions = Vec::new();
            for q in queries["queries"].as_array().unwrap() {
                let query_fp32: Vec<f32> = q["fp32"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                let ground_truth: Vec<usize> = q["fp32_ground_truth_top_k"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();

                let (hits_1, scanned_1) = index.search(&query_fp32, &vectors, 1, k);
                let ids_1: Vec<usize> = hits_1.iter().map(|(i, _)| *i).collect();
                recalls_nprobe1.push(recall_at_k(&ids_1, &ground_truth));
                scan_fractions.push(scanned_1 as f32 / vectors.len() as f32);
            }
            let mean_recall = recalls_nprobe1.iter().sum::<f32>() / recalls_nprobe1.len() as f32;
            let mean_scan_fraction =
                scan_fractions.iter().sum::<f32>() / scan_fractions.len() as f32;
            println!(
                "phase41_partitions k={num_partitions} nprobe=1 mean_recall@{k}={mean_recall} mean_scan_fraction={mean_scan_fraction}"
            );
        }
    }
}
