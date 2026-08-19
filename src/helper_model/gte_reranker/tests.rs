use std::path::Path;

use super::convert::convert_gte_reranker_safetensors;
use super::runtime::GteRerankerRuntime;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;

/// Real end-to-end qualification, same convention as Phase 37/38's
/// oracle tests: converts the actual downloaded GTE reranker
/// safetensors checkpoint into `.tqf`, runs the real tokenize ->
/// cross-encoder -> pool -> classify pipeline, and compares against
/// logits captured from the checkpoint's own official ONNX export.
#[test]
#[ignore = "requires the real gte-reranker-modernbert-base safetensors checkpoint, tokenizer.json, and a captured ONNX oracle JSON"]
fn real_checkpoint_matches_the_onnx_oracle() {
    let safetensors_path = std::env::var("TQF_GTE_SAFETENSORS").expect("set TQF_GTE_SAFETENSORS");
    let tokenizer_path = std::env::var("TQF_GTE_TOKENIZER").expect("set TQF_GTE_TOKENIZER");
    let oracle_path = std::env::var("TQF_GTE_ORACLE_JSON").expect("set TQF_GTE_ORACLE_JSON");
    let tqf_out_path = std::env::var("TQF_GTE_TQF_OUT").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("tqf-gte-reranker.tqf")
            .to_string_lossy()
            .to_string()
    });

    let report =
        convert_gte_reranker_safetensors(Path::new(&safetensors_path), Path::new(&tqf_out_path))
            .expect("conversion");
    println!(
        "gte_reranker_convert extents={} bytes={} sha256={}",
        report.extent_count,
        report.verified_output_bytes,
        report
            .source_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    let broker = MemoryBroker::new(Bytes(8 * 1024 * 1024 * 1024));
    let runtime = GteRerankerRuntime::load(
        Path::new(&tqf_out_path),
        Path::new(&tokenizer_path),
        &broker,
    )
    .expect("load runtime");

    let oracle: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&oracle_path).expect("open oracle"))
            .expect("parse oracle");
    let results = oracle["results"].as_array().unwrap();

    for entry in results {
        let query = entry["query"].as_str().unwrap();
        let doc = entry["doc"].as_str().unwrap();
        let expected_ids: Vec<u64> = entry["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let actual_ids: Vec<u64> = runtime
            .encode_pair_tokens(query, doc)
            .expect("encode")
            .into_iter()
            .map(|v| v as u64)
            .collect();
        assert_eq!(
            actual_ids, expected_ids,
            "token id mismatch for {query:?}/{doc:?}"
        );

        let expected_logit = entry["logits"][0][0].as_f64().unwrap() as f32;
        let actual_logit = runtime.score(query, doc).expect("score");
        let diff = (expected_logit - actual_logit).abs();
        println!(
            "gte_query {query:?} doc {doc:?} expected={expected_logit} actual={actual_logit} abs_diff={diff}"
        );
        assert!(
            diff < 0.05,
            "reranker logit too far from ONNX oracle: expected={expected_logit} actual={actual_logit} diff={diff}"
        );
    }
}
