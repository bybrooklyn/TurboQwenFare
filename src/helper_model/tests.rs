use std::io::Write;
use std::path::Path;

use serde_json::json;

use super::convert::convert_pplx_embed_safetensors;
use super::runtime::PplxEmbedRuntime;
use super::safetensors::SafetensorsFile;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;

fn write_synthetic_safetensors(path: &Path, name: &str, shape: &[u64], values: &[f32]) {
    let header = json!({
        name: {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0u64, (values.len() * 4) as u64],
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    for v in values {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}

#[test]
fn safetensors_reader_roundtrips_a_synthetic_f32_tensor() {
    let dir = std::env::temp_dir().join(format!("tqf-pplx-synth-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("synthetic.safetensors");
    let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    write_synthetic_safetensors(&path, "weight", &[3, 4], &values);

    let file = SafetensorsFile::open(&path).unwrap();
    let entry = file.entry("weight").unwrap();
    assert_eq!(entry.shape, vec![3, 4]);
    let read_back = file.read_f32("weight").unwrap();
    assert_eq!(read_back, values);
}

/// Real end-to-end qualification: converts the actual downloaded
/// pplx-embed-v1-0.6b `model.safetensors` into `.tqf`, runs the real
/// forward pass through the tokenizer/encoder/pooling/quantization
/// pipeline, and compares against embeddings captured from the model's
/// own official ONNX export (`docs/research/qualification/`'s
/// oracle-generation convention — see `raw-a-*` fixtures for the main
/// model's equivalent). Not committed as a fixture because the source
/// checkpoint is ~2.2 GiB; re-generate both files per the paths below.
#[test]
#[ignore = "requires the real pplx-embed-v1-0.6b safetensors checkpoint, tokenizer.json, and a captured ONNX oracle JSON"]
fn real_checkpoint_matches_the_onnx_oracle() {
    let safetensors_path = std::env::var("TQF_PPLX_SAFETENSORS").expect("set TQF_PPLX_SAFETENSORS");
    let tokenizer_path = std::env::var("TQF_PPLX_TOKENIZER").expect("set TQF_PPLX_TOKENIZER");
    let oracle_path = std::env::var("TQF_PPLX_ORACLE_JSON").expect("set TQF_PPLX_ORACLE_JSON");
    let tqf_out_path = std::env::var("TQF_PPLX_TQF_OUT").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("tqf-pplx-embed.tqf")
            .to_string_lossy()
            .to_string()
    });

    let report =
        convert_pplx_embed_safetensors(Path::new(&safetensors_path), Path::new(&tqf_out_path))
            .expect("conversion");
    println!(
        "pplx_embed_convert extents={} bytes={} sha256={}",
        report.extent_count,
        report.verified_output_bytes,
        report
            .source_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    let broker = MemoryBroker::new(Bytes(8 * 1024 * 1024 * 1024));
    let runtime = PplxEmbedRuntime::load(
        Path::new(&tqf_out_path),
        Path::new(&tokenizer_path),
        &broker,
    )
    .expect("load runtime");

    let oracle: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&oracle_path).expect("open oracle"))
            .expect("parse oracle");
    let texts = oracle["texts"].as_array().unwrap();
    let oracle_token_ids = oracle["token_ids"].as_array().unwrap();
    let oracle_fp32 = oracle["pooler_output"].as_array().unwrap();
    let oracle_int8 = oracle["pooler_output_int8"].as_array().unwrap();
    let oracle_binary = oracle["pooler_output_binary"].as_array().unwrap();

    for (i, text_value) in texts.iter().enumerate() {
        let text = text_value.as_str().unwrap();
        let expected_ids: Vec<u64> = oracle_token_ids[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let actual_ids: Vec<u64> = runtime
            .encode_tokens(text)
            .expect("encode")
            .into_iter()
            .map(|v| v as u64)
            .collect();
        assert_eq!(actual_ids, expected_ids, "token id mismatch for {text:?}");

        let embedding = runtime.embed(text, None).expect("embed");

        let expected_fp32: Vec<f32> = oracle_fp32[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (a, b) in embedding.fp32.iter().zip(&expected_fp32) {
            dot += (*a as f64) * (*b as f64);
            norm_a += (*a as f64) * (*a as f64);
            norm_b += (*b as f64) * (*b as f64);
        }
        let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
        println!("text {i:?} fp32 cosine similarity vs ONNX oracle: {cosine}");
        assert!(
            cosine > 0.999,
            "fp32 pooled embedding cosine similarity too low: {cosine}"
        );

        let expected_int8: Vec<i64> = oracle_int8[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let mismatches = embedding
            .int8
            .iter()
            .zip(&expected_int8)
            .filter(|(a, b)| (**a as i64 - **b).abs() > 1)
            .count();
        println!(
            "text {i:?} int8 mismatches (>1 off) out of {}: {mismatches}",
            embedding.int8.len()
        );
        assert!(
            mismatches < embedding.int8.len() / 20,
            "too many int8 quantized dims differ by more than 1: {mismatches}"
        );

        let expected_binary: Vec<f32> = oracle_binary[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let sign_mismatches = embedding
            .binary
            .iter()
            .zip(&expected_binary)
            .filter(|(a, b)| (**a > 0.0) != (**b > 0.0))
            .count();
        println!(
            "text {i:?} binary sign mismatches out of {}: {sign_mismatches}",
            embedding.binary.len()
        );
        assert!(
            sign_mismatches < embedding.binary.len() / 20,
            "too many binary sign bits differ: {sign_mismatches}"
        );
    }
}
