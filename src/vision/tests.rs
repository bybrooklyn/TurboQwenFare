use std::path::Path;

use super::convert::convert_vision_gguf;
use super::runtime::VisionRuntime;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;

/// Real end-to-end qualification, same convention as Phase 37/43's
/// oracle tests: converts the actual pinned mmproj GGUF into `.tqf`,
/// runs the real patch-embed -> 27-layer ViT -> merger pipeline on a
/// synthetic 96x96 all-`0.5` ("gray") image — which normalizes to
/// exactly zero pixels since `IMAGE_MEAN == IMAGE_STD == 0.5` — and
/// compares the final projected embeddings against sums/values captured
/// from a real `llama-mtmd-debug` run
/// (`-p encode --image gray -n 96`) against the same checkpoint.
#[test]
#[ignore = "requires the real pinned mmproj-Qwen3.6-35B-A3B-Q8_0.gguf checkpoint"]
fn real_checkpoint_matches_the_llama_cpp_oracle() {
    let mmproj_path = std::env::var("TQF_VISION_MMPROJ").expect("set TQF_VISION_MMPROJ");
    let tqf_out_path = std::env::var("TQF_VISION_TQF_OUT").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("tqf-vision.tqf")
            .to_string_lossy()
            .to_string()
    });

    let convert_broker = MemoryBroker::new(Bytes(1024 * 1024 * 1024));
    let report = convert_vision_gguf(
        Path::new(&mmproj_path),
        Path::new(&tqf_out_path),
        &convert_broker,
    )
    .expect("conversion");
    println!(
        "vision_convert extents={} bytes={} sha256={}",
        report.extent_count,
        report.verified_output_bytes,
        report
            .source_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    let broker = MemoryBroker::new(Bytes(4 * 1024 * 1024 * 1024));
    let runtime = VisionRuntime::load(Path::new(&tqf_out_path), &broker).expect("load");

    // "gray" fill = 0.5 per channel, fed as the *already-normalized*
    // model-ready tensor: `mtmd_debug_encode_image` (real llama.cpp
    // source, `tools/mtmd/mtmd.cpp`) copies this buffer straight into
    // `clip_image_f32` via `cpy_buf`, bypassing the normal raw-pixel
    // `(val - IMAGE_MEAN) / IMAGE_STD` preprocessing entirely — so the
    // real oracle's conv input is literally 0.5 everywhere, not 0.0
    // (confirmed by first computing patch_bias with an all-zero image,
    // getting exactly `patch_embd.bias` broadcast — sum 475.75, not the
    // oracle's 544.90 — then finding this debug-path bypass in
    // `mtmd.cpp`).
    let image = vec![0.5f32; 96 * 96 * 3];
    let merged = runtime.encode(&image, 96, 96);

    assert_eq!(merged.len(), 9, "6x6 patch grid / 2x2 merge = 9 tokens");
    assert_eq!(merged[0].len(), 2048);

    let total_sum: f64 = merged.iter().flatten().map(|&v| v as f64).sum();
    println!("total_sum={total_sum}");
    assert!(
        (total_sum - 17.626368).abs() < 0.5,
        "total sum {total_sum} too far from the real oracle's 17.626368"
    );

    let first_row = &merged[0];
    let expected_first = [-0.0791, -0.0221, -0.0972];
    let expected_last = [0.0007, -0.0624, -0.0724];
    for (got, want) in first_row[..3].iter().zip(expected_first) {
        assert!((got - want).abs() < 0.02, "got {got} want {want}");
    }
    for (got, want) in first_row[first_row.len() - 3..].iter().zip(expected_last) {
        assert!((got - want).abs() < 0.02, "got {got} want {want}");
    }
}

/// Structural (no real checkpoint required): a synthetic-fixture-sized
/// vision tower would need real weights to encode anything meaningful,
/// but the reorder/geometry helpers are pure functions and worth
/// unit-testing directly.
#[test]
fn merge_block_reorder_groups_2x2_patches_row_major() {
    use super::forward::reorder_to_merge_blocks;
    // 4x4 raster grid -> 4 merge blocks of 4 patches each.
    let raster: Vec<Vec<f32>> = (0..16).map(|i| vec![i as f32]).collect();
    let (reordered, coords) = reorder_to_merge_blocks(&raster, 4, 4);
    // First block (by=0,bx=0): patches at (0,0),(0,1),(1,0),(1,1) = raster
    // indices 0,1,4,5.
    assert_eq!(reordered[0][0], 0.0);
    assert_eq!(reordered[1][0], 1.0);
    assert_eq!(reordered[2][0], 4.0);
    assert_eq!(reordered[3][0], 5.0);
    assert_eq!(coords[0], (0, 0));
    assert_eq!(coords[1], (0, 1));
    assert_eq!(coords[2], (1, 0));
    assert_eq!(coords[3], (1, 1));
}
