# Phase 48: vision encoder

Spec Phase 48 deliverable (spec Part XIII, phase 48; Part I section 1,
`--enable-vision`). "Repack/lazy-load vision artifact, protocol
mapping, memory planner." The real `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf`
sidecar (`source::pinned::VISION_PROJECTOR_FILENAME`, ~614 MB) was
already pinned by an earlier phase but never read; this phase
downloads it, reverse-engineers its real architecture from the
checkpoint itself and from real llama.cpp source, implements the full
forward pass from scratch in Rust, and validates it against a real
independent oracle.

## What was built

**Real architecture facts, not assumed ones.** `general.architecture =
clip`, `clip.projector_type = qwen3vl_merger` in the real GGUF
metadata (read directly with Python's `gguf` library): a CLIP-style
Vision Transformer (1152 hidden, 16 heads, 27 layers, 4304
feed-forward, patch size 16, native 768x768/48x48-patch position
table, 2x2 spatial merge) feeding Qwen3-VL's `qwen3vl_merger`
projector (FC1 4608->4608, GELU, FC2 4608->2048). Every real GGUF
tensor name was inventoried directly (`v.patch_embd.{weight,weight.1,
bias}`, `v.position_embd.weight`, `v.blk.N.{ln1,ln2,attn_qkv,attn_out,
ffn_up,ffn_down}.{weight,bias}`, `v.post_ln.{weight,bias}`,
`mm.{0,2}.{weight,bias}`) rather than guessed from convention —
`src/vision/roles.rs`'s `VisionTensorRole::gguf_name` documents each.

**The graph itself was read from real llama.cpp source, not
reimplemented from memory of "how ViTs usually work."**
`tools/mtmd/models/qwen3vl.cpp`'s `clip_graph_qwen3vl::build()` and its
parent `clip_graph_qwen2vl::build_inp_with_temporal_merge` (a locally
patched llama.cpp checkout, read but never linked into this crate)
revealed three architectural specifics that would have been wrong by
default assumption:

1. **Patch embedding is two independent conv kernels summed over the
   same still-image input** (`v.patch_embd.weight` and `.weight.1`),
   not one — mathematically folded into one pre-summed kernel in
   `forward.rs::combined_patch_kernel` since convolution is linear in
   the kernel for a fixed input.
2. **Patches are reordered into 2x2-merge-block-major order
   immediately after the conv**, before the bias add, before the
   position-embedding add, and before every one of the 27 transformer
   layers — not raster order until a final reshape, as a naive port
   would assume. Confirmed two independent ways: `build()`'s own
   permute/reshape chain, and the real per-axis M-RoPE `positions`
   array construction (`case PROJECTOR_TYPE_QWEN3VL` in `clip.cpp`),
   which iterates patches in exactly this same block order.
   `forward.rs::reorder_to_merge_blocks` implements it, unit-tested
   directly (`vision::tests::merge_block_reorder_groups_2x2_patches_
   row_major`).
3. **Attention uses 2D vision M-RoPE** (`GGML_ROPE_TYPE_VISION`) on top
   of the absolute position embedding, not one or the other. The exact
   frequency/pairing scheme (first 18 of 36 rotation pairs use row
   position, the next 18 use column position, together consuming the
   *entire* 72-dim head — no NEOX-style pass-through tail) was derived
   directly from `ggml-cpu/ops.cpp`'s `ggml_mrope_cache_init`/
   `rotate_pairs`, not assumed; `forward.rs::apply_vision_rope`
   documents the derivation. The absolute position table itself is
   bilinear-resized from its native 48x48 grid with `align_corners`
   semantics (`clip.cpp`'s `resize_position_embeddings`, `ggml-cpu/
   ops.cpp`'s plain `GGML_SCALE_MODE_BILINEAR` branch) — implemented in
   `forward.rs::resize_position_embeddings`.

**The rest of the pipeline** (`src/vision/`): `geometry.rs` (every
constant cross-checked against the real GGUF metadata), `convert.rs`
(GGUF -> `.tqf` F32-passthrough repack, same convention as
`helper_model::convert`/`helper_model::gte_reranker::convert`, dtype
dispatch for the checkpoint's real mixed F32/F16/Q8_0 tensors),
`weights.rs` (loads the `.tqf` under a transient `MemoryOwner::
HelperModel`/`MemoryClass::Transient` broker lease — same pattern
Phase 37/43 established, so a text-only session that never sets
`--enable-vision` pays nothing), `forward.rs` (the full 27-layer
encoder + post-LN + merger), `runtime.rs` (ties load+encode behind one
call).

## Measured evidence

**Real oracle, not self-consistency.** Built `llama-mtmd-debug` from
the same locally patched llama.cpp checkout and ran it against the
real pinned Qwen3.6 LM checkpoint plus the real downloaded mmproj file
with a synthetic 96x96 "gray" test image
(`-p encode --image gray -n 96`), capturing the *complete* intermediate
trace (every op, not just the final output — the first attempt was
accidentally truncated by a `| tail -60` pipe and had to be rerun
without it). One real, non-obvious finding while building the test
harness: `llama-mtmd-debug`'s "gray" fixture feeds pixel value `0.5`
directly into the graph as the *already-normalized* model input
(`mtmd_debug_encode_image` in `tools/mtmd/mtmd.cpp` calls
`clip_image_f32::cpy_buf` straight from the raw `0.5`-filled buffer,
bypassing the normal `(raw - IMAGE_MEAN) / IMAGE_STD` preprocessing
entirely) — not `0.5` as a *raw* pixel that then normalizes to `0.0`.
This was caught, not assumed: a first test run using an all-zero input
tensor produced a `patch_bias` stage matching this crate's own loaded
`patch_embd.bias` tensor exactly (verified independently via Python's
`gguf` library) but not the oracle's `patch_bias` trace value — proving
the *bias tensor* was loaded correctly and the *test input* was wrong,
not the model logic.

With the corrected `0.5`-filled input, `vision::tests::
real_checkpoint_matches_the_llama_cpp_oracle` converts the real
checkpoint, runs the real encode pipeline, and matches the oracle's
real captured values at **every stage of the full 27-layer pipeline**,
bisected stage by stage during debugging (temporary instrumentation,
removed after validation) against the oracle's own intermediate
trace:

| Stage | Oracle sum | TQF sum |
|---|---|---|
| patch_bias (post-conv+bias, pre-position) | 544.8974 | 544.9025 |
| inp_pos_emb (post-position-embed) | 551.2897 | 551.2910 |
| layer_out-0 | 1079.1008 | 1079.0306 |
| layer_out-26 | 111449.375 | 111413.992 |
| norm_b-27 (post-LN) | -4.1833 | -4.1625 |
| ffn_up_b (merger FC1+bias) | -534.5928 | -536.4924 |
| ffn_gelu (merger GELU) | -204.3838 | -204.9383 |
| node_822 (final, FC2+bias) | 17.6264 | 17.7014 |

Agreement holds to within ordinary float non-associativity at every
checkpoint (same class of finding as `raw-a-512-divergence-
investigation.md` and Phase 27's TQKV divergence — accumulated
differences in summation order across a 27-layer, Q8_0-dequantizing
pipeline, not a structural defect), and the final per-token embedding
values match closely too (token 0's first three values: oracle
`[-0.0791, -0.0221, -0.0972]` vs TQF `[-0.0795, -0.0219, -0.0978]`).

**Zero regressions.** `cargo build`, `cargo test` (413 passing, +6 over
Phase 47's 407 for the new vision module's structural/geometry tests),
`cargo clippy`, and `cargo fmt --check` all stay clean. The one real
`#[ignore]`d checkpoint-gated oracle test
(`TQF_VISION_MMPROJ=/path/to/mmproj.gguf cargo test --release
vision::tests::real_checkpoint_matches_the_llama_cpp_oracle --
--ignored --nocapture`) passes against the real pinned checkpoint.

## Status and remaining work

- **Not wired into `--enable-vision`, the CLI flag, or the OpenAI
  multimodal content-part protocol mapping** (`vision: Vec<VisionInput>`
  already exists as a struct hint in `src/runtime/request.rs` from an
  earlier phase, unused). This phase delivers the runtime and its real
  oracle-validated qualification, not the end-to-end HTTP route — same
  scope boundary Phase 37/43 drew for their own helper models.
- **No real image decode/preprocessing** (JPEG/PNG -> resized, mean/std
  normalized float tensor). The oracle's own synthetic-fixture
  methodology (`--image gray`) sidesteps this by feeding pixel value
  `0.5` directly, so this phase's validation covers the *encoder graph*
  exactly but not a real-photo preprocessing path.
- **No memory-broker capacity planning test** for the lazy vision
  tower's footprint under a live 4G/2G broker (a real allocation *does*
  happen per-tensor via `MemoryOwner::HelperModel`/
  `MemoryClass::Transient` leases in `weights.rs`, but no test proves
  text-only steady-state memory is unaffected when `--enable-vision` is
  never set at the process level, since there's no CLI flag yet to
  gate on).
- Naive `O(patches × 256 × 3 × 1152)` scalar convolution and
  `O(seq_len²)` scalar attention — reference-tier correctness, not
  performance; consistent with this crate's established pattern of a
  correct scalar reference path before any SIMD/GPU optimization pass.
