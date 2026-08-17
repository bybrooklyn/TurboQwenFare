# Generated artifacts

Files here are produced by code, not hand-written. Each is regeneratable —
don't hand-edit them.

## `qwen36_tensor_inventory.json`

Produced by `src/dev/inventory.rs`'s `generate_inventory` +
`write_inventory_json`, from the `#[ignore]`d test
`regenerate_committed_tensor_inventory_artifact` (run explicitly with
`cargo test -- --ignored regenerate_committed_tensor_inventory_artifact`,
not part of a normal `cargo test`).

**This committed file remains a synthetic fixture, not the real model's
inventory.** The pinned 20+ GB checkpoint has since been downloaded and its
733 descriptors pass the ignored real-inventory and installed-container
topology tests recorded in `docs/research/canonical-source-manifest.md`.
The 26 entries here are still one representative tensor per known
logical role (embedding, one full-attention layer, one Gated DeltaNet
layer, router, shared/routed experts, ...), with placeholder `[32]`-element
shapes — enough to exercise and pin the classifier's behavior, not a real
model description.

For live qualification, run
`canonical_checkpoint_inventory_matches_the_fixed_graph` with
`TQF_CANONICAL_GGUF` pointing at the pinned source. Do not replace this small
redistributable fixture with checkpoint-derived weights or other large model
payloads.
