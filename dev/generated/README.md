# Generated artifacts

Files here are produced by code, not hand-written. Each is regeneratable —
don't hand-edit them.

## `qwen36_tensor_inventory.json`

Produced by `src/dev/inventory.rs`'s `generate_inventory` +
`write_inventory_json`, from the `#[ignore]`d test
`regenerate_committed_tensor_inventory_artifact` (run explicitly with
`cargo test -- --ignored regenerate_committed_tensor_inventory_artifact`,
not part of a normal `cargo test`).

**This is a synthetic fixture, not the real model's inventory.** This
environment has resolved the real canonical source pin (see
`docs/research/canonical-source-manifest.md`) but has not downloaded the
20+ GB checkpoint, so the real per-tensor GGUF names and shapes are
unconfirmed. The 26 entries here are one representative tensor per known
logical role (embedding, one full-attention layer, one Gated DeltaNet
layer, router, shared/routed experts, ...), with placeholder `[32]`-element
shapes — enough to exercise and pin the classifier's behavior, not a real
model description.

Once the real file is downloaded, re-run the generator against it
(`generate_inventory(&real_gguf_path)`); expect it to surface tensor names
this classifier doesn't recognize yet, especially for Gated DeltaNet
layers, whose real llama.cpp-convention names are the least confidently
guessed part of `src/dev/inventory.rs`'s classifier (see that file's
top-of-module doc comment).
