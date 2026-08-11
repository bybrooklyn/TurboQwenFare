fn main() {
    // Platform/backend wiring (Metal shader compilation, CUDA kernel
    // compilation, link flags) lands here as those backends are built out
    // (spec Part VII). Nothing to do yet at the crate-skeleton stage.
    println!("cargo::rerun-if-changed=build.rs");
}
