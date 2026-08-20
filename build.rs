use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    emit_backend_cfgs();
    build_swift_gui();
}

/// Collapses "the backend feature is enabled" and "this target can
/// actually use it" into one cfg each, so the ~30 backend-conditional
/// sites across the crate test a single, honest condition instead of
/// repeating `all(feature = "...", target_os = "...")`.
///
/// This is what lets `default = ["metal"]` stay in `Cargo.toml` while a
/// plain `cargo build`/`cargo test` still works on Linux: the feature is
/// on, but `metal-sys` is target-gated out of the dependency graph, so
/// `tqf_metal` is not set and the crate compiles against
/// `backend::reference`.
fn emit_backend_cfgs() {
    println!("cargo::rustc-check-cfg=cfg(tqf_metal)");
    println!("cargo::rustc-check-cfg=cfg(tqf_cuda)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if std::env::var("CARGO_FEATURE_METAL").is_ok() && target_os == "macos" {
        println!("cargo::rustc-cfg=tqf_metal");
    }
    // CUDA is Linux/Windows-only; it is also still a stub (spec phases 50
    // and 51 are not implemented), so this only keeps the condition
    // honest for when it is.
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() && target_os != "macos" {
        println!("cargo::rustc-cfg=tqf_cuda");
    }
}

fn build_swift_gui() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    // Real GUI compilation is opt-in behind a feature (see Cargo.toml's
    // `gui` feature) rather than unconditional even on macOS: it needs
    // a full Swift toolchain (`swift build`) present at build time,
    // which a headless-only CI/build environment may not have. Spec
    // §96's "one binary" rule is about the shipped product, not every
    // local `cargo build` invocation during development.
    if std::env::var("CARGO_FEATURE_GUI").is_err() {
        return;
    }

    println!("cargo::rerun-if-changed=swift/Package.swift");
    println!("cargo::rerun-if-changed=swift/Sources");

    let swift_dir = Path::new("swift");
    let status = Command::new("swift")
        .args(["build", "-c", "release", "--package-path"])
        .arg(swift_dir)
        .status()
        .expect("failed to invoke `swift build` — is the Swift toolchain installed?");
    assert!(status.success(), "swift build failed");

    let lib_dir = swift_dir.join(".build/release");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=TqfGUI");

    // Swift's runtime libraries live in the OS/toolchain's own
    // `usr/lib/swift` directories (confirmed via `swift build -v`'s
    // actual link invocation for a throwaway executable consuming this
    // same package — autolinking embeds the needed library names
    // directly in each object file, so no explicit `-lswiftCore` etc.
    // is needed, only the search paths and an rpath so the *runtime*
    // dylibs resolve at launch too, matching what `swift build` itself
    // passes).
    let sdk_path = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    if let Some(sdk_path) = sdk_path {
        println!("cargo:rustc-link-search=native={sdk_path}/usr/lib/swift");
    }
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    // AppKit/Foundation/SwiftUI hosting requires these frameworks to be
    // linked into the final Mach-O even though no Rust code references
    // their symbols directly.
    // `Observation` is a Swift-only module (not an Apple `.framework`
    // bundle), so it isn't in this list — it's already reachable via
    // the `-L /usr/lib/swift` search paths above.
    for framework in ["AppKit", "SwiftUI", "Foundation"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}
