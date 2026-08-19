use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

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
