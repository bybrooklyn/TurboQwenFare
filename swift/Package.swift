// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TqfGUI",
    platforms: [.macOS(.v14)],
    products: [
        // Static library: `build.rs` links this into the single `tqf`
        // Mach-O executable (spec §96 — "one binary rule").
        .library(name: "TqfGUI", type: .static, targets: ["TqfGUI"])
    ],
    targets: [
        .target(name: "TqfGUI", path: "Sources/TqfGUI")
        // A `TqfGUITests` XCTest/swift-testing target was attempted but
        // dropped: this development machine's Swift toolchain (a
        // swiftly-managed open-source toolchain, not Xcode's bundled
        // one — only Command Line Tools are installed, no full Xcode)
        // has neither `XCTest` nor `Testing` available to link against.
        // `TqfInferenceClient`'s HTTP/SSE logic is exercised indirectly
        // by `swift build` compiling cleanly and the real Rust-side
        // `cargo build --features gui`/`cargo test --features gui`
        // runs (see the Phase 46 qualification doc), not by a Swift
        // unit test suite.
    ]
)
