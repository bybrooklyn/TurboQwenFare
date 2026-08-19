//! SwiftUI bridge, linked into the same binary (spec Part XI sections
//! 96-98). Plain `tqf` opens this unless `--headless` is passed.
//!
//! The real adopted-source SwiftUI app lives under `swift/` at the
//! crate root (a separate Swift package, not Rust) and is compiled by
//! `build.rs` only when the `gui` Cargo feature is enabled — see that
//! file's own comment for why this is opt-in rather than unconditional
//! even on macOS. `launch` is a no-op when the feature is off, so
//! `--headless` (which never calls it at all) and a `gui`-less build
//! (whose `launch` does nothing if called) both keep the inference core
//! completely GUI-free, matching spec §98: "Headless mode skips the
//! Swift entrypoint entirely. No Swift inference runtime is
//! duplicated."

#[cfg(feature = "gui")]
mod ffi {
    extern "C" {
        pub fn tqf_launch_gui(base_url: *const std::os::raw::c_char);
    }
}

/// Hands the process's real main thread to the compiled-in SwiftUI app
/// (spec §98: "transfer the main thread to a C-callable Swift
/// entrypoint that creates NSApplication/SwiftUI hosting"). Blocks
/// until the user quits the app. Caller contract: must be invoked from
/// the actual main thread, after server/runtime startup has already
/// spawned its own worker threads — `AppKit`'s run loop owns the
/// calling thread for the rest of the process's life.
///
/// `base_url` is TQF's own local server address (e.g.
/// `http://127.0.0.1:11535`) that the GUI's `TqfInferenceClient` calls
/// over plain HTTP — no inference logic is duplicated on the Swift
/// side (spec §98's "No Swift inference runtime is duplicated").
#[cfg(feature = "gui")]
pub fn launch(base_url: &str) {
    let c_string = std::ffi::CString::new(base_url).expect("base_url must not contain a NUL byte");
    unsafe {
        ffi::tqf_launch_gui(c_string.as_ptr());
    }
}

/// No-op without the `gui` feature: nothing here ever links or calls
/// into Swift.
#[cfg(not(feature = "gui"))]
pub fn launch(_base_url: &str) {}

#[cfg(all(test, not(feature = "gui")))]
mod tests {
    /// Real regression test for the headless-separation contract (spec
    /// §98): without the `gui` feature compiled in, `launch` must be a
    /// true no-op — it returns immediately rather than blocking (the
    /// real `AppKit` entrypoint blocks in `NSApplication.run()` until
    /// the user quits), and calling it must not panic even with an
    /// address no server is actually listening on, since a no-op
    /// implementation never inspects `base_url` at all.
    #[test]
    fn launch_without_gui_feature_returns_immediately_and_never_panics() {
        let started = std::time::Instant::now();
        super::launch("not a real url, and nothing should ever look at it");
        assert!(
            started.elapsed() < std::time::Duration::from_millis(50),
            "a gui-less launch() must return immediately, not block"
        );
    }
}
