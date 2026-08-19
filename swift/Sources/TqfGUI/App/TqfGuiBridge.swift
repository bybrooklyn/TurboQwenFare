// New for TQF (not derived from NVMAI): the C-callable entrypoint spec
// §98 describes — "Rust startup initializes configuration/runtime/
// server worker threads, then transfers the main thread to a
// C-callable Swift entrypoint that creates NSApplication/SwiftUI
// hosting. Headless mode skips the Swift entrypoint entirely. No Swift
// inference runtime is duplicated" (this file makes zero inference
// calls — every token comes from an HTTP request to TQF's own Rust
// server via `TqfInferenceClient`).

import AppKit
import SwiftUI

/// A regular foreground app even when launched as a bare executable (no
/// `.app` bundle): Dock icon, click-to-activate, standard Quit menu.
/// Structurally the same delegate NVMAI's own `NVMAIMacApp.swift` uses
/// for the same reason, rewritten here since TQF's entry is a C
/// function, not a `@main App`.
private final class TqfAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

/// Called from Rust (`src/gui/macos/mod.rs`) after server/runtime
/// startup completes, on the process's real main thread — SwiftUI's
/// `NSApplication.run()` requires the main thread, so this function
/// must never be called from a Tokio worker thread. `base_url` is
/// TQF's own local server address (e.g. `http://127.0.0.1:11535`),
/// passed as a NUL-terminated UTF-8 C string owned by the caller for
/// the duration of this call only.
@_cdecl("tqf_launch_gui")
@MainActor
public func tqf_launch_gui(_ baseURLCString: UnsafePointer<CChar>?) {
    let urlString = baseURLCString.map { String(cString: $0) } ?? "http://127.0.0.1:11535"
    guard let baseURL = URL(string: urlString) else {
        fatalError("tqf_launch_gui: invalid base URL \(urlString)")
    }

    let app = NSApplication.shared
    let delegate = TqfAppDelegate()
    app.delegate = delegate

    let model = TqfAppModel(client: TqfInferenceClient(baseURL: baseURL))
    let hostingController = NSHostingController(rootView: RootView(model: model))
    let window = NSWindow(contentViewController: hostingController)
    window.title = "TurboQwenFare"
    window.setContentSize(NSSize(width: 900, height: 640))
    window.center()
    window.makeKeyAndOrderFront(nil)

    app.setActivationPolicy(.regular)
    app.activate()
    app.run()
}
