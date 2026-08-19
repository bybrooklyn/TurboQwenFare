// Adapted from NVMAI (https://github.com/), Apache License 2.0.
// Copied verbatim from NVMAIApp's MacPresentation/Mac.Components sources
// per spec §96 ("directly adapt selected Apache-2.0 TurboFieldfare/NVMAI
// SwiftUI source rather than visually imitating it"). No functional
// changes in this file.
//
import AppKit
import SwiftUI

public enum NVMAIMacTheme {
    public static let accentNSColor = NSColor(
        srgbRed: 106.0 / 255.0,
        green: 186.0 / 255.0,
        blue: 113.0 / 255.0,
        alpha: 1)

    public static let accentColor = Color(nsColor: accentNSColor)
}
