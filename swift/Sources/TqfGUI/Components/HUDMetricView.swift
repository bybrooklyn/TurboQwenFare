// Adapted from NVMAI (https://github.com/), Apache License 2.0.
// Copied verbatim from NVMAIApp's MacPresentation/Mac.Components sources
// per spec §96 ("directly adapt selected Apache-2.0 TurboFieldfare/NVMAI
// SwiftUI source rather than visually imitating it"). No functional
// changes in this file.
//
import SwiftUI

struct HUDMetricView: View {
    let value: String
    let label: String
    var animated = true

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .contentTransition(animated ? .numericText() : .identity)
                .animation(animated ? .snappy(duration: 0.25) : nil, value: value)
            Text(label)
                .font(.caption2)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 56)
    }
}
