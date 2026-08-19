// New for TQF (not derived from NVMAI's own much larger InspectorView,
// which surfaces NVMAI-runtime-specific per-kernel timing breakdowns
// that have no TQF equivalent — spec §47's "expandable engineering
// cockpit" for TQF surfaces the real metrics TQF's own server actually
// exposes today: `/v1/tqf/metrics`'s OS-sampled process memory and
// uptime). Uses the adopted `MetricFormat` formatters for consistent
// units/precision with NVMAI's own presentation conventions.

import SwiftUI

public struct InspectorView: View {
    let metrics: TqfMetrics?

    public init(metrics: TqfMetrics?) {
        self.metrics = metrics
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Inspector")
                .font(.headline)
            Divider()
            if let metrics {
                row("Uptime", MetricFormat.seconds(Double(metrics.uptimeSeconds)))
                row("Model installed", metrics.modelInstalled ? "yes" : "no")
                row("Resident memory", MetricFormat.memory(metrics.residentBytes))
                row("Peak resident", MetricFormat.memory(metrics.residentPeakBytes))
                row("Virtual size", MetricFormat.memory(metrics.virtualBytes))
            } else {
                Text("No metrics yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(minWidth: 220, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(.caption.monospacedDigit())
        }
    }
}
