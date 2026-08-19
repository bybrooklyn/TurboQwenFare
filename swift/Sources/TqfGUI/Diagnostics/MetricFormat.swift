// Adapted from NVMAI (https://github.com/), Apache License 2.0.
// Copied verbatim from NVMAIApp's Mac/Diagnostics sources per spec §96
// ("directly adapt selected Apache-2.0 TurboFieldfare/NVMAI SwiftUI
// source rather than visually imitating it"). No functional changes,
// except `enum MetricFormat` -> `public enum MetricFormat` (and its
// static methods -> public) so it's usable from outside this file
// across module boundaries, since NVMAI's original was internal-only
// within its own app target.
//
import Foundation

@MainActor
public enum MetricFormat {
    // D26: fixed POSIX locale so decimal separators and grouping do not vary
    // with the user's region settings.
    private static let posixLocale = Locale(identifier: "en_US_POSIX")

    private static let memoryFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter
    }()

    private static let storageFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useMB, .useGB, .useTB]
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter
    }()

    public static func seconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        if value < 1 {
            return String(format: "%.0f ms", locale: posixLocale, value * 1000)
        }
        return String(format: "%.2f s", locale: posixLocale, value)
    }

    public static func milliseconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        return "\(value.formatted(.number.precision(.fractionLength(1)))) ms"
    }

    public static func rate(_ value: Double) -> String {
        String(format: "%.1f", locale: posixLocale, value)
    }

    public static func percent(_ value: Double) -> String {
        String(format: "%.1f%%", locale: posixLocale, value)
    }

    public static func perToken(_ value: Double) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1))))/tok"
    }

    public static func megabytesPerToken(_ value: Double) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1)))) MB/tok"
    }

    public static func memory(_ bytes: UInt64?) -> String {
        guard let bytes else { return "\u{2014}" }
        return memoryFormatter.string(fromByteCount: Int64(bytes))
    }

    public static func storage(_ bytes: UInt64) -> String {
        storageFormatter.string(fromByteCount: Int64(clamping: bytes))
    }

}
