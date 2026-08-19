// New for TQF (not derived from NVMAI): a deliberately small state
// object, not a port of NVMAI's own 942-line AppModel. NVMAI's AppModel
// owns an in-process model-load lifecycle (its own Metal runtime loads
// weights directly); TQF's model is always server-owned (spec §22's
// request flow), so the GUI only needs "is the server reachable and
// generating," not a load/unload state machine of its own.

import Foundation
import Observation

@Observable
@MainActor
public final class TqfAppModel {
    public var promptText: String = ""
    public private(set) var outputText: String = ""
    public private(set) var isGenerating: Bool = false
    public private(set) var serverReachable: Bool = false
    public var error: String?

    /// spec §47: "Create simple default conversation/setup experience
    /// and expandable engineering cockpit." The simple experience is
    /// `RootView`'s prompt/output pane, always visible; this toggle
    /// reveals `InspectorView` alongside it. Read-only concern: this
    /// flag only ever changes what's *displayed*, never anything about
    /// how the server itself runs.
    public var showsInspector: Bool = false

    public private(set) var metrics: TqfMetrics?
    private var metricsPollTask: Task<Void, Never>?

    private let client: TqfInferenceClient
    private var generationTask: Task<Void, Never>?

    public init(client: TqfInferenceClient) {
        self.client = client
    }

    public func refreshServerStatus() async {
        serverReachable = await client.healthCheck()
    }

    /// spec §47: "The inspector consumes metrics; it must not change
    /// runtime policy directly except through supported configuration
    /// actions." This method only ever *reads* — there is no
    /// corresponding "set metric"/"mutate policy" call anywhere in this
    /// model, by construction, not just by convention.
    public func startMetricsPolling(interval: Duration = .seconds(2)) {
        metricsPollTask?.cancel()
        metricsPollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.metrics = await self.client.fetchMetrics()
                try? await Task.sleep(for: interval)
            }
        }
    }

    public func stopMetricsPolling() {
        metricsPollTask?.cancel()
        metricsPollTask = nil
    }

    public var canGenerate: Bool {
        !isGenerating && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public func generate() {
        guard canGenerate else { return }
        let prompt = promptText
        outputText = ""
        error = nil
        isGenerating = true
        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in client.streamChatCompletion(prompt: prompt) {
                    self.outputText += event.textDelta
                }
            } catch is CancellationError {
                // Cancellation is a normal user action, not an error.
            } catch {
                self.error = String(describing: error)
            }
            self.isGenerating = false
        }
    }

    public func cancel() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }
}
