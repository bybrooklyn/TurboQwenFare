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

    private let client: TqfInferenceClient
    private var generationTask: Task<Void, Never>?

    public init(client: TqfInferenceClient) {
        self.client = client
    }

    public func refreshServerStatus() async {
        serverReachable = await client.healthCheck()
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
