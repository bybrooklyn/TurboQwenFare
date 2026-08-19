// New for TQF (not derived from NVMAI): a thin HTTP client for TQF's own
// local OpenAI-compatible server (spec §96: "The Swift layer is
// intentionally thin and consumes localhost/control endpoints").

import Foundation

public enum TqfClientError: Error, Sendable {
    case serverUnreachable(String)
    case httpStatus(Int)
    case malformedStream
}

public struct TqfTokenEvent: Sendable {
    public let textDelta: String
}

/// spec §47's inspector metrics — mirrors `MetricsResponse` in
/// `src/server/tqf_api/mod.rs` field-for-field (real OS-sampled
/// process memory, not fabricated).
public struct TqfMetrics: Sendable, Decodable {
    public let uptimeSeconds: UInt64
    public let modelInstalled: Bool
    public let residentBytes: UInt64?
    public let virtualBytes: UInt64?
    public let residentPeakBytes: UInt64?

    enum CodingKeys: String, CodingKey {
        case uptimeSeconds = "uptime_seconds"
        case modelInstalled = "model_installed"
        case residentBytes = "resident_bytes"
        case virtualBytes = "virtual_bytes"
        case residentPeakBytes = "resident_peak_bytes"
    }
}

/// Streams one chat completion from TQF's real `/v1/chat/completions`
/// endpoint (`stream: true`), parsing the standard OpenAI SSE chunk
/// format (`data: {...}\n\n`, terminated by `data: [DONE]`) — the exact
/// wire format `src/server/openai/mod.rs`'s `stream_chat_completion`
/// produces, not a guessed shape.
public final class TqfInferenceClient: Sendable {
    private let baseURL: URL
    private let session: URLSession

    public init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    /// spec's `GET /health`-style check (`src/server/tqf_api`) — used by
    /// the GUI to show whether the local server is actually reachable
    /// before letting the user submit a prompt.
    public func healthCheck() async -> Bool {
        let url = baseURL.appendingPathComponent("health")
        guard let (_, response) = try? await session.data(from: url),
            let http = response as? HTTPURLResponse
        else {
            return false
        }
        return (200..<300).contains(http.statusCode)
    }

    /// Fetches spec §47's real inspector metrics from TQF's own
    /// `/v1/tqf/metrics`. Returns `nil` on any failure (server
    /// unreachable, malformed body) — the inspector treats a failed
    /// fetch as "no data yet," not an error banner, since polling
    /// failures are expected whenever the server briefly isn't up.
    public func fetchMetrics() async -> TqfMetrics? {
        let url = baseURL.appendingPathComponent("v1/tqf/metrics")
        guard let (data, response) = try? await session.data(from: url),
            let http = response as? HTTPURLResponse,
            (200..<300).contains(http.statusCode)
        else {
            return nil
        }
        return try? JSONDecoder().decode(TqfMetrics.self, from: data)
    }

    public func streamChatCompletion(prompt: String) -> AsyncThrowingStream<TqfTokenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(prompt: prompt, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        prompt: String,
        continuation: AsyncThrowingStream<TqfTokenEvent, Error>.Continuation
    ) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": "tqf",
            "stream": true,
            "messages": [["role": "user", "content": prompt]],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (bytes, response): (URLSession.AsyncBytes, URLResponse)
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch {
            throw TqfClientError.serverUnreachable(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw TqfClientError.httpStatus(code)
        }

        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst("data: ".count))
            if payload == "[DONE]" {
                continuation.finish()
                return
            }
            guard let data = payload.data(using: .utf8),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let choices = object["choices"] as? [[String: Any]],
                let first = choices.first,
                let delta = first["delta"] as? [String: Any]
            else {
                continue
            }
            if let content = delta["content"] as? String, !content.isEmpty {
                continuation.yield(TqfTokenEvent(textDelta: content))
            }
        }
        continuation.finish()
    }
}
