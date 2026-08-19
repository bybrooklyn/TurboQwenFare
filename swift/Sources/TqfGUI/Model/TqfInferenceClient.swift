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
