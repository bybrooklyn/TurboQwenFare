//! Protocol conformance fixtures (spec §260, and §331 which assigns them
//! to Phase 2b).
//!
//! **These fixtures are written from the specification, never from the
//! implementation's behavior.** That distinction is the entire point of
//! the module, and §331 states it as a rule because this project has
//! already been burned by the alternative: the pre-existing suite
//! contained a test asserting that `temperature: 0.7` returns HTTP 400.
//! It passed. Its name encoded the runtime's limitation as though it were
//! the requirement, so the test suite agreed with the server while the
//! server disagreed with §204 and with every real client.
//!
//! A fixture derived from current behavior cannot fail. So each fixture
//! here carries the spec section it encodes, and the reviewer's question
//! is "does this assertion follow from that section?" rather than "does
//! this pass?".
//!
//! A limitation is expressed as a fixture asserting the documented
//! *rejection* (§204: "reject rather than silently ignore"), never as a
//! fixture asserting the limitation is correct.

use std::net::SocketAddr;

use serde_json::Value;

use super::tests::{http_request, post_json, FixtureGenerator};

/// One conformance case: a spec citation, a request, and what that
/// section requires of the response.
struct Fixture {
    /// The section this encodes. Every fixture must cite one; a fixture
    /// that cannot name its source is by definition derived from the code.
    spec: &'static str,
    name: &'static str,
    method: Method,
    path: &'static str,
    body: &'static str,
    expect: &'static [Expect],
}

#[derive(Clone, Copy)]
enum Method {
    Get,
    Post,
}

enum Expect {
    Status(u16),
    /// A dotted path must be present and non-null, e.g. `choices.0.message`.
    Field(&'static str),
    /// A dotted path must equal this JSON literal.
    FieldEq(&'static str, &'static str),
    /// A dotted path must be absent or null.
    NoField(&'static str),
    Header(&'static str, &'static str),
    BodyContains(&'static str),
    BodyLacks(&'static str),
    /// Newline-delimited JSON: every non-empty line parses as a bare JSON
    /// object, with no SSE framing anywhere (spec §210, §333).
    NdjsonFraming,
    /// Server-sent events: `data:` payloads terminated by `[DONE]`.
    SseFraming,
    /// Every NDJSON line carries this field.
    EveryLineHas(&'static str),
    /// The last NDJSON line's field equals this literal.
    LastLineEq(&'static str, &'static str),
    /// At least this many NDJSON lines / SSE `data:` payloads.
    MinLines(usize),
}

// ---------------------------------------------------------------- helpers

fn split_response(raw: &str) -> (&str, String) {
    let (headers, body) = raw.split_once("\r\n\r\n").unwrap_or((raw, ""));
    // Streaming responses arrive with `Transfer-Encoding: chunked`, whose
    // hex length prefixes are transport framing, not payload. `curl`
    // strips them; this raw-socket harness has to do it itself, or every
    // chunk header shows up as a bogus NDJSON line.
    if headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        (headers, dechunk(body))
    } else {
        (headers, body.to_string())
    }
}

fn dechunk(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    loop {
        let Some((size_line, remainder)) = rest.split_once("\r\n") else {
            break;
        };
        let size =
            usize::from_str_radix(size_line.trim().split(';').next().unwrap_or("").trim(), 16);
        let Ok(size) = size else { break };
        if size == 0 || remainder.len() < size {
            break;
        }
        out.push_str(&remainder[..size]);
        rest = remainder[size..].trim_start_matches("\r\n");
    }
    out
}

fn status_code(raw: &str) -> u16 {
    raw.split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0)
}

/// Resolves `a.b.0.c` against a JSON value.
fn dotted<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        current = match segment.parse::<usize>() {
            Ok(index) => current.get(index)?,
            Err(_) => current.get(segment)?,
        };
    }
    Some(current)
}

/// Non-empty payload lines: NDJSON lines, or the payloads of SSE `data:`
/// frames with the `[DONE]` sentinel removed.
fn payload_lines(body: &str) -> Vec<&str> {
    if body.contains("data: ") {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| !line.trim().is_empty() && *line != "[DONE]")
            .collect()
    } else {
        body.lines()
            .filter(|line| !line.trim().is_empty())
            .collect()
    }
}

fn check(fixture: &Fixture, raw: &str) -> Vec<String> {
    let (headers, body) = split_response(raw);
    let body = body.as_str();
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let mut failures = Vec::new();
    let mut fail = |message: String| failures.push(message);

    for expectation in fixture.expect {
        match expectation {
            Expect::Status(want) => {
                let got = status_code(raw);
                if got != *want {
                    fail(format!("expected status {want}, got {got}"));
                }
            }
            Expect::Field(path) => match parsed.as_ref().and_then(|v| dotted(v, path)) {
                Some(Value::Null) | None => fail(format!("missing required field `{path}`")),
                Some(_) => {}
            },
            Expect::FieldEq(path, want) => {
                let want: Value = serde_json::from_str(want).expect("fixture literal must be JSON");
                match parsed.as_ref().and_then(|v| dotted(v, path)) {
                    Some(got) if *got == want => {}
                    Some(got) => fail(format!("`{path}`: expected {want}, got {got}")),
                    None => fail(format!("missing field `{path}` (expected {want})")),
                }
            }
            Expect::NoField(path) => {
                if let Some(got) = parsed.as_ref().and_then(|v| dotted(v, path)) {
                    if !got.is_null() {
                        fail(format!("`{path}` must be absent, got {got}"));
                    }
                }
            }
            Expect::Header(name, want) => {
                let found = headers.lines().any(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower.starts_with(&name.to_ascii_lowercase())
                        && lower.contains(&want.to_ascii_lowercase())
                });
                if !found {
                    fail(format!("header `{name}` must contain `{want}`"));
                }
            }
            Expect::BodyContains(needle) => {
                if !body.contains(needle) {
                    fail(format!("body must contain `{needle}`"));
                }
            }
            Expect::BodyLacks(needle) => {
                if body.contains(needle) {
                    fail(format!("body must not contain `{needle}`"));
                }
            }
            Expect::NdjsonFraming => {
                if body.contains("data: ") {
                    fail("NDJSON must not use SSE `data:` framing".to_string());
                }
                if body.contains("[DONE]") {
                    fail("NDJSON must not carry an SSE `[DONE]` sentinel".to_string());
                }
                for line in payload_lines(body) {
                    match serde_json::from_str::<Value>(line) {
                        Ok(value) if value.is_object() => {}
                        _ => fail(format!("NDJSON line is not a bare JSON object: {line}")),
                    }
                }
            }
            Expect::SseFraming => {
                if !body.contains("data: ") {
                    fail("SSE responses must use `data:` framing".to_string());
                }
                if !body.contains("[DONE]") {
                    fail("SSE responses must terminate with `[DONE]`".to_string());
                }
            }
            Expect::EveryLineHas(path) => {
                for line in payload_lines(body) {
                    let value: Value = match serde_json::from_str(line) {
                        Ok(value) => value,
                        Err(_) => continue,
                    };
                    if dotted(&value, path).is_none() {
                        fail(format!("line missing `{path}`: {line}"));
                    }
                }
            }
            Expect::LastLineEq(path, want) => {
                let want: Value = serde_json::from_str(want).expect("fixture literal must be JSON");
                match payload_lines(body).last() {
                    Some(line) => {
                        let value: Value = serde_json::from_str(line).unwrap_or(Value::Null);
                        match dotted(&value, path) {
                            Some(got) if *got == want => {}
                            Some(got) => {
                                fail(format!("last line `{path}`: expected {want}, got {got}"))
                            }
                            None => fail(format!("last line missing `{path}`: {line}")),
                        }
                    }
                    None => fail("response had no payload lines".to_string()),
                }
            }
            Expect::MinLines(min) => {
                let count = payload_lines(body).len();
                if count < *min {
                    fail(format!(
                        "expected at least {min} payload lines, got {count}"
                    ));
                }
            }
        }
    }
    failures
}

async fn run(fixtures: &[Fixture]) {
    // A realistic installed state: `serve.rs` always supplies a receipt
    // alongside a loaded generator, so a harness with `model_installed`
    // and no receipt would test a combination that cannot occur and would
    // 404 the inventory endpoints for the wrong reason.
    let addr: SocketAddr =
        super::tests::spawn_test_server_installed(std::sync::Arc::new(FixtureGenerator)).await;

    let mut report = Vec::new();
    for fixture in fixtures {
        let request = match fixture.method {
            Method::Get => super::tests::get(fixture.path),
            Method::Post => post_json(fixture.path, fixture.body),
        };
        let raw = http_request(addr, &request).await;
        let failures = check(fixture, &raw);
        if !failures.is_empty() {
            let (_, body) = split_response(&raw);
            let preview: String = body.chars().take(400).collect();
            report.push(format!(
                "  [{}] {}\n{}\n      --- response ---\n      {}",
                fixture.spec,
                fixture.name,
                failures
                    .iter()
                    .map(|f| format!("      {f}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                preview.replace('\n', "\n      ")
            ));
        }
    }

    assert!(
        report.is_empty(),
        "\n{} conformance failure(s):\n{}\n",
        report.len(),
        report.join("\n")
    );
}

// =====================================================================
// Fixtures. Each cites the section it encodes; the review question is
// "does this follow from that section?", not "does this pass?".
// =====================================================================

/// §70: OpenAI Chat Completions, Responses, embeddings, model listing.
/// §212: OpenAI surfaces return OpenAI-like errors.
const OPENAI: &[Fixture] = &[
    Fixture {
        spec: "§70",
        name: "GET /v1/models exposes the canonical model id",
        method: Method::Get,
        path: "/v1/models",
        body: "",
        expect: &[
            Expect::Status(200),
            Expect::FieldEq("object", r#""list""#),
            Expect::Field("data.0.id"),
            Expect::Field("data.0.created"),
            Expect::FieldEq("data.0.object", r#""model""#),
        ],
    },
    Fixture {
        spec: "§70",
        name: "chat completion carries id, created, usage, and a choice",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(200),
            Expect::Field("id"),
            Expect::Field("created"),
            Expect::FieldEq("object", r#""chat.completion""#),
            Expect::Field("choices.0.message.role"),
            Expect::Field("choices.0.finish_reason"),
            Expect::Field("usage.prompt_tokens"),
            Expect::Field("usage.completion_tokens"),
            Expect::Field("usage.total_tokens"),
        ],
    },
    // §204: parameters that ARE implemented must be accepted. Asserting a
    // 400 here would be the exact inversion §331 warns about.
    Fixture {
        spec: "§153/§204",
        name: "sampling parameters real clients send are accepted",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.8,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":42,"max_tokens":512,"stop":["\n\n"],"frequency_penalty":0.2,"presence_penalty":0.2}"#,
        expect: &[Expect::Status(200), Expect::Field("choices.0.message")],
    },
    // §204: reject rather than silently ignore. These are real limits, so
    // the fixture asserts the documented rejection.
    Fixture {
        spec: "§204",
        name: "n>1 is rejected, not silently reduced to one sequence",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#,
        expect: &[
            Expect::Status(400),
            Expect::Field("error.message"),
            Expect::FieldEq("error.type", r#""invalid_request_error""#),
        ],
    },
    Fixture {
        spec: "§204",
        name: "logprobs is rejected rather than approximated",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
        expect: &[Expect::Status(400), Expect::Field("error.message")],
    },
    Fixture {
        spec: "§203",
        name: "an Ollama-style tag resolves to the canonical model",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[Expect::Status(200), Expect::Field("choices.0.message")],
    },
    Fixture {
        spec: "§203/§212",
        name: "a genuinely unknown model is an OpenAI-shaped 400",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(400),
            Expect::Field("error.message"),
            Expect::FieldEq("error.type", r#""invalid_request_error""#),
        ],
    },
    Fixture {
        spec: "§70/§71",
        name: "streamed chat completion uses SSE and terminates with [DONE]",
        method: Method::Post,
        path: "/v1/chat/completions",
        body: r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        expect: &[
            Expect::Status(200),
            Expect::Header("content-type", "text/event-stream"),
            Expect::SseFraming,
            Expect::BodyContains("chat.completion.chunk"),
        ],
    },
    Fixture {
        spec: "§207",
        name: "Responses streaming uses typed events, not chat chunks",
        method: Method::Post,
        path: "/v1/responses",
        body: r#"{"input":"hi","stream":true}"#,
        expect: &[
            Expect::Status(200),
            Expect::BodyContains("response.created"),
            Expect::BodyContains("response.output_text.delta"),
            Expect::BodyContains("response.completed"),
            Expect::BodyLacks("chat.completion.chunk"),
        ],
    },
    // §86: the embedding surface is served only when an embedding model
    // exists. §335: an honest 501 naming the gap, never a silent stub.
    Fixture {
        spec: "§86/§335",
        name: "embeddings report the missing checkpoint rather than faking one",
        method: Method::Post,
        path: "/v1/embeddings",
        body: r#"{"input":"hello"}"#,
        expect: &[Expect::Status(501), Expect::Field("error.message")],
    },
];

/// §73 and §210, plus the framing details §333 calls out as the ones that
/// break real clients while `curl` still looks correct.
const OLLAMA: &[Fixture] = &[
    Fixture {
        spec: "§333",
        name: "GET / answers the liveness string clients probe first",
        method: Method::Get,
        path: "/",
        body: "",
        expect: &[
            Expect::Status(200),
            Expect::BodyContains("Ollama is running"),
        ],
    },
    Fixture {
        spec: "§333",
        name: "GET /api/version answers before any credential is presented",
        method: Method::Get,
        path: "/api/version",
        body: "",
        expect: &[Expect::Status(200), Expect::Field("version")],
    },
    Fixture {
        spec: "§210",
        name: "/api/tags reports a models array",
        method: Method::Get,
        path: "/api/tags",
        body: "",
        expect: &[Expect::Status(200), Expect::Field("models")],
    },
    Fixture {
        spec: "§210",
        name: "/api/ps reports running-model state",
        method: Method::Get,
        path: "/api/ps",
        body: "",
        expect: &[Expect::Status(200), Expect::Field("models")],
    },
    Fixture {
        spec: "§210",
        name: "/api/show returns model details for an Ollama-style tag",
        method: Method::Post,
        path: "/api/show",
        body: r#"{"model":"qwen3.6:35b"}"#,
        expect: &[
            Expect::Status(200),
            Expect::Field("details"),
            Expect::Field("model_info"),
            Expect::Field("capabilities"),
        ],
    },
    // Ollama nests the assistant turn under `message`, unlike OpenAI's
    // `choices[]`. A client reading `choices` gets nothing from Ollama and
    // vice versa, so the shape is load-bearing.
    Fixture {
        spec: "§210",
        name: "non-streaming /api/chat nests message{role,content} with done+timings",
        method: Method::Post,
        path: "/api/chat",
        body: r#"{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(200),
            Expect::Field("model"),
            Expect::Field("created_at"),
            Expect::Field("message.role"),
            Expect::FieldEq("done", "true"),
            Expect::Field("done_reason"),
            Expect::Field("total_duration"),
            Expect::Field("prompt_eval_count"),
            Expect::Field("eval_count"),
            Expect::Field("eval_duration"),
            Expect::NoField("choices"),
        ],
    },
    // The four framing details from §333, as one fixture: NDJSON framing,
    // bare JSON per line, and the terminal `done:true` object clients
    // block on rather than waiting for the socket to close.
    Fixture {
        spec: "§333",
        name: "streaming /api/chat is NDJSON with a terminal done object",
        method: Method::Post,
        path: "/api/chat",
        body: r#"{"model":"qwen3.6:35b","messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(200),
            Expect::Header("content-type", "application/x-ndjson"),
            Expect::NdjsonFraming,
            Expect::EveryLineHas("done"),
            Expect::LastLineEq("done", "true"),
            Expect::LastLineEq("done_reason", r#""stop""#),
        ],
    },
    // §333: `stream` defaults to true here, the opposite of OpenAI. The
    // request above omits it deliberately and must stream.
    Fixture {
        spec: "§333",
        name: "/api/generate uses `response`, not `message`",
        method: Method::Post,
        path: "/api/generate",
        body: r#"{"model":"qwen3.6:35b","prompt":"hi","stream":false}"#,
        expect: &[
            Expect::Status(200),
            Expect::Field("response"),
            Expect::FieldEq("done", "true"),
            Expect::NoField("message"),
        ],
    },
    Fixture {
        spec: "§204/§333",
        name: "a parameter that cannot be honored is rejected, not ignored",
        method: Method::Post,
        path: "/api/generate",
        body: r#"{"model":"qwen3.6:35b","prompt":"hi","raw":true,"stream":false}"#,
        expect: &[Expect::Status(400), Expect::Field("error")],
    },
    // Ollama ships mirostat:0 / tfs_z:1.0 / typical_p:1.0 as defaults —
    // the no-op values. Rejecting those would 400 half the ecosystem.
    Fixture {
        spec: "§333",
        name: "Ollama's own no-op defaults are accepted",
        method: Method::Post,
        path: "/api/chat",
        body: r#"{"model":"qwen3.6:35b","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"mirostat":0,"tfs_z":1.0,"typical_p":1.0,"temperature":0.8,"top_k":40}}"#,
        expect: &[Expect::Status(200), Expect::FieldEq("done", "true")],
    },
    // §210 says model-management endpoints are not required. §335 says an
    // honest error naming why beats an anonymous 404.
    Fixture {
        spec: "§210/§335",
        name: "model management is an explaining 501, not an anonymous 404",
        method: Method::Post,
        path: "/api/pull",
        body: r#"{"name":"llama3"}"#,
        expect: &[Expect::Status(501), Expect::Field("error")],
    },
    // §212: Ollama surfaces use Ollama's flat {"error": "..."} envelope,
    // not OpenAI's nested {"error":{"message":...}}.
    Fixture {
        spec: "§212",
        name: "Ollama errors use Ollama's flat envelope",
        method: Method::Post,
        path: "/api/chat",
        body: r#"{"model":"definitely-not-installed","stream":false,"messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(400),
            Expect::Field("error"),
            Expect::NoField("error.message"),
        ],
    },
];

/// §72: an Anthropic Messages facade sufficient for Claude Code.
/// §212: Anthropic surfaces return Anthropic-like errors.
const ANTHROPIC: &[Fixture] = &[
    Fixture {
        spec: "§72",
        name: "messages returns Anthropic's content-block shape",
        method: Method::Post,
        path: "/v1/messages",
        body: r#"{"model":"qwen3.6-35b-a3b","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(200),
            Expect::FieldEq("type", r#""message""#),
            Expect::FieldEq("role", r#""assistant""#),
            Expect::Field("content.0.type"),
            Expect::Field("stop_reason"),
            Expect::Field("usage.input_tokens"),
            Expect::Field("usage.output_tokens"),
            Expect::NoField("choices"),
        ],
    },
    // Anthropic's own API requires max_tokens; matching that keeps client
    // error handling identical against either endpoint.
    Fixture {
        spec: "§72",
        name: "absent max_tokens is rejected as the real API does",
        method: Method::Post,
        path: "/v1/messages",
        body: r#"{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(400),
            Expect::FieldEq("type", r#""error""#),
            Expect::Field("error.type"),
            Expect::Field("error.message"),
        ],
    },
    Fixture {
        spec: "§72",
        name: "count_tokens answers without running a generation",
        method: Method::Post,
        path: "/v1/messages/count_tokens",
        body: r#"{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[Expect::Status(200), Expect::Field("input_tokens")],
    },
    Fixture {
        spec: "§72/§71",
        name: "streaming uses Anthropic's own event names",
        method: Method::Post,
        path: "/v1/messages",
        body: r#"{"model":"qwen3.6-35b-a3b","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        expect: &[
            Expect::Status(200),
            Expect::BodyContains("message_start"),
            Expect::BodyContains("content_block_delta"),
            Expect::BodyContains("message_stop"),
            Expect::BodyLacks("chat.completion.chunk"),
        ],
    },
];

/// §211: native diagnostics live in their own namespace, separate from
/// the compatibility surfaces.
const NATIVE: &[Fixture] = &[
    Fixture {
        spec: "§211",
        name: "/health reports status and model state",
        method: Method::Get,
        path: "/health",
        body: "",
        expect: &[
            Expect::Status(200),
            Expect::FieldEq("status", r#""ok""#),
            Expect::Field("version"),
        ],
    },
    Fixture {
        spec: "§211",
        name: "/tqf/status is native, not under a compatibility namespace",
        method: Method::Get,
        path: "/tqf/status",
        body: "",
        expect: &[
            Expect::Status(200),
            Expect::Field("model.installed"),
            Expect::Field("model.loaded"),
            Expect::Field("backend"),
        ],
    },
    Fixture {
        spec: "§211",
        name: "/tqf/memory reports real broker accounting",
        method: Method::Get,
        path: "/tqf/memory",
        body: "",
        expect: &[Expect::Status(200)],
    },
];

// ------------------------------------------------------------- the tests

#[tokio::test]
async fn openai_surface_conforms() {
    run(OPENAI).await;
}

#[tokio::test]
async fn ollama_surface_conforms() {
    run(OLLAMA).await;
}

#[tokio::test]
async fn anthropic_surface_conforms() {
    run(ANTHROPIC).await;
}

#[tokio::test]
async fn native_surface_conforms() {
    run(NATIVE).await;
}

/// §331 makes "every fixture cites a section" a property of the suite
/// rather than a convention reviewers have to police.
#[test]
fn every_fixture_cites_a_specification_section() {
    for set in [OPENAI, OLLAMA, ANTHROPIC, NATIVE] {
        for fixture in set {
            assert!(
                fixture.spec.starts_with('§'),
                "fixture {:?} must cite the section it encodes",
                fixture.name
            );
            assert!(
                !fixture.expect.is_empty(),
                "fixture {:?} asserts nothing",
                fixture.name
            );
        }
    }
}
