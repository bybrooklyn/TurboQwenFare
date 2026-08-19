use std::collections::HashMap;
use std::path::PathBuf;

use super::protocol::JsonRpcRequest;
use super::server::handle_request;
use super::stdio::run_stdio_loop;
use super::tools::IndexState;
use crate::retrieval::lexical::LexicalIndex;

fn real_index_state() -> IndexState {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let relative_paths = [
        "src/memory/mod.rs",
        "src/retrieval/ignore.rs",
        "src/experts/policy.rs",
    ];
    let mut file_contents = HashMap::new();
    for path in relative_paths {
        let contents = std::fs::read_to_string(root.join(path)).unwrap();
        file_contents.insert(path.to_string(), contents);
    }
    let documents: Vec<(String, String)> = file_contents
        .iter()
        .map(|(p, c)| (p.clone(), c.clone()))
        .collect();
    IndexState {
        root: PathBuf::from(root),
        lexical: LexicalIndex::build(&documents),
        semantic: None,
        file_contents,
    }
}

fn request(id: i64, method: &str, params: serde_json::Value) -> JsonRpcRequest {
    serde_json::from_value(serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    }))
    .unwrap()
}

#[test]
fn initialize_reports_the_real_protocol_version_and_tools_capability() {
    let response = handle_request(None, &request(1, "initialize", serde_json::json!({})))
        .expect("initialize gets a response");
    let result = response.result.unwrap();
    assert_eq!(result["protocolVersion"], "2025-06-18");
    assert!(result["capabilities"]["tools"].is_object());
    assert_eq!(result["serverInfo"]["name"], "tqf");
}

#[test]
fn tools_list_advertises_all_seven_read_only_tools() {
    let response = handle_request(None, &request(2, "tools/list", serde_json::json!({})))
        .expect("tools/list gets a response");
    let tools = response.result.unwrap()["tools"]
        .as_array()
        .unwrap()
        .clone();
    let names: Vec<String> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    for expected in [
        "tqf_search",
        "tqf_symbol",
        "tqf_references",
        "tqf_callers",
        "tqf_tests",
        "tqf_file",
        "tqf_repo_map",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "missing {expected}: {names:?}"
        );
    }
}

#[test]
fn notifications_get_no_response() {
    let notification: JsonRpcRequest = serde_json::from_value(serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }))
    .unwrap();
    assert!(handle_request(None, &notification).is_none());
}

#[test]
fn unknown_method_is_a_protocol_error() {
    let response = handle_request(
        None,
        &request(3, "not/a/real/method", serde_json::json!({})),
    )
    .unwrap();
    assert!(response.result.is_none());
    assert_eq!(
        response.error.unwrap().code,
        super::protocol::METHOD_NOT_FOUND
    );
}

/// Spec §44: "Ensure retrieval is optional and server works normally
/// without an index." No `IndexState` at all must not error the
/// protocol — every read tool reports a clear, ordinary "no index"
/// result instead.
#[test]
fn server_works_normally_with_no_index_built() {
    for (tool, args) in [
        ("tqf_search", serde_json::json!({"query": "anything"})),
        ("tqf_symbol", serde_json::json!({"identifier": "Anything"})),
        ("tqf_file", serde_json::json!({"path": "src/lib.rs"})),
        ("tqf_repo_map", serde_json::json!({})),
    ] {
        let response = handle_request(
            None,
            &request(
                4,
                "tools/call",
                serde_json::json!({"name": tool, "arguments": args}),
            ),
        )
        .unwrap();
        let result = response
            .result
            .expect("no-index case is a normal result, not a protocol error");
        assert_eq!(
            result["isError"], false,
            "tool {tool} should not error with no index"
        );
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("No index"));
    }
}

/// Real end-to-end: builds a real lexical index over three of this
/// crate's own files and proves `tqf_search`/`tqf_symbol`/`tqf_file`/
/// `tqf_repo_map` all return real, correct answers through the exact
/// same `handle_request` path a client uses.
#[test]
fn real_index_tools_return_correct_real_answers() {
    let state = real_index_state();

    let search = handle_request(
        Some(&state),
        &request(
            5,
            "tools/call",
            serde_json::json!({"name": "tqf_search", "arguments": {"query": "MemoryBroker", "limit": 3}}),
        ),
    )
    .unwrap()
    .result
    .unwrap();
    assert_eq!(search["isError"], false);
    let search_text = search["content"][0]["text"].as_str().unwrap();
    assert!(search_text.contains("src/memory/mod.rs"), "{search_text}");

    let symbol = handle_request(
        Some(&state),
        &request(
            6,
            "tools/call",
            serde_json::json!({"name": "tqf_symbol", "arguments": {"identifier": "MemoryBroker"}}),
        ),
    )
    .unwrap()
    .result
    .unwrap();
    assert_eq!(symbol["content"][0]["text"], "src/memory/mod.rs");

    let file = handle_request(
        Some(&state),
        &request(
            7,
            "tools/call",
            serde_json::json!({"name": "tqf_file", "arguments": {"path": "src/experts/policy.rs"}}),
        ),
    )
    .unwrap()
    .result
    .unwrap();
    assert_eq!(
        file["content"][0]["text"],
        state.file_contents["src/experts/policy.rs"].as_str()
    );

    let repo_map = handle_request(
        Some(&state),
        &request(
            8,
            "tools/call",
            serde_json::json!({"name": "tqf_repo_map", "arguments": {}}),
        ),
    )
    .unwrap()
    .result
    .unwrap();
    let map_text = repo_map["content"][0]["text"].as_str().unwrap();
    assert!(map_text.contains("memory/"), "{map_text}");
    assert!(map_text.contains("retrieval/"), "{map_text}");
    assert!(map_text.contains("experts/"), "{map_text}");
}

/// spec §85/§95: these three tools need a real program graph this
/// build doesn't have; they must report that honestly (isError: true
/// with a real explanation) rather than fabricate results.
#[test]
fn graph_dependent_tools_honestly_report_the_capability_gap() {
    let state = real_index_state();
    for tool in ["tqf_references", "tqf_callers", "tqf_tests"] {
        let response = handle_request(
            Some(&state),
            &request(
                9,
                "tools/call",
                serde_json::json!({"name": tool, "arguments": {"identifier": "MemoryBroker"}}),
            ),
        )
        .unwrap()
        .result
        .unwrap();
        assert_eq!(
            response["isError"], true,
            "{tool} should honestly report it can't do this"
        );
        assert!(response["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("program graph"));
    }
}

/// Real wire-level test: a genuine newline-delimited JSON-RPC session
/// (initialize -> initialized notification -> tools/list ->
/// tools/call) fed through the actual stdio transport loop, not just
/// `handle_request` directly.
#[test]
fn stdio_transport_handles_a_real_session() {
    let state = real_index_state();
    let session = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"tqf_symbol","arguments":{"identifier":"MemoryBroker"}}}"#,
        "\n",
    );
    let mut output = Vec::new();
    run_stdio_loop(Some(&state), session.as_bytes(), &mut output).unwrap();
    let lines: Vec<&str> = std::str::from_utf8(&output).unwrap().lines().collect();
    // Three responses: initialize, tools/list, tools/call — the
    // notification in between produces no output line.
    assert_eq!(lines.len(), 3, "{lines:?}");
    let init: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    let call: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(call["result"]["content"][0]["text"], "src/memory/mod.rs");
}
