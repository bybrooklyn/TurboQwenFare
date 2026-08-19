//! The transport-agnostic MCP request handler: `initialize`,
//! `tools/list`, `tools/call`, and the `notifications/initialized`
//! notification (spec §95). Both the stdio and (future) streamable-HTTP
//! transports call `handle_request` — none of the MCP semantics live in
//! either transport.

use serde_json::Value;

use super::protocol::{
    JsonRpcRequest, JsonRpcResponse, INVALID_PARAMS, METHOD_NOT_FOUND, PROTOCOL_VERSION,
    SERVER_NAME, SERVER_VERSION,
};
use super::tools::{call_tool, tool_definitions, IndexState};

/// Handles one request/notification. Returns `None` for notifications
/// (no `id`, per JSON-RPC 2.0 — the caller must not write anything to
/// the transport for those).
pub fn handle_request(
    state: Option<&IndexState>,
    request: &JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let id = match &request.id {
        Some(id) => id.clone(),
        None => {
            // A notification (e.g. `notifications/initialized`) — spec
            // requires no response, but nothing about *this* server's
            // state actually depends on receiving it, since every tool
            // handler is stateless per call.
            return None;
        }
    };

    let response = match request.method.as_str() {
        "initialize" => JsonRpcResponse::ok(
            id,
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }),
        ),
        "tools/list" => {
            let tools: Vec<Value> = tool_definitions()
                .into_iter()
                .map(|t| serde_json::to_value(t).expect("tool definition serializes"))
                .collect();
            JsonRpcResponse::ok(id, serde_json::json!({"tools": tools}))
        }
        "tools/call" => {
            let Some(name) = request.params.get("name").and_then(|v| v.as_str()) else {
                return Some(JsonRpcResponse::err(
                    id,
                    INVALID_PARAMS,
                    "missing required \"name\"",
                ));
            };
            let empty = serde_json::json!({});
            let arguments = request.params.get("arguments").unwrap_or(&empty);
            match call_tool(state, name, arguments) {
                Ok(result) => JsonRpcResponse::ok(id, result),
                Err(message) if message.starts_with("unknown tool") => {
                    JsonRpcResponse::err(id, METHOD_NOT_FOUND, message)
                }
                Err(message) => JsonRpcResponse::err(id, INVALID_PARAMS, message),
            }
        }
        other => JsonRpcResponse::err(id, METHOD_NOT_FOUND, format!("unknown method: {other}")),
    };
    Some(response)
}
