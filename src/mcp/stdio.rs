//! MCP stdio transport (spec §95: "Support both stdio and streamable
//! HTTP MCP"). Newline-delimited JSON-RPC messages, per the MCP stdio
//! transport spec: one message per line on stdin, one response per
//! line on stdout. Generic over `BufRead`/`Write` so the real loop and
//! its tests exercise the exact same code — no separate "test harness"
//! that could drift from what actually runs against a real client.

use std::io::{BufRead, Write};

use super::protocol::JsonRpcRequest;
use super::server::handle_request;
use super::tools::IndexState;

/// Reads one JSON-RPC message per line from `input` until EOF, writes
/// each response (if any) as one line to `output`. A line that fails
/// to parse as a valid `JsonRpcRequest` is reported as a parse error on
/// a best-effort `id: null` response rather than killing the whole
/// session — one malformed line must not take down the connection.
pub fn run_stdio_loop<R: BufRead, W: Write>(
    state: Option<&IndexState>,
    input: R,
    mut output: W,
) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<JsonRpcRequest>(trimmed) {
            Ok(request) => handle_request(state, &request),
            Err(e) => Some(super::protocol::JsonRpcResponse::err(
                serde_json::Value::Null,
                -32700,
                format!("parse error: {e}"),
            )),
        };
        if let Some(response) = response {
            let serialized = serde_json::to_string(&response).expect("response serializes");
            output.write_all(serialized.as_bytes())?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    Ok(())
}
