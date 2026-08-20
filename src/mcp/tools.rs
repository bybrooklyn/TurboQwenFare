//! The read-only MCP tool surface (spec §95). "Retrieval tools remain
//! read-only initially. File edits/execution belong to the client
//! harness" — nothing here writes to disk or executes anything.
//!
//! Scope decision: `tqf_references`/`tqf_callers`/`tqf_tests` need a
//! real program graph (calls/refs/test-coverage edges), which Phase
//! 35/36/40 already decided not to build without a real parser/AST
//! (building fake results from regex matches would be worse than
//! reporting the gap honestly — the same call Phase 36 made for
//! structural chunking). Those three tools are wired into the protocol
//! (so `tools/list` accurately advertises what exists) but their
//! handlers report the real limitation via `isError: true` rather than
//! fabricating an answer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::retrieval::adaptive::module_of;
use crate::retrieval::flat::FlatVectorStore;
use crate::retrieval::hybrid::run_hybrid_query;
use crate::retrieval::lexical::LexicalIndex;

use super::protocol::{tool_text_result, ToolDefinition};

/// Everything a tool call needs. `semantic` is `None` unless a caller
/// has separately built one (the MCP server itself never loads the
/// pplx-embed helper model just to answer a lexical/exact query — spec
/// §85: "Identifier-like queries should hit exact/lexical/symbol paths
/// without loading the embedder").
///
/// `paths` is the indexed file list, not their contents. Holding every
/// file's text in memory would put the whole repository in RAM for the
/// life of the process to serve occasional `tqf_file` calls, and the
/// persisted index does not store contents anyway — it stores postings.
/// `tqf_file` reads from disk on demand instead, which also means it
/// returns the file as it is *now* rather than as it was when the index
/// was built.
pub struct IndexState {
    pub root: PathBuf,
    pub lexical: LexicalIndex,
    pub semantic: Option<FlatVectorStore>,
    pub paths: Vec<String>,
}

impl IndexState {
    /// Reads one indexed file.
    ///
    /// Two independent guards, because the path list is parsed from a
    /// file on disk and a tool argument is attacker-controlled: the path
    /// must be one this index actually contains, and the resolved
    /// location must still be inside the root. Either check alone would
    /// probably do; spec §95 makes this surface read-only, and a
    /// read-only surface that can be walked out of its root is not
    /// read-only in any useful sense.
    pub fn read_indexed_file(&self, path: &str) -> Result<String, String> {
        if !self.paths.iter().any(|known| known == path) {
            return Err(format!("File {path:?} is not in the index."));
        }
        let joined = self.root.join(path);
        let resolved = joined
            .canonicalize()
            .map_err(|e| format!("File {path:?} is indexed but unreadable: {e}"))?;
        if !within(&self.root, &resolved) {
            return Err(format!("File {path:?} resolves outside the indexed root."));
        }
        std::fs::read_to_string(&resolved)
            .map_err(|e| format!("File {path:?} is indexed but unreadable: {e}"))
    }
}

/// Containment after symlink resolution. The root is canonicalized here
/// rather than at construction so a caller cannot hand us a
/// pre-resolved-looking root that isn't.
fn within(root: &Path, resolved: &Path) -> bool {
    match root.canonicalize() {
        Ok(root) => resolved.starts_with(root),
        Err(_) => false,
    }
}

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "tqf_search",
            description: "Hybrid exact/lexical/semantic search over the indexed repository.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language or identifier query"},
                    "limit": {"type": "integer", "description": "Max results (default 8)"}
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "tqf_symbol",
            description: "Exact lookup of a case-sensitive identifier's defining/referencing files.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"]
            }),
        },
        ToolDefinition {
            name: "tqf_references",
            description: "List references to a symbol (requires a program graph; not available without a real parser).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"]
            }),
        },
        ToolDefinition {
            name: "tqf_callers",
            description: "List callers of a function (requires a program graph; not available without a real parser).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"]
            }),
        },
        ToolDefinition {
            name: "tqf_tests",
            description: "List tests covering a symbol (requires a program graph; not available without a real parser).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"identifier": {"type": "string"}},
                "required": ["identifier"]
            }),
        },
        ToolDefinition {
            name: "tqf_file",
            description: "Read one indexed file's content by repo-relative path.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "tqf_repo_map",
            description: "Summarize the indexed repository's top-level module/file layout.",
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        },
    ]
}

const NO_INDEX_MESSAGE: &str = "No index is built yet. The server is running normally without one — run `tqf sync <path>` to build a searchable index first.";

fn get_str<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(|v| v.as_str())
}

/// Dispatches one `tools/call`. Never returns a protocol-level error
/// for "no index built" or "requires a program graph" — those are
/// normal, expected outcomes reported as ordinary tool results (spec
/// §44: "Ensure retrieval is optional and server works normally
/// without an index").
pub fn call_tool(state: Option<&IndexState>, name: &str, params: &Value) -> Result<Value, String> {
    match name {
        "tqf_search" => {
            let query = get_str(params, "query").ok_or("missing required \"query\"")?;
            let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
            let Some(state) = state else {
                return Ok(tool_text_result(NO_INDEX_MESSAGE.to_string(), false));
            };
            let (_, _, fused) =
                run_hybrid_query(&state.lexical, state.semantic.as_ref(), query, None, limit, 60.0);
            let text = if fused.is_empty() {
                format!("No results for {query:?}.")
            } else {
                fused
                    .iter()
                    .take(limit)
                    .enumerate()
                    .map(|(i, c)| format!("{}. {} (rrf_score={:.4})", i + 1, c.chunk_id, c.rrf_score))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            Ok(tool_text_result(text, false))
        }
        "tqf_symbol" => {
            let identifier = get_str(params, "identifier").ok_or("missing required \"identifier\"")?;
            let Some(state) = state else {
                return Ok(tool_text_result(NO_INDEX_MESSAGE.to_string(), false));
            };
            let hits = state.lexical.exact_lookup(identifier);
            let text = if hits.is_empty() {
                format!("No exact matches for identifier {identifier:?}.")
            } else {
                hits.join("\n")
            };
            Ok(tool_text_result(text, false))
        }
        "tqf_references" | "tqf_callers" | "tqf_tests" => Ok(tool_text_result(
            format!(
                "{name} is not available: it needs a real program graph (calls/references/test-coverage edges from an AST), which this build does not have (no parser dependency was added — see the Phase 35/36 scope decision). This is a real capability gap, not a bug."
            ),
            true,
        )),
        "tqf_file" => {
            let path = get_str(params, "path").ok_or("missing required \"path\"")?;
            let Some(state) = state else {
                return Ok(tool_text_result(NO_INDEX_MESSAGE.to_string(), false));
            };
            match state.read_indexed_file(path) {
                Ok(contents) => Ok(tool_text_result(contents, false)),
                Err(why) => Ok(tool_text_result(why, true)),
            }
        }
        "tqf_repo_map" => {
            let Some(state) = state else {
                return Ok(tool_text_result(NO_INDEX_MESSAGE.to_string(), false));
            };
            let mut modules: BTreeMap<String, usize> = BTreeMap::new();
            for path in &state.paths {
                *modules.entry(module_of(path)).or_insert(0) += 1;
            }
            let text = modules
                .iter()
                .map(|(module, count)| format!("{module}/ ({count} file{})", if *count == 1 { "" } else { "s" }))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(tool_text_result(text, false))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}
