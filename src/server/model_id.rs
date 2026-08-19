//! Canonical model identity, shared by every compatibility surface.
//!
//! TQF serves exactly one model, but clients name it in many ways. Ollama
//! clients send registry-style tags (`qwen3.6:35b`,
//! `registry.ollama.ai/library/qwen3.6:latest`), OpenAI clients send
//! whatever their config file says, and an OpenAI-shaped client repointed
//! from an Ollama base URL sends Ollama spellings to `/v1` routes.
//!
//! Both adapters resolve through here rather than keeping their own alias
//! lists, because divergent alias sets per surface is exactly the
//! confusion spec §203's single canonical ID exists to prevent.

/// The one model this build serves. Responses normalize to this when
/// describing TQF's own inventory (spec §203).
pub const CANONICAL_MODEL_ID: &str = "qwen3.6-35b-a3b";

/// The Ollama-style tag TQF advertises in `/api/tags`, since that is the
/// form Ollama clients expect to see and then send back.
pub const OLLAMA_MODEL_TAG: &str = "qwen3.6:35b";

/// Names accepted for the served model, lowercased.
const ACCEPTED_NAMES: &[&str] = &[
    "qwen3.6-35b-a3b",
    "qwen3.6-35b",
    "qwen3.6",
    "qwen36",
    "qwen3.6-35b-a3b-gguf",
    "tqf",
];

/// Tags accepted alongside an accepted name. A tag naming a *different*
/// quantization is rejected rather than quietly served: TQF has one Q4_K_M
/// file, and answering a `:q8_0` request from it would misrepresent what
/// the client is talking to (spec §202).
const ACCEPTED_TAGS: &[&str] = &["latest", "35b", "35b-a3b", "q4", "q4_k_m", "a3b"];

/// Resolves a client-supplied model name to the canonical ID.
///
/// `None` and an empty string mean "whatever you serve", which is what
/// most clients send when they were configured with a base URL and no
/// model.
pub fn resolve(requested: Option<&str>) -> std::result::Result<&'static str, String> {
    let Some(requested) = requested.map(str::trim).filter(|name| !name.is_empty()) else {
        return Ok(CANONICAL_MODEL_ID);
    };

    // Strip a registry path (`registry.ollama.ai/library/qwen3.6:35b`,
    // `hf.co/owner/repo`) and any pinned digest suffix.
    let without_digest = requested.split("@sha256:").next().unwrap_or(requested);
    let last_segment = without_digest
        .rsplit('/')
        .next()
        .unwrap_or(without_digest)
        .to_ascii_lowercase();

    let (name, tag) = match last_segment.split_once(':') {
        Some((name, tag)) => (name, Some(tag)),
        None => (last_segment.as_str(), None),
    };

    if !ACCEPTED_NAMES.contains(&name) {
        return Err(format!(
            "model {requested:?} is not available; this server serves {CANONICAL_MODEL_ID:?} \
             (also addressable as {OLLAMA_MODEL_TAG:?})"
        ));
    }
    if let Some(tag) = tag {
        if !ACCEPTED_TAGS.contains(&tag) {
            return Err(format!(
                "model tag {tag:?} is not available; this server serves a single Q4_K_M \
                 checkpoint as {CANONICAL_MODEL_ID:?}"
            ));
        }
    }
    Ok(CANONICAL_MODEL_ID)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_or_empty_model_resolves_to_the_canonical_id() {
        assert_eq!(resolve(None).unwrap(), CANONICAL_MODEL_ID);
        assert_eq!(resolve(Some("")).unwrap(), CANONICAL_MODEL_ID);
        assert_eq!(resolve(Some("   ")).unwrap(), CANONICAL_MODEL_ID);
    }

    /// The spellings real clients actually send. Each of these used to be
    /// a 400 from the OpenAI adapter.
    #[test]
    fn ollama_style_tags_and_registry_paths_resolve() {
        for name in [
            "qwen3.6-35b-a3b",
            "Qwen3.6-35B-A3B",
            "qwen3.6:35b",
            "qwen3.6:latest",
            "qwen3.6-35b-a3b:q4_k_m",
            "qwen36",
            "tqf",
            "registry.ollama.ai/library/qwen3.6:35b",
            "hf.co/ggml-org/qwen3.6-35b-a3b-gguf",
            "qwen3.6:35b@sha256:0123456789abcdef",
        ] {
            assert_eq!(
                resolve(Some(name)).unwrap_or_else(|e| panic!("{name:?} rejected: {e}")),
                CANONICAL_MODEL_ID
            );
        }
    }

    #[test]
    fn a_different_model_is_rejected_with_a_message_naming_what_is_served() {
        let error = resolve(Some("llama3")).expect_err("must reject a different model");
        assert!(error.contains(CANONICAL_MODEL_ID), "{error}");
        assert!(error.contains("llama3"), "{error}");
    }

    /// Serving a Q4_K_M file in answer to a `:q8_0` request would
    /// misrepresent what the client is connected to.
    #[test]
    fn a_tag_naming_a_different_quantization_is_rejected() {
        for tag in ["qwen3.6:q8_0", "qwen3.6:fp16", "qwen3.6:70b"] {
            let error = resolve(Some(tag)).expect_err("must reject {tag}");
            assert!(error.contains("single Q4_K_M"), "{tag}: {error}");
        }
    }
}
