//! Qwen3.6 generation boundary shared by every HTTP protocol. This is not a
//! multi-model abstraction: the only permitted implementation executes the
//! fixed Qwen3.6 graph. Keeping its output shape here prevents OpenAI/SSE
//! framing or tool-tag parsing from leaking into token-critical decode.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use crate::error::{ProtocolError, Result, TqfError};
use crate::experts::ExpertCacheStats;
use crate::format::gguf;
use crate::ids::Bytes;
use crate::memory::MemoryBroker;
use crate::model::qwen36::runtime::{Qwen36BoundedReferenceRuntime, Qwen36ReferenceRuntime};
use crate::runtime::stream_decoder::{IncrementalOutputDecoder, StreamEvent, StreamFinish};
use crate::runtime::{NormalizedRequest, Role};
use crate::sampling::Sampler;
use crate::tokenizer::chat::{self, ChatMessage, ChatRole, ToolSpec};
use crate::tokenizer::chat::{THINK_END, THINK_START, TOOL_CALL_END, TOOL_CALL_START};
use crate::tokenizer::TqfTokenizer;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedToolCall {
    pub id: String,
    pub name: String,
    /// Kept in exact JSON text form for OpenAI compatibility and to avoid a
    /// parse/serialize round trip changing caller-visible argument bytes.
    pub arguments_json: String,
}

/// Real token counts and timings for one generation.
///
/// Ollama clients display tok/s computed from these, and OpenAI clients
/// bill and budget against `usage`. Reporting zeros would be a quiet lie
/// in a number users actually read, so these are measured, never
/// synthesized.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Wall time spent turning the prompt into a first token.
    pub prefill: std::time::Duration,
    /// Wall time spent producing the remaining tokens.
    pub decode: std::time::Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedOutput {
    pub text: String,
    pub tool_calls: Vec<GeneratedToolCall>,
    pub finish_reason: &'static str,
    pub usage: GenerationUsage,
}

impl GeneratedOutput {
    /// Separates ordinary assistant text from complete Qwen `<tool_call>`
    /// blocks. The pinned Qwen3.6 template emits XML function/parameter
    /// blocks; the older JSON body is still accepted for stored transcripts.
    /// Invalid payloads or an unmatched delimiter are a protocol error;
    /// forwarding malformed calls to coding clients would be worse than a
    /// visible failed generation.
    pub fn from_model_text(text: impl Into<String>) -> Result<Self> {
        let mut remaining = visible_after_thinking(text.into());
        let mut plain = String::new();
        let mut calls = Vec::new();
        while let Some(start) = remaining.find(TOOL_CALL_START) {
            plain.push_str(&remaining[..start]);
            let after_start = &remaining[start + TOOL_CALL_START.len()..];
            let end = after_start.find(TOOL_CALL_END).ok_or_else(|| {
                ProtocolError::Invalid(
                    "model emitted an unterminated <tool_call> block".to_string(),
                )
            })?;
            let body = after_start[..end].trim();
            calls.push(parse_tool_call(body, calls.len())?);
            remaining = after_start[end + TOOL_CALL_END.len()..].to_string();
        }
        plain.push_str(&remaining);
        let finish_reason = if calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        Ok(Self {
            text: plain,
            tool_calls: calls,
            finish_reason,
            usage: GenerationUsage::default(),
        })
    }

    /// Parses a continuation produced after the Qwen3.6 chat template has
    /// already opened `<think>`. Until the model emits `</think>`, every byte
    /// is private reasoning and must not reach a compatibility client.
    fn from_qwen_continuation(text: String) -> Result<Self> {
        if !text.contains(THINK_END) {
            return Self::from_model_text("");
        }
        Self::from_model_text(text)
    }
}

/// The generation prompt already opens `<think>`, so newly decoded text
/// normally begins with reasoning and contains only the closing tag. Keep
/// reasoning private from compatibility `content`; if a stored transcript
/// includes the opening tag, the same rule applies.
fn visible_after_thinking(text: String) -> String {
    if let Some(end) = text.find(THINK_END) {
        return text[end + THINK_END.len()..]
            .trim_start_matches(['\r', '\n'])
            .to_string();
    }
    text.strip_prefix(THINK_START)
        .map(|text| text.trim_start_matches(['\r', '\n']).to_string())
        .unwrap_or(text)
}

pub(crate) fn parse_tool_call(body: &str, index: usize) -> Result<GeneratedToolCall> {
    if body.starts_with('{') {
        let wire: ToolCallWire = serde_json::from_str(body).map_err(|error| {
            ProtocolError::Invalid(format!("model emitted invalid tool-call JSON: {error}"))
        })?;
        return build_tool_call(index, wire.name, wire.arguments);
    }

    let function = body.strip_prefix("<function=").ok_or_else(|| {
        ProtocolError::Invalid("model tool call is missing <function=name>".to_string())
    })?;
    let name_end = function.find('>').ok_or_else(|| {
        ProtocolError::Invalid("model tool call has an unterminated function name".to_string())
    })?;
    let name = &function[..name_end];
    let function_body = &function[name_end + 1..];
    let function_end = function_body.rfind("</function>").ok_or_else(|| {
        ProtocolError::Invalid("model tool call is missing </function>".to_string())
    })?;
    if !function_body[function_end + "</function>".len()..]
        .trim()
        .is_empty()
    {
        return Err(ProtocolError::Invalid(
            "model tool call has content after </function>".to_string(),
        )
        .into());
    }

    let mut parameters = &function_body[..function_end];
    let mut arguments = serde_json::Map::new();
    loop {
        parameters = parameters.trim_start_matches([' ', '\t', '\r', '\n']);
        if parameters.is_empty() {
            break;
        }
        let parameter = parameters.strip_prefix("<parameter=").ok_or_else(|| {
            ProtocolError::Invalid("model tool call contains malformed parameters".to_string())
        })?;
        let parameter_name_end = parameter.find('>').ok_or_else(|| {
            ProtocolError::Invalid("model tool call has an unterminated parameter name".to_string())
        })?;
        let parameter_name = parameter[..parameter_name_end].trim();
        if parameter_name.is_empty() || arguments.contains_key(parameter_name) {
            return Err(ProtocolError::Invalid(
                "model tool call has an empty or duplicate parameter name".to_string(),
            )
            .into());
        }
        let value_and_rest = parameter[parameter_name_end + 1..]
            .strip_prefix("\r\n")
            .or_else(|| parameter[parameter_name_end + 1..].strip_prefix('\n'))
            .unwrap_or(&parameter[parameter_name_end + 1..]);
        let close = value_and_rest.find("</parameter>").ok_or_else(|| {
            ProtocolError::Invalid(format!(
                "model tool call parameter {parameter_name} is unterminated"
            ))
        })?;
        let raw_value = value_and_rest[..close]
            .trim_end_matches(['\r', '\n'])
            .trim();
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        arguments.insert(parameter_name.to_string(), value);
        parameters = &value_and_rest[close + "</parameter>".len()..];
    }
    build_tool_call(
        index,
        name.to_string(),
        serde_json::Value::Object(arguments),
    )
}

fn build_tool_call(
    index: usize,
    name: String,
    arguments: serde_json::Value,
) -> Result<GeneratedToolCall> {
    if name.trim().is_empty() {
        return Err(ProtocolError::Invalid(
            "model emitted a tool call with an empty name".to_string(),
        )
        .into());
    }
    Ok(GeneratedToolCall {
        id: format!("call_{index}"),
        name,
        arguments_json: serde_json::to_string(&arguments).expect("JSON value serializes"),
    })
}

#[derive(Debug, Deserialize)]
struct ToolCallWire {
    name: String,
    arguments: serde_json::Value,
}

/// Qwen-specific model session, supplied after a trusted converted model has
/// loaded. It receives only normalized requests; no HTTP protocol is visible
/// to this boundary.
#[async_trait]
pub trait Qwen36Generator: Send + Sync {
    async fn generate(
        &self,
        request: NormalizedRequest,
        cancellation: CancellationToken,
    ) -> Result<GeneratedOutput>;

    /// Streams a generation, sending each [`StreamEvent`] as it is
    /// produced and returning the assembled output.
    ///
    /// The default implementation runs `generate()` and emits its result
    /// as one delta, so every existing implementor — including the test
    /// doubles — keeps working unchanged. Implementors that can really
    /// stream override it; [`Qwen36Generator::streams_incrementally`]
    /// reports which behavior a caller is getting.
    ///
    /// A send failure means the client is gone: implementations stop
    /// promptly rather than finishing a generation nobody will read.
    async fn generate_streaming(
        &self,
        request: NormalizedRequest,
        cancellation: CancellationToken,
        events: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<GeneratedOutput> {
        let output = self.generate(request, cancellation).await?;
        if !output.text.is_empty() {
            let _ = events
                .send(StreamEvent::TextDelta(output.text.clone()))
                .await;
        }
        for call in &output.tool_calls {
            let _ = events.send(StreamEvent::ToolCall(call.clone())).await;
        }
        Ok(output)
    }

    /// Real token count for a request's rendered prompt.
    ///
    /// Anthropic's `count_tokens` endpoint exists so a client can size a
    /// conversation before sending it, which only helps if the number is
    /// the tokenizer's rather than an estimate. Defaults to an error so
    /// an implementation without a tokenizer says so instead of guessing.
    /// Tokens the prompt would occupy, without generating.
    ///
    /// The default reports `Unsupported` rather than `ProtocolError::Invalid`:
    /// a generator that cannot count is a missing *capability*, not a
    /// malformed client request, and §212 maps those to different statuses.
    /// Conflating them told callers to fix a request that was already valid.
    fn count_prompt_tokens(&self, _request: &NormalizedRequest) -> Result<usize> {
        Err(crate::error::ModelError::Unsupported(
            "this generator cannot count tokens without generating".to_string(),
        )
        .into())
    }

    /// Whether `generate_streaming` really emits deltas during decode
    /// rather than one chunk at the end. Tests assert against this instead
    /// of guessing from timing.
    fn streams_incrementally(&self) -> bool {
        false
    }
}

/// An actual, fixed-graph Qwen3.6 generator for the high-memory resident
/// reference profile.  It is intentionally separate from the default server
/// path: Phase 14 makes this oracle useful for parity, while Phase 18 is what
/// makes the same graph fit the normal bounded-memory product contract.
pub struct Qwen36ResidentReferenceGenerator {
    /// `Arc` so the decode worker can hold it too: incremental
    /// detokenization runs inside the `spawn_blocking` loop, right beside
    /// the model step that produced the token.
    tokenizer: Arc<Mutex<TqfTokenizer>>,
    // Retains the conservative GGUF metadata/tokenizer allocation envelope
    // for as long as the derived tokenizer is live.
    _tokenizer_source: gguf::GgufFile,
    runtime: Arc<Mutex<QwenRuntimeInstance>>,
    max_context: usize,
}

enum QwenRuntimeInstance {
    Resident(Qwen36ReferenceRuntime),
    Bounded(Qwen36BoundedReferenceRuntime),
}

impl QwenRuntimeInstance {
    fn decode_step(
        &mut self,
        token: u32,
        sampler: &mut Sampler,
        history: &[u32],
    ) -> Result<crate::runtime::DecodeToken> {
        match self {
            Self::Resident(runtime) => runtime.decode_step(token, sampler, history),
            Self::Bounded(runtime) => runtime.decode_step(token, sampler, history),
        }
    }

    /// Phase 26 (spec §298): the resident runtimes prefill a prompt in
    /// chunked, layer-outer form with expert-set dedup; the bounded
    /// runtime keeps the per-token loop (its chunked form is a later
    /// step). Returns the greedy token after the final prompt token.
    fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        match self {
            Self::Resident(runtime) => runtime.prefill_greedy(prompt),
            Self::Bounded(runtime) => {
                let mut next = 0;
                for &token in prompt {
                    next = runtime.decode_greedy(token)?.token;
                }
                Ok(next)
            }
        }
    }

    fn reset_session(&mut self) {
        match self {
            Self::Resident(runtime) => runtime.reset_session(),
            Self::Bounded(runtime) => runtime.reset_session(),
        }
    }

    fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        match self {
            Self::Resident(runtime) => runtime.expert_cache_stats(),
            Self::Bounded(runtime) => Some(runtime.expert_cache_stats()),
        }
    }
}

/// Output cap when a request names none. Not a hard ceiling — a request
/// may ask for as much as the context window allows — just a bound that
/// keeps an unbounded client from decoding until the context is full.
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 512;

const DEV_DECODE_DIAGNOSTICS_ENV: &str = "TQF_DEV_DECODE_DIAGNOSTICS";

fn emit_decode_diagnostics(
    input_token: u32,
    decoded: &crate::runtime::DecodeToken,
    cache_before: Option<ExpertCacheStats>,
    cache_after: Option<ExpertCacheStats>,
) {
    if std::env::var(DEV_DECODE_DIAGNOSTICS_ENV).as_deref() != Ok("1") {
        return;
    }
    let timings = &decoded.diagnostics.timings;
    let layer_time = timings
        .layers
        .iter()
        .map(|(_, duration)| *duration)
        .sum::<std::time::Duration>();
    let raw_miss_bytes = cache_before
        .zip(cache_after)
        .map(|(before, after)| {
            after
                .raw_miss_bytes
                .0
                .saturating_sub(before.raw_miss_bytes.0)
        })
        .unwrap_or(0);
    tracing::info!(
        input_token,
        output_token = decoded.token,
        embedding_ms = timings.embedding.as_secs_f64() * 1000.0,
        layers_ms = layer_time.as_secs_f64() * 1000.0,
        final_norm_ms = timings.final_norm.as_secs_f64() * 1000.0,
        lm_head_ms = timings.lm_head.as_secs_f64() * 1000.0,
        sampling_ms = timings.sampling.as_secs_f64() * 1000.0,
        raw_expert_miss_bytes = raw_miss_bytes,
        "Qwen3.6 greedy decode diagnostics"
    );
    for layer in &decoded.diagnostics.per_layer_hashes {
        tracing::info!(
            layer = layer.layer.0,
            hash = %layer.hash.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
            "Qwen3.6 layer hash"
        );
    }
    for route in &decoded.diagnostics.router_trace {
        tracing::info!(
            layer = route.layer.0,
            expert_ids = ?route.route.ids.map(|expert| expert.0),
            weights = ?route.route.weights,
            "Qwen3.6 exact router trace"
        );
    }
}

impl Qwen36ResidentReferenceGenerator {
    pub fn open(
        tqf_path: &Path,
        tokenizer_gguf_path: &Path,
        memory_budget: Bytes,
        max_context: usize,
    ) -> Result<Self> {
        let broker = MemoryBroker::new(memory_budget);
        let gguf = gguf::open_with_broker(tokenizer_gguf_path, &broker)?;
        let tokenizer = TqfTokenizer::from_gguf(&gguf)?;
        let runtime = Qwen36ReferenceRuntime::open_resident(tqf_path, broker, max_context)?;
        Ok(Self {
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            _tokenizer_source: gguf,
            runtime: Arc::new(Mutex::new(QwenRuntimeInstance::Resident(runtime))),
            max_context,
        })
    }

    /// Opens the same protocol adapter over the whole-expert streaming
    /// reference graph. The historical type name is retained so callers have
    /// one fixed Qwen generator surface; this profile does not pin routed
    /// expert tensors.
    pub fn open_streaming(
        tqf_path: &Path,
        tokenizer_gguf_path: &Path,
        memory_budget: Bytes,
        max_context: usize,
        expert_cache_bytes: Bytes,
    ) -> Result<Self> {
        let broker = MemoryBroker::new(memory_budget);
        let gguf = gguf::open_with_broker(tokenizer_gguf_path, &broker)?;
        let tokenizer = TqfTokenizer::from_gguf(&gguf)?;
        let runtime =
            Qwen36BoundedReferenceRuntime::open(tqf_path, broker, max_context, expert_cache_bytes)?;
        Ok(Self {
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            _tokenizer_source: gguf,
            runtime: Arc::new(Mutex::new(QwenRuntimeInstance::Bounded(runtime))),
            max_context,
        })
    }

    /// Phase 25 profile (spec §297): the resident-core streaming runtime -
    /// attention/GDN/router/shared-expert weights stay resident and
    /// broker-accounted (~2.13 GiB for the canonical container), while
    /// routed Q4_K experts stream through the global cache. This removes
    /// the per-token re-reads of the bounded profile at the cost of
    /// pinning the core. Selected by the `TQF_DEV_RESIDENT_STREAMING`
    /// developer control; it becomes the default only after the M4
    /// assault records a measured end-to-end win.
    pub fn open_resident_streaming(
        tqf_path: &Path,
        tokenizer_gguf_path: &Path,
        memory_budget: Bytes,
        max_context: usize,
        expert_cache_bytes: Bytes,
    ) -> Result<Self> {
        let broker = MemoryBroker::new(memory_budget);
        let gguf = gguf::open_with_broker(tokenizer_gguf_path, &broker)?;
        let tokenizer = TqfTokenizer::from_gguf(&gguf)?;
        let runtime = Qwen36ReferenceRuntime::open_streaming(
            tqf_path,
            broker,
            max_context,
            expert_cache_bytes,
        )?;
        Ok(Self {
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            _tokenizer_source: gguf,
            runtime: Arc::new(Mutex::new(QwenRuntimeInstance::Resident(runtime))),
            max_context,
        })
    }

    fn prompt_tokens(&self, request: &NormalizedRequest) -> Result<Vec<u32>> {
        if !request.vision.is_empty() {
            return Err(ProtocolError::Invalid(
                "vision inputs require the later vision execution path".to_string(),
            )
            .into());
        }
        let messages = request
            .messages
            .iter()
            .map(|message| {
                let mut rendered = ChatMessage::text(
                    match message.role {
                        Role::System => ChatRole::System,
                        Role::User => ChatRole::User,
                        Role::Assistant => ChatRole::Assistant,
                        Role::Tool => ChatRole::Tool,
                    },
                    message.content.clone(),
                );
                rendered.tool_calls = message
                    .tool_calls
                    .iter()
                    .map(|call| chat::ToolCall {
                        name: call.name.clone(),
                        arguments_json: call.arguments_json.clone(),
                    })
                    .collect();
                rendered
            })
            .collect::<Vec<_>>();
        let tools = request
            .tools
            .iter()
            .map(|tool| ToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters_json_schema: tool.parameters_json_schema.clone(),
            })
            .collect::<Vec<_>>();
        self.tokenizer
            .lock()
            .expect("tokenizer mutex poisoned")
            .encode(&chat::render(&messages, &tools, true), false)
    }
}

#[async_trait]
impl Qwen36Generator for Qwen36ResidentReferenceGenerator {
    async fn generate(
        &self,
        request: NormalizedRequest,
        cancellation: CancellationToken,
    ) -> Result<GeneratedOutput> {
        self.run_generation(request, cancellation, None).await
    }

    async fn generate_streaming(
        &self,
        request: NormalizedRequest,
        cancellation: CancellationToken,
        events: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<GeneratedOutput> {
        self.run_generation(request, cancellation, Some(events))
            .await
    }

    fn count_prompt_tokens(&self, request: &NormalizedRequest) -> Result<usize> {
        Ok(self.prompt_tokens(request)?.len())
    }

    fn streams_incrementally(&self) -> bool {
        true
    }
}

impl Qwen36ResidentReferenceGenerator {
    /// The single decode loop behind both `generate` and
    /// `generate_streaming`.
    ///
    /// Sharing it is deliberate: two loops would eventually disagree, and
    /// "the streamed answer differs from the batch answer" is exactly the
    /// bug class spec §71 warns about. The only difference between the two
    /// entry points is whether `events` is `Some`.
    ///
    /// Threading: all model work stays inside `spawn_blocking` (spec §25 —
    /// the decode loop must not run on the Tokio executor). Only decoded
    /// stream events cross back, via `blocking_send`, which applies
    /// backpressure to the *blocking* thread rather than a Tokio worker.
    async fn run_generation(
        &self,
        request: NormalizedRequest,
        cancellation: CancellationToken,
        events: Option<tokio::sync::mpsc::Sender<StreamEvent>>,
    ) -> Result<GeneratedOutput> {
        let prompt = self.prompt_tokens(&request)?;
        if prompt.is_empty() {
            return Err(ProtocolError::Invalid("empty tokenized prompt".to_string()).into());
        }
        if prompt.len() >= self.max_context {
            return Err(ProtocolError::Invalid(format!(
                "tokenized prompt ({} tokens) leaves no room for output within the context \
                 limit of {}",
                prompt.len(),
                self.max_context
            ))
            .into());
        }
        // The real bound is the context window, not an arbitrary constant.
        // A `.min(256)` here would silently truncate any longer request now
        // that the HTTP layer no longer rejects one — turning a clear 400
        // into a mysteriously short answer.
        let headroom = self.max_context.saturating_sub(prompt.len());
        let maximum = request
            .sampling
            .max_output_tokens
            .map(|requested| requested as usize)
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
            .min(headroom);
        let prompt_tokens = prompt.len() as u32;
        if maximum == 0 {
            return Ok(GeneratedOutput {
                usage: GenerationUsage {
                    prompt_tokens,
                    ..GenerationUsage::default()
                },
                ..GeneratedOutput::from_model_text("")?
            });
        }

        let eos = self
            .tokenizer
            .lock()
            .expect("tokenizer mutex poisoned")
            .eos_token_id;
        let mut sampler = Sampler::new(&request.sampling);
        let mut decoder =
            IncrementalOutputDecoder::new(request.sampling.stop_sequences.clone(), true);
        let runtime = Arc::clone(&self.runtime);
        let tokenizer = Arc::clone(&self.tokenizer);
        let cancellation_for_decode = cancellation.clone();

        let decoded = tokio::task::spawn_blocking(move || -> Result<DecodeRun> {
            let mut runtime = runtime.lock().expect("Qwen runtime mutex poisoned");
            runtime.reset_session();
            if cancellation_for_decode.is_cancelled() {
                return Err(TqfError::Cancelled);
            }

            // Phase 26: chunked prefill with expert-set dedup (the resident
            // runtimes) or the per-token reference loop.
            let prefill_started = std::time::Instant::now();
            let cache_before = runtime.expert_cache_stats();
            let next = runtime.prefill(&prompt)?;
            let prefill = prefill_started.elapsed();
            let cache_after = runtime.expert_cache_stats();
            tracing::info!(
                prompt_tokens = prompt.len(),
                prefill_ms = prefill.as_secs_f64() * 1000.0,
                "Qwen3.6 prefill"
            );
            if let (Some(before), Some(after)) = (cache_before, cache_after) {
                tracing::info!(
                    prefill_expert_hits = after.hits.saturating_sub(before.hits),
                    prefill_expert_misses = after.misses.saturating_sub(before.misses),
                    prefill_raw_miss_bytes = after
                        .raw_miss_bytes
                        .0
                        .saturating_sub(before.raw_miss_bytes.0),
                    prefill_demand_io_ms =
                        after.demand_io_nanos.saturating_sub(before.demand_io_nanos) as f64 / 1e6,
                    "Qwen3.6 prefill expert I/O"
                );
            }

            let decode_started = std::time::Instant::now();
            let mut next = next;
            let mut generated = Vec::with_capacity(maximum);
            let mut stream_state = crate::tokenizer::DecodeStreamState::default();
            let mut collected = Vec::new();
            let mut reached_eos = false;
            let mut client_gone = false;

            for _ in 0..maximum {
                if cancellation_for_decode.is_cancelled() {
                    return Err(TqfError::Cancelled);
                }
                generated.push(next);
                if Some(next) == eos {
                    reached_eos = true;
                    break;
                }

                // Decode this token to text and turn it into whatever is
                // safe to show, before running the next model step: a
                // client should see token N while the GPU works on N+1.
                let text = tokenizer
                    .lock()
                    .expect("tokenizer mutex poisoned")
                    .decode_step(&mut stream_state, next)?;
                if let Some(text) = text {
                    for event in decoder.push(&text)? {
                        collected.push(event.clone());
                        if let Some(sender) = &events {
                            // A closed receiver means the client hung up.
                            // Stop rather than finish a generation nobody
                            // is reading — it holds the single generation
                            // slot the whole time (spec §75).
                            if sender.blocking_send(event).is_err() {
                                client_gone = true;
                                break;
                            }
                        }
                    }
                }
                if client_gone || decoder.stopped() {
                    break;
                }

                let input = next;
                let cache_before = runtime.expert_cache_stats();
                // `generated` is this request's history, which is what the
                // repetition penalties score against.
                let decoded = runtime.decode_step(input, &mut sampler, &generated)?;
                let cache_after = runtime.expert_cache_stats();
                emit_decode_diagnostics(input, &decoded, cache_before, cache_after);
                next = decoded.token;
            }

            let (tail, finish) = decoder.finish()?;
            for event in tail {
                collected.push(event.clone());
                if let Some(sender) = &events {
                    if sender.blocking_send(event).is_err() {
                        client_gone = true;
                        break;
                    }
                }
            }

            Ok(DecodeRun {
                events: collected,
                finish,
                reached_eos,
                stopped: decoder.stopped(),
                client_gone,
                completion_tokens: generated.len() as u32,
                prefill,
                decode: decode_started.elapsed(),
            })
        })
        .await
        .map_err(|error| {
            TqfError::Internal(crate::error::InternalError {
                incident_id: format!("resident-reference-generate-{error}"),
                message: "reference decode worker panicked or was cancelled".to_string(),
            })
        })??;

        Ok(decoded.into_output(prompt_tokens))
    }
}

/// What one decode loop produced, before it is shaped into a
/// `GeneratedOutput`.
struct DecodeRun {
    events: Vec<StreamEvent>,
    finish: StreamFinish,
    reached_eos: bool,
    stopped: bool,
    client_gone: bool,
    completion_tokens: u32,
    prefill: std::time::Duration,
    decode: std::time::Duration,
}

impl DecodeRun {
    fn into_output(self, prompt_tokens: u32) -> GeneratedOutput {
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for event in self.events {
            match event {
                // Reasoning is deliberately not part of `text`: the
                // compatibility surfaces expose assistant content, and
                // `<think>` output is not that (see `visible_after_thinking`).
                StreamEvent::Reasoning(_) => {}
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::ToolCall(call) => tool_calls.push(call),
            }
        }

        // Ran to the token budget without EOS, a stop match, or a client
        // disconnect: the answer was cut off, and clients rely on
        // "length" to know that.
        let finish_reason = match self.finish {
            StreamFinish::ToolCalls => "tool_calls",
            StreamFinish::Length => "length",
            StreamFinish::Stop if !self.reached_eos && !self.stopped && !self.client_gone => {
                "length"
            }
            StreamFinish::Stop => "stop",
        };

        GeneratedOutput {
            text,
            tool_calls,
            finish_reason,
            usage: GenerationUsage {
                prompt_tokens,
                completion_tokens: self.completion_tokens,
                prefill: self.prefill,
                decode: self.decode,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_tool_calls_and_preserves_plain_assistant_text() {
        let output = GeneratedOutput::from_model_text(
            "Working.<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.rs\"}}</tool_call>Done.",
        )
        .unwrap();
        assert_eq!(output.text, "Working.Done.");
        assert_eq!(output.finish_reason, "tool_calls");
        assert_eq!(output.tool_calls[0].name, "read_file");
        assert_eq!(output.tool_calls[0].arguments_json, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn extracts_canonical_qwen_xml_tool_calls_and_hides_thinking() {
        let output = GeneratedOutput::from_model_text(
            "private reasoning\n</think>\n\nWorking.\n<tool_call>\n<function=read_file>\n<parameter=path>\nCargo.toml\n</parameter>\n<parameter=line>\n12\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap();
        assert_eq!(output.text.trim(), "Working.");
        assert_eq!(output.finish_reason, "tool_calls");
        assert_eq!(output.tool_calls[0].name, "read_file");
        assert_eq!(
            output.tool_calls[0].arguments_json,
            r#"{"path":"Cargo.toml","line":12}"#
        );
    }

    #[test]
    fn hides_an_incomplete_qwen_reasoning_continuation() {
        let output = GeneratedOutput::from_qwen_continuation(
            "private reasoning that reached the output limit".to_string(),
        )
        .unwrap();
        assert_eq!(output.text, "");
        assert!(output.tool_calls.is_empty());
    }

    #[test]
    fn rejects_unterminated_or_invalid_tool_calls() {
        assert!(GeneratedOutput::from_model_text("<tool_call>{}").is_err());
        assert!(GeneratedOutput::from_model_text("<tool_call>wat</tool_call>").is_err());
        assert!(GeneratedOutput::from_model_text(
            "<tool_call><function=x><parameter=a>1</function></tool_call>"
        )
        .is_err());
    }
}
