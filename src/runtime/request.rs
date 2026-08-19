//! Normalized internal request representation (spec Part IV, section 26).
//! Every protocol surface converts into this one shape before reaching the
//! scheduler, so the model loop never branches on protocol.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolFlavor {
    OpenAiResponses,
    OpenAiChatCompletions,
    OpenAiEmbeddings,
    Anthropic,
    Ollama,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone)]
pub struct MessageToolCall {
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<MessageToolCall>,
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json_schema: String,
}

/// The one internal sampling representation (spec §153). Every protocol
/// adapter normalizes its own parameter names into this before the model
/// loop sees them, so no protocol-specific spelling reaches the decoder.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// `0.0` means greedy, and is the default. That is deliberate: with a
    /// non-greedy default, an adapter that forgot to set sampling would
    /// silently start sampling, which would quietly invalidate every
    /// greedy-parity qualification record. See `crate::sampling`.
    pub temperature: f32,
    pub top_p: f32,
    /// Keep only the `k` highest-logit candidates. `None` disables.
    pub top_k: Option<u32>,
    /// Relative probability floor, as a fraction of the most likely
    /// candidate. `None` disables.
    pub min_p: Option<f32>,
    /// Explicit RNG seed for reproducible stochastic sampling.
    pub seed: Option<u64>,
    /// llama.cpp/Ollama-style repetition penalty; `1.0` disables.
    pub repeat_penalty: f32,
    /// How many recent tokens the repetition penalties consider.
    pub repeat_last_n: usize,
    /// OpenAI-style penalties; `0.0` disables.
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// Sequences that end generation when produced. Matched against
    /// decoded text, not token ids, because a stop string need not align
    /// to token boundaries (spec §205).
    pub stop_sequences: Vec<String>,
    pub max_output_tokens: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            seed: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            max_output_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RetrievalPolicy {
    #[default]
    Disabled,
    Auto,
}

#[derive(Debug, Clone)]
pub struct VisionInput {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

/// The one internal request shape the runtime understands. OpenAI/Anthropic
/// /Ollama handlers each build one of these at the protocol boundary; the
/// scheduler and model loop never see protocol-specific types.
#[derive(Debug, Clone)]
pub struct NormalizedRequest {
    pub protocol: ProtocolFlavor,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub sampling: SamplingParams,
    pub logical_context_limit: usize,
    pub retrieval: RetrievalPolicy,
    pub vision: Vec<VisionInput>,
    pub stream: bool,
}

impl NormalizedRequest {
    /// Builds a request with no tools/vision/retrieval — what every current
    /// protocol handler needs until tool calling, retrieval, and vision
    /// wiring exist (later phases).
    pub fn new(protocol: ProtocolFlavor, messages: Vec<Message>, stream: bool) -> Self {
        Self {
            protocol,
            messages,
            tools: Vec::new(),
            sampling: SamplingParams::default(),
            logical_context_limit: 0,
            retrieval: RetrievalPolicy::default(),
            vision: Vec::new(),
            stream,
        }
    }
}
