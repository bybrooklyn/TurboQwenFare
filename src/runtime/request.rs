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
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json_schema: String,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_output_tokens: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
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
