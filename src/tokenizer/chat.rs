//! Qwen chat-template rendering (spec §281 phase 9 golden-fixture list:
//! "normal system/user/assistant; developer guidance; historical thinking
//! if supported; tool definitions/results; vision placeholders;
//! Unicode/byte fallback"). This is a native rendering of the pinned
//! Qwen3.6 template's text/tool subset, including its XML function-call
//! wire format and thinking prefix,
//! reimplemented natively in Rust rather than interpreting the
//! GGUF-embedded Jinja2 template string — a general Jinja engine
//! dependency for one fixed model family's fixed template shape is out of
//! scope (spec §114: no generic frameworks where a native implementation
//! suffices).

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const TOOL_CALL_START: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const TOOL_RESPONSE_START: &str = "<tool_response>";
pub const TOOL_RESPONSE_END: &str = "</tool_response>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";

const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    /// Folded into the same wire role as `System` when rendered — Qwen's
    /// ChatML template has no distinct "developer" turn, and TQF's own
    /// `runtime::request::Role` doesn't carry one either — but kept as its
    /// own variant here so callers preserve *why* a system-turn message
    /// exists until render time.
    Developer,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    fn wire_role(self) -> &'static str {
        match self {
            ChatRole::System | ChatRole::Developer => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            // Tool results are fed back as a user-turn wrapping
            // `<tool_response>`, matching Qwen's published template (no
            // dedicated "tool" role token in the pinned checkpoint).
            ChatRole::Tool => "user",
        }
    }
}

#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    /// An out-of-band image reference (spec §321 vision: pixel/embedding
    /// wiring is a later phase). The template only needs to emit the
    /// marker token(s) the model was trained to expect around each image.
    Image,
}

#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ContentPart>,
    /// Reasoning content from a *prior, already-committed* assistant turn
    /// (spec: "historical thinking if supported"). The template never
    /// generates a `<think>` opening tag for a *new* assistant turn; the
    /// generation prompt below does that exactly as the pinned template does.
    pub thinking: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(content.into())],
            thinking: None,
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// A JSON Schema document, already serialized (the caller/protocol
    /// layer owns JSON schema construction; the template just embeds it).
    pub parameters_json_schema: String,
}

/// Renders a full conversation into the exact text the tokenizer should
/// encode. `add_generation_prompt` opens a trailing, unclosed assistant
/// turn for the model to continue — the standard shape for an inference
/// request as opposed to encoding a completed transcript.
pub fn render(messages: &[ChatMessage], tools: &[ToolSpec], add_generation_prompt: bool) -> String {
    let mut out = String::new();

    let leading_system_text = messages
        .first()
        .filter(|m| matches!(m.role, ChatRole::System | ChatRole::Developer))
        .map(|m| render_content(&m.content));

    if !tools.is_empty() {
        out.push_str(IM_START);
        out.push_str("system\n");
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools {
            let parameters =
                serde_json::from_str::<serde_json::Value>(&tool.parameters_json_schema)
                    .unwrap_or_else(|_| serde_json::json!({"type": "object"}));
            let wire = serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                }
            });
            out.push('\n');
            out.push_str(&wire.to_string());
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_INSTRUCTIONS);
        if let Some(text) = &leading_system_text {
            if !text.trim().is_empty() {
                out.push_str("\n\n");
                out.push_str(text.trim());
            }
        }
        out.push_str(IM_END);
        out.push('\n');
    }

    let skip_first = !tools.is_empty() && leading_system_text.is_some();
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        if i == 0 && skip_first {
            i += 1;
            continue;
        }
        if message.role == ChatRole::Tool {
            out.push_str(IM_START);
            out.push_str("user");
            while i < messages.len() && messages[i].role == ChatRole::Tool {
                out.push('\n');
                out.push_str(TOOL_RESPONSE_START);
                out.push('\n');
                out.push_str(render_content(&messages[i].content).trim());
                out.push('\n');
                out.push_str(TOOL_RESPONSE_END);
                i += 1;
            }
            out.push_str(IM_END);
            out.push('\n');
            continue;
        }
        render_message(&mut out, message);
        i += 1;
    }

    if add_generation_prompt {
        out.push_str(IM_START);
        out.push_str("assistant\n");
        out.push_str(THINK_START);
        out.push('\n');
    }

    out
}

fn render_message(out: &mut String, message: &ChatMessage) {
    out.push_str(IM_START);
    out.push_str(message.role.wire_role());
    out.push('\n');

    if message.role != ChatRole::Tool {
        if let Some(thinking) = &message.thinking {
            out.push_str(THINK_START);
            out.push('\n');
            out.push_str(thinking);
            out.push('\n');
            out.push_str(THINK_END);
            out.push('\n');
        }
        let content = render_content(&message.content);
        let content = content.trim();
        out.push_str(content);
        for (index, call) in message.tool_calls.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            } else if !content.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(TOOL_CALL_START);
            out.push('\n');
            out.push_str("<function=");
            out.push_str(&call.name);
            out.push_str(">\n");
            if let Ok(serde_json::Value::Object(arguments)) =
                serde_json::from_str::<serde_json::Value>(&call.arguments_json)
            {
                for (name, value) in arguments {
                    out.push_str("<parameter=");
                    out.push_str(&name);
                    out.push_str(">\n");
                    match value {
                        serde_json::Value::String(value) => out.push_str(&value),
                        value => out.push_str(&value.to_string()),
                    }
                    out.push_str("\n</parameter>\n");
                }
            }
            out.push_str("</function>\n");
            out.push_str(TOOL_CALL_END);
        }
    }

    out.push_str(IM_END);
    out.push('\n');
}

fn render_content(parts: &[ContentPart]) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            ContentPart::Text(text) => out.push_str(text),
            ContentPart::Image => {
                out.push_str(VISION_START);
                out.push_str(IMAGE_PAD);
                out.push_str(VISION_END);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_system_user_assistant() {
        let messages = vec![
            ChatMessage::text(ChatRole::System, "You are helpful."),
            ChatMessage::text(ChatRole::User, "Hi"),
            ChatMessage::text(ChatRole::Assistant, "Hello!"),
        ];
        let rendered = render(&messages, &[], false);
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n"
        );
    }

    #[test]
    fn add_generation_prompt_opens_trailing_assistant_turn() {
        let messages = vec![ChatMessage::text(ChatRole::User, "Hi")];
        let rendered = render(&messages, &[], true);
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn developer_guidance_renders_as_a_system_turn() {
        let messages = vec![ChatMessage::text(ChatRole::Developer, "Be terse.")];
        let rendered = render(&messages, &[], false);
        assert_eq!(rendered, "<|im_start|>system\nBe terse.<|im_end|>\n");
    }

    #[test]
    fn historical_thinking_is_wrapped_in_think_tags() {
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "42");
        assistant.thinking = Some("6 * 7 = 42".to_string());
        let rendered = render(&[assistant], &[], false);
        assert_eq!(
            rendered,
            "<|im_start|>assistant\n<think>\n6 * 7 = 42\n</think>\n42<|im_end|>\n"
        );
    }

    #[test]
    fn tool_definitions_are_embedded_in_the_system_turn() {
        let tools = vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "Look up the weather".to_string(),
            parameters_json_schema: r#"{"type":"object","properties":{}}"#.to_string(),
        }];
        let messages = vec![ChatMessage::text(ChatRole::User, "What's the weather?")];
        let rendered = render(&messages, &tools, false);
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("\"name\":\"get_weather\""));
        assert!(rendered.contains("</tools>"));
        assert!(rendered.contains("<function=example_function_name>"));
    }

    #[test]
    fn tool_call_and_tool_result_round_trip_shape() {
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "");
        assistant.tool_calls.push(ToolCall {
            name: "get_weather".to_string(),
            arguments_json: r#"{"city":"Boston"}"#.to_string(),
        });
        let tool_result = ChatMessage::text(ChatRole::Tool, r#"{"tempF":72}"#);

        let rendered = render(&[assistant, tool_result], &[], false);
        assert!(rendered.contains("<tool_call>"));
        assert!(rendered.contains("<function=get_weather>"));
        assert!(rendered.contains("<parameter=city>\nBoston\n</parameter>"));
        assert!(rendered.contains("</tool_call>"));
        assert!(rendered.contains("<tool_response>"));
        assert!(rendered.contains(r#"{"tempF":72}"#));
        assert!(rendered.contains("</tool_response>"));
    }

    #[test]
    fn multiple_tool_calls_follow_the_pinned_single_newline_separator() {
        let mut assistant = ChatMessage::text(ChatRole::Assistant, "planning");
        assistant.tool_calls = vec![
            ToolCall {
                name: "first".to_string(),
                arguments_json: "{}".to_string(),
            },
            ToolCall {
                name: "second".to_string(),
                arguments_json: "{}".to_string(),
            },
        ];
        let rendered = render(&[assistant], &[], false);
        assert!(rendered.contains("planning\n\n<tool_call>"));
        assert!(rendered.contains("</tool_call>\n<tool_call>\n<function=second>"));
    }

    #[test]
    fn vision_placeholder_emits_marker_tokens() {
        let message = ChatMessage {
            role: ChatRole::User,
            content: vec![
                ContentPart::Text("What's in this image? ".to_string()),
                ContentPart::Image,
            ],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let rendered = render(&[message], &[], false);
        assert!(rendered.contains(VISION_START));
        assert!(rendered.contains(IMAGE_PAD));
        assert!(rendered.contains(VISION_END));
    }
}
