//! Turns a raw model token stream into the deltas a client may actually
//! see (spec §71, §205).
//!
//! Emitting decoded text straight through is wrong in four independent
//! ways, and this type exists to be the single place all four are handled
//! — and, being model-free, the single place all four are testable:
//!
//! 1. **UTF-8 boundaries.** A codepoint routinely spans two tokens. The
//!    tokenizer's incremental decode already withholds an incomplete one
//!    (see `TqfTokenizer::decode_step`), so this type receives only
//!    complete text — but it must not undo that by splitting mid-`char`
//!    itself, which is why every truncation here lands on a
//!    `char_boundary`.
//! 2. **Reasoning.** The Qwen3.6 prompt ends with an open `<think>`, so
//!    the first tokens of every generation are private reasoning. They are
//!    reported as [`StreamEvent::Reasoning`] rather than dropped silently
//!    or leaked as content — on a reasoning model this can run for
//!    hundreds of tokens, and a client that streams nothing that whole
//!    time looks hung.
//! 3. **Tool calls.** A partially-written `<tool_call>` block must never
//!    reach a client as visible text; it is buffered until it closes and
//!    then emitted as one parsed [`StreamEvent::ToolCall`].
//! 4. **Stop sequences.** A stop string need not align to token
//!    boundaries, so matching happens on decoded text and the matcher
//!    retains the longest suffix that could still become a match.
//!
//! The invariant behind all of it: **a byte already emitted cannot be
//! retracted.** So text is held back whenever its tail could still turn
//! out to be the start of a delimiter or a stop sequence.

use crate::error::Result;
use crate::runtime::generation::{parse_tool_call, GeneratedToolCall};
use crate::tokenizer::chat::{THINK_END, TOOL_CALL_END, TOOL_CALL_START};

/// One externally visible step of a generation.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Private chain-of-thought, before `</think>`. Adapters decide
    /// whether to surface it; the OpenAI `content` field must not.
    Reasoning(String),
    /// Assistant-visible text.
    TextDelta(String),
    /// A complete, parsed tool call.
    ToolCall(GeneratedToolCall),
}

/// Why a generation ended, in the crate's existing vocabulary
/// (`GeneratedOutput::finish_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFinish {
    Stop,
    ToolCalls,
    Length,
}

impl StreamFinish {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::ToolCalls => "tool_calls",
            Self::Length => "length",
        }
    }
}

#[derive(Debug)]
pub struct IncrementalOutputDecoder {
    /// Before `</think>`: everything is reasoning.
    thinking: bool,
    /// Decoded text received but not yet safe to emit.
    pending: String,
    in_tool_call: bool,
    tool_buffer: String,
    tool_call_count: usize,
    stop_sequences: Vec<String>,
    /// Set once a stop sequence matched; further input is discarded so a
    /// late token cannot append past the stop.
    stopped: bool,
    /// Reproduces `visible_after_thinking`'s `trim_start_matches(['\r',
    /// '\n'])`, which the non-streaming path applies to the text right
    /// after `</think>`.
    trimming_leading_newlines: bool,
}

impl IncrementalOutputDecoder {
    /// `thinking` is true for a live generation, whose prompt already
    /// opened `<think>`. Replaying stored text that contains no thinking
    /// block passes false.
    pub fn new(stop_sequences: Vec<String>, thinking: bool) -> Self {
        Self {
            thinking,
            pending: String::new(),
            in_tool_call: false,
            tool_buffer: String::new(),
            tool_call_count: 0,
            stop_sequences: stop_sequences
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect(),
            stopped: false,
            trimming_leading_newlines: false,
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Feeds newly decoded text and returns whatever became safe to emit.
    pub fn push(&mut self, text: &str) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        if self.stopped || text.is_empty() {
            return Ok(events);
        }
        self.pending.push_str(text);
        self.drain(&mut events, false)?;
        Ok(events)
    }

    /// Flushes whatever is still held back, now that no more text is
    /// coming and nothing held can still grow into a delimiter.
    ///
    /// An unterminated `<tool_call>` is the one case that differs from the
    /// non-streaming `GeneratedOutput::from_model_text`, which raises a
    /// protocol error. A stream cannot retract the bytes it already sent,
    /// so it reports `Length` and drops the partial block rather than
    /// leaking half a tool call as visible text.
    pub fn finish(&mut self) -> Result<(Vec<StreamEvent>, StreamFinish)> {
        let mut events = Vec::new();
        if !self.stopped {
            self.drain(&mut events, true)?;
        }

        let finish = if self.in_tool_call {
            StreamFinish::Length
        } else if self.tool_call_count > 0 {
            StreamFinish::ToolCalls
        } else {
            StreamFinish::Stop
        };
        Ok((events, finish))
    }

    /// `final_flush` releases text that was only being held in case it
    /// grew into a delimiter.
    fn drain(&mut self, events: &mut Vec<StreamEvent>, final_flush: bool) -> Result<()> {
        loop {
            if self.stopped {
                self.pending.clear();
                return Ok(());
            }

            if self.thinking {
                if let Some(at) = self.pending.find(THINK_END) {
                    let reasoning: String = self.pending[..at].into();
                    if !reasoning.is_empty() {
                        events.push(StreamEvent::Reasoning(reasoning));
                    }
                    self.pending = self.pending[at + THINK_END.len()..].to_string();
                    self.thinking = false;
                    self.trimming_leading_newlines = true;
                    continue;
                }
                // `</think>` can itself arrive split across tokens.
                let hold = suffix_that_could_grow_into(&self.pending, &[THINK_END]);
                let emit = self.take_prefix(self.pending.len() - hold, final_flush);
                if !emit.is_empty() {
                    events.push(StreamEvent::Reasoning(emit));
                }
                return Ok(());
            }

            if self.in_tool_call {
                if let Some(at) = self.tool_buffer_end() {
                    let body = self.tool_buffer[..at].trim().to_string();
                    let call = parse_tool_call(&body, self.tool_call_count)?;
                    self.tool_call_count += 1;
                    events.push(StreamEvent::ToolCall(call));
                    self.pending = self.tool_buffer[at + TOOL_CALL_END.len()..].to_string();
                    self.tool_buffer.clear();
                    self.in_tool_call = false;
                    continue;
                }
                // Still open: hold everything. A partial block must never
                // be emitted as text.
                self.tool_buffer.push_str(&self.pending);
                self.pending.clear();
                return Ok(());
            }

            if self.trimming_leading_newlines {
                let trimmed = self.pending.trim_start_matches(['\r', '\n']);
                if trimmed.len() != self.pending.len() {
                    self.pending = trimmed.to_string();
                }
                // Only stop trimming once there is a non-newline byte to
                // prove the run ended; otherwise a newline arriving in the
                // next chunk would slip through.
                if !self.pending.is_empty() {
                    self.trimming_leading_newlines = false;
                } else if !final_flush {
                    return Ok(());
                }
            }

            // A completed stop sequence ends the generation at the match.
            if let Some((at, _)) = self.first_stop_match() {
                let emit: String = self.pending[..at].into();
                if !emit.is_empty() {
                    events.push(StreamEvent::TextDelta(emit));
                }
                self.pending.clear();
                self.stopped = true;
                return Ok(());
            }

            if let Some(at) = self.pending.find(TOOL_CALL_START) {
                let emit: String = self.pending[..at].into();
                if !emit.is_empty() {
                    events.push(StreamEvent::TextDelta(emit));
                }
                self.tool_buffer = self.pending[at + TOOL_CALL_START.len()..].to_string();
                self.pending.clear();
                self.in_tool_call = true;
                continue;
            }

            // Nothing matched yet, so emit everything except the tail that
            // could still become a tool-call opener or a stop sequence.
            let mut watched: Vec<&str> = vec![TOOL_CALL_START];
            watched.extend(self.stop_sequences.iter().map(String::as_str));
            let hold = suffix_that_could_grow_into(&self.pending, &watched);
            let emit = self.take_prefix(self.pending.len() - hold, final_flush);
            if !emit.is_empty() {
                events.push(StreamEvent::TextDelta(emit));
            }
            return Ok(());
        }
    }

    /// Removes and returns `pending[..upto]`, rounded down to a `char`
    /// boundary. `final_flush` releases everything, since nothing held can
    /// still grow.
    fn take_prefix(&mut self, upto: usize, final_flush: bool) -> String {
        let upto = if final_flush {
            self.pending.len()
        } else {
            upto
        };
        let mut boundary = upto.min(self.pending.len());
        while boundary > 0 && !self.pending.is_char_boundary(boundary) {
            boundary -= 1;
        }
        if boundary == 0 {
            return String::new();
        }
        let head: String = self.pending[..boundary].into();
        self.pending = self.pending[boundary..].to_string();
        head
    }

    fn tool_buffer_end(&mut self) -> Option<usize> {
        // The closing tag may span the buffer/pending split.
        self.tool_buffer.push_str(&self.pending);
        self.pending.clear();
        self.tool_buffer.find(TOOL_CALL_END)
    }

    /// Earliest completed stop-sequence match in `pending`.
    fn first_stop_match(&self) -> Option<(usize, &str)> {
        self.stop_sequences
            .iter()
            .filter_map(|stop| {
                self.pending
                    .find(stop.as_str())
                    .map(|at| (at, stop.as_str()))
            })
            .min_by_key(|(at, _)| *at)
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of any
/// `watched` string — i.e. exactly how much must be held back so no
/// partial delimiter or stop sequence is ever emitted.
///
/// Computing it per-call rather than always holding `max_len - 1` bytes
/// means ordinary text (which usually shares no prefix with `<tool_call>`)
/// streams with no delay at all.
fn suffix_that_could_grow_into(text: &str, watched: &[&str]) -> usize {
    let mut longest = 0;
    for candidate in watched {
        let max = candidate.len().saturating_sub(1).min(text.len());
        for length in (1..=max).rev() {
            if length <= longest {
                break;
            }
            let start = text.len() - length;
            if !text.is_char_boundary(start) {
                continue;
            }
            if candidate.as_bytes().starts_with(&text.as_bytes()[start..]) {
                longest = length;
                break;
            }
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::generation::GeneratedOutput;

    /// Feeds `text` one chunk at a time and collects everything emitted.
    fn run(chunks: &[&str], stops: &[&str]) -> (Vec<StreamEvent>, StreamFinish) {
        let mut decoder =
            IncrementalOutputDecoder::new(stops.iter().map(|s| s.to_string()).collect(), true);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.push(chunk).expect("push must succeed"));
        }
        let (tail, finish) = decoder.finish().expect("finish must succeed");
        events.extend(tail);
        (events, finish)
    }

    fn visible(events: &[StreamEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(events: &[StreamEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Reasoning(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn tool_calls(events: &[StreamEvent]) -> Vec<&GeneratedToolCall> {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    /// Splits a string into every possible chunking at `size` boundaries,
    /// so a test can prove behavior is independent of where tokens land.
    fn chunked(text: &str, size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if current.len() >= size {
                chunks.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    }

    fn run_chunked(text: &str, size: usize, stops: &[&str]) -> (Vec<StreamEvent>, StreamFinish) {
        let owned = chunked(text, size);
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        run(&refs, stops)
    }

    // ---------------------------------------------------------- reasoning

    #[test]
    fn nothing_visible_is_emitted_before_the_think_close_tag() {
        let (events, _) = run(&["reasoning about it", " some more"], &[]);
        assert_eq!(visible(&events), "");
        assert_eq!(reasoning(&events), "reasoning about it some more");
    }

    #[test]
    fn text_after_the_think_close_tag_becomes_visible() {
        let (events, finish) = run(&["thinking</think>\n\nHello there"], &[]);
        assert_eq!(reasoning(&events), "thinking");
        assert_eq!(visible(&events), "Hello there");
        assert_eq!(finish, StreamFinish::Stop);
    }

    /// `</think>` arriving one character at a time must still gate
    /// correctly — this is the boundary case that a naive `contains` check
    /// on each chunk gets wrong.
    #[test]
    fn a_think_close_tag_split_across_chunks_still_gates() {
        for size in 1..=8 {
            let (events, _) = run_chunked("mulling</think>visible text", size, &[]);
            assert_eq!(visible(&events), "visible text", "chunk size {size}");
            assert_eq!(reasoning(&events), "mulling", "chunk size {size}");
        }
    }

    /// The non-streaming path trims `\r\n` right after `</think>`; the
    /// streaming path must produce the same visible text.
    #[test]
    fn leading_newlines_after_thinking_are_trimmed_even_across_chunks() {
        let (events, _) = run(&["x</think>", "\n", "\n", "Answer"], &[]);
        assert_eq!(visible(&events), "Answer");
    }

    // ------------------------------------------------------------- stops

    #[test]
    fn a_stop_sequence_truncates_at_the_match_and_ends_generation() {
        let (events, _) = run(&["</think>keep this STOPdrop this"], &["STOP"]);
        assert_eq!(visible(&events), "keep this ");
    }

    /// The case a chunk-local `contains` check cannot catch.
    #[test]
    fn a_stop_sequence_spanning_chunks_still_matches() {
        for size in 1..=6 {
            let (events, _) = run_chunked("</think>alpha ENDbeta", size, &["END"]);
            assert_eq!(visible(&events), "alpha ", "chunk size {size}");
        }
    }

    /// A held-back suffix that turns out *not* to be a stop must be
    /// released, not swallowed. Without this, text ending in a stop prefix
    /// would silently vanish.
    #[test]
    fn a_stop_prefix_that_does_not_complete_is_released() {
        let (events, _) = run(&["</think>done ST"], &["STOP"]);
        assert_eq!(visible(&events), "done ST");

        let (events, _) = run(&["</think>done ST", "ART"], &["STOP"]);
        assert_eq!(visible(&events), "done START");
    }

    #[test]
    fn nothing_after_a_stop_is_emitted() {
        let mut decoder = IncrementalOutputDecoder::new(vec!["HALT".into()], true);
        let mut events = decoder.push("</think>before HALT").unwrap();
        assert!(decoder.stopped());
        events.extend(decoder.push("after").unwrap());
        assert_eq!(visible(&events), "before ");
    }

    // -------------------------------------------------------- tool calls

    #[test]
    fn a_complete_tool_call_emits_one_event_and_no_visible_text() {
        let (events, finish) = run(
            &[r#"</think><tool_call>{"name":"ls","arguments":{"path":"/"}}</tool_call>"#],
            &[],
        );
        assert_eq!(visible(&events), "");
        let calls = tool_calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
        assert_eq!(finish, StreamFinish::ToolCalls);
    }

    /// The hazard this whole buffer exists for: a client must never see
    /// raw `<tool_call>` JSON rendered as assistant text.
    #[test]
    fn a_partial_tool_call_never_leaks_as_visible_text() {
        let mut decoder = IncrementalOutputDecoder::new(Vec::new(), true);
        let mut events = Vec::new();
        for chunk in [
            "</think>Let me check. ",
            "<tool_call>",
            r#"{"name":"ls","#,
            r#""arguments":{"path":"/"}}"#,
        ] {
            events.extend(decoder.push(chunk).unwrap());
        }
        // The block is still open, so only the prose before it is visible.
        assert_eq!(visible(&events), "Let me check. ");
        assert!(!visible(&events).contains("tool_call"));
        assert!(!visible(&events).contains("arguments"));
        assert!(tool_calls(&events).is_empty());
    }

    /// A generation that runs out of budget mid-block reports `Length` and
    /// drops the partial rather than leaking it — a stream cannot retract
    /// bytes, so it cannot raise the protocol error the batch path does.
    #[test]
    fn an_unterminated_tool_call_reports_length_and_leaks_nothing() {
        let (events, finish) = run(&[r#"</think>hi <tool_call>{"name":"l"#], &[]);
        assert_eq!(visible(&events), "hi ");
        assert!(tool_calls(&events).is_empty());
        assert_eq!(finish, StreamFinish::Length);
    }

    #[test]
    fn a_tool_call_split_across_many_chunks_yields_one_event() {
        for size in 1..=7 {
            let (events, _) = run_chunked(
                r#"</think>ok <tool_call>{"name":"grep","arguments":{"q":"x"}}</tool_call> done"#,
                size,
                &[],
            );
            let calls = tool_calls(&events);
            assert_eq!(calls.len(), 1, "chunk size {size}");
            assert_eq!(calls[0].name, "grep", "chunk size {size}");
            assert_eq!(visible(&events), "ok  done", "chunk size {size}");
        }
    }

    #[test]
    fn the_xml_function_tool_call_form_is_parsed_too() {
        let (events, _) = run(
            &["</think><tool_call><function=ls><parameter=path>/tmp</parameter></function></tool_call>"],
            &[],
        );
        let calls = tool_calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
    }

    // ------------------------------------------------------------- UTF-8

    /// Multi-byte text must survive arbitrary chunking without a
    /// replacement character or a split `char`.
    #[test]
    fn multibyte_text_is_never_split_mid_character() {
        let text = "</think>日本語のテキストと絵文字 🎉🚀 と混在";
        for size in 1..=12 {
            let (events, _) = run_chunked(text, size, &[]);
            let out = visible(&events);
            assert!(!out.contains('\u{FFFD}'), "chunk size {size}: {out}");
            assert_eq!(
                out, "日本語のテキストと絵文字 🎉🚀 と混在",
                "chunk size {size}"
            );
        }
    }

    #[test]
    fn holding_back_a_delimiter_prefix_respects_char_boundaries() {
        // '<' can start `<tool_call>`, and the surrounding text is
        // multi-byte, so a byte-wise hold would split a character.
        let (events, _) = run(&["</think>テスト<"], &[]);
        assert_eq!(visible(&events), "テスト<");
    }

    // ------------------------------------------------------- differential

    /// The property that keeps the two paths honest: streaming a
    /// generation and batch-parsing the same text must produce the same
    /// visible content and the same tool calls. Without this, "streaming
    /// says X, non-streaming says Y" is a whole live bug class.
    #[test]
    fn streamed_output_matches_the_batch_parser_on_the_same_text() {
        let cases = [
            "</think>plain answer",
            "</think>\n\ntrimmed answer",
            "thinking first</think>then answering",
            r#"</think>before <tool_call>{"name":"a","arguments":{}}</tool_call> after"#,
            r#"</think><tool_call>{"name":"a","arguments":{"x":1}}</tool_call><tool_call>{"name":"b","arguments":{"y":2}}</tool_call>"#,
            "</think>unicode 日本語 🎉 mixed",
        ];

        for text in cases {
            let batch = GeneratedOutput::from_model_text(text).expect("batch parse");
            for size in [1usize, 3, 7, 64] {
                let (events, _) = run_chunked(text, size, &[]);
                assert_eq!(
                    visible(&events),
                    batch.text,
                    "text {text:?} at chunk size {size}"
                );
                assert_eq!(
                    tool_calls(&events).len(),
                    batch.tool_calls.len(),
                    "tool call count for {text:?} at chunk size {size}"
                );
                for (streamed, expected) in tool_calls(&events).iter().zip(&batch.tool_calls) {
                    assert_eq!(streamed.name, expected.name);
                    assert_eq!(streamed.arguments_json, expected.arguments_json);
                }
            }
        }
    }

    /// Ordinary text must not be delayed waiting for a delimiter that
    /// cannot occur — otherwise every generation streams in one burst at
    /// the end, which is the bug this module replaces.
    #[test]
    fn ordinary_text_is_emitted_immediately_rather_than_buffered() {
        let mut decoder = IncrementalOutputDecoder::new(vec!["STOP".into()], true);
        assert!(decoder.push("</think>Hello").unwrap().len() == 1);
        let events = decoder.push(" world").unwrap();
        assert_eq!(
            visible(&events),
            " world",
            "second chunk must emit on arrival"
        );
    }

    #[test]
    fn an_empty_push_produces_no_events() {
        let mut decoder = IncrementalOutputDecoder::new(Vec::new(), true);
        assert!(decoder.push("").unwrap().is_empty());
    }
}
