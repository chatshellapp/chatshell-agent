//! Tool-call history reconstruction.
//!
//! When an assistant turn invoked tools, the replayable LLM projection is not
//! a single message: it is an assistant message carrying `tool_calls`,
//! followed by one tool-role message per executed call, followed by the final
//! assistant text. Both clients encoded this protocol - desktop inline in
//! `commands/chat/message_builder.rs`, iOS in
//! `Services/LLM/AgentClientHistory.swift` - and the iOS copy produced a
//! cross-turn tool-history bug.
//!
//! [`pair_history`] is the single implementation. Clients flatten their
//! persistence rows into [`HistoryRow`]s (over JSON through the FFI on iOS)
//! and convert the returned [`ChatMessage`]s into their local types.

use serde::{Deserialize, Serialize};

use crate::types::{ChatMessage, ChatMessageImage, ChatToolCall, MessageRole};

/// Role of one flattened history row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryRole {
    System,
    User,
    Assistant,
    /// A standalone stored tool-result row. Passed through as a single
    /// tool-role message (legacy iOS storage shape; desktop keeps tool
    /// results inside assistant rows instead).
    Tool,
}

/// One persisted tool invocation, as stored by the client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRow {
    /// Tool-call item identifier.
    pub id: String,
    /// OpenAI Responses API call identifier, when the provider supplied one.
    #[serde(default)]
    pub call_id: Option<String>,
    pub name: String,
    /// Raw JSON arguments the model passed.
    pub arguments_json: String,
    /// Executed output. `None` when the invocation never completed.
    #[serde(default)]
    pub output: Option<String>,
}

/// One flattened history row from client persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryRow {
    pub role: HistoryRole,
    /// Message content. For assistant rows with tool calls this is the final
    /// text emitted after the calls.
    pub content: String,
    /// Assistant text emitted before the first tool call (desktop recovers
    /// it from ordered content blocks; clients without that data pass `""`).
    #[serde(default)]
    pub pre_tool_content: String,
    /// Reasoning content attached to assistant rows.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRow>,
    /// Images attached to user rows (e.g. the pending turn's attachments).
    #[serde(default)]
    pub images: Vec<ChatMessageImage>,
}

/// Expand flattened history rows into the replayable message sequence.
///
/// For an assistant row with tool calls, emits:
///
/// 1. an assistant message carrying `tool_calls` (content =
///    [`HistoryRow::pre_tool_content`], which may be empty),
/// 2. one tool-role message per tool call that has an output
///    (`tool_call_id` = call id, `tool_result_call_id` = Responses API
///    `call_id` when present),
/// 3. a trailing assistant message with the final text, when non-empty.
pub fn pair_history(rows: &[HistoryRow]) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(rows.len());
    for row in rows {
        match row.role {
            HistoryRole::System => messages.push(simple(MessageRole::System, &row.content)),
            HistoryRole::User => {
                let mut message = simple(MessageRole::User, &row.content);
                message.images = row.images.clone();
                messages.push(message);
            }
            HistoryRole::Tool => messages.push(simple(MessageRole::Tool, &row.content)),
            HistoryRole::Assistant if row.tool_calls.is_empty() => {
                let mut message = simple(MessageRole::Assistant, &row.content);
                message.reasoning_content = row.reasoning_content.clone();
                messages.push(message);
            }
            HistoryRole::Assistant => {
                // 1) Assistant message carrying the tool calls.
                messages.push(ChatMessage {
                    role: MessageRole::Assistant,
                    content: row.pre_tool_content.clone(),
                    reasoning_content: row.reasoning_content.clone(),
                    images: Vec::new(),
                    tool_calls: row
                        .tool_calls
                        .iter()
                        .map(|tc| ChatToolCall {
                            id: tc.id.clone(),
                            call_id: tc.call_id.clone(),
                            name: tc.name.clone(),
                            arguments_json: tc.arguments_json.clone(),
                        })
                        .collect(),
                    tool_call_id: None,
                    tool_result_call_id: None,
                });

                // 2) Tool result messages for completed invocations.
                for tc in &row.tool_calls {
                    if let Some(output) = &tc.output {
                        messages.push(ChatMessage {
                            role: MessageRole::Tool,
                            content: output.clone(),
                            reasoning_content: None,
                            images: Vec::new(),
                            tool_calls: Vec::new(),
                            tool_call_id: Some(tc.id.clone()),
                            tool_result_call_id: tc.call_id.clone(),
                        });
                    }
                }

                // 3) Final assistant text after the tool calls, if non-empty.
                if !row.content.trim().is_empty() {
                    messages.push(simple(MessageRole::Assistant, &row.content));
                }
            }
        }
    }
    messages
}

fn simple(role: MessageRole, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.to_string(),
        reasoning_content: None,
        images: Vec::new(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_result_call_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(role: HistoryRole, content: &str) -> HistoryRow {
        HistoryRow {
            role,
            content: content.to_string(),
            pre_tool_content: String::new(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            images: Vec::new(),
        }
    }

    fn tc(id: &str, output: Option<&str>) -> ToolCallRow {
        ToolCallRow {
            id: id.to_string(),
            call_id: None,
            name: "bash".to_string(),
            arguments_json: "{}".to_string(),
            output: output.map(str::to_string),
        }
    }

    #[test]
    fn plain_rows_pass_through() {
        let messages = pair_history(&[
            row(HistoryRole::System, "be brief"),
            row(HistoryRole::User, "hello"),
            row(HistoryRole::Assistant, "hi"),
        ]);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[2].role, MessageRole::Assistant);
        assert_eq!(messages[2].content, "hi");
    }

    #[test]
    fn tool_rows_pass_through() {
        let messages = pair_history(&[row(HistoryRole::Tool, "raw result")]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::Tool);
        assert_eq!(messages[0].content, "raw result");
    }

    #[test]
    fn user_images_carry_through() {
        let mut user = row(HistoryRole::User, "look");
        user.images = vec![ChatMessageImage {
            base64: "abc".to_string(),
            media_type: "image/png".to_string(),
        }];
        let messages = pair_history(&[user]);
        assert_eq!(messages[0].images.len(), 1);
        assert_eq!(messages[0].images[0].base64, "abc");
    }

    #[test]
    fn assistant_without_tools_keeps_reasoning() {
        let messages = pair_history(&[HistoryRow {
            reasoning_content: Some("thinking".to_string()),
            ..row(HistoryRole::Assistant, "answer")
        }]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].reasoning_content.as_deref(), Some("thinking"));
    }

    #[test]
    fn tool_call_turn_expands_three_part() {
        let mut call = tc("t1", Some("ok"));
        call.call_id = Some("call-1".to_string());
        let messages = pair_history(&[HistoryRow {
            content: "final answer".to_string(),
            pre_tool_content: "let me check".to_string(),
            reasoning_content: Some("why".to_string()),
            tool_calls: vec![call],
            ..row(HistoryRole::Assistant, "")
        }]);
        assert_eq!(messages.len(), 3);

        // 1) assistant carrying tool_calls, pre-tool text, reasoning
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[0].content, "let me check");
        assert_eq!(messages[0].reasoning_content.as_deref(), Some("why"));
        assert_eq!(messages[0].tool_calls.len(), 1);
        assert_eq!(messages[0].tool_calls[0].id, "t1");
        assert_eq!(messages[0].tool_calls[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(messages[0].tool_calls[0].arguments_json, "{}");

        // 2) tool result keyed by both identifiers
        assert_eq!(messages[1].role, MessageRole::Tool);
        assert_eq!(messages[1].content, "ok");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(messages[1].tool_result_call_id.as_deref(), Some("call-1"));

        // 3) final assistant text
        assert_eq!(messages[2].role, MessageRole::Assistant);
        assert_eq!(messages[2].content, "final answer");
        assert_eq!(messages[2].reasoning_content, None);
    }

    #[test]
    fn output_less_tool_call_emits_no_tool_message() {
        let messages = pair_history(&[HistoryRow {
            tool_calls: vec![tc("t1", None), tc("t2", Some("done"))],
            ..row(HistoryRole::Assistant, "")
        }]);
        // assistant + one tool result; no final text (empty content)
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].tool_calls.len(), 2);
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("t2"));
    }

    #[test]
    fn whitespace_only_final_text_is_skipped() {
        let messages = pair_history(&[HistoryRow {
            tool_calls: vec![tc("t1", Some("ok"))],
            ..row(HistoryRole::Assistant, "   \n  ")
        }]);
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn multi_turn_interleaving() {
        let messages = pair_history(&[
            row(HistoryRole::User, "q1"),
            HistoryRow {
                tool_calls: vec![tc("t1", Some("r1"))],
                ..row(HistoryRole::Assistant, "a1")
            },
            row(HistoryRole::User, "q2"),
        ]);
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[2].role, MessageRole::Tool);
        assert_eq!(messages[3].role, MessageRole::Assistant);
        assert_eq!(messages[4].role, MessageRole::User);
    }

    #[test]
    fn rows_round_trip_through_json() {
        // The FFI boundary serializes rows as JSON; verify the DTO shape.
        let rows = vec![
            row(HistoryRole::User, "hi"),
            HistoryRow {
                content: "done".to_string(),
                pre_tool_content: "checking".to_string(),
                reasoning_content: Some("hmm".to_string()),
                tool_calls: vec![tc("t1", Some("out"))],
                ..row(HistoryRole::Assistant, "")
            },
        ];
        let json = serde_json::to_string(&rows).unwrap();
        assert!(json.contains("\"pre_tool_content\":\"checking\""));
        assert!(json.contains("\"arguments_json\":\"{}\""));
        assert!(json.contains("\"role\":\"assistant\""));
        let back: Vec<HistoryRow> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rows);
    }
}
