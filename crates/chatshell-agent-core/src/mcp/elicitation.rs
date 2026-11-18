//! Elicitation (`elicitation/create`): a server asks the user for input
//! during a call, as a schema-described form or a URL to open.
//!
//! The platform shows the dialog ([`McpElicitationPrompter`]); this module
//! owns the rules every platform shares (INV-2, INV-1):
//! - never auto-approved: without a prompter the request is declined;
//! - a timeout or a server-side cancellation answers `decline`;
//! - an accepted form is validated against the requested schema and its
//!   strings are scanned (blocked paths) before anything returns to the
//!   server; an invalid or blocked answer is declined.

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use rmcp::model::{ElicitRequestParams, ElicitResult, ElicitationAction};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::path_policy;

/// How long the user has to answer before the request is declined.
pub const ELICITATION_TIMEOUT: Duration = Duration::from_secs(120);

/// What the server asks for, in platform-neutral form.
#[derive(Debug, Clone, PartialEq)]
pub enum McpElicitationRequest {
    /// Fill in a form. `schema` is the requested JSON Schema (an object with
    /// primitive-typed properties).
    Form { message: String, schema: Value },
    /// Open `url` in the browser (only on the user's click).
    Url { message: String, url: String },
}

/// The user's answer.
#[derive(Debug, Clone, PartialEq)]
pub enum McpElicitationAnswer {
    /// Form submitted with `content` (an object), or URL opened (`Null`).
    Accept(Value),
    /// Declined; the call may continue without the input.
    Decline,
    /// Cancelled the whole operation.
    Cancel,
}

/// Platform dialog for elicitation requests.
pub trait McpElicitationPrompter: Send + Sync + 'static {
    /// Ask the user. `server` is the user-facing server name. The future
    /// is dropped when the request times out or the server cancels it; the
    /// prompter should then dismiss its dialog (see `dismissed`).
    fn prompt(
        &self,
        server: String,
        request: McpElicitationRequest,
        dismissed: CancellationToken,
    ) -> BoxFuture<'static, McpElicitationAnswer>;
}

/// Convert the wire request; `None` for a mode this client does not know
/// (declined).
pub fn to_request(params: &ElicitRequestParams) -> Option<McpElicitationRequest> {
    Some(match params {
        ElicitRequestParams::FormElicitationParams {
            message,
            requested_schema,
            ..
        } => McpElicitationRequest::Form {
            message: message.clone(),
            schema: serde_json::to_value(requested_schema).unwrap_or(Value::Null),
        },
        ElicitRequestParams::UrlElicitationParams { message, url, .. } => {
            McpElicitationRequest::Url {
                message: message.clone(),
                url: url.clone(),
            }
        }
        _ => return None,
    })
}

fn type_matches(kind: &str, value: &Value) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_f64().is_some_and(|n| n.fract() == 0.0),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        _ => false,
    }
}

/// Allowed values of an enum-like definition (`enum`, or `oneOf`/`anyOf`
/// of `const`s), if it is one.
fn allowed_values(definition: &Map<String, Value>) -> Option<Vec<Value>> {
    if let Some(values) = definition.get("enum").and_then(Value::as_array) {
        return Some(values.clone());
    }
    for key in ["oneOf", "anyOf"] {
        if let Some(options) = definition.get(key).and_then(Value::as_array) {
            let consts: Vec<Value> = options
                .iter()
                .filter_map(|o| o.get("const").cloned())
                .collect();
            if !consts.is_empty() {
                return Some(consts);
            }
        }
    }
    None
}

fn check_property(name: &str, definition: &Value, value: &Value) -> Result<(), String> {
    let Some(definition) = definition.as_object() else {
        return Err(format!("'{name}' has no usable definition"));
    };
    let kind = definition
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("string");
    if !type_matches(kind, value) {
        return Err(format!("'{name}' must be a {kind}"));
    }
    if let Some(allowed) = allowed_values(definition) {
        let ok = match value {
            Value::Array(items) => items.iter().all(|item| allowed.contains(item)),
            other => allowed.contains(other),
        };
        if !ok {
            return Err(format!("'{name}' is not one of the allowed values"));
        }
    }
    if kind == "array" {
        let items = definition.get("items").and_then(Value::as_object);
        if let (Some(items), Value::Array(values)) = (items, value)
            && let Some(allowed) = allowed_values(items)
            && !values.iter().all(|v| allowed.contains(v))
        {
            return Err(format!("'{name}' contains a value that is not allowed"));
        }
    }
    if let Some(text) = value.as_str() {
        let len = text.chars().count() as u64;
        if definition
            .get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|min| len < min)
        {
            return Err(format!("'{name}' is too short"));
        }
        if definition
            .get("maxLength")
            .and_then(Value::as_u64)
            .is_some_and(|max| len > max)
        {
            return Err(format!("'{name}' is too long"));
        }
    }
    if let Some(n) = value.as_f64() {
        if definition
            .get("minimum")
            .and_then(Value::as_f64)
            .is_some_and(|min| n < min)
        {
            return Err(format!("'{name}' is below the minimum"));
        }
        if definition
            .get("maximum")
            .and_then(Value::as_f64)
            .is_some_and(|max| n > max)
        {
            return Err(format!("'{name}' is above the maximum"));
        }
    }
    Ok(())
}

/// Validate form content against the requested schema: an object, only
/// declared properties, required ones present, primitive types, enums and
/// bounds respected.
pub fn validate_form_content(schema: &Value, content: &Value) -> Result<(), String> {
    let Some(content) = content.as_object() else {
        return Err("the answer must be an object".to_string());
    };
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (name, value) in content {
        let Some(definition) = properties.get(name) else {
            return Err(format!("'{name}' was not requested"));
        };
        check_property(name, definition, value)?;
    }
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !content.contains_key(name) {
                return Err(format!("'{name}' is required"));
            }
        }
    }
    Ok(())
}

fn collect_strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
    match value {
        Value::String(s) => out.push(s),
        Value::Array(items) => items.iter().for_each(|v| collect_strings(v, out)),
        Value::Object(map) => map.values().for_each(|v| collect_strings(v, out)),
        _ => {}
    }
}

/// INV-1: scan an answer's strings before it reaches the server.
pub fn scan_answer(content: &Value) -> Result<(), String> {
    let mut strings = Vec::new();
    collect_strings(content, &mut strings);
    for s in strings {
        path_policy::check_blocked_path(s)?;
    }
    Ok(())
}

fn decline() -> ElicitResult {
    ElicitResult::new(ElicitationAction::Decline)
}

/// Resolve one elicitation: ask, enforce the timeout and the server's
/// cancellation, then validate and scan.
pub async fn resolve(
    prompter: Option<Arc<dyn McpElicitationPrompter>>,
    server: &str,
    params: &ElicitRequestParams,
    server_cancelled: CancellationToken,
    timeout: Duration,
) -> ElicitResult {
    let Some(prompter) = prompter else {
        return decline();
    };
    let Some(request) = to_request(params) else {
        return decline();
    };
    let dismissed = CancellationToken::new();
    let ask = prompter.prompt(server.to_string(), request.clone(), dismissed.clone());
    let answer = tokio::select! {
        answer = ask => Some(answer),
        _ = tokio::time::sleep(timeout) => None,
        _ = server_cancelled.cancelled() => None,
    };
    let Some(answer) = answer else {
        dismissed.cancel();
        tracing::info!("MCP elicitation from '{server}' closed without an answer; declining");
        return decline();
    };
    match answer {
        McpElicitationAnswer::Decline => decline(),
        McpElicitationAnswer::Cancel => ElicitResult::new(ElicitationAction::Cancel),
        McpElicitationAnswer::Accept(content) => match &request {
            McpElicitationRequest::Url { .. } => ElicitResult::new(ElicitationAction::Accept),
            McpElicitationRequest::Form { schema, .. } => {
                if let Err(reason) =
                    validate_form_content(schema, &content).and_then(|_| scan_answer(&content))
                {
                    tracing::warn!("MCP elicitation answer for '{server}' rejected: {reason}");
                    return decline();
                }
                ElicitResult::new(ElicitationAction::Accept).with_content(content)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 10},
                "age": {"type": "integer", "minimum": 0},
                "plan": {"type": "string", "enum": ["free", "pro"]},
                "notify": {"type": "boolean"}
            },
            "required": ["name"]
        })
    }

    #[test]
    fn valid_content_passes() {
        let content = json!({"name": "Ada", "age": 36, "plan": "pro", "notify": true});
        assert!(validate_form_content(&schema(), &content).is_ok());
    }

    #[test]
    fn invalid_content_is_rejected() {
        let s = schema();
        for bad in [
            json!({"age": 3}),
            json!({"name": "Ada", "extra": 1}),
            json!({"name": 5}),
            json!({"name": "Ada", "age": 1.5}),
            json!({"name": "Ada", "age": -1}),
            json!({"name": "Ada", "plan": "enterprise"}),
            json!({"name": ""}),
            json!({"name": "a very long name"}),
            json!("not an object"),
        ] {
            assert!(validate_form_content(&s, &bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn one_of_consts_and_arrays_are_enums() {
        let s = json!({"type": "object", "properties": {
            "color": {"type": "string", "oneOf": [{"const": "red"}, {"const": "blue"}]},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
        }});
        assert!(validate_form_content(&s, &json!({"color": "red", "tags": ["a"]})).is_ok());
        assert!(validate_form_content(&s, &json!({"color": "green"})).is_err());
        assert!(validate_form_content(&s, &json!({"tags": ["c"]})).is_err());
    }

    #[test]
    fn blocked_paths_in_answers_are_caught() {
        assert!(scan_answer(&json!({"path": "~/.ssh/id_rsa"})).is_err());
        assert!(scan_answer(&json!({"name": "Ada"})).is_ok());
    }

    struct Fixed(McpElicitationAnswer);
    impl McpElicitationPrompter for Fixed {
        fn prompt(
            &self,
            _server: String,
            _request: McpElicitationRequest,
            _dismissed: CancellationToken,
        ) -> BoxFuture<'static, McpElicitationAnswer> {
            let answer = self.0.clone();
            Box::pin(async move { answer })
        }
    }

    struct Never;
    impl McpElicitationPrompter for Never {
        fn prompt(
            &self,
            _server: String,
            _request: McpElicitationRequest,
            dismissed: CancellationToken,
        ) -> BoxFuture<'static, McpElicitationAnswer> {
            Box::pin(async move {
                dismissed.cancelled().await;
                McpElicitationAnswer::Accept(json!({}))
            })
        }
    }

    fn form() -> ElicitRequestParams {
        serde_json::from_value(json!({
            "mode": "form",
            "message": "Who are you?",
            "requestedSchema": schema()
        }))
        .unwrap()
    }

    async fn run(prompter: Option<Arc<dyn McpElicitationPrompter>>) -> ElicitResult {
        resolve(
            prompter,
            "srv",
            &form(),
            CancellationToken::new(),
            Duration::from_millis(100),
        )
        .await
    }

    #[tokio::test]
    async fn without_a_prompter_the_request_is_declined() {
        assert_eq!(run(None).await.action, ElicitationAction::Decline);
    }

    #[tokio::test]
    async fn accepted_valid_form_returns_content() {
        let r = run(Some(Arc::new(Fixed(McpElicitationAnswer::Accept(
            json!({"name": "Ada"}),
        )))))
        .await;
        assert_eq!(r.action, ElicitationAction::Accept);
        assert_eq!(r.content, Some(json!({"name": "Ada"})));
    }

    #[tokio::test]
    async fn invalid_or_blocked_answers_are_declined() {
        for content in [json!({"age": 1}), json!({"name": "~/.ssh/id_rsa"})] {
            let r = run(Some(Arc::new(Fixed(McpElicitationAnswer::Accept(content))))).await;
            assert_eq!(r.action, ElicitationAction::Decline);
            assert_eq!(r.content, None);
        }
    }

    #[tokio::test]
    async fn timeout_declines_and_dismisses() {
        let r = run(Some(Arc::new(Never))).await;
        assert_eq!(r.action, ElicitationAction::Decline);
    }

    #[tokio::test]
    async fn server_cancellation_declines() {
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let r = resolve(
            Some(Arc::new(Never)),
            "srv",
            &form(),
            cancelled,
            Duration::from_secs(60),
        )
        .await;
        assert_eq!(r.action, ElicitationAction::Decline);
    }

    #[tokio::test]
    async fn cancel_and_url_accept() {
        let r = run(Some(Arc::new(Fixed(McpElicitationAnswer::Cancel)))).await;
        assert_eq!(r.action, ElicitationAction::Cancel);
        let url: ElicitRequestParams = serde_json::from_value(json!({
            "mode": "url", "message": "Sign in", "url": "https://example.com/auth",
            "elicitationId": "e1"
        }))
        .unwrap();
        let r = resolve(
            Some(Arc::new(Fixed(McpElicitationAnswer::Accept(Value::Null)))),
            "srv",
            &url,
            CancellationToken::new(),
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(r.action, ElicitationAction::Accept);
        assert_eq!(r.content, None);
    }
}
