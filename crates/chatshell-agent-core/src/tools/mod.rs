//! LLM tool engines over the [`crate::tool_fs::Fs`] seam (ADR 03 Layer 1).
//!
//! Each module hosts one engine: pure decision logic plus IO routed through
//! the `Fs` trait, constructed with [`crate::tool_fs::StdFs`] by default.
//! Hosts register the rig `Tool` impls directly (desktop's agent builder) or
//! surface the schemas to client-side tool callers (iOS).

pub mod bash;
pub mod bash_session_manager;
pub mod edit;
pub mod glob;
pub mod grep;
pub mod image_question;
pub mod kill_shell;
pub mod read;
pub mod write;

pub use bash::{BashArgs, BashError, BashTool, SharedBashSession, TempFileList};
pub use bash_session_manager::BashSessionManager;
pub use edit::{EditArgs, EditError, EditTool};
pub use glob::{GlobArgs, GlobError, GlobTool};
pub use grep::{GrepArgs, GrepError, GrepTool};
pub use image_question::{ImageInput, ImageQuestionResponder};
pub use kill_shell::{KillShellArgs, KillShellError, KillShellTool};
pub use read::{ReadArgs, ReadError, ReadTool, UrlByteFetcher, UrlBytes};
pub use write::{WriteArgs, WriteError, WriteTool};

/// Classify a tool's IO-facing error into a rig execution error whose
/// model-visible feedback stays actionable.
///
/// rig's default `from_error` redacts any non-`ToolExecutionError` to the
/// error kind's generic feedback — for `Other` that is literally
/// "the tool failed". A macOS TCC denial on `~/Downloads` then reads as an
/// opaque failure and the model burns turns retrying blind (observed: five
/// minutes of pdftotext/pypdf/xattr flailing on a permission-denied PDF).
/// Mapping permission denials to [`ToolErrorKind::PermissionDenied`] with
/// guidance text lets the model stop and tell the user what to grant.
pub(crate) fn classify_tool_error(message: String) -> rig::agent::tool::ToolExecutionError {
    use rig::agent::tool::{ToolErrorKind, ToolExecutionError};

    let lower = message.to_ascii_lowercase();
    let permission_denied = lower.contains("operation not permitted")
        || lower.contains("permission denied")
        || lower.contains("read-only file system");
    if permission_denied {
        return ToolExecutionError::new(ToolErrorKind::PermissionDenied, message)
            .with_model_feedback(
                "the OS denied access to this path (macOS privacy/TCC or file \
             permissions). Do not retry the same path: ask the user to grant \
             the app access in System Settings > Privacy & Security (Files and \
             Folders or Full Disk Access), or have the file copied into an \
             accessible folder first",
            );
    }
    ToolExecutionError::other(message)
}
