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
pub use read::{ReadArgs, ReadError, ReadTool};
pub use write::{WriteArgs, WriteError, WriteTool};
