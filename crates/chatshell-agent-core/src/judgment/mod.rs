//! Auto-mode judgment requests (permission v3, Phase A).
//!
//! In `auto`, every call the v2 gate would have prompted for resolves to
//! [`crate::permission::GateDecision::Classify`] instead, carrying a
//! [`JudgmentRequest`]: which judgment source fired, the v2 prompt that
//! source produced (the interim renderer and the ineligible-provider path
//! still show it), and deterministic [`Evidence`] about the action.
//!
//! Pure: no I/O beyond the path-policy lookups the resolver already does,
//! no clock, no provider call. The host performs the judgment.
//!
//! The evidence never contains assistant text or tool results (the
//! injection surface): per-tool allowlists pick the exposed argument
//! fields, so free-text fields such as bash `description` never reach the
//! judge.

pub mod evidence;
pub mod transcript;

pub use evidence::{
    BashEvidence, Evidence, JailProfile, NetworkClass, PathBoundary, PathEvidence, PathTier,
};
pub use transcript::{AssembledTranscript, TranscriptBudget, TranscriptEntry, assemble};

use crate::permission::{ForcedPrompt, ForcedPromptKind};
use serde::{Deserialize, Serialize};

/// Why a call needs a judgment: the five v2 forced-prompt kinds, the two
/// floor kinds (sensitive writes, deletions outside the workspace), the
/// four mode/tier matrix arms, and plain bash without a sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgmentSource {
    /// Bash classifier `Warn` verdict.
    Security,
    /// The AST layer could not analyze the command.
    Unparsed,
    /// Bash network egress.
    Network,
    /// Fetch into a private or local network.
    PrivateNetwork,
    /// File-tool read of an ask-tier path.
    SensitiveRead,
    /// Write into a write ask-tier path (credential store, persistence
    /// location, agent settings, host data).
    SensitiveWrite,
    /// Bash deletes an existing path outside the workspace.
    OutsideDeletion,
    /// File-tool write outside the workspace.
    OutOfWorkspaceWrite,
    /// MCP tool call not covered by a human-minted grant.
    McpCall,
    /// `kill_shell`.
    KillShell,
    /// A tool the gate does not know (exec tier, fail-safe).
    UnknownTool,
    /// Bash in `auto` with no sandbox backend.
    UnsandboxedBash,
}

impl From<ForcedPromptKind> for JudgmentSource {
    fn from(kind: ForcedPromptKind) -> Self {
        match kind {
            ForcedPromptKind::Security => JudgmentSource::Security,
            ForcedPromptKind::Unparsed => JudgmentSource::Unparsed,
            ForcedPromptKind::Network => JudgmentSource::Network,
            ForcedPromptKind::PrivateNetwork => JudgmentSource::PrivateNetwork,
            ForcedPromptKind::SensitiveRead => JudgmentSource::SensitiveRead,
            ForcedPromptKind::SensitiveWrite => JudgmentSource::SensitiveWrite,
            ForcedPromptKind::OutsideDeletion => JudgmentSource::OutsideDeletion,
        }
    }
}

/// What the gate hands the judge for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgmentRequest {
    /// Human-readable target of the call (as on approval cards).
    pub subject: String,
    pub source: JudgmentSource,
    /// The forced prompt the v2 gate showed for this call, if the source
    /// is a forced kind. `None` for matrix arms (their v2 prompt offered
    /// grants).
    pub forced: Option<ForcedPrompt>,
    pub evidence: Evidence,
}
