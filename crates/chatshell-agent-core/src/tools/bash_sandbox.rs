//! Sandbox wiring for the bash tool: per-conversation sandbox settings,
//! the escalation hook the host implements, and denial detection.
//!
//! When a sandboxed command fails in a way that looks like the sandbox
//! blocked it, the tool asks the host's [`SandboxEscalator`] (the desktop
//! shows an approval card) whether to retry once outside the sandbox,
//! allow network for the rest of the session, or deny. The outcome is
//! annotated on the tool result, so the transcript records every approved
//! escape and the model sees why a denied command failed.

use crate::sandbox::SandboxBackend;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// What the sandbox appears to have blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationKind {
    /// A filesystem operation outside the writable roots or on a
    /// protected path.
    Filesystem,
    /// A non-loopback network connection.
    Network,
}

/// One escalation question for the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EscalationRequest {
    pub command: String,
    pub kind: EscalationKind,
    /// The output line that triggered the request.
    pub detail: String,
}

/// The host's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationDecision {
    /// Run this command once more, outside the sandbox.
    RetryUnsandboxed,
    /// Restart the session with network access for the rest of it, then
    /// run the command again.
    AllowNetworkSession,
    /// Keep the sandboxed result.
    Deny,
}

/// Host hook answering escalation requests (an approval prompt on
/// desktop). Unanswered or cancelled requests must resolve to `Deny`.
pub trait SandboxEscalator: Send + Sync {
    fn escalate<'a>(
        &'a self,
        request: EscalationRequest,
    ) -> Pin<Box<dyn Future<Output = EscalationDecision> + Send + 'a>>;
}

/// Per-conversation sandbox settings for the bash tool.
#[derive(Clone)]
pub struct BashSandbox {
    /// Sandbox launcher; `None` when this machine has none (the tool then
    /// runs unconfined and says so in the result; the gate has already
    /// prompted for the command in every mode that enables the sandbox).
    pub backend: Option<SandboxBackend>,
    /// Writable scratch directory for this conversation (also `TMPDIR`).
    pub scratch_dir: PathBuf,
    /// "Allow network this session" flag, shared with the host so it can
    /// reset it on mode switch and conversation deletion.
    pub network_allowed: Arc<AtomicBool>,
    /// Escalation hook; `None` means sandbox denials are reported only.
    pub escalator: Option<Arc<dyn SandboxEscalator>>,
}

impl std::fmt::Debug for BashSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashSandbox")
            .field("backend", &self.backend)
            .field("scratch_dir", &self.scratch_dir)
            .field("network_allowed", &self.network_allowed)
            .field("escalator", &self.escalator.is_some())
            .finish()
    }
}

/// Output fragments typical of a blocked network connection.
const NETWORK_MARKERS: &[&str] = &[
    "could not resolve host",
    "could not resolve",
    "nodename nor servname provided",
    "temporary failure in name resolution",
    "name or service not known",
    "network is unreachable",
    "failed to connect to",
    "couldn't connect to server",
    "getaddrinfo enotfound",
    "eai_again",
    "no route to host",
];

/// Output fragments typical of a blocked filesystem operation (Seatbelt
/// returns EPERM; bubblewrap's read-only binds return EROFS).
const FILESYSTEM_MARKERS: &[&str] = &[
    "operation not permitted",
    "read-only file system",
    "sandbox-exec",
    "deny(1)",
];

/// Classify a failed sandboxed command's output. Network markers win
/// when network is still blocked; otherwise filesystem markers.
pub fn detect_denial(output: &str, network_allowed: bool) -> Option<(EscalationKind, String)> {
    let find = |markers: &[&str]| {
        output.lines().find_map(|line| {
            let lower = line.to_ascii_lowercase();
            markers
                .iter()
                .any(|marker| lower.contains(marker))
                .then(|| line.trim().chars().take(240).collect::<String>())
        })
    };
    if !network_allowed && let Some(line) = find(NETWORK_MARKERS) {
        return Some((EscalationKind::Network, line));
    }
    find(FILESYSTEM_MARKERS).map(|line| (EscalationKind::Filesystem, line))
}

/// Result annotation: the user approved one run outside the sandbox.
pub const NOTE_RETRIED_UNSANDBOXED: &str = "[sandbox escalation: the OS sandbox blocked this command; \
the user approved running it once outside the sandbox. Result below is from that run.]";

/// Result annotation: the user allowed network for the session.
pub const NOTE_NETWORK_ALLOWED: &str = "[sandbox escalation: the user allowed network access for \
the rest of this session. The shell was restarted, so earlier cd/export state is gone.]";

/// Result annotation: the sandbox blocked the command and nobody approved
/// an escape.
pub fn note_denied(kind: EscalationKind, answered: bool) -> String {
    let what = match kind {
        EscalationKind::Network => "network access",
        EscalationKind::Filesystem => "a filesystem operation outside the workspace",
    };
    if answered {
        format!(
            "[sandbox: the OS sandbox blocked {what}, and the user declined to run it outside the \
             sandbox. Do not retry the same command; ask the user how to proceed.]"
        )
    } else {
        format!("[sandbox: the OS sandbox blocked {what}.]")
    }
}

/// Result annotation: no sandbox backend exists on this machine.
pub const NOTE_SANDBOX_UNAVAILABLE: &str = "[sandbox unavailable on this system: the command ran \
without OS confinement after the user approved it.]";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_denial_is_detected_until_network_is_allowed() {
        let out = "curl: (6) Could not resolve host: example.com";
        assert_eq!(
            detect_denial(out, false).map(|(k, _)| k),
            Some(EscalationKind::Network)
        );
        assert_eq!(detect_denial(out, true), None);
    }

    #[test]
    fn filesystem_denial_is_detected() {
        let out = "bash: /Users/x/outside.txt: Operation not permitted";
        let (kind, detail) = detect_denial(out, false).unwrap();
        assert_eq!(kind, EscalationKind::Filesystem);
        assert!(detail.contains("outside.txt"));
        assert_eq!(
            detect_denial("touch: /x: Read-only file system", true).map(|(k, _)| k),
            Some(EscalationKind::Filesystem)
        );
    }

    #[test]
    fn ordinary_failures_are_not_denials() {
        assert_eq!(
            detect_denial("error[E0425]: cannot find value", false),
            None
        );
        assert_eq!(detect_denial("No such file or directory", false), None);
    }
}

/// End-to-end tests against the real macOS Seatbelt sandbox.
#[cfg(all(test, target_os = "macos"))]
mod seatbelt_integration {
    use super::*;
    use crate::sandbox::{SandboxBackend, detect_backend};
    use crate::tools::bash::{BashArgs, BashTool};
    use rig::agent::tool::{Tool as _, ToolContext};
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    struct Recorder {
        decision: EscalationDecision,
        requests: Mutex<Vec<EscalationRequest>>,
    }

    impl SandboxEscalator for Recorder {
        fn escalate<'a>(
            &'a self,
            request: EscalationRequest,
        ) -> Pin<Box<dyn Future<Output = EscalationDecision> + Send + 'a>> {
            self.requests.lock().unwrap().push(request);
            Box::pin(async move { self.decision })
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        outside: PathBuf,
        tool: BashTool,
        recorder: Arc<Recorder>,
        network: Arc<AtomicBool>,
    }

    fn fixture(decision: EscalationDecision) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let root = base.join("work");
        std::fs::create_dir_all(&root).unwrap();
        let recorder = Arc::new(Recorder {
            decision,
            requests: Mutex::new(Vec::new()),
        });
        let network = Arc::new(AtomicBool::new(false));
        let tool = BashTool::with_working_directory(root.display().to_string())
            .with_env_passthrough(Vec::new())
            .with_sandbox(BashSandbox {
                backend: Some(SandboxBackend::Seatbelt),
                scratch_dir: base.join("scratch"),
                network_allowed: network.clone(),
                escalator: Some(recorder.clone()),
            });
        Fixture {
            outside: base.join("outside.txt"),
            _dir: dir,
            root,
            tool,
            recorder,
            network,
        }
    }

    async fn run(tool: &BashTool, command: &str) -> String {
        tool.call(
            &mut ToolContext::new(),
            BashArgs {
                command: Some(command.to_string()),
                timeout: Some(30),
                description: None,
                workdir: None,
            },
        )
        .await
        .unwrap_or_else(|e| format!("ERR {e}"))
    }

    fn requests(f: &Fixture) -> Vec<EscalationRequest> {
        f.recorder.requests.lock().unwrap().clone()
    }

    #[tokio::test]
    async fn backend_is_detected() {
        assert_eq!(detect_backend(), Some(SandboxBackend::Seatbelt));
    }

    #[tokio::test]
    async fn writes_inside_root_and_scratch_succeed() {
        let f = fixture(EscalationDecision::Deny);
        let out = run(
            &f.tool,
            "echo hi > inside.txt && echo ok > \"$TMPDIR/t\" && cat inside.txt",
        )
        .await;
        assert!(out.contains("hi"), "{out}");
        assert!(f.root.join("inside.txt").exists());
        assert!(requests(&f).is_empty());
    }

    #[tokio::test]
    async fn ordinary_tools_work_inside_the_sandbox() {
        let f = fixture(EscalationDecision::Deny);
        let out = run(
            &f.tool,
            "mkdir -p src && printf 'a\\nb\\n' > src/x.txt && find . -name '*.txt' | sed 's/x/y/' \
             && git --version && (cd src && wc -l x.txt) && ls / > /dev/null && echo all-ok",
        )
        .await;
        assert!(out.contains("all-ok"), "{out}");
        assert!(requests(&f).is_empty(), "{out}");
    }

    #[tokio::test]
    async fn write_outside_root_is_denied_and_escalated() {
        let f = fixture(EscalationDecision::Deny);
        let out = run(&f.tool, &format!("echo hi > '{}'", f.outside.display())).await;
        assert!(!f.outside.exists(), "write escaped the sandbox: {out}");
        let reqs = requests(&f);
        assert_eq!(reqs.len(), 1, "{out}");
        assert_eq!(reqs[0].kind, EscalationKind::Filesystem);
        assert!(out.contains("declined"), "{out}");
    }

    #[tokio::test]
    async fn approved_retry_runs_once_outside_the_sandbox() {
        let f = fixture(EscalationDecision::RetryUnsandboxed);
        let out = run(&f.tool, &format!("echo hi > '{}'", f.outside.display())).await;
        assert!(f.outside.exists(), "{out}");
        assert!(out.contains("outside the sandbox"), "{out}");
        // The persistent session stays sandboxed afterwards.
        std::fs::remove_file(&f.outside).unwrap();
        let f2_out = run(&f.tool, &format!("touch '{}'", f.outside.display())).await;
        assert!(requests(&f).len() == 2, "{f2_out}");
    }

    #[tokio::test]
    async fn workspace_root_cannot_be_renamed() {
        // Make the parent writable too (it is the scratch root), so only
        // the anchor deny on the workspace root can stop the rename.
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let root = base.join("work");
        std::fs::create_dir_all(&root).unwrap();
        let tool = BashTool::with_working_directory(root.display().to_string()).with_sandbox(
            BashSandbox {
                backend: Some(SandboxBackend::Seatbelt),
                scratch_dir: base.clone(),
                network_allowed: Arc::new(AtomicBool::new(false)),
                escalator: None,
            },
        );
        // Control: a sibling directory under the writable parent can move.
        std::fs::create_dir_all(base.join("other")).unwrap();
        let control = run(
            &tool,
            &format!("mv '{0}/other' '{0}/other-moved'", base.display()),
        )
        .await;
        assert!(
            base.join("other-moved").is_dir(),
            "control rename failed: {control}"
        );
        let out = run(
            &tool,
            &format!("mv '{0}/work' '{0}/work-moved'", base.display()),
        )
        .await;
        assert!(root.is_dir(), "{out}");
        assert!(!base.join("work-moved").exists(), "{out}");
    }

    #[tokio::test]
    async fn blocked_file_names_are_unreadable() {
        let f = fixture(EscalationDecision::Deny);
        std::fs::write(f.root.join(".env"), "SECRET=1").unwrap();
        let out = run(&f.tool, "cat .env").await;
        assert!(!out.contains("SECRET=1"), "{out}");
    }

    #[tokio::test]
    async fn git_hooks_stay_read_only() {
        let f = fixture(EscalationDecision::Deny);
        // Initialize outside the sandbox: the hooks directory must exist
        // for the deny to be meaningful.
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&f.root)
            .status()
            .unwrap();
        assert!(init.success());
        assert!(f.root.join(".git/hooks").is_dir());
        // Indirect target: the textual write-target check cannot see it,
        // so only the OS sandbox stands in the way.
        let out = run(
            &f.tool,
            "h=.git/hooks; echo evil > \"$h/pre-commit\"; git status --short; echo done",
        )
        .await;
        assert!(!f.root.join(".git/hooks/pre-commit").exists(), "{out}");
        assert!(out.contains("done"), "{out}");
    }

    #[tokio::test]
    async fn network_is_blocked_and_escalates_as_network() {
        let f = fixture(EscalationDecision::Deny);
        let out = run(
            &f.tool,
            "curl -sS --max-time 5 https://example.com -o /dev/null",
        )
        .await;
        let reqs = requests(&f);
        assert_eq!(reqs.len(), 1, "{out}");
        assert_eq!(reqs[0].kind, EscalationKind::Network, "{out}");
        assert!(!f.network.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn allowing_network_restarts_the_session_with_network() {
        let f = fixture(EscalationDecision::AllowNetworkSession);
        let out = run(
            &f.tool,
            "curl -sS --max-time 5 https://example.com -o /dev/null",
        )
        .await;
        assert!(f.network.load(Ordering::SeqCst), "{out}");
        assert!(out.contains("allowed network access"), "{out}");
    }

    #[tokio::test]
    async fn environment_is_minimized() {
        let f = fixture(EscalationDecision::Deny);
        // Cargo sets this for the test process; bash must not inherit it.
        if std::env::var("CARGO_MANIFEST_DIR").is_err() {
            return;
        }
        let out = run(&f.tool, "echo \"sentinel=${CARGO_MANIFEST_DIR:-absent}\"").await;
        assert!(out.contains("sentinel=absent"), "{out}");
        let passed = BashTool::with_working_directory(f.root.display().to_string())
            .with_env_passthrough(vec!["CARGO_MANIFEST_DIR".to_string()]);
        let out = run(&passed, "echo \"sentinel=${CARGO_MANIFEST_DIR:-absent}\"").await;
        assert!(!out.contains("sentinel=absent"), "{out}");
    }

    #[tokio::test]
    async fn missing_backend_runs_unconfined_with_a_note() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let tool = BashTool::with_working_directory(base.display().to_string()).with_sandbox(
            BashSandbox {
                backend: None,
                scratch_dir: base.join("scratch"),
                network_allowed: Arc::new(AtomicBool::new(false)),
                escalator: None,
            },
        );
        let out = run(&tool, "echo ran").await;
        assert!(out.contains("ran"));
        assert!(out.contains("sandbox unavailable"), "{out}");
    }
}
