//! Sandbox wiring for the bash tool: per-conversation sandbox settings,
//! the escalation hook the host implements, and denial detection.
//!
//! When a sandboxed command fails in a way that looks like the sandbox
//! blocked it, the tool asks the host's [`SandboxEscalator`] (the desktop
//! shows an approval card) whether to retry once outside the sandbox,
//! allow network for the rest of the session, or deny. The outcome is
//! annotated on the tool result, so the transcript records every approved
//! escape and the model sees why a denied command failed.

use crate::path_policy::ReadClassifier;
use crate::sandbox::SandboxBackend;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// What the sandbox appears to have blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationKind {
    /// A filesystem operation outside the writable roots or on a
    /// protected path (including reads of `path_policy`-denied files).
    Filesystem,
    /// A non-loopback network connection.
    Network,
    /// Stored credentials (the keychain, a credential helper) that the
    /// sandbox keeps out of reach, e.g. `git push` over HTTPS.
    Credentials,
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
    /// Keep the sandboxed result; the host reported the denial to the
    /// user without asking (auto mode), who can still run the command
    /// outside the sandbox once.
    Reported,
}

/// A per-execution change to the jail, authorized by an auto-mode allow
/// or a run-anyway tap. Never a grant: exactly one execution of the
/// command consumes it, and the session returns to its normal profile
/// right after.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionDelta {
    /// Allow non-loopback network for this execution.
    pub network: bool,
    /// Extra writable roots for this execution (existing directories).
    pub writable_roots: Vec<PathBuf>,
    /// Run this execution outside the sandbox (run-anyway on a sandbox
    /// denial; FR-18's retry-outside-the-sandbox-once).
    pub unsandboxed: bool,
}

impl ExecutionDelta {
    pub fn is_empty(&self) -> bool {
        !self.network && self.writable_roots.is_empty() && !self.unsandboxed
    }

    /// Human-readable summary (result notes, the decision log).
    pub fn describe(&self) -> String {
        if self.unsandboxed {
            return "outside the sandbox".to_string();
        }
        let mut parts = Vec::new();
        if self.network {
            parts.push("network access".to_string());
        }
        if !self.writable_roots.is_empty() {
            let roots: Vec<String> = self
                .writable_roots
                .iter()
                .map(|root| root.display().to_string())
                .collect();
            parts.push(format!("write access to {}", roots.join(", ")));
        }
        parts.join(" and ")
    }
}

/// Pending per-execution deltas, keyed by the exact command text. Shared
/// by the permission gate (which authorizes) and the bash tool (which
/// consumes); the model has no way to add entries.
#[derive(Debug, Clone, Default)]
pub struct ExecutionDeltas {
    entries: Arc<Mutex<Vec<(String, ExecutionDelta)>>>,
}

impl ExecutionDeltas {
    /// Authorize `delta` for the next execution of `command`.
    pub fn authorize(&self, command: &str, delta: ExecutionDelta) {
        if delta.is_empty() {
            return;
        }
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((command.to_string(), delta));
    }

    /// Consume the delta authorized for `command`, if any.
    pub fn take(&self, command: &str) -> Option<ExecutionDelta> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let index = entries.iter().position(|(key, _)| key == command)?;
        Some(entries.remove(index).1)
    }

    /// Drop every pending delta (turn ended).
    pub fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    /// Per-execution deltas authorized by the permission gate.
    pub deltas: ExecutionDeltas,
}

impl std::fmt::Debug for BashSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashSandbox")
            .field("backend", &self.backend)
            .field("scratch_dir", &self.scratch_dir)
            .field("network_allowed", &self.network_allowed)
            .field("escalator", &self.escalator.is_some())
            .field("deltas", &self.deltas.len())
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

/// bubblewrap's read masks fail with EACCES. Ordinary permission errors
/// say the same, so this marker counts only on a line that names a path
/// the read policy blocks.
const MASK_MARKER: &str = "permission denied";

/// Output fragments of a credential lookup that failed because the
/// keychain or credential helper is out of reach (git falls back to an
/// interactive prompt, which a non-interactive session cannot answer).
const CREDENTIAL_MARKERS: &[&str] = &["could not read username for", "could not read password for"];

/// Classify a failed sandboxed command's output. Network markers win
/// when network is still blocked; then filesystem markers (including a
/// permission error on a policy-blocked path); then credential markers.
pub fn detect_denial(output: &str, network_allowed: bool) -> Option<(EscalationKind, String)> {
    let detail = |line: &str| line.trim().chars().take(240).collect::<String>();
    let find = |markers: &[&str]| {
        output.lines().find_map(|line| {
            let lower = line.to_ascii_lowercase();
            markers
                .iter()
                .any(|marker| lower.contains(marker))
                .then(|| detail(line))
        })
    };
    if !network_allowed && let Some(line) = find(NETWORK_MARKERS) {
        return Some((EscalationKind::Network, line));
    }
    if let Some(line) = find(FILESYSTEM_MARKERS) {
        return Some((EscalationKind::Filesystem, line));
    }
    let policy = ReadClassifier::new();
    if let Some(line) = output.lines().find(|line| {
        line.to_ascii_lowercase().contains(MASK_MARKER) && names_blocked_path(line, &policy)
    }) {
        return Some((EscalationKind::Filesystem, detail(line)));
    }
    find(CREDENTIAL_MARKERS).map(|line| (EscalationKind::Credentials, line))
}

/// Whether an error line names a path the read policy blocks
/// (`cat: /home/u/.zshrc: Permission denied`,
/// `ls: cannot open directory '/home/u/.ssh': Permission denied`).
fn names_blocked_path(line: &str, policy: &ReadClassifier) -> bool {
    line.split_whitespace()
        .map(|token| {
            token.trim_matches(|c: char| matches!(c, '\'' | '"' | ':' | ',' | '`' | '‘' | '’'))
        })
        .filter(|token| token.contains('/') || token.starts_with('.'))
        .any(|token| {
            policy
                .classify_lexical(std::path::Path::new(token))
                .is_some()
        })
}

/// Result annotation: the user approved one run outside the sandbox.
pub const NOTE_RETRIED_UNSANDBOXED: &str = "[sandbox escalation: the OS sandbox blocked this command; \
the user approved running it once outside the sandbox. Result below is from that run.]";

/// Result annotation: run once outside the sandbox after the user tapped
/// run-anyway on an auto-mode sandbox denial.
pub const NOTE_RUN_ANYWAY_UNSANDBOXED: &str = "[auto mode: the user chose to run this command \
once outside the sandbox after the sandbox blocked it. Result below is from that run.]";

/// Result annotation: the command ran with a per-execution delta.
pub fn note_delta(delta: &ExecutionDelta) -> String {
    format!(
        "[auto mode: this command ran with {} for this one execution. The shell was restarted \
         for it and again after, so earlier cd/export state is gone.]",
        delta.describe()
    )
}

/// Result annotation: auto mode reported a sandbox denial to the user.
pub fn note_reported(kind: EscalationKind) -> String {
    format!(
        "[sandbox: the OS sandbox blocked {}. Auto mode reported this to the user, who can run \
         the command outside the sandbox once. Do not work around the sandbox; use a different \
         approach or ask the user.]",
        blocked_what(kind)
    )
}

fn blocked_what(kind: EscalationKind) -> &'static str {
    match kind {
        EscalationKind::Network => "network access",
        EscalationKind::Filesystem => {
            "a filesystem operation (a write outside the workspace, or a protected path)"
        }
        EscalationKind::Credentials => {
            "access to stored credentials (keychain or credential helper)"
        }
    }
}

/// Result annotation: the user allowed network for the session.
pub const NOTE_NETWORK_ALLOWED: &str = "[sandbox escalation: the user allowed network access for \
the rest of this session. The shell was restarted, so earlier cd/export state is gone.]";

/// Result annotation: the sandbox blocked the command and nobody approved
/// an escape.
pub fn note_denied(kind: EscalationKind, answered: bool) -> String {
    let what = blocked_what(kind);
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
    fn deltas_are_consumed_by_exactly_one_matching_execution() {
        let deltas = ExecutionDeltas::default();
        deltas.authorize("ls", ExecutionDelta::default());
        assert!(deltas.is_empty(), "empty deltas are not stored");
        let network = ExecutionDelta {
            network: true,
            ..ExecutionDelta::default()
        };
        deltas.authorize("curl https://example.com", network.clone());
        assert_eq!(deltas.take("curl https://example.org"), None);
        assert_eq!(deltas.take("curl https://example.com"), Some(network));
        assert_eq!(deltas.take("curl https://example.com"), None);
        assert_eq!(
            ExecutionDelta {
                network: true,
                writable_roots: vec![PathBuf::from("/a")],
                unsandboxed: false
            }
            .describe(),
            "network access and write access to /a"
        );
    }

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
    fn permission_denied_counts_only_on_blocked_paths() {
        let home = dirs::home_dir().unwrap();
        let masked = format!("cat: {}: Permission denied", home.join(".zshrc").display());
        assert_eq!(
            detect_denial(&masked, false).map(|(k, _)| k),
            Some(EscalationKind::Filesystem)
        );
        let dir = format!(
            "ls: cannot open directory '{}': Permission denied",
            home.join(".ssh").display()
        );
        assert_eq!(
            detect_denial(&dir, false).map(|(k, _)| k),
            Some(EscalationKind::Filesystem)
        );
        assert_eq!(
            detect_denial("cat: app/.env: Permission denied", false).map(|(k, _)| k),
            Some(EscalationKind::Filesystem)
        );
        assert_eq!(
            detect_denial("cat: /var/log/system.log: Permission denied", false),
            None
        );
        assert_eq!(detect_denial("./build.sh: Permission denied", false), None);
    }

    #[test]
    fn credential_failures_escalate_as_credentials() {
        let out = "fatal: could not read Username for 'https://github.com': Device not configured";
        let (kind, detail) = detect_denial(out, true).unwrap();
        assert_eq!(kind, EscalationKind::Credentials);
        assert!(detail.contains("github.com"));
        assert!(note_denied(kind, true).contains("stored credentials"));
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

/// End-to-end tests against the real OS sandbox: Seatbelt on macOS,
/// bubblewrap on Linux. On Linux the tests skip when bubblewrap is
/// missing unless `CHATSHELL_REQUIRE_BWRAP=1` (set in the Linux container
/// run) turns a missing backend into a failure.
#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod os_sandbox_integration {
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
        deltas: ExecutionDeltas,
    }

    /// The backend under test, or `None` to skip.
    fn backend() -> Option<SandboxBackend> {
        let detected = detect_backend();
        if detected.is_none() {
            assert!(
                cfg!(target_os = "linux")
                    && std::env::var("CHATSHELL_REQUIRE_BWRAP").as_deref() != Ok("1"),
                "no sandbox backend detected"
            );
            eprintln!("skipping: bubblewrap not installed");
        }
        detected
    }

    fn fixture(decision: EscalationDecision) -> Option<Fixture> {
        let backend = backend()?;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let root = base.join("work");
        std::fs::create_dir_all(&root).unwrap();
        let recorder = Arc::new(Recorder {
            decision,
            requests: Mutex::new(Vec::new()),
        });
        let network = Arc::new(AtomicBool::new(false));
        let deltas = ExecutionDeltas::default();
        let tool = BashTool::with_working_directory(root.display().to_string())
            .with_env_passthrough(Vec::new())
            .with_sandbox(BashSandbox {
                backend: Some(backend),
                scratch_dir: base.join("scratch"),
                network_allowed: network.clone(),
                escalator: Some(recorder.clone()),
                deltas: deltas.clone(),
            });
        Some(Fixture {
            outside: base.join("outside.txt"),
            _dir: dir,
            root,
            tool,
            recorder,
            network,
            deltas,
        })
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
        if cfg!(target_os = "macos") {
            assert_eq!(detect_backend(), Some(SandboxBackend::Seatbelt));
        } else if std::env::var("CHATSHELL_REQUIRE_BWRAP").as_deref() == Ok("1") {
            assert!(matches!(
                detect_backend(),
                Some(SandboxBackend::Bubblewrap(_))
            ));
        }
    }

    #[tokio::test]
    async fn writes_inside_root_and_scratch_succeed() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
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
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
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
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        let out = run(&f.tool, &format!("echo hi > '{}'", f.outside.display())).await;
        assert!(!f.outside.exists(), "write escaped the sandbox: {out}");
        let reqs = requests(&f);
        assert_eq!(reqs.len(), 1, "{out}");
        assert_eq!(reqs[0].kind, EscalationKind::Filesystem);
        assert!(out.contains("declined"), "{out}");
    }

    #[tokio::test]
    async fn a_writable_root_delta_applies_to_one_execution_only() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        let outside_dir = f.outside.parent().unwrap().join("granted");
        std::fs::create_dir_all(&outside_dir).unwrap();
        let target = outside_dir.join("a.txt");
        let command = format!("echo hi > '{}'", target.display());
        f.deltas.authorize(
            &command,
            ExecutionDelta {
                writable_roots: vec![outside_dir.clone()],
                ..ExecutionDelta::default()
            },
        );
        let out = run(&f.tool, &command).await;
        assert!(target.exists(), "{out}");
        assert!(out.contains("for this one execution"), "{out}");
        assert!(requests(&f).is_empty(), "{out}");
        assert!(f.deltas.is_empty(), "consumed");
        // The same command again: the session is back to its profile.
        std::fs::remove_file(&target).unwrap();
        let out = run(&f.tool, &command).await;
        assert!(!target.exists(), "{out}");
        assert_eq!(requests(&f).len(), 1);
        assert!(!f.network.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn an_unsandboxed_delta_runs_once_outside_the_sandbox() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        let command = format!("echo hi > '{}'", f.outside.display());
        f.deltas.authorize(
            &command,
            ExecutionDelta {
                unsandboxed: true,
                ..ExecutionDelta::default()
            },
        );
        let out = run(&f.tool, &command).await;
        assert!(f.outside.exists(), "{out}");
        assert!(out.contains("once outside the sandbox"), "{out}");
        assert!(requests(&f).is_empty());
        std::fs::remove_file(&f.outside).unwrap();
        run(&f.tool, &command).await;
        assert!(!f.outside.exists());
    }

    #[tokio::test]
    async fn a_reported_denial_keeps_the_sandboxed_result() {
        let Some(f) = fixture(EscalationDecision::Reported) else {
            return;
        };
        let out = run(&f.tool, &format!("echo hi > '{}'", f.outside.display())).await;
        assert!(!f.outside.exists());
        assert_eq!(requests(&f).len(), 1);
        assert!(out.contains("Auto mode reported this to the user"), "{out}");
    }

    #[tokio::test]
    async fn approved_retry_runs_once_outside_the_sandbox() {
        let Some(f) = fixture(EscalationDecision::RetryUnsandboxed) else {
            return;
        };
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
        let Some(backend) = backend() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let root = base.join("work");
        std::fs::create_dir_all(&root).unwrap();
        let tool = BashTool::with_working_directory(root.display().to_string()).with_sandbox(
            BashSandbox {
                backend: Some(backend),
                scratch_dir: base.clone(),
                network_allowed: Arc::new(AtomicBool::new(false)),
                escalator: None,
                deltas: ExecutionDeltas::default(),
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
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        std::fs::write(f.root.join(".env"), "SECRET=1").unwrap();
        let out = run(&f.tool, "cat .env").await;
        assert!(!out.contains("SECRET=1"), "{out}");
    }

    #[tokio::test]
    async fn policy_denied_reads_fail_and_escalate() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        // A registered private directory (as the desktop registers its
        // data directory) and project secrets by name pattern.
        let private = f.outside.with_file_name("app-data");
        std::fs::create_dir_all(&private).unwrap();
        std::fs::write(private.join("data.db"), "OTHER-CONVERSATION").unwrap();
        // Registered before the first command, which spawns the session.
        crate::path_policy::register_private_dir(&private);
        std::fs::write(f.root.join(".env.test"), "SECRET=env-test").unwrap();
        std::fs::write(f.root.join(".env.example"), "SECRET=template").unwrap();
        let out = run(&f.tool, "cat .env.example; cat .env.test").await;
        assert!(out.contains("SECRET=template"), "{out}");
        assert!(!out.contains("env-test"), "{out}");
        let reqs = requests(&f);
        assert_eq!(reqs.len(), 1, "{out}");
        assert_eq!(reqs[0].kind, EscalationKind::Filesystem, "{out}");
        let out = run(
            &f.tool,
            &format!(
                "cat '{}' ; ls '{}'",
                private.join("data.db").display(),
                private.display()
            ),
        )
        .await;
        assert!(!out.contains("OTHER-CONVERSATION"), "{out}");
        assert!(!out.contains("data.db\n"), "{out}");
        assert_eq!(requests(&f).len(), 2, "{out}");
    }

    #[tokio::test]
    async fn git_hooks_stay_read_only() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
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
    async fn git_dir_cannot_be_moved_aside_to_plant_hooks() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&f.root)
            .status()
            .unwrap();
        assert!(init.success());
        // Rename-around: move `.git` aside, plant a hook and a config
        // through the new name, move it back. Each step is attempted even
        // if an earlier one fails.
        let out = run(
            &f.tool,
            "mv .git .git-bak; mkdir -p .git-bak/hooks; \
             echo evil > .git-bak/hooks/pre-commit; \
             printf '[core]\\n\\thooksPath = /tmp/evil\\n' >> .git-bak/config; \
             mv .git-bak .git; echo done",
        )
        .await;
        assert!(out.contains("done"), "{out}");
        assert!(f.root.join(".git").is_dir(), "{out}");
        assert!(!f.root.join(".git-bak").exists(), "{out}");
        assert!(!f.root.join(".git/hooks/pre-commit").exists(), "{out}");
        let config = std::fs::read_to_string(f.root.join(".git/config")).unwrap();
        assert!(!config.contains("hooksPath"), "{config}");
        // In-place git work still succeeds.
        let out = run(
            &f.tool,
            "echo x > f.txt && git add f.txt && git -c user.name=t -c user.email=t@t \
             commit -qm init && git log --oneline | wc -l",
        )
        .await;
        assert!(out.trim().ends_with('1'), "{out}");
        // Removing and recreating `.git` fails the same way (this
        // deletes the unprotected repository contents, so it runs last).
        let out = run(
            &f.tool,
            "rm -rf .git 2>/dev/null; mkdir -p .git/hooks && echo evil > .git/hooks/post-checkout; echo done",
        )
        .await;
        assert!(!f.root.join(".git/hooks/post-checkout").exists(), "{out}");
    }

    #[tokio::test]
    async fn host_processes_are_not_visible() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        if !cfg!(target_os = "linux") {
            return;
        }
        let host_pid = std::process::id();
        let out = run(
            &f.tool,
            &format!(
                "test -e /proc/{host_pid}/environ && echo visible || echo hidden; \
                 kill -0 {host_pid} 2>/dev/null && echo signalable || echo unsignalable"
            ),
        )
        .await;
        assert!(out.contains("hidden"), "{out}");
        assert!(out.contains("unsignalable"), "{out}");
    }

    /// Reads another process's argv and environment through
    /// `sysctl(KERN_PROCARGS2)`, the call behind `ps eww`.
    const PROCARGS_PROBE: &str = r#"
import ctypes, sys
libc = ctypes.CDLL(None, use_errno=True)
mib = (ctypes.c_int * 3)(1, 49, int(sys.argv[1]))
size = ctypes.c_size_t(1 << 20)
buf = ctypes.create_string_buffer(size.value)
if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
    print("procargs-denied")
elif b"CARGO_MANIFEST_DIR=" in buf.raw[: size.value]:
    print("procargs-leaked")
else:
    print("procargs-empty")
"#;

    #[tokio::test]
    async fn other_process_environments_are_unreadable() {
        if !cfg!(target_os = "macos") || std::env::var("CARGO_MANIFEST_DIR").is_err() {
            return;
        }
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        std::fs::write(f.root.join("probe.py"), PROCARGS_PROBE).unwrap();
        let has_python = run(&f.tool, "command -v python3 >/dev/null && echo py-ok").await;
        if !has_python.contains("py-ok") {
            eprintln!("skipping: python3 not available");
            return;
        }
        // Control: outside the sandbox the probe sees this test process's
        // environment (an unsigned binary started by cargo).
        let host_pid = std::process::id();
        let control = std::process::Command::new("python3")
            .args(["-I", "probe.py", &host_pid.to_string()])
            .current_dir(&f.root)
            .output()
            .unwrap();
        let control = String::from_utf8_lossy(&control.stdout);
        assert!(control.contains("procargs-leaked"), "control: {control}");
        let out = run(&f.tool, &format!("python3 -I probe.py {host_pid}")).await;
        assert!(out.contains("procargs-denied"), "{out}");
        // Processes inside the sandbox stay inspectable.
        let out = run(
            &f.tool,
            "CARGO_MANIFEST_DIR=x bash -c 'python3 -I probe.py $$'",
        )
        .await;
        assert!(out.contains("procargs-leaked"), "{out}");
    }

    #[tokio::test]
    async fn tls_works_without_the_keychain_service() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
        f.network.store(true, Ordering::SeqCst);
        let out = run(
            &f.tool,
            "curl -sS --max-time 15 https://example.com -o /dev/null && echo curl-ok; \
             git ls-remote https://github.com/git/git HEAD >/dev/null && echo git-ok",
        )
        .await;
        assert!(out.contains("curl-ok"), "{out}");
        assert!(out.contains("git-ok"), "{out}");
        assert!(requests(&f).is_empty(), "{out}");
    }

    #[tokio::test]
    async fn network_is_blocked_and_escalates_as_network() {
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
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
        let Some(f) = fixture(EscalationDecision::AllowNetworkSession) else {
            return;
        };
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
        let Some(f) = fixture(EscalationDecision::Deny) else {
            return;
        };
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
                deltas: ExecutionDeltas::default(),
            },
        );
        let out = run(&tool, "echo ran").await;
        assert!(out.contains("ran"));
        assert!(out.contains("sandbox unavailable"), "{out}");
    }
}
