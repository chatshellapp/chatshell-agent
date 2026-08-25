//! Bash tool for LLM agents
//!
//! Provides a persistent bash session where environment variables, working directory,
//! and other state persist between calls. The session is lazily created on first use
//! and automatically recreated if it dies or times out.

use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;

use crate::bash_security::{self, SecurityVerdict};
use crate::process_api::{ShellProcess, ShellSpawner, TokioSpawner};
use crate::tool_fs::{Fs, StdFs};

/// Maximum output length returned to the LLM (chars)
const MAX_OUTPUT_CHARS: usize = 20_000;

/// Characters kept from the start when truncating
const TRUNCATE_HEAD_CHARS: usize = 5_000;

/// Characters kept from the end when truncating
const TRUNCATE_TAIL_CHARS: usize = 10_000;

/// Stop accumulating output beyond this to prevent OOM
const MAX_CAPTURE_BYTES: usize = 100_000;

const DEFAULT_TIMEOUT_SECS: u64 = 120;

// ---------------------------------------------------------------------------
// BashSession – persistent child process
// ---------------------------------------------------------------------------

struct CommandOutput {
    output: String,
    exit_code: i32,
    /// True when the command exceeded its timeout: `output` is the partial
    /// capture and `exit_code` is meaningless. The session was killed and
    /// will be recreated on the next call.
    timed_out: bool,
    /// True when capture stopped at [`MAX_CAPTURE_BYTES`].
    capture_truncated: bool,
}

pub struct BashSession<P: ShellProcess> {
    proc: P,
}

/// Shared session handle for a spawner `S`, used by `BashTool`,
/// `KillShellTool`, and the session manager.
pub type SharedSession<S> = Arc<TokioMutex<Option<BashSession<<S as ShellSpawner>::Shell>>>>;

/// Concrete shared handle over the default tokio process backend.
pub type SharedBashSession = SharedSession<TokioSpawner>;

/// Shared list of temp file paths created during output truncation.
pub type TempFileList = Arc<Mutex<Vec<PathBuf>>>;

impl<P: ShellProcess> fmt::Debug for BashSession<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BashSession")
            .field("pid", &self.proc.pid())
            .finish()
    }
}

impl<P: ShellProcess> BashSession<P> {
    async fn new<S: ShellSpawner<Shell = P>>(spawner: &S, cwd: &str) -> Result<Self, BashError> {
        let mut session = Self {
            proc: spawner
                .spawn("bash", std::path::Path::new(cwd))
                .await
                .map_err(|e| BashError(format!("Failed to spawn bash: {}", e)))?,
        };
        session.initialize().await?;
        Ok(session)
    }

    /// Merge stderr into stdout, disable prompts, wait for ready marker.
    async fn initialize(&mut self) -> Result<(), BashError> {
        let marker = Self::make_marker("READY");
        let init = format!(
            "exec 2>&1\nexport PS1=''\nexport PS2=''\nexport PROMPT_COMMAND=''\necho {marker}\n"
        );

        self.proc
            .write_stdin(init.as_bytes())
            .await
            .map_err(|e| BashError(format!("Init write failed: {}", e)))?;
        self.proc
            .flush_stdin()
            .await
            .map_err(|e| BashError(format!("Init flush failed: {}", e)))?;

        // Discard any bashrc / profile output until we see the marker
        tokio::time::timeout(Duration::from_secs(10), async {
            let mut buf = Vec::new();
            loop {
                buf.clear();
                let n = self
                    .proc
                    .read_line(&mut buf)
                    .await
                    .map_err(|e| BashError(format!("Init read failed: {}", e)))?;
                if n == 0 {
                    return Err(BashError("Session ended during init".into()));
                }
                if String::from_utf8_lossy(&buf).trim() == marker {
                    return Ok(());
                }
            }
        })
        .await
        .map_err(|_| BashError("Session init timed out".into()))?
    }

    /// Execute a command, returning merged stdout+stderr and the exit code.
    async fn execute(
        &mut self,
        command: &str,
        timeout: Duration,
    ) -> Result<CommandOutput, BashError> {
        let marker = Self::make_marker("DONE");

        // The marker echo must be a separate command so $? captures the
        // exit code of the user's command, not something internal.
        let script = format!("{command}\necho {marker}$?\n");

        self.proc
            .write_stdin(script.as_bytes())
            .await
            .map_err(|e| BashError(format!("Write failed: {}", e)))?;
        self.proc
            .flush_stdin()
            .await
            .map_err(|e| BashError(format!("Flush failed: {}", e)))?;

        // `output`/`capture_truncated` stay owned by this frame; the timeout
        // future only borrows them, so a timeout keeps the partial capture.
        let mut output = String::new();
        let mut capture_truncated = false;

        let result: Result<Result<(String, i32), BashError>, _> =
            tokio::time::timeout(timeout, async {
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    let n = self
                        .proc
                        .read_line(&mut buf)
                        .await
                        .map_err(|e| BashError(format!("Read failed: {}", e)))?;
                    if n == 0 {
                        return Err(BashError("Session ended unexpectedly".into()));
                    }

                    let line = String::from_utf8_lossy(&buf);

                    if let Some(rest) = line.strip_prefix(marker.as_str()) {
                        let exit_code = rest.trim().parse::<i32>().unwrap_or(-1);
                        return Ok((std::mem::take(&mut output), exit_code));
                    }

                    if !capture_truncated {
                        output.push_str(&line);
                        if output.len() > MAX_CAPTURE_BYTES {
                            capture_truncated = true;
                        }
                    }
                }
            })
            .await;

        match result {
            Ok(Ok((captured, exit_code))) => Ok(CommandOutput {
                output: captured,
                exit_code,
                timed_out: false,
                capture_truncated,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // Kill the process to free resources; session will be recreated.
                // The partial capture survives and is returned to the caller:
                // a timeout is a labeled partial, not a hard failure.
                self.proc.kill().await;
                Ok(CommandOutput {
                    output: std::mem::take(&mut output),
                    exit_code: -1,
                    timed_out: true,
                    capture_truncated,
                })
            }
        }
    }

    fn is_alive(&mut self) -> bool {
        self.proc.is_alive()
    }

    pub async fn kill(&mut self) {
        self.proc.kill().await;
    }

    /// Synchronous kill for use in app shutdown hooks where async is unavailable.
    pub fn kill_sync(&mut self) {
        if self.is_alive() {
            self.proc.start_kill();
        }
    }

    fn make_marker(label: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        format!(
            "__CHATSHELL_{}_{:x}_{}_{}__",
            label,
            nanos,
            std::process::id(),
            seq
        )
    }
}

impl<P: ShellProcess> Drop for BashSession<P> {
    fn drop(&mut self) {
        if self.is_alive() {
            self.proc.start_kill();
        }
    }
}

// ---------------------------------------------------------------------------
// BashTool
// ---------------------------------------------------------------------------

/// Arguments for the bash tool
#[derive(Debug, Clone, Deserialize)]
pub struct BashArgs {
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub workdir: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("Bash error: {0}")]
pub struct BashError(String);

/// Persistent-session bash tool.
///
/// Environment variables and working directory survive across calls.
/// The session is created lazily and recreated automatically on failure.
fn default_fs() -> std::sync::Arc<dyn Fs> {
    std::sync::Arc::new(StdFs)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BashTool<S: ShellSpawner = TokioSpawner> {
    #[serde(skip_serializing_if = "Option::is_none")]
    default_cwd: Option<String>,
    #[serde(skip)]
    spawner: S,
    #[serde(skip)]
    session: SharedSession<S>,
    #[serde(skip)]
    temp_files: TempFileList,
    #[serde(skip)]
    abort_notify: Option<Arc<tokio::sync::Notify>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_root: Option<PathBuf>,
    #[serde(skip, default = "default_fs")]
    fs: std::sync::Arc<dyn Fs>,
}

impl<S: ShellSpawner> fmt::Debug for BashTool<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BashTool")
            .field("default_cwd", &self.default_cwd)
            .finish()
    }
}

impl<S: ShellSpawner + Default> Default for BashTool<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ShellSpawner + Default> BashTool<S> {
    pub fn new() -> Self {
        Self {
            default_cwd: None,
            spawner: S::default(),
            session: Arc::new(TokioMutex::new(None)),
            temp_files: Arc::new(Mutex::new(Vec::new())),
            abort_notify: None,
            project_root: None,
            fs: std::sync::Arc::new(StdFs),
        }
    }

    pub fn with_working_directory(cwd: String) -> Self {
        Self {
            default_cwd: Some(cwd),
            ..Self::new()
        }
    }

    pub fn with_session(session: SharedSession<S>, cwd: Option<String>) -> Self {
        Self {
            default_cwd: cwd,
            session,
            ..Self::new()
        }
    }

    /// Route temp-file spill and cleanup through a custom [`Fs`].
    pub fn with_fs(mut self, fs: std::sync::Arc<dyn Fs>) -> Self {
        self.fs = fs;
        self
    }

    pub fn with_project_root(mut self, root: PathBuf) -> Self {
        self.project_root = Some(root);
        self
    }

    /// Set a shared temp file tracker (call before the tool is consumed by the agent builder).
    pub fn with_temp_file_tracker(mut self, tracker: TempFileList) -> Self {
        self.temp_files = tracker;
        self
    }

    /// Set the abort notification handle for cooperative cancellation.
    pub fn with_abort_notify(mut self, notify: Arc<tokio::sync::Notify>) -> Self {
        self.abort_notify = Some(notify);
        self
    }

    pub fn session_handle(&self) -> SharedSession<S> {
        self.session.clone()
    }

    #[allow(dead_code)]
    pub fn temp_files_handle(&self) -> TempFileList {
        self.temp_files.clone()
    }

    pub(crate) fn resolve_cwd(&self) -> String {
        self.default_cwd
            .clone()
            .or_else(|| dirs::home_dir().map(|p| p.to_string_lossy().to_string()))
            .unwrap_or_else(|| ".".to_string())
    }

    /// Ensure `*guard` contains a live session, creating one if needed.
    async fn ensure_session(
        &self,
        guard: &mut tokio::sync::MutexGuard<'_, Option<BashSession<S::Shell>>>,
    ) -> Result<(), BashError> {
        if let Some(ref mut s) = **guard {
            if s.is_alive() {
                return Ok(());
            }
            tracing::warn!("🖥️ [bash] Session died, creating a new one");
        }

        let cwd = self.resolve_cwd();
        tracing::info!("🖥️ [bash] Starting persistent session in: {}", cwd);
        **guard = Some(BashSession::new(&self.spawner, &cwd).await?);
        Ok(())
    }

    fn check_dangerous(command: &str) -> Option<&'static str> {
        bash_security::check_dangerous(command)
    }

    /// Keep the first `TRUNCATE_HEAD_CHARS` and last `TRUNCATE_TAIL_CHARS`,
    /// replacing the middle with a summary line.
    /// When truncation occurs, saves the full output to a temp file and returns
    /// `(truncated_text, Some(temp_file_path))`.
    fn smart_truncate(output: &str) -> (String, Option<String>) {
        if output.len() <= MAX_OUTPUT_CHARS {
            return (output.to_string(), None);
        }

        let temp_path = Self::save_full_output_to_temp(&StdFs, output);

        let head_end = Self::char_boundary_at_or_before(output, TRUNCATE_HEAD_CHARS);
        let tail_start = Self::char_boundary_at_or_after(
            output,
            output.len().saturating_sub(TRUNCATE_TAIL_CHARS),
        );

        let truncated = if head_end >= tail_start {
            let end = Self::char_boundary_at_or_before(output, MAX_OUTPUT_CHARS);
            format!(
                "{}...\n[output truncated, {} chars total]",
                &output[..end],
                output.len()
            )
        } else {
            let omitted_lines = output[head_end..tail_start].lines().count();
            format!(
                "{}\n\n... [{} lines omitted, {} chars total] ...\n\n{}",
                output[..head_end].trim_end(),
                omitted_lines,
                output.len(),
                output[tail_start..].trim_start()
            )
        };

        (truncated, temp_path)
    }

    fn save_full_output_to_temp(fs: &dyn Fs, output: &str) -> Option<String> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let filename = format!("chatshell_bash_output_{:x}.txt", nanos);
        let path = std::env::temp_dir().join(filename);
        match fs.write(&path, output.as_bytes()) {
            Ok(()) => {
                tracing::info!("🖥️ [bash] Full output saved to: {}", path.display());
                Some(path.to_string_lossy().to_string())
            }
            Err(e) => {
                tracing::warn!("🖥️ [bash] Failed to save full output: {}", e);
                None
            }
        }
    }

    /// Delete all temp files tracked by the given handle.
    pub fn cleanup_temp_files(handle: &TempFileList) {
        if let Ok(mut files) = handle.lock() {
            for path in files.drain(..) {
                if let Err(e) = StdFs.remove_file(&path) {
                    tracing::debug!(
                        "🧹 [bash] Failed to clean temp file {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }
    }

    fn char_boundary_at_or_before(s: &str, pos: usize) -> usize {
        let mut i = pos.min(s.len());
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    fn char_boundary_at_or_after(s: &str, pos: usize) -> usize {
        let mut i = pos.min(s.len());
        while i < s.len() && !s.is_char_boundary(i) {
            i += 1;
        }
        i
    }
}

impl<S: ShellSpawner + Default> Tool for BashTool<S> {
    const NAME: &'static str = "bash";

    type Error = BashError;
    type Args = BashArgs;
    type Output = String;

    fn description(&self) -> String {
        let cwd_note = if let Some(ref dir) = self.default_cwd {
            format!(" The session starts in: {}.", dir)
        } else {
            String::new()
        };
        format!(
            "Execute a bash command in a persistent session. \
                 Environment variables and working directory persist between calls. \
                 Use `cd` to change directories. \
                 Returns combined stdout and stderr. \
                 Always prefer non-interactive commands (avoid commands that require user input).{cwd_note}"
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 120, max: 600)"
                },
                "description": {
                    "type": "string",
                    "description": "A concise (5-10 word) human-readable description of what the command does"
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory to run this command in. The command runs in a sub-shell so the session's cwd is not changed."
                }
            },
            "required": ["command"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        tracing::info!(
            "🔧 [tool-call] bash: command={:?} timeout={:?} description={:?} workdir={:?}",
            args.command,
            args.timeout,
            args.description,
            args.workdir
        );

        let mut guard = self.session.lock().await;

        let command = args
            .command
            .as_deref()
            .filter(|c| !c.is_empty())
            .ok_or_else(|| BashError("No command provided".into()))?;

        // --- security checks (dangerous patterns + compound analysis + injection) ---
        let security_warning = match bash_security::check_command(command, Self::check_dangerous) {
            SecurityVerdict::Block(msg) => {
                tracing::warn!("🛡️ [bash] Command blocked: {}", command);
                return Err(BashError(msg));
            }
            SecurityVerdict::Warn(msg) => {
                tracing::warn!("🛡️ [bash] Security warning: {} — {}", msg, command);
                Some(msg)
            }
            SecurityVerdict::Allow => None,
        };

        // --- write-target boundary check (redirect targets + write commands) ---
        if let Err(msg) = bash_security::check_write_targets(
            command,
            self.project_root.as_deref(),
            self.default_cwd.as_deref(),
        ) {
            tracing::warn!("🛡️ [bash] Write target blocked: {} — {}", msg, command);
            return Err(BashError(msg));
        }

        // Wrap in a sub-shell if workdir is specified so the session's cwd is unaffected
        let effective_command = if let Some(ref dir) = args.workdir {
            format!("(cd '{}' && {})", dir.replace('\'', "'\\''"), command)
        } else {
            command.to_string()
        };

        // --- ensure session ---
        self.ensure_session(&mut guard).await?;

        let timeout_secs = args.timeout.unwrap_or(DEFAULT_TIMEOUT_SECS).clamp(1, 600);

        let session = guard.as_mut().unwrap();
        let result = if let Some(ref abort) = self.abort_notify {
            tokio::select! {
                biased;
                _ = abort.notified() => {
                    tracing::info!("🛑 [bash] Command aborted: generation was stopped");
                    if let Some(mut s) = guard.take() {
                        s.kill().await;
                    }
                    Err(BashError("Command aborted: generation was stopped.".into()))
                }
                r = session.execute(&effective_command, Duration::from_secs(timeout_secs)) => r,
            }
        } else {
            session
                .execute(&effective_command, Duration::from_secs(timeout_secs))
                .await
        };

        match result {
            Ok(cmd) => {
                tracing::info!(
                    "🔧 [tool-result] bash: exit_code={} output_len={}",
                    cmd.exit_code,
                    cmd.output.len()
                );

                let (text, temp_file) = Self::smart_truncate(cmd.output.trim_end());
                if let Some(ref path_str) = temp_file
                    && let Ok(mut files) = self.temp_files.lock()
                {
                    files.push(PathBuf::from(path_str));
                }
                let file_notice = temp_file
                    .map(|p| {
                        format!(
                            "\n[full output saved to {} -- use the read tool to view]",
                            p
                        )
                    })
                    .unwrap_or_default();

                let warn_prefix = security_warning
                    .as_ref()
                    .map(|w| format!("{}\n", w))
                    .unwrap_or_default();

                let timeout_note = if cmd.timed_out {
                    format!(
                        "[command timed out after {}s — session was reset; \
                         partial output below (the command may still have been running)]\n",
                        timeout_secs
                    )
                } else {
                    String::new()
                };
                let capture_note = if cmd.capture_truncated {
                    format!("\n[output capture stopped at {} bytes]", MAX_CAPTURE_BYTES)
                } else {
                    String::new()
                };

                if cmd.timed_out {
                    return Ok(format!(
                        "{}{}{}{}{}",
                        warn_prefix, timeout_note, text, capture_note, file_notice
                    ));
                }

                if cmd.exit_code == 0 {
                    if text.is_empty() {
                        Ok(format!("{}(no output){}", warn_prefix, capture_note))
                    } else {
                        Ok(format!(
                            "{}{}{}{}",
                            warn_prefix, text, capture_note, file_notice
                        ))
                    }
                } else if text.is_empty() {
                    Ok(format!(
                        "{}[exit code: {}]{}{}",
                        warn_prefix, cmd.exit_code, capture_note, file_notice
                    ))
                } else {
                    Ok(format!(
                        "{}[exit code: {}]\n{}{}{}",
                        warn_prefix, cmd.exit_code, text, capture_note, file_notice
                    ))
                }
            }
            Err(e) => {
                // Mark the session dead so it gets recreated next time.
                if let Some(ref mut s) = *guard
                    && !s.is_alive()
                {
                    tracing::warn!("🖥️ [bash] Session dead after error, will recreate");
                    *guard = None;
                }
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_tool_creation() {
        let tool: BashTool = BashTool::new();
        assert_eq!(<BashTool as Tool>::NAME, "bash");
        let _ = tool;
    }

    #[test]
    fn test_bash_args_command_only() {
        let args: BashArgs = serde_json::from_str(r#"{"command": "echo hello"}"#).unwrap();
        assert_eq!(args.command, Some("echo hello".to_string()));
        assert_eq!(args.timeout, None);
    }

    #[test]
    fn test_bash_args_with_timeout() {
        let args: BashArgs =
            serde_json::from_str(r#"{"command": "sleep 5", "timeout": 10}"#).unwrap();
        assert_eq!(args.command, Some("sleep 5".to_string()));
        assert_eq!(args.timeout, Some(10));
    }

    #[test]
    fn test_bash_args_with_description() {
        let args: BashArgs = serde_json::from_str(
            r#"{"command": "ls -la", "description": "List directory contents"}"#,
        )
        .unwrap();
        assert_eq!(args.command, Some("ls -la".to_string()));
        assert_eq!(
            args.description,
            Some("List directory contents".to_string())
        );
    }

    #[test]
    fn test_dangerous_command_detection() {
        // Root-targeting commands
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm -rf /").is_some());
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm -rf /*").is_some());
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm -fr /").is_some());
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm -rf / && echo done").is_some());
        assert!(BashTool::<TokioSpawner>::check_dangerous("chmod -R 777 /*").is_some());

        // Other dangerous patterns
        assert!(BashTool::<TokioSpawner>::check_dangerous("sudo mkfs.ext4 /dev/sda1").is_some());
        assert!(BashTool::<TokioSpawner>::check_dangerous(":(){:|:&};:").is_some());

        // Safe commands (specific paths, not root)
        assert!(BashTool::<TokioSpawner>::check_dangerous("echo hello").is_none());
        assert!(BashTool::<TokioSpawner>::check_dangerous("ls -la").is_none());
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm -rf /tmp/mydir").is_none());
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm -rf /var/log/old").is_none());
        assert!(BashTool::<TokioSpawner>::check_dangerous("rm file.txt").is_none());
        assert!(BashTool::<TokioSpawner>::check_dangerous("chmod -R 777 /tmp").is_none());
    }

    #[tokio::test]
    async fn timeout_returns_partial_output() {
        let mut session = BashSession::new(&TokioSpawner, "/tmp").await.unwrap();

        let out = session
            .execute(
                "echo partial-before-timeout; sleep 30",
                Duration::from_secs(1),
            )
            .await
            .unwrap();

        assert!(out.timed_out, "flagged as timed out");
        assert_eq!(out.exit_code, -1);
        assert!(
            out.output.contains("partial-before-timeout"),
            "partial capture survives: {:?}",
            out.output
        );
        assert!(!session.is_alive(), "session killed on timeout");
    }

    #[tokio::test]
    async fn fast_command_still_reports_exit_code() {
        let mut session = BashSession::new(&TokioSpawner, "/tmp").await.unwrap();
        let out = session
            .execute("echo ok; (exit 3)", Duration::from_secs(10))
            .await
            .unwrap();

        assert!(!out.timed_out);
        assert_eq!(out.exit_code, 3);
        assert_eq!(out.output.trim(), "ok");
    }

    #[test]
    fn test_smart_truncate_short() {
        let short = "hello world\n";
        let (result, temp) = BashTool::<TokioSpawner>::smart_truncate(short);
        assert_eq!(result, short);
        assert!(temp.is_none());
    }

    #[test]
    fn test_smart_truncate_long() {
        let long: String = (0..5000).map(|i| format!("line {}\n", i)).collect();
        let (result, temp) = BashTool::<TokioSpawner>::smart_truncate(&long);
        assert!(result.len() < long.len());
        assert!(result.contains("lines omitted"));
        assert!(result.contains("line 0"));
        assert!(result.contains("line 4999"));
        assert!(temp.is_some());
        // Clean up temp file
        if let Some(ref path) = temp {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn test_smart_truncate_preserves_char_boundaries() {
        for piece in ["你好世界\n", "こんにちは世界\n", "안녕하세요 세계\n"] {
            let long: String = piece.repeat(10_000);
            let (result, temp) = BashTool::<TokioSpawner>::smart_truncate(&long);
            assert!(result.len() < long.len());
            assert!(result.chars().all(|c| c != '\u{FFFD}'));
            if let Some(path) = &temp {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}
