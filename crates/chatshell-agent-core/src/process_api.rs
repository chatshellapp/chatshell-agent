//! Process seam for the bash tool family (ADR 03 Layer 2).
//!
//! The bash engine in `tools::bash` is protocol logic (init handshake, marker
//! framing, exit-code capture, output budgets, abort) over a shell process
//! with piped stdin/stdout. Everything process-specific lives behind
//! [`ShellProcess`]/[`ShellSpawner`]; [`TokioShell`] is the tokio-backed
//! implementation any Unix-ish Rust host (desktop today, Android if it
//! enables shell per the ADR 03 matrix) uses as-is. Platforms that forbid
//! child processes (iOS) simply never construct the bash tool.

use std::future::Future;
use std::io;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// How long `kill` waits for the process to exit before giving up.
const KILL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// A live shell process with piped stdin/stdout (stderr is the host's call:
/// the bash engine merges it inside the shell via `exec 2>&1`).
pub trait ShellProcess: Send + 'static {
    /// Write bytes to the process's stdin.
    fn write_stdin(&mut self, data: &[u8]) -> impl Future<Output = io::Result<()>> + Send;
    /// Flush stdin.
    fn flush_stdin(&mut self) -> impl Future<Output = io::Result<()>> + Send;
    /// Read one `\n`-terminated line (or EOF, `Ok(0)`), appending to `buf`.
    fn read_line(&mut self, buf: &mut Vec<u8>) -> impl Future<Output = io::Result<usize>> + Send;
    /// Process id, for diagnostics.
    fn pid(&self) -> Option<u32>;
    /// True while the process has not exited.
    fn is_alive(&mut self) -> bool;
    /// Request termination without waiting (sync; safe in drop/shutdown).
    fn start_kill(&mut self);

    /// Terminate and wait (bounded by [`KILL_WAIT_TIMEOUT`]). Shared
    /// implementation: signal first, then a timed reap.
    fn kill(&mut self) -> impl Future<Output = ()> + Send {
        async move {
            self.start_kill();
            let _ = tokio::time::timeout(KILL_WAIT_TIMEOUT, self.wait_dead()).await;
        }
    }

    /// Wait for exit (unbounded).
    fn wait_dead(&mut self) -> impl Future<Output = io::Result<()>> + Send;
}

/// Factory spawning shell processes.
pub trait ShellSpawner: Send + Sync + Clone + 'static {
    type Shell: ShellProcess;

    /// Spawn `program` in `cwd` with piped stdin/stdout and null stderr.
    fn spawn(
        &self,
        program: &str,
        cwd: &Path,
    ) -> impl Future<Output = io::Result<Self::Shell>> + Send;
}

/// tokio-backed [`ShellProcess`] over a real child process.
pub struct TokioShell {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

/// Default spawner: any Unix-ish host with a real `bash` on PATH.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioSpawner;

impl ShellSpawner for TokioSpawner {
    type Shell = TokioShell;

    async fn spawn(&self, program: &str, cwd: &Path) -> io::Result<TokioShell> {
        let mut cmd = Command::new(program);
        cmd.current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr is merged into stdout by the shell itself (`exec 2>&1`)
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("failed to open stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("failed to open stdout"))?;
        Ok(TokioShell {
            child,
            stdin,
            reader: BufReader::new(stdout),
        })
    }
}

impl ShellProcess for TokioShell {
    async fn write_stdin(&mut self, data: &[u8]) -> io::Result<()> {
        self.stdin.write_all(data).await
    }

    async fn flush_stdin(&mut self) -> io::Result<()> {
        self.stdin.flush().await
    }

    async fn read_line(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.reader.read_until(b'\n', buf).await
    }

    fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn start_kill(&mut self) {
        let _ = self.child.start_kill();
    }

    async fn wait_dead(&mut self) -> io::Result<()> {
        self.child.wait().await.map(|_| ())
    }
}
