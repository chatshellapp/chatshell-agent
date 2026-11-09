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
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;

/// How long `kill` waits for the process to exit before giving up.
const KILL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// How long [`ShellProcess::exit_report`] waits for the process to exit
/// and its stderr to drain.
const EXIT_REPORT_TIMEOUT: Duration = Duration::from_secs(2);

/// Bytes of launcher stderr kept for [`ShellProcess::exit_report`] (the
/// tail; older output is dropped).
const STDERR_TAIL_BYTES: usize = 4096;

/// Characters of stderr included in an exit report.
const REPORT_STDERR_CHARS: usize = 1000;

/// A live shell process with piped stdin/stdout. The bash engine merges
/// the shell's stderr into stdout via `exec 2>&1`, so the process's own
/// stderr carries only what is written before that: the launcher's
/// messages (a sandbox that fails to start) and early shell errors.
pub trait ShellProcess: Send + 'static {
    /// Write bytes to the process's stdin.
    fn write_stdin(&mut self, data: &[u8]) -> impl Future<Output = io::Result<()>> + Send;
    /// Flush stdin.
    fn flush_stdin(&mut self) -> impl Future<Output = io::Result<()>> + Send;
    /// Read one `\n`-terminated line (or EOF, `Ok(0)`), appending to `buf`.
    fn read_line(&mut self, buf: &mut Vec<u8>) -> impl Future<Output = io::Result<usize>> + Send;
    /// Process id, for diagnostics.
    fn pid(&self) -> Option<u32>;
    /// The shell's current directory as the OS reports it, when the
    /// process is the shell itself (see [`shell_cwd`]). Unlike anything the
    /// shell prints, the shell cannot misreport it.
    fn cwd(&self) -> Option<PathBuf> {
        self.pid().and_then(shell_cwd)
    }
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

    /// Why the process ended early, for errors during session startup:
    /// its exit status and the tail of its stderr. Waits briefly for the
    /// process to exit. `None` when nothing is known.
    fn exit_report(&mut self) -> impl Future<Output = Option<String>> + Send {
        async { None }
    }
}

/// Everything needed to launch a shell process. The bash tool builds one
/// per session: a sandbox wraps `program`/`args` (e.g. `sandbox-exec ...
/// -- bash`) and env minimization replaces the inherited environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnSpec {
    /// Executable to run (an absolute path for sandbox launchers).
    pub program: String,
    /// Arguments after `program`.
    pub args: Vec<String>,
    /// Working directory of the new process.
    pub cwd: PathBuf,
    /// `None`: inherit the host environment. `Some`: start from an empty
    /// environment containing exactly these variables.
    pub env: Option<Vec<(String, String)>>,
}

impl SpawnSpec {
    /// Run `program` with no arguments in `cwd`, inheriting the environment.
    pub fn new(program: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: cwd.into(),
            env: None,
        }
    }
}

/// Factory spawning shell processes.
pub trait ShellSpawner: Send + Sync + Clone + 'static {
    type Shell: ShellProcess;

    /// Spawn `spec` with piped stdin/stdout. Stderr is captured for
    /// [`ShellProcess::exit_report`] where the implementation supports it.
    fn spawn(&self, spec: &SpawnSpec) -> impl Future<Output = io::Result<Self::Shell>> + Send;
}

/// The current directory of process `pid`, read from the OS, when the
/// process is a `bash` (unconfined bash, or Seatbelt's `sandbox-exec`,
/// which execs bash in place). `None` for any other executable — a
/// launcher that forks the shell (bubblewrap) has a directory of its own
/// — and on platforms without a reader.
pub fn shell_cwd(pid: u32) -> Option<PathBuf> {
    let (exe, cwd) = process_exe_and_cwd(pid)?;
    let name = exe.file_name()?.to_str()?;
    (name == "bash" || name.starts_with("bash-")).then_some(cwd)
}

#[cfg(target_os = "linux")]
fn process_exe_and_cwd(pid: u32) -> Option<(PathBuf, PathBuf)> {
    let proc = PathBuf::from(format!("/proc/{pid}"));
    let exe = std::fs::read_link(proc.join("exe")).ok()?;
    let cwd = std::fs::read_link(proc.join("cwd")).ok()?;
    Some((exe, cwd))
}

#[cfg(target_os = "macos")]
fn process_exe_and_cwd(pid: u32) -> Option<(PathBuf, PathBuf)> {
    use std::ffi::CStr;
    use std::os::unix::ffi::OsStrExt;
    let pid = libc::c_int::try_from(pid).ok()?;

    let mut exe = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    // SAFETY: the buffer is writable for the size passed.
    let len = unsafe { libc::proc_pidpath(pid, exe.as_mut_ptr().cast(), exe.len() as u32) };
    if len <= 0 {
        return None;
    }
    exe.truncate(len as usize);

    let size = std::mem::size_of::<libc::proc_vnodepathinfo>();
    // SAFETY: a zeroed plain-data struct is valid; proc_pidinfo writes at
    // most `size` bytes into it.
    let mut info: libc::proc_vnodepathinfo = unsafe { std::mem::zeroed() };
    let written = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDVNODEPATHINFO,
            0,
            (&mut info as *mut libc::proc_vnodepathinfo).cast(),
            size as libc::c_int,
        )
    };
    if written as usize != size {
        return None;
    }
    // `vip_path` is a NUL-terminated MAXPATHLEN buffer split into rows.
    // SAFETY: the rows are contiguous `c_char`s; read as bytes.
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(
            info.pvi_cdir.vip_path.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&info.pvi_cdir.vip_path),
        )
    };
    let cwd = CStr::from_bytes_until_nul(raw).ok()?.to_bytes();
    if cwd.is_empty() {
        return None;
    }
    Some((
        PathBuf::from(std::ffi::OsStr::from_bytes(&exe)),
        PathBuf::from(std::ffi::OsStr::from_bytes(cwd)),
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_exe_and_cwd(_pid: u32) -> Option<(PathBuf, PathBuf)> {
    None
}

/// tokio-backed [`ShellProcess`] over a real child process.
pub struct TokioShell {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    stderr_drain: Option<JoinHandle<()>>,
}

/// Read `stderr` to EOF, keeping the last [`STDERR_TAIL_BYTES`] bytes.
/// Draining continuously means a chatty process never blocks on a full
/// pipe.
async fn drain_stderr(mut stderr: ChildStderr, tail: Arc<Mutex<Vec<u8>>>) {
    let mut chunk = [0u8; 1024];
    while let Ok(n) = stderr.read(&mut chunk).await {
        if n == 0 {
            break;
        }
        let mut tail = tail.lock().unwrap_or_else(|e| e.into_inner());
        tail.extend_from_slice(&chunk[..n]);
        let excess = tail.len().saturating_sub(STDERR_TAIL_BYTES);
        tail.drain(..excess);
    }
}

/// Render captured stderr for an error message: lossy UTF-8, control
/// characters (terminal escapes) removed, whitespace collapsed, and
/// bounded to the last [`REPORT_STDERR_CHARS`] characters.
fn render_stderr(bytes: &[u8]) -> String {
    let text: String = String::from_utf8_lossy(bytes)
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let skip = text.chars().count().saturating_sub(REPORT_STDERR_CHARS);
    text.chars().skip(skip).collect()
}

/// Default spawner: any Unix-ish host with a real `bash` on PATH.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioSpawner;

impl ShellSpawner for TokioSpawner {
    type Shell = TokioShell;

    async fn spawn(&self, spec: &SpawnSpec) -> io::Result<TokioShell> {
        let mut cmd = Command::new(&spec.program);
        cmd.args(&spec.args);
        if let Some(env) = &spec.env {
            cmd.env_clear().envs(env.iter().map(|(k, v)| (k, v)));
        }
        cmd.current_dir(&spec.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The shell merges its stderr into stdout itself (`exec 2>&1`);
            // what arrives here before that is kept for `exit_report`.
            .stderr(Stdio::piped())
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
        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        let stderr_drain = child
            .stderr
            .take()
            .map(|stderr| tokio::spawn(drain_stderr(stderr, stderr_tail.clone())));
        Ok(TokioShell {
            child,
            stdin,
            reader: BufReader::new(stdout),
            stderr_tail,
            stderr_drain,
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

    async fn exit_report(&mut self) -> Option<String> {
        let status = tokio::time::timeout(EXIT_REPORT_TIMEOUT, self.child.wait())
            .await
            .ok()
            .and_then(Result::ok);
        // The process has exited (or is stuck): its stderr reaches EOF once
        // every holder of the pipe is gone.
        if let Some(drain) = self.stderr_drain.as_mut()
            && tokio::time::timeout(EXIT_REPORT_TIMEOUT, drain)
                .await
                .is_ok()
        {
            self.stderr_drain = None;
        }
        let stderr = {
            let tail = self.stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
            render_stderr(&tail)
        };
        let status = status.map(|status| match status.code() {
            Some(code) => format!("exit status {code}"),
            None => format!("{status}"),
        });
        match (status, stderr.is_empty()) {
            (None, true) => None,
            (Some(status), true) => Some(status),
            (None, false) => Some(stderr),
            (Some(status), false) => Some(format!("{status}: {stderr}")),
        }
    }
}

impl Drop for TokioShell {
    fn drop(&mut self) {
        if let Some(drain) = self.stderr_drain.take() {
            drain.abort();
        }
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_os_reports_the_shell_directory() {
        let start = tempfile::tempdir().unwrap();
        let start = start.path().canonicalize().unwrap();
        let mut shell = TokioSpawner
            .spawn(&SpawnSpec::new("bash", &start))
            .await
            .unwrap();
        assert_eq!(shell.cwd(), Some(start.clone()));
        shell.write_stdin(b"cd /; echo done\n").await.unwrap();
        shell.flush_stdin().await.unwrap();
        let mut line = Vec::new();
        shell.read_line(&mut line).await.unwrap();
        assert_eq!(shell.cwd(), Some(PathBuf::from("/")));
        shell.kill().await;
    }

    #[tokio::test]
    async fn a_process_that_is_not_bash_has_no_shell_directory() {
        let mut child = tokio::process::Command::new("sleep")
            .arg("5")
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        assert_eq!(child.id().and_then(shell_cwd), None);
        let _ = child.start_kill();
    }
}
