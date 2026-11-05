//! OS sandbox for the bash session (macOS Seatbelt, Linux bubblewrap).
//!
//! [`wrap`] turns the bash [`SpawnSpec`] into one that launches bash under
//! the platform sandbox with a fixed profile:
//!
//! - writes confined to the writable roots (the conversation working
//!   directory and a per-conversation scratch directory), with the
//!   `path_policy` home write-blocklist and each root's `.git/hooks` and
//!   `.git/config` denied on top (code-execution vectors);
//! - reads open except the `path_policy` read-blocklist (credential
//!   files, `.env`-style secrets);
//! - network limited to loopback unless `allow_network` is set.
//!
//! macOS: `/usr/bin/sandbox-exec` is invoked by absolute path (a `PATH`
//! entry can never substitute it) and every path travels as a `-D`
//! parameter; the profile text contains only constants. Each writable root
//! carries an anchor deny on unlinking/renaming the root itself, and each
//! root's `.git` entry is write-denied by name (no rename, removal,
//! replacement, or creation), so the `.git` carveouts cannot be relocated
//! from inside the sandbox: moving `.git` aside, planting a hook, and
//! moving it back fails at the first step.
//!
//! Linux: bubblewrap from a fixed system location; read-only bind of `/`,
//! writable binds of the roots, masks over blocked files, a private pid
//! namespace, and a new session (no controlling terminal). A `.git`
//! directory is bound onto itself so it becomes a mount point, which the
//! kernel refuses to rename or remove (`EBUSY`); a `.git` file (worktree
//! pointer) is bound read-only.
//!
//! # Known Linux limitations
//!
//! bubblewrap works with mounts, so it can only cover paths that exist
//! when the session starts and cannot match by file name. Seatbelt has
//! neither limit (it matches paths and name patterns at access time), so
//! these gaps are Linux-only:
//!
//! - **Read blocklist by name.** Blocked file names (`.env`,
//!   `credentials.json`, ...) are found by a scan of the writable roots at
//!   spawn time, bounded by depth (`MASK_SCAN_MAX_DEPTH`), an entry
//!   budget (`MASK_SCAN_LIMIT`), and the skipped directories
//!   (`MASK_SCAN_SKIP`). Such a file beyond those bounds, or created
//!   during the session, is readable by bash. Home entries (`~/.ssh/id_*`,
//!   ...) are likewise masked only if they exist at spawn time.
//! - **`.git` anchoring.** A symlinked `.git` cannot be anchored
//!   (bubblewrap refuses symlink mount destinations), and a `.git` absent
//!   at spawn time cannot be either (mounting would create one on the
//!   host). In both cases bash can create or replace `.git` with hooks.
//! - **Nested repositories** (a `.git` below a root, not at it) are not
//!   anchored on either platform.
//!
//! The file tools' `path_policy` checks still apply to their own reads
//! and writes. Bash gets only a textual check of direct write targets,
//! which indirect targets (`h=.git/hooks; echo x > "$h/pre-commit"`)
//! pass, so none of these gaps is closed for bash by in-tool checks.

use crate::grant_keys;
use crate::path_policy;
use crate::process_api::SpawnSpec;
use std::path::{Path, PathBuf};

/// Absolute path of the macOS sandbox launcher.
pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Fixed locations searched for bubblewrap (never `PATH`).
pub const BWRAP_PATHS: &[&str] = &["/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap"];

/// Upper bound on entries visited when scanning writable roots for
/// blocked file names (bubblewrap cannot mask by pattern).
const MASK_SCAN_LIMIT: usize = 20_000;

/// Deepest directory level the mask scan descends to below a root.
const MASK_SCAN_MAX_DEPTH: usize = 8;

/// Directories skipped by the mask scan: large build and dependency
/// trees, where scanning costs startup time and project secrets rarely
/// live. A blocked name inside them stays readable on Linux (see the
/// module docs).
const MASK_SCAN_SKIP: &[&str] = &[".git", "node_modules", "target", ".venv", "venv", "dist"];

/// Repository entry under each writable root whose name is anchored.
const GIT_DIR: &str = ".git";

/// Paths inside [`GIT_DIR`] that stay read-only (code-execution vectors).
const GIT_PROTECTED: &[&str] = &["hooks", "config"];

/// An available sandbox launcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxBackend {
    /// macOS Seatbelt via [`SANDBOX_EXEC`].
    Seatbelt,
    /// Linux bubblewrap at this absolute path.
    Bubblewrap(PathBuf),
}

/// The sandbox backend usable on this machine, if any.
///
/// On Linux, bubblewrap must also pass a one-time probe of the namespace
/// setup the jail needs (user and pid namespaces plus a private procfs).
/// It fails where unprivileged user namespaces are disabled, or inside
/// containers whose `/proc` is partially masked (the kernel refuses a
/// fresh procfs mount there). Such a machine reports no backend, so
/// callers degrade exactly as when bubblewrap is not installed, instead
/// of every bash session dying at startup.
pub fn detect_backend() -> Option<SandboxBackend> {
    if cfg!(target_os = "macos") && Path::new(SANDBOX_EXEC).is_file() {
        return Some(SandboxBackend::Seatbelt);
    }
    if cfg!(target_os = "linux") {
        static PROBED: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
        return PROBED
            .get_or_init(|| {
                BWRAP_PATHS
                    .iter()
                    .map(Path::new)
                    .find(|path| path.is_file())
                    .filter(|path| bubblewrap_works(path))
                    .map(Path::to_path_buf)
            })
            .clone()
            .map(SandboxBackend::Bubblewrap);
    }
    None
}

/// Fixed locations searched for the probe's no-op program.
const TRUE_PATHS: &[&str] = &["/usr/bin/true", "/bin/true"];

/// Whether bubblewrap can set up the namespaces [`bubblewrap_spec`] uses.
fn bubblewrap_works(bwrap: &Path) -> bool {
    let Some(true_bin) = TRUE_PATHS.iter().find(|p| Path::new(p).is_file()) else {
        return false;
    };
    std::process::Command::new(bwrap)
        .args([
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--unshare-pid",
            "--proc",
            "/proc",
            "--unshare-net",
            "--new-session",
            "--",
            true_bin,
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// What the sandboxed bash may do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicy {
    /// Directories bash may write (each must exist).
    pub writable_roots: Vec<PathBuf>,
    /// Allow non-loopback network access.
    pub allow_network: bool,
    /// User home directory, for home-relative blocklists.
    pub home: Option<PathBuf>,
}

/// Why a policy cannot be sandboxed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SandboxError {
    #[error("writable root {0} is not an existing absolute directory")]
    InvalidRoot(PathBuf),
    #[error("refusing to make the filesystem root writable")]
    RootIsFilesystemRoot,
}

/// Wrap `inner` (the bash launch) so it runs inside the sandbox.
pub fn wrap(
    backend: &SandboxBackend,
    policy: &SandboxPolicy,
    inner: SpawnSpec,
) -> Result<SpawnSpec, SandboxError> {
    let roots = canonical_roots(&policy.writable_roots)?;
    match backend {
        SandboxBackend::Seatbelt => Ok(seatbelt_spec(policy, &roots, inner)),
        SandboxBackend::Bubblewrap(bwrap) => Ok(bubblewrap_spec(bwrap, policy, &roots, inner)),
    }
}

fn canonical_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
    let mut out = Vec::with_capacity(roots.len());
    for root in roots {
        if !root.is_absolute() {
            return Err(SandboxError::InvalidRoot(root.clone()));
        }
        let canonical = root
            .canonicalize()
            .ok()
            .filter(|path| path.is_dir())
            .ok_or_else(|| SandboxError::InvalidRoot(root.clone()))?;
        if canonical.parent().is_none() {
            return Err(SandboxError::RootIsFilesystemRoot);
        }
        if !out.contains(&canonical) {
            out.push(canonical);
        }
    }
    Ok(out)
}

/// A home-relative read-blocklist entry, classified.
enum HomeEntry {
    /// Whole path (file or directory).
    Path(PathBuf),
    /// Entries of `dir` whose name starts with `prefix` (`.ssh/id_`).
    Prefix { dir: PathBuf, prefix: String },
}

fn home_read_entries(home: &Path) -> Vec<HomeEntry> {
    path_policy::READ_BLOCKED_HOME_PREFIXES
        .iter()
        .map(|entry| {
            let trimmed = entry.trim_end_matches('/');
            if !entry.ends_with('/') && trimmed.ends_with('_') {
                let (dir, prefix) = trimmed.rsplit_once('/').unwrap_or(("", trimmed));
                HomeEntry::Prefix {
                    dir: home.join(dir),
                    prefix: prefix.to_string(),
                }
            } else {
                HomeEntry::Path(home.join(trimmed))
            }
        })
        .collect()
}

/// Resolve a blocklist path the way the kernel will see it (`/etc` is
/// `/private/etc` on macOS), tolerating paths that do not exist yet.
fn resolved(path: &Path) -> PathBuf {
    path.to_str()
        .and_then(grant_keys::grant_path)
        .unwrap_or_else(|| path.to_path_buf())
}

fn home_write_denied_paths(home: Option<&Path>) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = home
        .map(|home| {
            path_policy::WRITE_BLOCKED_HOME
                .iter()
                .map(|entry| resolved(&home.join(entry.trim_end_matches('/'))))
                .collect()
        })
        .unwrap_or_default();
    paths.dedup();
    paths
}

// ---------------------------------------------------------------------------
// macOS Seatbelt
// ---------------------------------------------------------------------------

/// Base policy: deny by default, allow what ordinary command-line tools
/// need (process control, sysctl reads, tty/null devices, POSIX IPC, the
/// directory/user lookup and TLS trust services).
///
/// Another process's argv and environment (`kern.procargs2`, the `ps eww`
/// path) are readable if either the sysctl or `process-info-pidinfo` is
/// allowed, and `(deny default)` does not cover pidinfo. Both are denied
/// and pidinfo is re-allowed only inside the sandbox, so environments of
/// processes outside it (which carry API keys) stay unreadable.
///
/// The keychain's `com.apple.SecurityServer` service is not reachable:
/// file denies do not stop keychain IPC.
const SEATBELT_BASE: &str = r#"(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow sysctl-read)
(deny sysctl-read (sysctl-name-regex #"procargs"))
(deny process-info-pidinfo)
(allow process-info-pidinfo (target same-sandbox))
(allow user-preference-read)
(allow file-read*)
(allow file-read-metadata)
(allow file-write-data file-ioctl
  (literal "/dev/null")
  (literal "/dev/zero")
  (literal "/dev/tty")
  (literal "/dev/dtracehelper")
  (regex #"^/dev/fd/[0-9]+$"))
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+$"))
(allow ipc-posix-sem)
(allow ipc-posix-shm)
(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.system.opendirectoryd.membership")
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.PowerManagement.control")
  (global-name "com.apple.system.logger")
  (global-name "com.apple.logd")
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.ocspd"))
"#;

/// Network rules when non-loopback access is allowed.
const SEATBELT_NETWORK_ON: &str = r#"(allow network*)
(allow system-socket)
(allow mach-lookup
  (global-name "com.apple.dnssd.service")
  (global-name "com.apple.networkd")
  (global-name "com.apple.SystemConfiguration.DNSConfiguration")
  (global-name "com.apple.SystemConfiguration.configd"))
"#;

/// Network rules otherwise: loopback only (local dev servers, databases).
const SEATBELT_NETWORK_LOOPBACK: &str = r#"(allow network-outbound (remote ip "localhost:*"))
(allow network-inbound (local ip "localhost:*"))
(allow network-bind (local ip "localhost:*"))
"#;

/// Escape a constant for an SBPL `#"..."` regex literal.
fn sbpl_regex_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for c in text.chars() {
        if r".^$|?*+()[]{}\".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Builds the profile text and its `-D` parameters together so every
/// path appears only as a parameter reference.
struct SeatbeltBuilder {
    profile: String,
    params: Vec<(String, PathBuf)>,
}

impl SeatbeltBuilder {
    fn param(&mut self, prefix: &str, path: PathBuf) -> String {
        let name = format!("{prefix}_{}", self.params.len());
        self.params.push((name.clone(), path));
        name
    }
}

fn seatbelt_profile(policy: &SandboxPolicy, roots: &[PathBuf]) -> SeatbeltBuilder {
    let mut b = SeatbeltBuilder {
        profile: String::from(SEATBELT_BASE),
        params: Vec::new(),
    };

    b.profile.push_str(if policy.allow_network {
        SEATBELT_NETWORK_ON
    } else {
        SEATBELT_NETWORK_LOOPBACK
    });

    // Writable roots.
    for root in roots {
        let name = b.param("WRITABLE_ROOT", root.clone());
        b.profile.push_str(&format!(
            "(allow file-write* (subpath (param \"{name}\")))\n"
        ));
    }

    // Denies come after the allows: the last matching rule wins.
    let git_protected = roots.iter().flat_map(|root| {
        GIT_PROTECTED
            .iter()
            .map(move |entry| root.join(GIT_DIR).join(entry))
    });
    for path in home_write_denied_paths(policy.home.as_deref())
        .into_iter()
        .chain(git_protected)
    {
        let name = b.param("WRITE_DENY", path);
        b.profile.push_str(&format!(
            "(deny file-write* (subpath (param \"{name}\")))\n"
        ));
    }
    // Anchor denies: the root directory itself cannot be removed or
    // renamed, and its `.git` entry cannot be renamed, removed, created,
    // or rewritten (a `.git` file is a gitdir pointer), so the carveouts
    // above always name the repository git will use. Entries inside
    // `.git` are separate paths, so ordinary git operations still work.
    for root in roots {
        let name = b.param("ANCHOR", root.clone());
        b.profile.push_str(&format!(
            "(deny file-write-unlink (literal (param \"{name}\")))\n"
        ));
        let name = b.param("GIT_ANCHOR", root.join(GIT_DIR));
        b.profile.push_str(&format!(
            "(deny file-write* (literal (param \"{name}\")))\n"
        ));
    }

    // Read blocklist.
    if let Some(home) = &policy.home {
        for entry in home_read_entries(home) {
            match entry {
                HomeEntry::Path(path) => {
                    let name = b.param("READ_DENY", resolved(&path));
                    b.profile
                        .push_str(&format!("(deny file-read* (subpath (param \"{name}\")))\n"));
                }
                HomeEntry::Prefix { dir, prefix } => {
                    let name = b.param("READ_DENY_DIR", resolved(&dir));
                    b.profile.push_str(&format!(
                        "(deny file-read* (require-all (subpath (param \"{name}\")) \
                         (regex #\"/{}[^/]*$\")))\n",
                        sbpl_regex_escape(&prefix)
                    ));
                }
            }
        }
    }
    for absolute in path_policy::READ_BLOCKED_ABSOLUTE {
        let name = b.param("READ_DENY", resolved(Path::new(absolute)));
        b.profile
            .push_str(&format!("(deny file-read* (literal (param \"{name}\")))\n"));
    }
    for file_name in path_policy::READ_BLOCKED_FILENAMES {
        b.profile.push_str(&format!(
            "(deny file-read* (regex #\"/{}$\"))\n",
            sbpl_regex_escape(file_name)
        ));
    }
    b
}

fn seatbelt_spec(policy: &SandboxPolicy, roots: &[PathBuf], inner: SpawnSpec) -> SpawnSpec {
    let built = seatbelt_profile(policy, roots);
    let mut args = vec!["-p".to_string(), built.profile];
    for (name, path) in built.params {
        args.push("-D".to_string());
        args.push(format!("{name}={}", path.display()));
    }
    args.push("--".to_string());
    args.push(inner.program);
    args.extend(inner.args);
    SpawnSpec {
        program: SANDBOX_EXEC.to_string(),
        args,
        cwd: inner.cwd,
        env: inner.env,
    }
}

// ---------------------------------------------------------------------------
// Linux bubblewrap
// ---------------------------------------------------------------------------

fn bubblewrap_spec(
    bwrap: &Path,
    policy: &SandboxPolicy,
    roots: &[PathBuf],
    inner: SpawnSpec,
) -> SpawnSpec {
    let mut args: Vec<String> = vec![
        "--die-with-parent".into(),
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
        "--dev".into(),
        "/dev".into(),
        // A private pid namespace makes `--proc` mount a fresh procfs:
        // host processes (and their environments) stay invisible and
        // cannot be signaled or traced.
        "--unshare-pid".into(),
        "--proc".into(),
        "/proc".into(),
        // A new session detaches bash from the app's controlling
        // terminal: no TIOCSTI keystroke injection into it, and no
        // signals to its foreground process group.
        "--new-session".into(),
    ];
    let push3 = |args: &mut Vec<String>, flag: &str, a: &Path, b: &Path| {
        args.push(flag.to_string());
        args.push(a.display().to_string());
        args.push(b.display().to_string());
    };
    for root in roots {
        push3(&mut args, "--bind", root, root);
    }
    for path in home_write_denied_paths(policy.home.as_deref()) {
        if path.exists() {
            push3(&mut args, "--ro-bind", &path, &path);
        }
    }
    for root in roots {
        protect_git_dir(&mut args, &root.join(GIT_DIR));
    }
    for path in bubblewrap_read_masks(policy, roots) {
        if path.is_dir() {
            args.push("--tmpfs".into());
            args.push(path.display().to_string());
        } else if path.exists() {
            push3(&mut args, "--ro-bind", Path::new("/dev/null"), &path);
        }
    }
    if !policy.allow_network {
        args.push("--unshare-net".into());
    }
    args.push("--chdir".into());
    args.push(inner.cwd.display().to_string());
    args.push("--".into());
    args.push(inner.program);
    args.extend(inner.args);
    SpawnSpec {
        program: bwrap.display().to_string(),
        args,
        cwd: inner.cwd,
        env: inner.env,
    }
}

/// Anchor a root's `.git` and keep its code-execution paths read-only.
///
/// bubblewrap cannot deny by name, so the anchor is a mount: a `.git`
/// directory is bound onto itself (a mount point cannot be renamed or
/// removed from inside the namespace), then `hooks` and `config` are bound
/// read-only on top. A missing `hooks` directory gets a read-only tmpfs
/// (bubblewrap creates the empty directory, as `git init` would). A `.git`
/// file is bound read-only. Symlinked or absent entries are skipped:
/// bubblewrap refuses symlink destinations, and mounting on a missing
/// `.git` would create one on the host.
fn protect_git_dir(args: &mut Vec<String>, git: &Path) {
    let Ok(meta) = std::fs::symlink_metadata(git) else {
        return;
    };
    let git_s = git.display().to_string();
    if meta.is_file() {
        args.extend(["--ro-bind".into(), git_s.clone(), git_s]);
        return;
    }
    if !meta.is_dir() {
        return;
    }
    args.extend(["--bind".into(), git_s.clone(), git_s]);
    for entry in GIT_PROTECTED {
        let path = git.join(entry);
        let path_s = path.display().to_string();
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if !meta.file_type().is_symlink() => {
                args.extend(["--ro-bind".into(), path_s.clone(), path_s]);
            }
            Err(_) if *entry == "hooks" => {
                args.extend([
                    "--tmpfs".into(),
                    path_s.clone(),
                    "--remount-ro".into(),
                    path_s,
                ]);
            }
            _ => {}
        }
    }
}

/// Existing paths bubblewrap must mask: home entries, absolute entries,
/// and blocked file names found under the writable roots. bubblewrap
/// cannot mask by pattern, so this is a spawn-time snapshot: files outside
/// the scan bounds or created later stay readable by bash (see the module
/// docs; the in-tool `path_policy` check covers only the file tools).
fn bubblewrap_read_masks(policy: &SandboxPolicy, roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut masks = Vec::new();
    if let Some(home) = &policy.home {
        for entry in home_read_entries(home) {
            match entry {
                HomeEntry::Path(path) => masks.push(path),
                HomeEntry::Prefix { dir, prefix } => {
                    if let Ok(entries) = std::fs::read_dir(&dir) {
                        masks.extend(
                            entries
                                .flatten()
                                .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
                                .map(|e| e.path()),
                        );
                    }
                }
            }
        }
    }
    masks.extend(path_policy::READ_BLOCKED_ABSOLUTE.iter().map(PathBuf::from));
    let mut budget = MASK_SCAN_LIMIT;
    for root in roots {
        scan_blocked_names(root, 0, &mut budget, &mut masks);
    }
    masks.sort();
    masks.dedup();
    masks
}

fn scan_blocked_names(dir: &Path, depth: usize, budget: &mut usize, out: &mut Vec<PathBuf>) {
    if depth > MASK_SCAN_MAX_DEPTH || *budget == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            if !MASK_SCAN_SKIP.contains(&name.as_ref()) {
                scan_blocked_names(&entry.path(), depth + 1, budget, out);
            }
        } else if path_policy::READ_BLOCKED_FILENAMES.contains(&name.as_ref()) {
            out.push(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(root: &Path, network: bool) -> SandboxPolicy {
        SandboxPolicy {
            writable_roots: vec![root.to_path_buf()],
            allow_network: network,
            home: Some(PathBuf::from("/Users/tester")),
        }
    }

    fn inner(cwd: &Path) -> SpawnSpec {
        SpawnSpec::new("bash", cwd)
    }

    #[test]
    fn seatbelt_passes_paths_only_as_parameters() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let spec = wrap(
            &SandboxBackend::Seatbelt,
            &policy(&root, false),
            inner(&root),
        )
        .unwrap();
        assert_eq!(spec.program, SANDBOX_EXEC);
        let profile = &spec.args[1];
        assert!(!profile.contains(&root.display().to_string()));
        assert!(!profile.contains("/Users/tester"));
        assert!(
            spec.args
                .iter()
                .any(|arg| arg == &format!("WRITABLE_ROOT_0={}", root.display()))
        );
        let dash = spec.args.iter().position(|a| a == "--").unwrap();
        assert_eq!(spec.args[dash + 1], "bash");
    }

    #[test]
    fn seatbelt_network_rules_follow_policy() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let off = wrap(
            &SandboxBackend::Seatbelt,
            &policy(&root, false),
            inner(&root),
        )
        .unwrap();
        assert!(off.args[1].contains("localhost:*"));
        assert!(!off.args[1].contains("(allow network*)"));
        let on = wrap(
            &SandboxBackend::Seatbelt,
            &policy(&root, true),
            inner(&root),
        )
        .unwrap();
        assert!(on.args[1].contains("(allow network*)"));
    }

    #[test]
    fn seatbelt_profile_has_anchor_and_read_denies() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let spec = wrap(
            &SandboxBackend::Seatbelt,
            &policy(&root, false),
            inner(&root),
        )
        .unwrap();
        let profile = &spec.args[1];
        assert!(profile.contains("(deny file-write-unlink (literal (param \"ANCHOR_"));
        assert!(profile.contains(r#"(regex #"/\.env$")"#));
        assert!(profile.contains(r#"(regex #"/id_[^/]*$")"#));
    }

    #[test]
    fn invalid_roots_are_rejected() {
        let backend = SandboxBackend::Seatbelt;
        let mut p = policy(Path::new("relative"), false);
        assert!(matches!(
            wrap(&backend, &p, inner(Path::new("/"))),
            Err(SandboxError::InvalidRoot(_))
        ));
        p.writable_roots = vec![PathBuf::from("/")];
        assert_eq!(
            wrap(&backend, &p, inner(Path::new("/"))),
            Err(SandboxError::RootIsFilesystemRoot)
        );
        p.writable_roots = vec![PathBuf::from("/definitely/not/here")];
        assert!(wrap(&backend, &p, inner(Path::new("/"))).is_err());
    }

    #[test]
    fn bubblewrap_argv_binds_roots_and_masks_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::write(root.join("app/.env"), "SECRET=1").unwrap();
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let spec = wrap(&backend, &policy(&root, false), inner(&root)).unwrap();
        assert_eq!(spec.program, "/usr/bin/bwrap");
        let args = spec.args.join(" ");
        let root_s = root.display().to_string();
        assert!(args.contains(&format!("--bind {root_s} {root_s}")));
        assert!(args.contains(&format!("--ro-bind /dev/null {root_s}/app/.env")));
        assert!(args.contains("--unshare-net"));
        let on = wrap(&backend, &policy(&root, true), inner(&root)).unwrap();
        assert!(!on.args.iter().any(|a| a == "--unshare-net"));
    }

    #[test]
    fn seatbelt_anchors_git_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let spec = wrap(
            &SandboxBackend::Seatbelt,
            &policy(&root, false),
            inner(&root),
        )
        .unwrap();
        let profile = &spec.args[1];
        let git = root.join(".git");
        let param = spec
            .args
            .iter()
            .find_map(|arg| {
                arg.strip_suffix(&format!("={}", git.display()))
                    .filter(|name| name.starts_with("GIT_ANCHOR_"))
            })
            .expect("git anchor parameter");
        let rule = format!("(deny file-write* (literal (param \"{param}\")))");
        let rule_at = profile.find(&rule).expect("git anchor rule");
        let allow_at = profile.find("(allow file-write* (subpath").unwrap();
        assert!(rule_at > allow_at, "anchor must follow the allows");
        for protected in ["hooks", "config"] {
            let path = format!("={}", git.join(protected).display());
            assert!(spec.args.iter().any(|a| a.ends_with(&path)), "{protected}");
        }
    }

    #[test]
    fn bubblewrap_anchors_git_directory_and_isolates_pids() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config"), "").unwrap();
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let spec = wrap(&backend, &policy(&root, false), inner(&root)).unwrap();
        let args = spec.args.join(" ");
        let git = root.join(".git").display().to_string();
        let anchor = format!("--bind {git} {git}");
        let hooks = format!("--tmpfs {git}/hooks --remount-ro {git}/hooks");
        let config = format!("--ro-bind {git}/config {git}/config");
        let root_bind = args
            .find(&format!("--bind {0} {0} ", root.display()))
            .unwrap();
        let anchor_at = args.find(&anchor).expect("git self-bind");
        let hooks_at = args.find(&hooks).expect("read-only hooks tmpfs");
        let config_at = args.find(&config).expect("read-only config");
        assert!(root_bind < anchor_at && anchor_at < hooks_at && anchor_at < config_at);
        assert!(spec.args.iter().any(|a| a == "--unshare-pid"));
        let pid = spec.args.iter().position(|a| a == "--unshare-pid").unwrap();
        let proc = spec.args.iter().position(|a| a == "--proc").unwrap();
        assert!(pid < proc);
    }

    #[test]
    fn seatbelt_hides_other_process_environments_and_the_keychain() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let spec = wrap(
            &SandboxBackend::Seatbelt,
            &policy(&root, true),
            inner(&root),
        )
        .unwrap();
        let profile = &spec.args[1];
        let allow = profile.find("(allow sysctl-read)\n").unwrap();
        let deny = profile
            .find(r#"(deny sysctl-read (sysctl-name-regex #"procargs"))"#)
            .expect("procargs deny");
        let pid_deny = profile.find("(deny process-info-pidinfo)").unwrap();
        let pid_allow = profile
            .find("(allow process-info-pidinfo (target same-sandbox))")
            .unwrap();
        assert!(allow < deny && pid_deny < pid_allow);
        assert!(!profile.contains("com.apple.SecurityServer"));
    }

    #[test]
    fn bubblewrap_starts_a_new_session() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let spec = wrap(&backend, &policy(&root, false), inner(&root)).unwrap();
        let session = spec.args.iter().position(|a| a == "--new-session");
        let dash = spec.args.iter().position(|a| a == "--").unwrap();
        assert!(session.is_some_and(|at| at < dash), "{:?}", spec.args);
    }

    #[test]
    fn bubblewrap_binds_git_file_read_only_and_skips_missing_git() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let spec = wrap(&backend, &policy(&root, false), inner(&root)).unwrap();
        assert!(!spec.args.join(" ").contains(".git"));
        std::fs::write(root.join(".git"), "gitdir: /elsewhere\n").unwrap();
        let spec = wrap(&backend, &policy(&root, false), inner(&root)).unwrap();
        let git = root.join(".git").display().to_string();
        let args = spec.args.join(" ");
        assert!(args.contains(&format!("--ro-bind {git} {git}")), "{args}");
        assert!(!args.contains(&format!("{git}/hooks")), "{args}");
    }

    #[test]
    fn regex_escape_covers_metacharacters() {
        assert_eq!(sbpl_regex_escape(".env.local"), r"\.env\.local");
        assert_eq!(sbpl_regex_escape("a+b"), r"a\+b");
    }
}
