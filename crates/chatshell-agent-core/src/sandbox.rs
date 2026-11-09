//! OS sandbox for the bash session (macOS Seatbelt, Linux bubblewrap).
//!
//! [`wrap`] turns the bash [`SpawnSpec`] into one that launches bash under
//! the platform sandbox with a fixed profile:
//!
//! - writes confined to the writable roots (the conversation working
//!   directory and a per-conversation scratch directory), with the
//!   `path_policy` home write-blocklist and each root's `.git/hooks` and
//!   `.git/config` denied on top (code-execution vectors);
//! - reads open except the `path_policy` read deny groups (credentials,
//!   keychains, shell configuration and history, browser and personal
//!   data, project secrets by file name) and the host's registered private
//!   directories. Ask-tier groups are denied too: the jail cannot ask, so
//!   a denial escalates instead;
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
//! writable binds of the roots, masks over blocked files (mode-0000, so a
//! read fails with `EACCES` instead of returning empty content), a private pid
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
    /// Extra directories that must not be read (the host's own data
    /// directory; see `path_policy::register_private_dir`).
    pub private_dirs: Vec<PathBuf>,
    /// The temp channel: the conversation scratch directory when literal
    /// `/tmp` is writable inside the jail. Linux binds the scratch over
    /// `/tmp` (`/tmp` *is* the scratch); macOS cannot mount, so it allows
    /// writes to the OS temporary directory (`/private/tmp`) instead.
    /// `None` when the channel is off (see [`temp_channel_available`]).
    pub temp_scratch: Option<PathBuf>,
}

/// The literal temporary directory and its resolved spelling.
pub const TMP_SPELLINGS: &[&str] = &["/tmp", "/private/tmp"];

/// Whether the jail can make literal `/tmp` writable for `roots`.
///
/// macOS allows the OS temporary directory directly. Linux binds the
/// scratch over `/tmp`, which would hide any writable root living under
/// `/tmp` (the desktop's scratch fallback does), so the channel is off
/// then. Other platforms have no jail. The launch spec, the jail-profile
/// evidence, and the analysis temp roots all use this one predicate.
pub fn temp_channel_available(roots: &[PathBuf]) -> bool {
    if cfg!(target_os = "macos") {
        return true;
    }
    if !cfg!(target_os = "linux") {
        return false;
    }
    !roots.iter().any(|root| under_tmp(root))
}

fn under_tmp(path: &Path) -> bool {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    TMP_SPELLINGS
        .iter()
        .any(|tmp| canonical.starts_with(tmp) || path.starts_with(tmp))
}

/// How the temp channel appears to bash, for the jail-profile evidence.
pub fn describe_temp_channel(policy: &SandboxPolicy) -> Option<String> {
    policy.temp_scratch.as_ref()?;
    Some(if cfg!(target_os = "linux") {
        "/tmp is an alias of the conversation scratch directory".to_string()
    } else {
        "/tmp (/private/tmp, the shared OS temporary directory) is writable".to_string()
    })
}

impl SandboxPolicy {
    /// The bash session policy: writable roots are the explicit working
    /// directory (never a home fallback) plus the scratch directory. The
    /// bash tool launches with this, and the permission gate describes it
    /// to the auto-mode judge, so both always agree.
    pub fn for_bash(workspace: Option<&Path>, scratch_dir: &Path, allow_network: bool) -> Self {
        let mut writable_roots: Vec<PathBuf> = workspace.iter().map(|w| w.to_path_buf()).collect();
        writable_roots.push(scratch_dir.to_path_buf());
        let temp_scratch =
            temp_channel_available(&writable_roots).then(|| scratch_dir.to_path_buf());
        SandboxPolicy {
            writable_roots,
            allow_network,
            home: dirs::home_dir(),
            private_dirs: crate::path_policy::private_dirs(),
            temp_scratch,
        }
    }
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

/// The jail's read denies: every `path_policy` deny group (both tiers:
/// the jail cannot ask, it denies and the escalation prompt takes over)
/// plus the policy's private directories.
#[derive(Debug, Default)]
struct ReadDenies {
    /// Whole paths (files or directory subtrees), kernel spelling.
    paths: Vec<PathBuf>,
    /// File names denied anywhere.
    names: Vec<&'static str>,
    /// File-name prefixes denied anywhere, with exempt names.
    prefixes: Vec<(&'static str, &'static [&'static str])>,
}

fn read_denies(policy: &SandboxPolicy, roots: &[PathBuf]) -> ReadDenies {
    let mut denies = ReadDenies::default();
    for group in path_policy::active_read_groups() {
        for entry in group.entries {
            match *entry {
                path_policy::DenyEntry::Home(rel) => {
                    if let Some(home) = &policy.home {
                        denies.paths.push(resolved(&home.join(rel)));
                    }
                }
                path_policy::DenyEntry::Absolute(abs) => {
                    denies.paths.push(resolved(Path::new(abs)));
                }
                path_policy::DenyEntry::Name(name) => denies.names.push(name),
                path_policy::DenyEntry::NamePrefix { prefix, except } => {
                    denies.prefixes.push((prefix, except));
                }
            }
        }
    }
    denies
        .paths
        .extend(policy.private_dirs.iter().map(|dir| resolved(dir)));
    // A deny covering a writable root would make the workspace itself
    // unreadable (and, under bubblewrap, hide it): skip it. The file
    // tools still refuse those paths.
    denies
        .paths
        .retain(|deny| !roots.iter().any(|root| root.starts_with(deny)));
    denies.paths.sort();
    denies.paths.dedup();
    denies
}

/// Whether a file name is denied by name or name prefix.
fn name_denied(denies: &ReadDenies, name: &str) -> bool {
    denies.names.contains(&name)
        || denies.prefixes.iter().any(|(prefix, except)| {
            name.starts_with(prefix) && name.len() > prefix.len() && !except.contains(&name)
        })
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

    // The temp channel: literal `/tmp` writes, by the resolved spelling
    // (Seatbelt matches the kernel path).
    if policy.temp_scratch.is_some() {
        let name = b.param("TEMP_ROOT", resolved(Path::new("/tmp")));
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

    // Read denies. Name rules first: each name-prefix deny is followed by
    // allows for its exempt names (`.env.example`), and the path denies
    // after them win over those allows (the last matching rule wins).
    let denies = read_denies(policy, roots);
    for (prefix, except) in &denies.prefixes {
        b.profile.push_str(&format!(
            "(deny file-read* (regex #\"/{}[^/]+$\"))\n",
            sbpl_regex_escape(prefix)
        ));
        for exempt in *except {
            b.profile.push_str(&format!(
                "(allow file-read* (regex #\"/{}$\"))\n",
                sbpl_regex_escape(exempt)
            ));
        }
    }
    for file_name in &denies.names {
        b.profile.push_str(&format!(
            "(deny file-read* (regex #\"/{}$\"))\n",
            sbpl_regex_escape(file_name)
        ));
    }
    for path in denies.paths {
        let name = b.param("READ_DENY", path);
        b.profile
            .push_str(&format!("(deny file-read* (subpath (param \"{name}\")))\n"));
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
    // The temp channel: the scratch is bound over `/tmp`, so `/tmp` is the
    // scratch. Never when a writable root lives under `/tmp` (the overlay
    // would hide it).
    if let Some(scratch) = &policy.temp_scratch
        && let Ok(scratch) = scratch.canonicalize()
        && !roots.iter().any(|root| under_tmp(root))
    {
        push3(&mut args, "--bind", &scratch, Path::new("/tmp"));
    }
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
    // Masks fail reads loudly (EACCES): an unreadable empty directory or
    // file, not an empty one that reads as blank content.
    let placeholder = deny_placeholder();
    let file_mask = placeholder.as_deref().unwrap_or(Path::new("/dev/null"));
    for path in bubblewrap_read_masks(policy, roots) {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            args.extend([
                "--perms".into(),
                "0000".into(),
                "--tmpfs".into(),
                path.display().to_string(),
            ]);
        } else if meta.is_file() {
            push3(&mut args, "--ro-bind", file_mask, &path);
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

/// Existing paths bubblewrap must mask: the deny paths, and blocked file
/// names found under the writable roots. bubblewrap cannot mask by
/// pattern, so this is a spawn-time snapshot: files outside the scan
/// bounds or created later stay readable by bash (see the module docs;
/// the in-tool `path_policy` check covers only the file tools). Masks
/// inside an already-masked directory are dropped (the outer mask hides
/// them, and nothing can be mounted inside an unreadable tmpfs).
fn bubblewrap_read_masks(policy: &SandboxPolicy, roots: &[PathBuf]) -> Vec<PathBuf> {
    let denies = read_denies(policy, roots);
    let mut masks: Vec<PathBuf> = denies
        .paths
        .iter()
        .filter(|path| std::fs::symlink_metadata(path).is_ok_and(|m| !m.file_type().is_symlink()))
        .cloned()
        .collect();
    let mut budget = MASK_SCAN_LIMIT;
    for root in roots {
        scan_blocked_names(&denies, root, 0, &mut budget, &mut masks);
    }
    masks.sort();
    masks.dedup();
    let mut kept: Vec<PathBuf> = Vec::with_capacity(masks.len());
    for mask in masks {
        if !kept
            .iter()
            .any(|outer| mask != *outer && mask.starts_with(outer))
        {
            kept.push(mask);
        }
    }
    kept
}

fn scan_blocked_names(
    denies: &ReadDenies,
    dir: &Path,
    depth: usize,
    budget: &mut usize,
    out: &mut Vec<PathBuf>,
) {
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
                scan_blocked_names(denies, &entry.path(), depth + 1, budget, out);
            }
        } else if kind.is_file() && name_denied(denies, &name) {
            out.push(entry.path());
        }
    }
}

/// An empty, mode-0000 file owned by this user, bound over masked files
/// so reading them fails with `EACCES`. Lives in a private per-user
/// directory (`$XDG_RUNTIME_DIR`, else the temp dir); `None` (fall back
/// to `/dev/null`, which reads as empty) if it cannot be set up safely.
#[cfg(target_os = "linux")]
fn deny_placeholder() -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    static PLACEHOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PLACEHOLDER
        .get_or_init(|| {
            // `/proc/self` is owned by the calling process's uid.
            let uid = std::fs::metadata("/proc/self").ok()?.uid();
            let base = std::env::var_os("XDG_RUNTIME_DIR")
                .map(PathBuf::from)
                .filter(|dir| dir.is_absolute())
                .unwrap_or_else(std::env::temp_dir);
            let dir = base.join(format!("chatshell-sandbox-{uid}"));
            let _ = std::fs::DirBuilder::new().mode(0o700).create(&dir);
            let dir_meta = std::fs::symlink_metadata(&dir).ok()?;
            if !dir_meta.is_dir() || dir_meta.uid() != uid || dir_meta.mode() & 0o077 != 0 {
                return None;
            }
            let file = dir.join("deny-placeholder");
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o000)
                .open(&file);
            let meta = std::fs::symlink_metadata(&file).ok()?;
            if !meta.is_file() || meta.uid() != uid || meta.len() != 0 {
                return None;
            }
            if meta.permissions().mode() & 0o777 != 0 {
                std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o000)).ok()?;
            }
            Some(file)
        })
        .clone()
}

#[cfg(not(target_os = "linux"))]
fn deny_placeholder() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(root: &Path, network: bool) -> SandboxPolicy {
        SandboxPolicy {
            writable_roots: vec![root.to_path_buf()],
            allow_network: network,
            home: Some(PathBuf::from("/Users/tester")),
            private_dirs: Vec::new(),
            temp_scratch: None,
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
        let prefix = profile
            .find(r#"(deny file-read* (regex #"/\.env\.[^/]+$"))"#)
            .expect("env prefix deny");
        let exempt = profile
            .find(r#"(allow file-read* (regex #"/\.env\.example$"))"#)
            .expect("template exemption");
        let path_deny = profile
            .find("(deny file-read* (subpath (param \"READ_DENY_")
            .unwrap();
        assert!(
            prefix < exempt && exempt < path_deny,
            "path denies must win"
        );
        assert!(
            spec.args.iter().any(|a| a.ends_with("=/Users/tester/.ssh")),
            "whole ~/.ssh denied"
        );
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
        let file_mask = deny_placeholder()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/dev/null".into());
        assert!(args.contains(&format!("--ro-bind {file_mask} {root_s}/app/.env")));
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
    fn private_dirs_are_denied_unless_they_contain_a_root() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let root = base.join("work");
        std::fs::create_dir_all(&root).unwrap();
        let mut p = policy(&root, false);
        p.private_dirs = vec![base.join("app-data"), base.clone()];
        let spec = wrap(&SandboxBackend::Seatbelt, &p, inner(&root)).unwrap();
        let app_data = format!("={}", base.join("app-data").display());
        assert!(
            spec.args
                .iter()
                .any(|a| a.starts_with("READ_DENY_") && a.ends_with(&app_data))
        );
        // `base` contains the writable root: denying it would make the
        // workspace unreadable, so it is skipped.
        let base_deny = format!("={}", base.display());
        assert!(
            !spec
                .args
                .iter()
                .any(|a| a.starts_with("READ_DENY_") && a.ends_with(&base_deny))
        );
    }

    #[test]
    fn bubblewrap_masks_fail_reads_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let home = base.join("home");
        let root = base.join("work");
        std::fs::create_dir_all(home.join(".ssh")).unwrap();
        std::fs::create_dir_all(home.join(".aws/sso")).unwrap();
        std::fs::write(home.join(".zshrc"), "export KEY=1").unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(".env.test"), "SECRET=1").unwrap();
        std::fs::write(root.join(".env.example"), "SECRET=").unwrap();
        let mut p = policy(&root, false);
        p.home = Some(home.clone());
        // Nested under the `~/.aws` mask: must be dropped.
        p.private_dirs = vec![home.join(".aws/sso")];
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let spec = wrap(&backend, &p, inner(&root)).unwrap();
        let args = spec.args.join(" ");
        let ssh = home.join(".ssh").display().to_string();
        assert!(
            args.contains(&format!("--perms 0000 --tmpfs {ssh}")),
            "{args}"
        );
        let aws = home.join(".aws").display().to_string();
        assert!(
            args.contains(&format!("--perms 0000 --tmpfs {aws} ")),
            "{args}"
        );
        assert!(!args.contains(&format!("{aws}/sso")), "{args}");
        let zshrc = home.join(".zshrc").display().to_string();
        let file_mask = deny_placeholder()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/dev/null".into());
        assert!(
            args.contains(&format!("--ro-bind {file_mask} {zshrc}")),
            "{args}"
        );
        let env_test = root.join(".env.test").display().to_string();
        assert!(
            args.contains(&format!("--ro-bind {file_mask} {env_test}")),
            "{args}"
        );
        assert!(!args.contains(".env.example"), "{args}");
    }

    #[test]
    fn regex_escape_covers_metacharacters() {
        assert_eq!(sbpl_regex_escape(".env.local"), r"\.env\.local");
        assert_eq!(sbpl_regex_escape("a+b"), r"a\+b");
    }

    #[test]
    fn bubblewrap_binds_the_scratch_over_tmp() {
        let dir = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let root = dir.path().canonicalize().unwrap();
        let scratch = root.join("scratch");
        std::fs::create_dir_all(&scratch).unwrap();
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let mut p = policy(&root, false);
        p.writable_roots.push(scratch.clone());
        p.temp_scratch = Some(scratch.clone());
        let args = wrap(&backend, &p, inner(&root)).unwrap().args.join(" ");
        assert!(args.contains(&format!("--bind {} /tmp", scratch.display())));
        // An alias, never a writable-root bind of /tmp itself.
        assert!(!args.contains("--bind /tmp /tmp"));
        // Without the channel, /tmp stays as the read-only base shows it.
        p.temp_scratch = None;
        let args = wrap(&backend, &p, inner(&root)).unwrap().args.join(" ");
        assert!(!args.contains(" /tmp "));
    }

    #[test]
    fn the_tmp_overlay_never_hides_a_root_under_tmp() {
        let dir = tempfile::tempdir_in("/tmp").unwrap();
        let root = dir.path().canonicalize().unwrap();
        let backend = SandboxBackend::Bubblewrap(PathBuf::from("/usr/bin/bwrap"));
        let mut p = policy(&root, false);
        p.temp_scratch = Some(root.clone());
        let args = wrap(&backend, &p, inner(&root)).unwrap().args.join(" ");
        assert!(!args.contains(&format!("--bind {} /tmp", root.display())));
        if cfg!(target_os = "linux") {
            assert!(!temp_channel_available(std::slice::from_ref(&root)));
            let policy = SandboxPolicy::for_bash(None, &root, false);
            assert_eq!(policy.temp_scratch, None);
            assert_eq!(describe_temp_channel(&policy), None);
        }
    }

    #[test]
    fn seatbelt_allows_the_resolved_tmp_with_the_channel() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mut p = policy(&root, false);
        p.temp_scratch = Some(root.clone());
        let spec = wrap(&SandboxBackend::Seatbelt, &p, inner(&root)).unwrap();
        let tmp = resolved(Path::new("/tmp")).display().to_string();
        assert!(
            spec.args
                .iter()
                .any(|a| a.starts_with("TEMP_ROOT_") && a.ends_with(&format!("={tmp}")))
        );
        assert!(spec.args[1].contains("(allow file-write* (subpath (param \"TEMP_ROOT_"));
        p.temp_scratch = None;
        let spec = wrap(&SandboxBackend::Seatbelt, &p, inner(&root)).unwrap();
        assert!(!spec.args[1].contains("TEMP_ROOT"));
    }

    #[test]
    fn the_temp_channel_follows_the_platform() {
        let home = dirs::home_dir().unwrap();
        let policy = SandboxPolicy::for_bash(None, &home.join("scratch"), false);
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert_eq!(policy.temp_scratch, Some(home.join("scratch")));
            assert!(describe_temp_channel(&policy).is_some());
        } else {
            assert_eq!(policy.temp_scratch, None);
        }
        // The channel is never a writable root.
        assert!(!policy.writable_roots.iter().any(|r| r.starts_with("/tmp")));
    }
}
