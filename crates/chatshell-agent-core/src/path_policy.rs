//! Shared path security policy for all tools.
//!
//! Provides read and write access checks that protect sensitive files and
//! enforce project-directory boundaries. Used by built-in tools (read, write,
//! edit, grep, glob) directly, and by MCP STDIO argument scanning via
//! heuristic helpers.

use std::path::{Component, Path, PathBuf};

// ---------------------------------------------------------------------------
// Read deny groups
// ---------------------------------------------------------------------------

/// How a read deny group is enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadTier {
    /// Never readable by the agent: file tools refuse, the jail denies.
    Deny,
    /// Readable after the user approves the specific call: the permission
    /// gate forces a prompt for file-tool reads, the jail denies (and
    /// escalates). The file tools themselves allow it, because the gate
    /// has already asked; `yolo` and gate-less hosts read silently.
    Ask,
}

/// Platforms a deny group applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Any,
    MacOs,
    Linux,
}

impl Platform {
    /// Whether the group applies on the platform this binary runs on.
    pub fn current(self) -> bool {
        match self {
            Platform::Any => true,
            Platform::MacOs => cfg!(target_os = "macos"),
            Platform::Linux => cfg!(target_os = "linux"),
        }
    }
}

/// One entry of a read deny group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyEntry {
    /// Home-relative path: the file, or the directory and everything in it.
    Home(&'static str),
    /// Absolute path: the file, or the directory and everything in it.
    Absolute(&'static str),
    /// A file with exactly this name, anywhere.
    Name(&'static str),
    /// Files whose name starts with `prefix`, anywhere, except the names
    /// in `except` (templates such as `.env.example`).
    NamePrefix {
        prefix: &'static str,
        except: &'static [&'static str],
    },
}

/// A named set of read denies with one rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenyGroup {
    /// Stable identifier (tests, logs).
    pub id: &'static str,
    /// Human-readable description, used in denial messages.
    pub label: &'static str,
    pub tier: ReadTier,
    pub platform: Platform,
    pub entries: &'static [DenyEntry],
}

use DenyEntry::{Absolute, Home, Name, NamePrefix};

/// Every read deny group. The file tools (`check_read`), the permission
/// gate (ask tier), and the OS jail (`sandbox`) all consume this table.
pub const READ_DENY_GROUPS: &[DenyGroup] = &[
    DenyGroup {
        id: "credentials",
        label: "cloud and tool credentials",
        tier: ReadTier::Deny,
        platform: Platform::Any,
        entries: &[
            Home(".ssh"),
            Home(".gnupg"),
            Home(".aws"),
            Home(".azure"),
            Home(".config/gcloud"),
            Home(".gcloud"),
            Home(".kube"),
            Home(".docker"),
            Home(".git-credentials"),
            Home(".netrc"),
            Home(".npmrc"),
            Home(".pypirc"),
            Home(".bunfig.toml"),
            Home(".config/bun/bunfig.toml"),
            Home(".vault-token"),
            Home(".terraform.d"),
            Home(".config/op"),
            Home(".credentials"),
            Home(".secrets"),
            Home(".keys"),
            Home(".pki"),
            Home(".config/gh"),
            Home(".cargo/credentials.toml"),
            Home(".cargo/credentials"),
            Home(".gem/credentials"),
            Home(".pgpass"),
            Home(".my.cnf"),
            Home(".config/rclone/rclone.conf"),
        ],
    },
    DenyGroup {
        id: "password_stores",
        label: "password manager or keychain",
        tier: ReadTier::Deny,
        platform: Platform::Any,
        entries: &[Home(".password-store"), Home(".1password"), Home(".op")],
    },
    DenyGroup {
        id: "password_stores_macos",
        label: "password manager or keychain",
        tier: ReadTier::Deny,
        platform: Platform::MacOs,
        entries: &[
            Home("Library/Keychains"),
            Absolute("/Library/Keychains"),
            Home("Library/Group Containers/2BUA8C4S2C.com.1password"),
            Home("Library/Application Support/1Password"),
            Home("Library/Containers/com.1password.1password"),
        ],
    },
    DenyGroup {
        id: "password_stores_linux",
        label: "password manager or keychain",
        tier: ReadTier::Deny,
        platform: Platform::Linux,
        entries: &[Home(".local/share/keyrings")],
    },
    DenyGroup {
        id: "shell_configs",
        label: "shell configuration",
        tier: ReadTier::Ask,
        platform: Platform::Any,
        entries: &[
            Home(".zshrc"),
            Home(".zprofile"),
            Home(".zshenv"),
            Home(".zlogin"),
            Home(".zlogout"),
            Home(".bashrc"),
            Home(".bash_profile"),
            Home(".bash_login"),
            Home(".bash_logout"),
            Home(".profile"),
            // `~/.env` and `~/.envrc` are deny-tier through the
            // `project_secrets` file-name rules.
            Home(".config/fish"),
        ],
    },
    DenyGroup {
        id: "shell_history",
        label: "shell or REPL history",
        tier: ReadTier::Deny,
        platform: Platform::Any,
        entries: &[
            Home(".bash_history"),
            Home(".zsh_history"),
            Home(".history"),
            Home(".python_history"),
            Home(".node_repl_history"),
            Home(".psql_history"),
            Home(".mysql_history"),
            Home(".sqlite_history"),
            Home(".local/share/fish/fish_history"),
        ],
    },
    DenyGroup {
        id: "browser_data_macos",
        label: "browser data",
        tier: ReadTier::Deny,
        platform: Platform::MacOs,
        entries: &[
            Home("Library/Application Support/Google/Chrome"),
            Home("Library/Application Support/Chromium"),
            Home("Library/Application Support/Firefox"),
            Home("Library/Application Support/Microsoft Edge"),
            Home("Library/Application Support/Arc"),
            Home("Library/Application Support/BraveSoftware"),
            Home("Library/Application Support/Vivaldi"),
            Home("Library/Application Support/com.operasoftware.Opera"),
            Home("Library/Safari"),
        ],
    },
    DenyGroup {
        id: "browser_data_linux",
        label: "browser data",
        tier: ReadTier::Deny,
        platform: Platform::Linux,
        entries: &[
            Home(".config/google-chrome"),
            Home(".config/chromium"),
            Home(".mozilla/firefox"),
            Home(".config/microsoft-edge"),
            Home(".config/BraveSoftware"),
            Home(".config/vivaldi"),
            Home(".config/opera"),
        ],
    },
    DenyGroup {
        id: "personal_data_macos",
        label: "personal data (messages, mail, device backups)",
        tier: ReadTier::Deny,
        platform: Platform::MacOs,
        entries: &[
            Home("Library/Messages"),
            Home("Library/Mail"),
            Home("Library/Cookies"),
            Home("Library/Containers/com.apple.Safari"),
            Home("Library/Application Support/MobileSync"),
        ],
    },
    DenyGroup {
        id: "agent_credentials",
        label: "AI tool credentials",
        tier: ReadTier::Deny,
        platform: Platform::Any,
        entries: &[
            Home(".codex/auth.json"),
            Home(".claude/.credentials.json"),
            Home(".config/github-copilot"),
        ],
    },
    DenyGroup {
        id: "project_secrets",
        label: "secrets file",
        tier: ReadTier::Deny,
        platform: Platform::Any,
        entries: &[
            Name(".env"),
            NamePrefix {
                prefix: ".env.",
                except: &[".env.example", ".env.sample", ".env.template", ".env.dist"],
            },
            Name(".envrc"),
            Name(".netrc"),
            Name(".git-credentials"),
            Name(".pgpass"),
            Name("credentials.json"),
            Name("service-account.json"),
            Name("service_account.json"),
            Name("secrets.yaml"),
            Name("secrets.yml"),
        ],
    },
    DenyGroup {
        id: "system_secrets",
        label: "sensitive system file",
        tier: ReadTier::Deny,
        platform: Platform::Any,
        entries: &[
            Absolute("/etc/shadow"),
            Absolute("/etc/gshadow"),
            Absolute("/etc/master.passwd"),
        ],
    },
];

/// Label of the runtime-registered private directories (see
/// [`register_private_dir`]).
pub const PRIVATE_DIR_LABEL: &str = "ChatShell app data";

/// Directories registered at runtime as never readable (ChatShell's own
/// data directory: other conversations, settings).
static PRIVATE_DIRS: std::sync::RwLock<Vec<PathBuf>> = std::sync::RwLock::new(Vec::new());

/// Register a directory the agent must never read (deny tier). Hosts call
/// this once at startup with their data directory.
pub fn register_private_dir(dir: &Path) {
    let mut dirs = PRIVATE_DIRS.write().unwrap_or_else(|e| e.into_inner());
    let dir = normalize_path(dir);
    if !dirs.contains(&dir) {
        dirs.push(dir);
    }
}

/// The registered private directories.
pub fn private_dirs() -> Vec<PathBuf> {
    PRIVATE_DIRS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// The deny groups that apply on this platform.
pub fn active_read_groups() -> impl Iterator<Item = &'static DenyGroup> {
    READ_DENY_GROUPS.iter().filter(|g| g.platform.current())
}

/// A read that hit a deny group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlock {
    pub tier: ReadTier,
    /// The group label (or [`PRIVATE_DIR_LABEL`]).
    pub label: &'static str,
    /// The matched entry, for messages (`~/.aws`, `.env.test`, ...).
    pub matched: String,
}

impl ReadBlock {
    /// The error message file tools return for a deny-tier hit.
    pub fn message(&self) -> String {
        format!("Read blocked: {} ({})", self.label, self.matched)
    }
}

/// Classify a read of `path` against the deny groups and the registered
/// private directories. Both the lexical path and its symlink-resolved
/// form are checked (a project symlink to `~/.zshrc` is caught), and the
/// strictest hit wins: deny over ask.
pub fn classify_read(path: &Path) -> Option<ReadBlock> {
    ReadClassifier::new().classify(path)
}

/// [`classify_read`] with the home directory, absolute entries, and
/// private directories resolved once, for classifying many paths.
pub struct ReadClassifier {
    homes: Vec<PathBuf>,
    /// Absolute entries with their symlink-resolved spelling.
    absolutes: Vec<(&'static str, Option<PathBuf>)>,
    private: Vec<PathBuf>,
}

impl Default for ReadClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadClassifier {
    pub fn new() -> Self {
        Self::with(home_dir().as_deref(), private_dirs())
    }

    fn with(home: Option<&Path>, private: Vec<PathBuf>) -> Self {
        let homes = home
            .into_iter()
            .flat_map(|home| {
                let canonical = std::fs::canonicalize(home).ok();
                std::iter::once(home.to_path_buf()).chain(canonical)
            })
            .collect();
        let absolutes = active_read_groups()
            .flat_map(|group| group.entries)
            .filter_map(|entry| match entry {
                Absolute(abs) => Some((*abs, std::fs::canonicalize(abs).ok())),
                _ => None,
            })
            .collect();
        Self {
            homes,
            absolutes,
            private,
        }
    }

    /// Classify `path` and its symlink-resolved form.
    pub fn classify(&self, path: &Path) -> Option<ReadBlock> {
        let lexical = normalize_path(path);
        let resolved = resolve_existing(&lexical).filter(|r| *r != lexical);
        let first = self.classify_lexical(&lexical);
        let second = resolved.and_then(|r| self.classify_lexical(&r));
        stricter(first, second)
    }

    /// Classify `path` as spelled, without resolving symlinks (for paths
    /// produced by a walk that does not follow links, under a root that
    /// was itself classified with [`ReadClassifier::classify`]).
    pub fn classify_lexical(&self, path: &Path) -> Option<ReadBlock> {
        let mut best = None;
        for dir in &self.private {
            if path.starts_with(dir) {
                best = stricter(
                    best,
                    Some(ReadBlock {
                        tier: ReadTier::Deny,
                        label: PRIVATE_DIR_LABEL,
                        matched: dir.display().to_string(),
                    }),
                );
            }
        }
        for group in active_read_groups() {
            for entry in group.entries {
                if let Some(matched) = self.entry_matches(entry, path) {
                    best = stricter(
                        best,
                        Some(ReadBlock {
                            tier: group.tier,
                            label: group.label,
                            matched,
                        }),
                    );
                }
            }
        }
        best
    }

    /// The matched form of `entry` if it covers `path`. Path entries match
    /// whole components (`~/.aws` covers `~/.aws/sso/cache` but not
    /// `~/.aws-vault`).
    fn entry_matches(&self, entry: &DenyEntry, path: &Path) -> Option<String> {
        let name = || path.file_name().and_then(|n| n.to_str());
        match entry {
            Home(rel) => self
                .homes
                .iter()
                .any(|home| path.starts_with(home.join(rel)))
                .then(|| format!("~/{rel}")),
            Absolute(abs) => {
                let resolved = self
                    .absolutes
                    .iter()
                    .find(|(a, _)| a == abs)
                    .and_then(|(_, r)| r.as_deref());
                (path.starts_with(abs) || resolved.is_some_and(|r| path.starts_with(r)))
                    .then(|| abs.to_string())
            }
            Name(blocked) => (name()? == *blocked).then(|| blocked.to_string()),
            NamePrefix { prefix, except } => {
                let name = name()?;
                (name.starts_with(prefix) && name.len() > prefix.len() && !except.contains(&name))
                    .then(|| name.to_string())
            }
        }
    }
}

/// The stricter of two optional blocks: any block over none, deny over
/// ask, the first on a tie.
fn stricter(a: Option<ReadBlock>, b: Option<ReadBlock>) -> Option<ReadBlock> {
    match (a, b) {
        (Some(a), Some(b)) if a.tier == ReadTier::Ask && b.tier == ReadTier::Deny => Some(b),
        (Some(a), _) => Some(a),
        (None, b) => b,
    }
}

/// `path` with symlinks resolved through its deepest existing ancestor
/// (the path itself may not exist). `None` when nothing resolves.
fn resolve_existing(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(mut resolved) = std::fs::canonicalize(&existing) {
            for part in tail.iter().rev() {
                resolved.push(part);
            }
            return Some(resolved);
        }
        tail.push(existing.file_name()?.to_os_string());
        if !existing.pop() {
            return None;
        }
    }
}

// ---------------------------------------------------------------------------
// Write blocklist
// ---------------------------------------------------------------------------

/// Relative path prefixes inside a repository that must never be written.
const WRITE_BLOCKED_REPO_PREFIXES: &[&str] = &[
    ".git/hooks/",
    ".git/hooks",
    ".git/config",
    ".git/objects/",
    ".git/objects",
    ".git/refs/",
    ".git/refs",
    ".git/HEAD",
    ".git/index",
];

/// Home-relative paths that must never be written.
pub(crate) const WRITE_BLOCKED_HOME: &[&str] = &[
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".bash_login",
    ".zshenv",
    ".zlogin",
    ".zprofile",
    ".ssh/",
    ".gnupg/",
    ".gitconfig",
];

/// Absolute path prefixes that must never be written.
const WRITE_BLOCKED_ABSOLUTE: &[&str] =
    &["/etc/", "/usr/", "/bin/", "/sbin/", "/System/", "/Library/"];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check whether a file tool may read `path`.
///
/// Returns `Err(reason)` for a deny-tier hit (see [`classify_read`]).
/// Ask-tier hits pass: the permission gate forces a prompt for them before
/// the tool runs. No directory boundary is enforced for reads — the LLM
/// needs to read broadly for context.
pub fn check_read(path: &Path, _project_root: Option<&Path>) -> Result<(), String> {
    match classify_read(path) {
        Some(block) if block.tier == ReadTier::Deny => Err(block.message()),
        // Ask tier: the permission gate prompted for this call already.
        _ => Ok(()),
    }
}

/// Check whether writing to `path` is allowed.
///
/// Two layers:
/// 1. Absolute blocklist — certain paths are never writable.
/// 2. Project boundary — when `project_root` is `Some`, writes must land
///    inside that directory tree.
pub fn check_write(path: &Path, project_root: Option<&Path>) -> Result<(), String> {
    let normalized = normalize_path(path);
    let path_str = normalized.to_string_lossy();

    // Layer 1a: absolute path blocklist
    for blocked in WRITE_BLOCKED_ABSOLUTE {
        if path_str.starts_with(blocked) {
            return Err(format!(
                "Write blocked: protected system directory ({})",
                path.display()
            ));
        }
    }

    // Layer 1b: home-relative blocklist
    if let Some(home) = home_dir() {
        let home_str = home.to_string_lossy();
        if let Some(rel) = path_str.strip_prefix(home_str.as_ref()) {
            let rel = rel.trim_start_matches('/');
            for blocked in WRITE_BLOCKED_HOME {
                if rel == blocked.trim_end_matches('/') || rel.starts_with(blocked) {
                    return Err(format!("Write blocked: protected user file (~/{blocked})"));
                }
            }
        }
    }

    // Layer 1c: repo-internal paths (checked relative to the path itself)
    for blocked in WRITE_BLOCKED_REPO_PREFIXES {
        if contains_component_prefix(&path_str, blocked) {
            return Err(format!(
                "Write blocked: protected repository path ({blocked})"
            ));
        }
    }

    // Layer 2: project boundary
    if let Some(root) = project_root {
        let root_normalized = normalize_path(root);
        if !normalized.starts_with(&root_normalized) {
            return Err(format!(
                "Write blocked: outside project directory ({} is not under {})",
                path.display(),
                root.display()
            ));
        }
    }

    Ok(())
}

/// Heuristic: does the string look like a filesystem path?
///
/// Used by MCP argument scanning to decide whether to run path checks on a
/// string value.
pub fn looks_like_path(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed.len() > 4096 {
        return false;
    }
    trimmed.starts_with('/')
        || trimmed.starts_with("~/")
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
}

/// Check a raw string against both read and write blocklists.
///
/// Used by MCP STDIO argument scanning where we don't know whether the string
/// will be used for reading or writing.  If it matches *either* blocklist, the
/// call is rejected.
pub fn check_blocked_path(s: &str) -> Result<(), String> {
    if !looks_like_path(s) {
        return Ok(());
    }

    let path = expand_tilde(s);
    check_read(&path, None)?;
    check_write(&path, None)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(s)
}

/// Normalize a path by resolving `.` and `..` components lexically (without
/// touching the filesystem).  This prevents `../../etc/shadow` bypasses
/// while still working for paths that don't exist yet (write/create).
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// Check if a path string contains a component prefix like `.git/hooks/`
/// anywhere in its path (not just at the start).
fn contains_component_prefix(path_str: &str, prefix: &str) -> bool {
    // Check at end of path
    if path_str.ends_with(prefix.trim_end_matches('/')) {
        // Make sure it's a component boundary
        let before = &path_str[..path_str.len() - prefix.trim_end_matches('/').len()];
        if before.is_empty() || before.ends_with('/') {
            return true;
        }
    }

    // Check with trailing slash (prefix is a directory)
    let search = if prefix.ends_with('/') {
        format!("/{prefix}")
    } else {
        format!("/{prefix}/")
    };
    if path_str.contains(&search) {
        return true;
    }

    // Check at start of path
    path_str.starts_with(prefix) || path_str.starts_with(&format!("/{prefix}"))
}

fn home_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- check_read -------------------------------------------------------

    #[test]
    fn read_blocks_etc_shadow() {
        assert!(check_read(Path::new("/etc/shadow"), None).is_err());
        assert!(check_read(Path::new("/etc/gshadow"), None).is_err());
        assert!(check_read(Path::new("/etc/master.passwd"), None).is_err());
    }

    #[test]
    fn read_blocks_ssh_keys() {
        let home = home_dir().unwrap();
        assert!(check_read(&home.join(".ssh/id_rsa"), None).is_err());
        assert!(check_read(&home.join(".ssh/id_ed25519"), None).is_err());
        assert!(check_read(&home.join(".ssh/id_ecdsa"), None).is_err());
    }

    #[test]
    fn read_blocks_sensitive_filenames() {
        assert!(check_read(Path::new("/some/project/.env"), None).is_err());
        assert!(check_read(Path::new("/app/.env.local"), None).is_err());
        assert!(check_read(Path::new("/app/.env.production"), None).is_err());
        assert!(check_read(Path::new("/app/credentials.json"), None).is_err());
        assert!(check_read(Path::new("/app/secrets.yaml"), None).is_err());
    }

    #[test]
    fn read_blocks_cloud_credentials() {
        let home = home_dir().unwrap();
        assert!(check_read(&home.join(".aws/credentials"), None).is_err());
        assert!(check_read(&home.join(".docker/config.json"), None).is_err());
        assert!(check_read(&home.join(".kube/config"), None).is_err());
        assert!(check_read(&home.join(".npmrc"), None).is_err());
    }

    #[test]
    fn read_allows_normal_files() {
        assert!(check_read(Path::new("/tmp/test.rs"), None).is_ok());
        assert!(check_read(Path::new("/usr/include/stdio.h"), None).is_ok());
        assert!(check_read(Path::new("/home/user/project/src/main.rs"), None).is_ok());
    }

    #[test]
    fn read_denies_the_whole_ssh_directory() {
        let home = home_dir().unwrap();
        assert!(check_read(&home.join(".ssh/config"), None).is_err());
        assert!(check_read(&home.join(".ssh"), None).is_err());
    }

    #[test]
    fn home_entries_match_whole_components() {
        let home = home_dir().unwrap();
        assert!(check_read(&home.join(".aws/sso/cache/token.json"), None).is_err());
        assert!(check_read(&home.join(".aws-vault/notes.txt"), None).is_ok());
        assert!(check_read(&home.join(".npmrc-backup"), None).is_ok());
        assert!(check_read(&home.join(".config/gcloud/credentials.db"), None).is_err());
        assert!(check_read(&home.join(".config/gcloudish"), None).is_ok());
    }

    #[test]
    fn new_groups_are_denied() {
        let home = home_dir().unwrap();
        for rel in [
            ".git-credentials",
            ".netrc",
            ".zsh_history",
            ".config/gh/hosts.yml",
            ".cargo/credentials.toml",
            ".password-store/x.gpg",
            ".codex/auth.json",
            ".env",
        ] {
            let block = classify_read(&home.join(rel)).unwrap_or_else(|| panic!("{rel}"));
            assert_eq!(block.tier, ReadTier::Deny, "{rel}");
        }
        if cfg!(target_os = "macos") {
            assert!(check_read(&home.join("Library/Keychains/login.keychain-db"), None).is_err());
            assert!(check_read(&home.join("Library/Messages/chat.db"), None).is_err());
            assert!(check_read(&home.join("Library/Safari/History.db"), None).is_err());
        }
        if cfg!(target_os = "linux") {
            assert!(check_read(&home.join(".mozilla/firefox/x/cookies.sqlite"), None).is_err());
        }
    }

    #[test]
    fn shell_configs_are_ask_tier() {
        let home = home_dir().unwrap();
        let block = classify_read(&home.join(".zshrc")).unwrap();
        assert_eq!(block.tier, ReadTier::Ask);
        assert_eq!(block.label, "shell configuration");
        // The tools allow it: the gate prompts for ask-tier reads.
        assert!(check_read(&home.join(".zshrc"), None).is_ok());
        assert!(check_read(&home.join(".config/fish/config.fish"), None).is_ok());
        assert!(classify_read(&home.join(".config/fish/config.fish")).is_some());
    }

    #[test]
    fn env_variants_are_denied_but_templates_are_not() {
        for name in [
            ".env.test",
            ".env.prod",
            ".env.ci",
            ".env.local",
            ".envrc",
            ".pgpass",
        ] {
            assert!(
                check_read(&Path::new("/app").join(name), None).is_err(),
                "{name}"
            );
        }
        for name in [
            ".env.example",
            ".env.sample",
            ".env.template",
            ".env.dist",
            ".envoy",
            "env.ts",
        ] {
            assert!(
                check_read(&Path::new("/app").join(name), None).is_ok(),
                "{name}"
            );
        }
    }

    #[test]
    fn symlinks_to_blocked_files_are_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".aws")).unwrap();
        std::fs::write(home.join(".aws/credentials"), "secret").unwrap();
        std::fs::write(home.join(".zshrc"), "export KEY=1").unwrap();
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(home.join(".aws/credentials"), project.join("notes"))
                .unwrap();
            std::os::unix::fs::symlink(home.join(".zshrc"), project.join("rc")).unwrap();
            std::os::unix::fs::symlink(home.join(".aws"), project.join("cloud")).unwrap();
            let policy = ReadClassifier::with(Some(&home), Vec::new());
            assert_eq!(
                policy.classify(&project.join("notes")).unwrap().tier,
                ReadTier::Deny
            );
            assert_eq!(
                policy.classify(&project.join("rc")).unwrap().tier,
                ReadTier::Ask
            );
            assert_eq!(
                policy
                    .classify(&project.join("cloud/credentials"))
                    .unwrap()
                    .tier,
                ReadTier::Deny
            );
            assert!(policy.classify_lexical(&project.join("notes")).is_none());
        }
    }

    #[test]
    fn private_dirs_are_denied() {
        let dir = tempfile::tempdir().unwrap();
        let private = dir.path().join("app-data");
        let policy = ReadClassifier::with(None, vec![private.clone()]);
        let block = policy.classify(&private.join("data.db")).unwrap();
        assert_eq!(block.tier, ReadTier::Deny);
        assert_eq!(block.label, PRIVATE_DIR_LABEL);
        assert!(policy.classify(&dir.path().join("app-data-2/x")).is_none());
    }

    #[test]
    fn deny_wins_over_ask() {
        let a = Some(ReadBlock {
            tier: ReadTier::Ask,
            label: "a",
            matched: "a".into(),
        });
        let d = Some(ReadBlock {
            tier: ReadTier::Deny,
            label: "d",
            matched: "d".into(),
        });
        assert_eq!(stricter(a.clone(), d.clone()).unwrap().label, "d");
        assert_eq!(stricter(d.clone(), a.clone()).unwrap().label, "d");
        assert_eq!(stricter(None, a.clone()).unwrap().label, "a");
    }

    #[test]
    fn every_group_has_entries_and_a_unique_id() {
        let mut ids = std::collections::HashSet::new();
        for group in READ_DENY_GROUPS {
            assert!(!group.entries.is_empty(), "{}", group.id);
            assert!(ids.insert(group.id), "duplicate id {}", group.id);
        }
    }

    // ---- check_write ------------------------------------------------------

    #[test]
    fn write_blocks_system_dirs() {
        assert!(check_write(Path::new("/etc/nginx/nginx.conf"), None).is_err());
        assert!(check_write(Path::new("/usr/local/bin/app"), None).is_err());
        assert!(check_write(Path::new("/bin/sh"), None).is_err());
        assert!(check_write(Path::new("/sbin/init"), None).is_err());
    }

    #[test]
    fn write_blocks_shell_configs() {
        let home = home_dir().unwrap();
        assert!(check_write(&home.join(".bashrc"), None).is_err());
        assert!(check_write(&home.join(".zshrc"), None).is_err());
        assert!(check_write(&home.join(".profile"), None).is_err());
        assert!(check_write(&home.join(".bash_profile"), None).is_err());
        assert!(check_write(&home.join(".zshenv"), None).is_err());
    }

    #[test]
    fn write_blocks_git_internals() {
        assert!(check_write(Path::new("/project/.git/hooks/pre-commit"), None).is_err());
        assert!(check_write(Path::new("/project/.git/config"), None).is_err());
        assert!(check_write(Path::new("/project/.git/objects/ab/cdef"), None).is_err());
        assert!(check_write(Path::new("/project/.git/refs/heads/main"), None).is_err());
        assert!(check_write(Path::new("/project/.git/HEAD"), None).is_err());
    }

    #[test]
    fn write_blocks_ssh_dir() {
        let home = home_dir().unwrap();
        assert!(check_write(&home.join(".ssh/authorized_keys"), None).is_err());
        assert!(check_write(&home.join(".ssh/id_rsa"), None).is_err());
    }

    #[test]
    fn write_allows_within_project() {
        let root = Path::new("/home/user/project");
        assert!(check_write(Path::new("/home/user/project/src/main.rs"), Some(root)).is_ok());
        assert!(check_write(Path::new("/home/user/project/new-file.txt"), Some(root)).is_ok());
    }

    #[test]
    fn write_blocks_outside_project() {
        let root = Path::new("/home/user/project");
        assert!(check_write(Path::new("/home/user/other/file.txt"), Some(root)).is_err());
        assert!(check_write(Path::new("/tmp/file.txt"), Some(root)).is_err());
    }

    #[test]
    fn write_traversal_blocked() {
        let root = Path::new("/home/user/project");
        assert!(check_write(Path::new("/home/user/project/../../etc/passwd"), Some(root)).is_err());
    }

    #[test]
    fn write_no_project_root_only_blocklist() {
        assert!(check_write(Path::new("/tmp/safe-file.txt"), None).is_ok());
        assert!(check_write(Path::new("/home/user/project/file.rs"), None).is_ok());
    }

    // ---- looks_like_path --------------------------------------------------

    #[test]
    fn looks_like_path_positive() {
        assert!(looks_like_path("/etc/shadow"));
        assert!(looks_like_path("~/Documents/file.txt"));
        assert!(looks_like_path("./relative/path"));
        assert!(looks_like_path("../parent/path"));
    }

    #[test]
    fn looks_like_path_negative() {
        assert!(!looks_like_path("hello world"));
        assert!(!looks_like_path("npm install"));
        assert!(!looks_like_path("SELECT * FROM users"));
        assert!(!looks_like_path(""));
    }

    // ---- check_blocked_path -----------------------------------------------

    #[test]
    fn blocked_path_catches_ssh_key() {
        let home = home_dir().unwrap();
        let path_str = format!("{}{}", home.display(), "/.ssh/id_rsa");
        assert!(check_blocked_path(&path_str).is_err());
    }

    #[test]
    fn blocked_path_catches_tilde_ssh() {
        assert!(check_blocked_path("~/.ssh/id_rsa").is_err());
    }

    #[test]
    fn blocked_path_catches_git_hooks() {
        assert!(check_blocked_path("/project/.git/hooks/pre-commit").is_err());
    }

    #[test]
    fn blocked_path_allows_normal() {
        assert!(check_blocked_path("/tmp/test.txt").is_ok());
        assert!(check_blocked_path("./src/main.rs").is_ok());
    }

    #[test]
    fn blocked_path_skips_non_paths() {
        assert!(check_blocked_path("hello world").is_ok());
        assert!(check_blocked_path("SELECT * FROM users").is_ok());
    }

    // ---- normalize_path ---------------------------------------------------

    #[test]
    fn normalize_resolves_dotdot() {
        assert_eq!(
            normalize_path(Path::new("/home/user/project/../../etc/shadow")),
            PathBuf::from("/home/etc/shadow")
        );
    }

    #[test]
    fn normalize_resolves_dot() {
        assert_eq!(
            normalize_path(Path::new("/home/./user/./project")),
            PathBuf::from("/home/user/project")
        );
    }

    #[test]
    fn normalize_preserves_clean_path() {
        assert_eq!(
            normalize_path(Path::new("/home/user/project/src/main.rs")),
            PathBuf::from("/home/user/project/src/main.rs")
        );
    }

    // ---- macOS-specific ---------------------------------------------------

    #[test]
    fn write_blocks_macos_system_dirs() {
        assert!(check_write(Path::new("/System/Library/something"), None).is_err());
        assert!(check_write(Path::new("/Library/LaunchDaemons/com.evil.plist"), None).is_err());
    }
}
