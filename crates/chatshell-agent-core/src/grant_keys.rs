//! Grant-key derivation for the approval gate.
//!
//! A grant key names exactly what an approval covers. Keys are plain
//! strings so session grants (RAM) and remembered grants (per-device
//! SQLite) share one representation:
//!
//! - `bash:<base>` or `bash:<base> <subcommand>` — one per compound segment
//! - `tool:<name>` — a whole tool (session scope only for file tools)
//! - `mcp:<server>/<tool>` — one MCP tool on one server
//! - `file:<tier>\0<path>` / `dir:<tier>\0<path>` — tier-pinned file grants
//!
//! Bash keys are narrowed to base + first subcommand for tools whose
//! subcommand space ranges from harmless to destructive (`git status` vs
//! `git push --force`). Leading global options are skipped; an option the
//! table does not know makes the segment unkeyable, because an unknown
//! option may consume the next word and a wrong guess would let one
//! approved subcommand cover another.

use crate::bash_security;
use std::path::{Component, Path, PathBuf};

/// How the subcommand of a listed base is found.
enum SubcommandStyle {
    /// First non-option word after the global options.
    Word {
        /// Global options that consume the following word.
        value_options: &'static [&'static str],
        /// Global options that take no value.
        flag_options: &'static [&'static str],
        /// Leading `+<toolchain>`-style words to skip (`cargo +nightly`).
        skip_plus_words: bool,
    },
    /// The operation flag is the subcommand (`pacman -Syu` -> `-S`).
    PacmanOperation,
}

/// Bases keyed with their first subcommand. Append-only.
const SUBCOMMAND_BASES: &[(&str, SubcommandStyle)] = &[
    (
        "git",
        SubcommandStyle::Word {
            value_options: &[
                "-C",
                "-c",
                "--git-dir",
                "--work-tree",
                "--namespace",
                "--config-env",
                "--super-prefix",
            ],
            flag_options: &[
                "-p",
                "--paginate",
                "-P",
                "--no-pager",
                "--bare",
                "--no-replace-objects",
                "--literal-pathspecs",
                "--glob-pathspecs",
                "--noglob-pathspecs",
                "--icase-pathspecs",
                "--no-optional-locks",
            ],
            skip_plus_words: false,
        },
    ),
    (
        "npm",
        SubcommandStyle::Word {
            value_options: &[
                "--prefix",
                "-w",
                "--workspace",
                "--userconfig",
                "--registry",
                "--cache",
                "--loglevel",
            ],
            flag_options: &[
                "-g",
                "--global",
                "-s",
                "--silent",
                "-q",
                "--quiet",
                "--verbose",
                "-ws",
                "--workspaces",
                "--no-color",
            ],
            skip_plus_words: false,
        },
    ),
    (
        "pnpm",
        SubcommandStyle::Word {
            value_options: &["-C", "--dir", "--filter", "-F", "--workspace-dir"],
            flag_options: &[
                "-r",
                "--recursive",
                "-w",
                "--workspace-root",
                "-s",
                "--silent",
                "--color",
                "--no-color",
            ],
            skip_plus_words: false,
        },
    ),
    (
        "yarn",
        SubcommandStyle::Word {
            value_options: &["--cwd"],
            flag_options: &["-s", "--silent", "--verbose", "--no-progress"],
            skip_plus_words: false,
        },
    ),
    (
        "cargo",
        SubcommandStyle::Word {
            value_options: &["-C", "--config", "-Z", "--color"],
            flag_options: &[
                "-v",
                "-vv",
                "--verbose",
                "-q",
                "--quiet",
                "--frozen",
                "--locked",
                "--offline",
            ],
            skip_plus_words: true,
        },
    ),
    (
        "pip",
        SubcommandStyle::Word {
            value_options: PIP_VALUE_OPTIONS,
            flag_options: PIP_FLAG_OPTIONS,
            skip_plus_words: false,
        },
    ),
    (
        "pip3",
        SubcommandStyle::Word {
            value_options: PIP_VALUE_OPTIONS,
            flag_options: PIP_FLAG_OPTIONS,
            skip_plus_words: false,
        },
    ),
    (
        "uv",
        SubcommandStyle::Word {
            value_options: &[
                "--directory",
                "--project",
                "--cache-dir",
                "--config-file",
                "--color",
            ],
            flag_options: &[
                "-q",
                "--quiet",
                "-v",
                "--verbose",
                "--offline",
                "--no-cache",
                "-n",
                "--no-progress",
                "--native-tls",
            ],
            skip_plus_words: false,
        },
    ),
    (
        "poetry",
        SubcommandStyle::Word {
            value_options: &["-C", "--directory", "-P", "--project"],
            flag_options: &[
                "-q",
                "--quiet",
                "-v",
                "-vv",
                "-vvv",
                "--verbose",
                "-n",
                "--no-interaction",
                "--ansi",
                "--no-ansi",
                "--no-plugins",
                "--no-cache",
            ],
            skip_plus_words: false,
        },
    ),
    (
        "docker",
        SubcommandStyle::Word {
            value_options: &[
                "--context",
                "-c",
                "-H",
                "--host",
                "--config",
                "-l",
                "--log-level",
            ],
            flag_options: &["-D", "--debug", "--tls", "--tlsverify"],
            skip_plus_words: false,
        },
    ),
    (
        "podman",
        SubcommandStyle::Word {
            value_options: &[
                "--connection",
                "-c",
                "--url",
                "--root",
                "--runroot",
                "--log-level",
            ],
            flag_options: &["--remote", "-r"],
            skip_plus_words: false,
        },
    ),
    (
        "kubectl",
        SubcommandStyle::Word {
            value_options: &[
                "-n",
                "--namespace",
                "--context",
                "--kubeconfig",
                "--cluster",
                "--user",
                "-s",
                "--server",
                "--as",
                "--token",
            ],
            flag_options: &["--insecure-skip-tls-verify"],
            skip_plus_words: false,
        },
    ),
    (
        "systemctl",
        SubcommandStyle::Word {
            value_options: &[
                "-H",
                "--host",
                "-M",
                "--machine",
                "-t",
                "--type",
                "--state",
                "-p",
                "--property",
            ],
            flag_options: &[
                "--user",
                "--system",
                "--global",
                "--no-pager",
                "-q",
                "--quiet",
                "--no-block",
                "-a",
                "--all",
                "-l",
                "--full",
            ],
            skip_plus_words: false,
        },
    ),
    (
        "brew",
        SubcommandStyle::Word {
            value_options: &[],
            flag_options: &["-v", "--verbose", "-d", "--debug", "-q", "--quiet"],
            skip_plus_words: false,
        },
    ),
    (
        "apt",
        SubcommandStyle::Word {
            value_options: APT_VALUE_OPTIONS,
            flag_options: APT_FLAG_OPTIONS,
            skip_plus_words: false,
        },
    ),
    (
        "apt-get",
        SubcommandStyle::Word {
            value_options: APT_VALUE_OPTIONS,
            flag_options: APT_FLAG_OPTIONS,
            skip_plus_words: false,
        },
    ),
    (
        "dnf",
        SubcommandStyle::Word {
            value_options: &[
                "-c",
                "--config",
                "--installroot",
                "--releasever",
                "--repo",
                "--enablerepo",
                "--disablerepo",
                "-x",
                "--exclude",
            ],
            flag_options: &[
                "-y",
                "--assumeyes",
                "-q",
                "--quiet",
                "-v",
                "--verbose",
                "--refresh",
                "-C",
                "--cacheonly",
            ],
            skip_plus_words: false,
        },
    ),
    ("pacman", SubcommandStyle::PacmanOperation),
];

const PIP_VALUE_OPTIONS: &[&str] = &["--python", "--log", "--proxy", "--cache-dir", "--cert"];
const PIP_FLAG_OPTIONS: &[&str] = &[
    "-v",
    "--verbose",
    "-q",
    "--quiet",
    "--isolated",
    "--no-cache-dir",
    "--disable-pip-version-check",
    "--require-virtualenv",
    "--no-color",
];
const APT_VALUE_OPTIONS: &[&str] = &["-o", "-c", "-t"];
const APT_FLAG_OPTIONS: &[&str] = &[
    "-y",
    "--yes",
    "--assume-yes",
    "-q",
    "-qq",
    "--quiet",
    "-s",
    "--simulate",
    "--no-install-recommends",
];

/// Words that may precede the real base command without changing it.
fn is_prefix_word(word: &str) -> bool {
    matches!(word, "command" | "builtin" | "noglob" | "nocorrect")
}

/// Bash grant keys for `command` (without the `bash:` prefix): one per
/// compound segment, sorted and deduplicated. `None` when any segment
/// cannot be keyed (empty, or a listed base whose subcommand cannot be
/// identified) — such commands are never grant-coverable.
pub fn bash_grant_keys(command: &str) -> Option<Vec<String>> {
    let segments = bash_security::split_compound_command(command);
    if segments.is_empty() {
        return None;
    }
    let mut keys = Vec::with_capacity(segments.len());
    for segment in segments {
        let stripped = bash_security::strip_safe_wrappers(segment);
        let words: Vec<&str> = stripped
            .split_whitespace()
            .skip_while(|word| is_prefix_word(word))
            .collect();
        keys.push(segment_key(&words)?);
    }
    keys.sort_unstable();
    keys.dedup();
    Some(keys)
}

fn segment_key(words: &[&str]) -> Option<String> {
    let (&base, rest) = words.split_first()?;
    let program = base.rsplit('/').next().unwrap_or(base);
    let Some((_, style)) = SUBCOMMAND_BASES.iter().find(|(name, _)| *name == program) else {
        return Some(base.to_string());
    };
    let subcommand = match style {
        SubcommandStyle::Word {
            value_options,
            flag_options,
            skip_plus_words,
        } => word_subcommand(rest, value_options, flag_options, *skip_plus_words)?,
        SubcommandStyle::PacmanOperation => pacman_operation(rest)?,
    };
    Some(format!("{base} {subcommand}"))
}

fn word_subcommand(
    words: &[&str],
    value_options: &[&str],
    flag_options: &[&str],
    skip_plus_words: bool,
) -> Option<String> {
    let mut iter = words.iter();
    while let Some(&word) = iter.next() {
        if skip_plus_words && word.starts_with('+') {
            continue;
        }
        if !word.starts_with('-') {
            return Some(word.to_string());
        }
        if word == "--" {
            return None;
        }
        // `--opt=value` carries its own value.
        if word.starts_with("--") && word.contains('=') {
            continue;
        }
        if value_options.contains(&word) {
            iter.next()?;
            continue;
        }
        if flag_options.contains(&word) {
            continue;
        }
        // Unknown option shape: it may consume the next word, so the
        // subcommand cannot be identified safely.
        return None;
    }
    None
}

fn pacman_operation(words: &[&str]) -> Option<String> {
    for word in words {
        if let Some(long) = word.strip_prefix("--") {
            let op = match long {
                "sync" => 'S',
                "remove" => 'R',
                "query" => 'Q',
                "upgrade" => 'U',
                "database" => 'D',
                "files" => 'F',
                "deptest" => 'T',
                _ => continue,
            };
            return Some(format!("-{op}"));
        }
        if let Some(short) = word.strip_prefix('-')
            && let Some(op) = short.chars().find(char::is_ascii_uppercase)
        {
            return Some(format!("-{op}"));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// File grants (tier-pinned, one-way coverage)
// ---------------------------------------------------------------------------

/// Tier of a file grant. A write grant covers reads of the same target;
/// a read grant never covers writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileTier {
    Read,
    Write,
}

impl FileTier {
    fn as_str(self) -> &'static str {
        match self {
            FileTier::Read => "read",
            FileTier::Write => "write",
        }
    }

    /// Grant tiers that cover a call of this tier.
    fn covering(self) -> &'static [FileTier] {
        match self {
            FileTier::Read => &[FileTier::Read, FileTier::Write],
            FileTier::Write => &[FileTier::Write],
        }
    }
}

/// Key granting `tier` on exactly one file.
pub fn file_key(tier: FileTier, path: &Path) -> String {
    format!("file:{}\0{}", tier.as_str(), path.display())
}

/// Key granting `tier` on the direct entries of one directory (not on
/// nested subdirectories).
pub fn dir_key(tier: FileTier, dir: &Path) -> String {
    format!("dir:{}\0{}", tier.as_str(), dir.display())
}

/// Keys any of which authorizes a `tier` call on `path` (already
/// resolved with [`grant_path`]): file keys for the path itself and
/// directory keys for its parent, for every covering tier.
pub fn covering_file_keys(tier: FileTier, path: &Path) -> Vec<String> {
    let mut keys = Vec::new();
    for covering in tier.covering() {
        keys.push(file_key(*covering, path));
        if let Some(parent) = path.parent() {
            keys.push(dir_key(*covering, parent));
        }
    }
    keys
}

/// Canonical form of a file-tool path for grant keys: absolute, `.`/`..`
/// folded lexically, then symlinks resolved through the deepest existing
/// ancestor (the target itself may not exist yet). `None` for relative
/// paths: their meaning depends on a cwd the gate does not know, so they
/// are never grant-keyed.
pub fn grant_path(path: &str) -> Option<PathBuf> {
    let path = Path::new(path);
    if !path.is_absolute() {
        return None;
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other.as_os_str()),
        }
    }
    let mut existing = normalized.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(resolved) = std::fs::canonicalize(&existing) {
            let mut out = resolved;
            for part in tail.iter().rev() {
                out.push(part);
            }
            return Some(out);
        }
        let name = existing.file_name()?.to_os_string();
        tail.push(name);
        if !existing.pop() {
            return Some(normalized);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(command: &str) -> Option<Vec<String>> {
        bash_grant_keys(command)
    }

    fn one(command: &str) -> String {
        let keys = keys(command).unwrap_or_else(|| panic!("{command} should be keyable"));
        assert_eq!(keys.len(), 1, "{command}: {keys:?}");
        keys.into_iter().next().unwrap()
    }

    #[test]
    fn unlisted_bases_keep_one_token_keys() {
        assert_eq!(one("ls -la"), "ls");
        assert_eq!(one("nice -n 10 make test"), "make");
        assert_eq!(one("GOOS=linux go build"), "go");
    }

    #[test]
    fn listed_bases_key_with_first_subcommand() {
        assert_eq!(one("git status"), "git status");
        assert_eq!(one("git push --force"), "git push");
        assert_eq!(one("npm install"), "npm install");
        assert_eq!(one("pnpm test"), "pnpm test");
        assert_eq!(one("yarn add left-pad"), "yarn add");
        assert_eq!(one("cargo build --release"), "cargo build");
        assert_eq!(one("pip install requests"), "pip install");
        assert_eq!(one("pip3 uninstall requests"), "pip3 uninstall");
        assert_eq!(one("uv sync"), "uv sync");
        assert_eq!(one("poetry add httpx"), "poetry add");
        assert_eq!(one("docker rm -f web"), "docker rm");
        assert_eq!(one("podman ps"), "podman ps");
        assert_eq!(one("kubectl delete pod x"), "kubectl delete");
        assert_eq!(one("systemctl restart nginx"), "systemctl restart");
        assert_eq!(one("brew install jq"), "brew install");
        assert_eq!(one("apt install curl"), "apt install");
        assert_eq!(one("apt-get remove curl"), "apt-get remove");
        assert_eq!(one("dnf install curl"), "dnf install");
        assert_eq!(one("pacman -Syu"), "pacman -S");
        assert_eq!(one("pacman -Rns foo"), "pacman -R");
        assert_eq!(one("pacman --query"), "pacman -Q");
    }

    #[test]
    fn global_options_are_skipped() {
        assert_eq!(one("git -C repo push --force"), "git push");
        assert_eq!(one("git -c user.name=x commit -m hi"), "git commit");
        assert_eq!(one("git --git-dir=/tmp/x/.git log"), "git log");
        assert_eq!(one("git --no-pager diff"), "git diff");
        assert_eq!(one("cargo +nightly build"), "cargo build");
        assert_eq!(one("docker --context prod rm web"), "docker rm");
        assert_eq!(one("kubectl -n kube-system delete pod x"), "kubectl delete");
        assert_eq!(one("kubectl --context prod get pods"), "kubectl get");
        assert_eq!(one("npm --prefix web run build"), "npm run");
        assert_eq!(one("pnpm -C web --filter app test"), "pnpm test");
        assert_eq!(
            one("apt-get -y -o Dpkg::Options=x install foo"),
            "apt-get install"
        );
        assert_eq!(one("systemctl --user restart foo"), "systemctl restart");
    }

    #[test]
    fn unidentifiable_subcommand_is_not_keyable() {
        assert_eq!(keys("git"), None);
        assert_eq!(keys("git --version"), None);
        assert_eq!(keys("git --frobnicate push"), None);
        assert_eq!(keys("git -C"), None);
        assert_eq!(keys("pacman"), None);
        assert_eq!(keys(""), None);
        // One unkeyable segment makes the whole compound unkeyable.
        assert_eq!(keys("ls && git --weird push"), None);
    }

    #[test]
    fn compounds_key_every_segment() {
        assert_eq!(
            keys("cargo test && rm -rf target/"),
            Some(vec!["cargo test".to_string(), "rm".to_string()])
        );
        assert_eq!(
            keys("git status && git push | tee log"),
            Some(vec![
                "git push".to_string(),
                "git status".to_string(),
                "tee".to_string()
            ])
        );
    }

    #[test]
    fn path_prefixed_base_keeps_its_path() {
        assert_eq!(one("/usr/bin/git push"), "/usr/bin/git push");
        assert_eq!(one("./git push"), "./git push");
    }

    #[test]
    fn write_grant_covers_read_but_not_the_reverse() {
        let path = Path::new("/w/src/a.rs");
        let read_keys = covering_file_keys(FileTier::Read, path);
        let write_keys = covering_file_keys(FileTier::Write, path);
        assert!(read_keys.contains(&file_key(FileTier::Write, path)));
        assert!(read_keys.contains(&file_key(FileTier::Read, path)));
        assert!(write_keys.contains(&file_key(FileTier::Write, path)));
        assert!(!write_keys.contains(&file_key(FileTier::Read, path)));
        assert!(!write_keys.contains(&dir_key(FileTier::Read, Path::new("/w/src"))));
    }

    #[test]
    fn directory_grants_cover_direct_entries_only() {
        let dir = dir_key(FileTier::Write, Path::new("/w/src"));
        assert!(covering_file_keys(FileTier::Write, Path::new("/w/src/a.rs")).contains(&dir));
        assert!(
            !covering_file_keys(FileTier::Write, Path::new("/w/src/nested/a.rs")).contains(&dir)
        );
        assert!(!covering_file_keys(FileTier::Write, Path::new("/w/a.rs")).contains(&dir));
    }

    #[test]
    fn grant_path_requires_absolute_and_folds_dots() {
        assert_eq!(grant_path("relative/a.rs"), None);
        let tmp = std::env::temp_dir().canonicalize().unwrap();
        let raw = format!("{}/x/../does-not-exist/./f.txt", tmp.display());
        assert_eq!(
            grant_path(&raw),
            Some(tmp.join("does-not-exist").join("f.txt"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn grant_path_resolves_symlinked_ancestors() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("grant-keys-{}", std::process::id()));
        let real = root.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = root.join("link");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let via_link = format!("{}/new.txt", link.display());
        assert_eq!(grant_path(&via_link), Some(real.join("new.txt")));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
