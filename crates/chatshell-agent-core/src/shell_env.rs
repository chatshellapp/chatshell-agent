//! Environment of the bash child process.
//!
//! The bash session no longer inherits the full app environment (API keys,
//! tokens, and other secrets the app process happens to carry). It starts
//! from an empty environment holding a small allowlist plus the variables
//! the user chose to pass through on this device.
//!
//! Which variables exist is a property of the device, not of a
//! conversation, so the passthrough list is a per-device file next to the
//! other device-local stores and never syncs.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Variables every bash session receives (when set in the app).
pub const BASE_ALLOWLIST: &[&str] = &[
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TZ", "TMPDIR",
];

/// Variables commonly needed by developer workflows; offered as
/// suggestions when they are set in the app environment. Append-only.
pub const SUGGESTED_PASSTHROUGH: &[&str] = &[
    "SSH_AUTH_SOCK",
    "JAVA_HOME",
    "GOPATH",
    "GOROOT",
    "NVM_DIR",
    "NVM_BIN",
    "PYENV_ROOT",
    "RUSTUP_HOME",
    "CARGO_HOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "ANDROID_HOME",
    "DOCKER_HOST",
    "EDITOR",
    "LC_CTYPE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
];

/// File name of the per-device config inside the app data directory.
pub const CONFIG_FILE_NAME: &str = "bash_environment.json";

/// Name fragments that mark a variable as a likely secret. Such names are
/// never suggested and never named in hints (a user may still pass one
/// through explicitly).
const SECRET_MARKERS: &[&str] = &["KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL"];

/// True when `name` looks like it holds a secret.
pub fn is_secret_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    SECRET_MARKERS.iter().any(|marker| upper.contains(marker))
}

/// True for a syntactically valid environment variable name.
pub fn is_valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Environment for a bash session: the allowlist plus `passthrough`,
/// each taken from `lookup` (the app environment) when set. Sorted by
/// name, deduplicated.
pub fn minimized_env(
    passthrough: &[String],
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<(String, String)> {
    let names: BTreeSet<&str> = BASE_ALLOWLIST
        .iter()
        .copied()
        .chain(passthrough.iter().map(String::as_str))
        .filter(|name| is_valid_name(name))
        .collect();
    names
        .into_iter()
        .filter_map(|name| lookup(name).map(|value| (name.to_string(), value)))
        .collect()
}

/// Suggested passthrough names that are set in the app environment and
/// not already allowed.
pub fn suggestions(lookup: impl Fn(&str) -> Option<String>) -> Vec<String> {
    SUGGESTED_PASSTHROUGH
        .iter()
        .filter(|name| !BASE_ALLOWLIST.contains(name) && !is_secret_name(name))
        .filter(|name| lookup(name).is_some())
        .map(|name| name.to_string())
        .collect()
}

/// When a failed command's output names a variable that is set in the app
/// but withheld from bash, return a hint pointing at the passthrough
/// setting. Secret-looking names are never mentioned.
pub fn missing_variable_hint(
    output: &str,
    passthrough: &[String],
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let mut missing: BTreeSet<&str> = BTreeSet::new();
    for token in output.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        // Candidates: known suggestion names, or SCREAMING_SNAKE names.
        let screaming = token.len() >= 3
            && token.contains('_')
            && is_valid_name(token)
            && !token.chars().any(|c| c.is_ascii_lowercase());
        if !screaming && !SUGGESTED_PASSTHROUGH.contains(&token) {
            continue;
        }
        if BASE_ALLOWLIST.contains(&token)
            || passthrough.iter().any(|name| name == token)
            || is_secret_name(token)
        {
            continue;
        }
        if lookup(token).is_some() {
            missing.insert(token);
        }
    }
    if missing.is_empty() {
        return None;
    }
    let names: Vec<&str> = missing.into_iter().collect();
    Some(format!(
        "[hint: {} {} set in the app environment but not passed to bash. \
         The user can add {} under Settings > Built-in Tools > Bash environment.]",
        names.join(", "),
        if names.len() == 1 { "is" } else { "are" },
        if names.len() == 1 { "it" } else { "them" },
    ))
}

/// Persisted per-device bash environment settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BashEnvConfig {
    /// Extra variable names passed from the app environment to bash.
    #[serde(default)]
    pub passthrough: Vec<String>,
}

/// Handle to the per-device config file.
#[derive(Debug, Clone)]
pub struct BashEnvStore {
    path: PathBuf,
}

impl BashEnvStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load the config; a missing file is the default (empty) config.
    pub fn load(&self) -> Result<BashEnvConfig> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", self.path.display())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(BashEnvConfig::default()),
            Err(err) => Err(err).with_context(|| format!("read {}", self.path.display())),
        }
    }

    /// Save `passthrough` (invalid names rejected, duplicates dropped,
    /// sorted). Returns the stored config.
    pub fn save(&self, passthrough: &[String]) -> Result<BashEnvConfig> {
        if let Some(bad) = passthrough
            .iter()
            .map(|name| name.trim())
            .find(|name| !is_valid_name(name))
        {
            anyhow::bail!("invalid environment variable name: {bad:?}");
        }
        let names: BTreeSet<String> = passthrough
            .iter()
            .map(|name| name.trim().to_string())
            .collect();
        let config = BashEnvConfig {
            passthrough: names.into_iter().collect(),
        };
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&config)?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn minimized_env_keeps_only_allowlist_and_passthrough() {
        let lookup = env(&[
            ("PATH", "/usr/bin"),
            ("HOME", "/Users/x"),
            ("OPENAI_API_KEY", "sk-secret"),
            ("CHATSHELL_SENTINEL", "leak"),
            ("JAVA_HOME", "/opt/jdk"),
        ]);
        let built = minimized_env(&["JAVA_HOME".to_string()], &lookup);
        let names: Vec<&str> = built.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, vec!["HOME", "JAVA_HOME", "PATH"]);
        assert!(!built.iter().any(|(k, _)| k == "OPENAI_API_KEY"));
        assert!(!built.iter().any(|(k, _)| k == "CHATSHELL_SENTINEL"));
    }

    #[test]
    fn invalid_passthrough_names_are_ignored() {
        let lookup = env(&[("A=B", "x")]);
        assert!(minimized_env(&["A=B".to_string()], &lookup).is_empty());
    }

    #[test]
    fn suggestions_exclude_unset_allowlisted_and_secret_names() {
        let lookup = env(&[
            ("JAVA_HOME", "/opt/jdk"),
            ("SSH_AUTH_SOCK", "/tmp/agent"),
            ("PATH", "/usr/bin"),
        ]);
        assert_eq!(suggestions(&lookup), vec!["SSH_AUTH_SOCK", "JAVA_HOME"]);
        assert!(
            SUGGESTED_PASSTHROUGH
                .iter()
                .all(|name| !is_secret_name(name))
        );
    }

    #[test]
    fn hint_names_withheld_variables_but_never_secrets() {
        let lookup = env(&[
            ("JAVA_HOME", "/opt/jdk"),
            ("GITHUB_TOKEN", "ghp_x"),
            ("PATH", "/usr/bin"),
        ]);
        let hint = missing_variable_hint(
            "ERROR: JAVA_HOME is not set; GITHUB_TOKEN missing; PATH ok",
            &[],
            &lookup,
        )
        .unwrap();
        assert!(hint.contains("JAVA_HOME"));
        assert!(!hint.contains("GITHUB_TOKEN"));
        assert!(!hint.contains("PATH"));
        // Already passed through: no hint.
        assert_eq!(
            missing_variable_hint("JAVA_HOME is not set", &["JAVA_HOME".into()], &lookup),
            None
        );
        // Not set in the app either: nothing to suggest.
        assert_eq!(
            missing_variable_hint("GOPATH is not set", &[], &lookup),
            None
        );
    }

    #[test]
    fn store_round_trips_and_validates() {
        let dir = tempfile::tempdir().unwrap();
        let store = BashEnvStore::new(dir.path().join(CONFIG_FILE_NAME));
        assert_eq!(store.load().unwrap(), BashEnvConfig::default());
        let saved = store
            .save(&[
                "JAVA_HOME".into(),
                " SSH_AUTH_SOCK ".into(),
                "JAVA_HOME".into(),
            ])
            .unwrap();
        assert_eq!(saved.passthrough, vec!["JAVA_HOME", "SSH_AUTH_SOCK"]);
        assert_eq!(store.load().unwrap(), saved);
        assert!(store.save(&["BAD NAME".into()]).is_err());
    }
}
