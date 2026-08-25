//! Process-wide anchor state shared by the read/write/edit tools: content
//! tags minted on read/write and validated on edit (upstream
//! `file-snapshot-store` semantics), the no-op loop guard, and the
//! exclusive edit lock. Tools are constructed per agent build, so this state
//! must live outside them.

use std::path::Path;
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::edit_anchor::{AnchorStatus, NoopLoopGuard, NoopRecordResult, ObservedFiles};

struct AnchorState {
    observed: ObservedFiles,
    noop: NoopLoopGuard,
}

static STATE: LazyLock<Mutex<AnchorState>> = LazyLock::new(|| {
    Mutex::new(AnchorState {
        observed: ObservedFiles::new(),
        noop: NoopLoopGuard::new(),
    })
});

fn lock() -> MutexGuard<'static, AnchorState> {
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Stable key: canonicalize when the path exists on disk, else use it as-is.
fn path_key(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Mint a fresh anchor for `content` at `path` (record + 4-hex tag), or
/// `None` above the snapshot size cap.
pub fn record_anchor(path: &Path, content: &str) -> Option<String> {
    lock().observed.record(&path_key(path), content)
}

/// Compare current content against the last-served anchor.
pub fn check_anchor(path: &Path, content: &str) -> AnchorStatus {
    lock().observed.check(&path_key(path), content)
}

/// Record a byte-identical no-op edit attempt (upstream noop-loop-guard).
pub fn record_noop(path: &Path, input_hash: u64) -> NoopRecordResult {
    lock().noop.record(&path_key(path), input_hash)
}

/// Clear the no-op counter after a real commit on `path`.
pub fn reset_noop(path: &Path) {
    lock().noop.reset(&path_key(path));
}

/// Exclusive edit lock (upstream: edit concurrency = "exclusive"). The edit
/// tool's whole read-modify-write is synchronous, so a std mutex held across
/// it is safe; concurrent edits on one file cannot interleave.
pub static EDIT_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "chatshell-anchor-state-test-{}-{}",
            name,
            std::process::id()
        ));
        std::fs::write(&path, "content").unwrap();
        path
    }

    #[test]
    fn record_and_check_roundtrip() {
        let path = fixture("roundtrip");
        let tag = record_anchor(&path, "content").unwrap();
        assert_eq!(check_anchor(&path, "content"), AnchorStatus::Fresh);
        assert_eq!(check_anchor(&path, "changed"), AnchorStatus::Drifted);
        assert_eq!(tag.len(), 4);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn unobserved_paths_are_unobserved() {
        let path = fixture("missing-anchor");
        std::fs::remove_file(&path).ok();
        assert_eq!(check_anchor(&path, "anything"), AnchorStatus::Unobserved);
    }

    #[test]
    fn noop_counter_escalates_and_resets() {
        let path = fixture("noop");
        let hash = 42;
        assert!(!record_noop(&path, hash).escalate);
        assert!(!record_noop(&path, hash).escalate);
        assert!(record_noop(&path, hash).escalate);
        reset_noop(&path);
        assert!(!record_noop(&path, hash).escalate);
        std::fs::remove_file(path).ok();
    }
}
