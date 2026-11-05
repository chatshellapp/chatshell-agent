//! Copy-based file history for the write and edit tools.
//!
//! Before a turn's first write to a file, the file's current state is
//! recorded under `<conversation dir>/<turn id>/`: a backup copy when it
//! existed, a "did not exist" marker when the turn creates it. Restoring a
//! turn rewinds it and every later turn (newest first), which leaves every
//! touched file as it was before that turn. Changes made through bash are
//! not captured. History is device-local and never synced.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Files above this size are not backed up (the entry records it, and a
/// restore reports the file as not restorable).
pub const MAX_BACKUP_BYTES: u64 = 20 * 1024 * 1024;
const INDEX_FILE: &str = "index.jsonl";

/// One file's state before a turn first wrote it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub path: PathBuf,
    /// The file existed before the turn (false: the turn created it).
    pub existed: bool,
    /// Backup file name inside the turn directory (`None` for created
    /// files, or when the file was too large to back up).
    pub backup: Option<String>,
    /// Milliseconds since the Unix epoch when the entry was recorded.
    pub at_ms: u64,
}

/// What a restore did.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct RestoreReport {
    /// Files put back to their earlier content.
    pub restored: Vec<PathBuf>,
    /// Files removed because the rewound turns created them.
    pub removed: Vec<PathBuf>,
    /// Files that could not be restored, with the reason.
    pub failed: Vec<(PathBuf, String)>,
}

/// The capture handle the tools hold for one turn.
#[derive(Debug)]
pub struct FileHistory {
    turn_dir: PathBuf,
    lock: Mutex<()>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn safe_component(id: &str) -> Option<&str> {
    let ok = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    ok.then_some(id)
}

fn read_index(turn_dir: &Path) -> Vec<HistoryEntry> {
    let Ok(text) = fs::read_to_string(turn_dir.join(INDEX_FILE)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

impl FileHistory {
    /// History for `turn_id` under `conversation_dir`. `None` when the
    /// turn id is not a safe directory name.
    pub fn new(conversation_dir: &Path, turn_id: &str) -> Option<Self> {
        Some(Self {
            turn_dir: conversation_dir.join(safe_component(turn_id)?),
            lock: Mutex::new(()),
        })
    }

    /// Record `path`'s state before its first write in this turn.
    /// Best-effort: a failure is logged and never blocks the write.
    pub fn capture(&self, path: &Path) {
        if let Err(e) = self.try_capture(path) {
            tracing::warn!(
                "🗂️ [file_history] Could not back up {}: {e}",
                path.display()
            );
        }
    }

    fn try_capture(&self, path: &Path) -> std::io::Result<()> {
        let path = std::path::absolute(path)?;
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let entries = read_index(&self.turn_dir);
        if entries.iter().any(|entry| entry.path == path) {
            return Ok(());
        }
        fs::create_dir_all(&self.turn_dir)?;
        let backup = match fs::metadata(&path) {
            Ok(meta) if meta.is_file() && meta.len() <= MAX_BACKUP_BYTES => {
                let name = format!("{}.bak", entries.len());
                fs::copy(&path, self.turn_dir.join(&name))?;
                Some(name)
            }
            _ => None,
        };
        let entry = HistoryEntry {
            existed: path.exists(),
            path,
            backup,
            at_ms: now_ms(),
        };
        let mut index = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.turn_dir.join(INDEX_FILE))?;
        writeln!(index, "{}", serde_json::to_string(&entry)?)?;
        Ok(())
    }
}

/// The entries recorded for `turn_id`.
pub fn turn_entries(conversation_dir: &Path, turn_id: &str) -> Vec<HistoryEntry> {
    safe_component(turn_id)
        .map(|turn| read_index(&conversation_dir.join(turn)))
        .unwrap_or_default()
}

/// Turn directories with their first-entry time, oldest first.
fn turns(conversation_dir: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(dirs) = fs::read_dir(conversation_dir) else {
        return Vec::new();
    };
    let mut turns: Vec<(u64, PathBuf)> = dirs
        .flatten()
        .map(|d| d.path())
        .filter(|p| p.is_dir())
        .filter_map(|dir| {
            let first = read_index(&dir).first()?.at_ms;
            Some((first, dir))
        })
        .collect();
    turns.sort();
    turns
}

/// Rewind `turn_id` and every later turn: each touched file returns to
/// its state before `turn_id`. The rewound turns' history is consumed.
pub fn restore_from(conversation_dir: &Path, turn_id: &str) -> RestoreReport {
    let mut report = RestoreReport::default();
    let Some(turn) = safe_component(turn_id) else {
        return report;
    };
    let target = conversation_dir.join(turn);
    let all = turns(conversation_dir);
    let Some(start) = all
        .iter()
        .find(|(_, dir)| *dir == target)
        .map(|(at, _)| *at)
    else {
        return report;
    };
    let rewound: Vec<PathBuf> = all
        .into_iter()
        .filter(|(at, _)| *at >= start)
        .map(|(_, dir)| dir)
        .collect();

    // Newest first; within a turn, last capture first. The final write to
    // each path is the oldest state, i.e. the one before `turn_id`.
    let mut outcome: std::collections::BTreeMap<PathBuf, Result<bool, String>> =
        std::collections::BTreeMap::new();
    for dir in rewound.iter().rev() {
        for entry in read_index(dir).into_iter().rev() {
            let result = restore_entry(dir, &entry);
            outcome.insert(entry.path.clone(), result);
        }
    }
    for (path, result) in outcome {
        match result {
            Ok(true) => report.restored.push(path),
            Ok(false) => report.removed.push(path),
            Err(reason) => report.failed.push((path, reason)),
        }
    }
    for dir in rewound {
        let _ = fs::remove_dir_all(dir);
    }
    report
}

/// `Ok(true)`: content restored; `Ok(false)`: created file removed.
fn restore_entry(turn_dir: &Path, entry: &HistoryEntry) -> Result<bool, String> {
    if !entry.existed {
        return match fs::remove_file(&entry.path) {
            Ok(()) => Ok(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.to_string()),
        };
    }
    let Some(backup) = &entry.backup else {
        return Err(format!(
            "no backup (files over {} MB are not backed up)",
            MAX_BACKUP_BYTES / (1024 * 1024)
        ));
    };
    if let Some(parent) = entry.path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::copy(turn_dir.join(backup), &entry.path)
        .map(|_| true)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoring_a_turn_rewinds_it_and_every_later_turn() {
        let work = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let a = work.path().join("a.txt");
        let b = work.path().join("b.txt");
        fs::write(&a, "a0").unwrap();

        // Turn 1: edits a, creates b.
        let t1 = FileHistory::new(history.path(), "turn-1").unwrap();
        t1.capture(&a);
        fs::write(&a, "a1").unwrap();
        t1.capture(&b);
        fs::write(&b, "b1").unwrap();
        // A second write in the same turn keeps the first snapshot.
        t1.capture(&a);
        fs::write(&a, "a1b").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Turn 2: edits both.
        let t2 = FileHistory::new(history.path(), "turn-2").unwrap();
        t2.capture(&a);
        fs::write(&a, "a2").unwrap();
        t2.capture(&b);
        fs::write(&b, "b2").unwrap();

        assert_eq!(turn_entries(history.path(), "turn-1").len(), 2);

        // Restoring turn 2 only rewinds turn 2.
        let report = restore_from(history.path(), "turn-2");
        assert_eq!(fs::read_to_string(&a).unwrap(), "a1b");
        assert_eq!(fs::read_to_string(&b).unwrap(), "b1");
        assert_eq!(report.restored.len(), 2);
        assert!(
            turn_entries(history.path(), "turn-2").is_empty(),
            "consumed"
        );

        // Turn 3 then a restore of turn 1: back to before turn 1.
        let t3 = FileHistory::new(history.path(), "turn-3").unwrap();
        t3.capture(&a);
        fs::write(&a, "a3").unwrap();
        let report = restore_from(history.path(), "turn-1");
        assert_eq!(fs::read_to_string(&a).unwrap(), "a0");
        assert!(!b.exists(), "the file turn 1 created is removed");
        assert_eq!(report.removed, vec![std::path::absolute(&b).unwrap()]);
        assert!(report.failed.is_empty());
    }

    #[test]
    fn unsafe_turn_ids_are_refused() {
        let history = tempfile::tempdir().unwrap();
        assert!(FileHistory::new(history.path(), "../x").is_none());
        assert!(FileHistory::new(history.path(), "").is_none());
        assert_eq!(
            restore_from(history.path(), "../x"),
            RestoreReport::default()
        );
    }

    #[tokio::test]
    async fn the_write_tool_captures_before_writing() {
        use crate::tools::WriteTool;
        use rig::agent::tool::Tool as _;
        let work = tempfile::tempdir().unwrap();
        let history = tempfile::tempdir().unwrap();
        let handle = std::sync::Arc::new(FileHistory::new(history.path(), "t").unwrap());
        let path = work.path().join("new.txt");
        let mut ctx = rig::agent::tool::ToolContext::default();

        let write = WriteTool::new().with_file_history(handle);
        let args = serde_json::from_value(serde_json::json!({
            "path": path.display().to_string(),
            "content": "hello\n",
        }))
        .unwrap();
        write.call(&mut ctx, args).await.unwrap();

        let entries = turn_entries(history.path(), "t");
        assert_eq!(entries.len(), 1);
        assert!(!entries[0].existed);
        restore_from(history.path(), "t");
        assert!(!path.exists());
    }
}
