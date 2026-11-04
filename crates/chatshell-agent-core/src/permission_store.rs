//! Remembered approvals: a per-device SQLite file of grant keys.
//!
//! Lives in its own database file (never the canonical DB): the sync
//! snapshot is a whole-database `VACUUM INTO`, so a table inside the
//! canonical DB would travel to other devices. A remembered grant must
//! only ever take effect on the device where the user approved it.
//!
//! Lookups open a read-only connection and never create the file; only
//! recording an approval creates it. The table is capped at
//! [`MAX_GRANTS`] rows; recording beyond the cap evicts the oldest
//! approvals.

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags, params};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of remembered grants per device.
pub const MAX_GRANTS: usize = 512;

/// File name of the store inside the app data directory.
pub const STORE_FILE_NAME: &str = "permission_grants.db";

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS permission_grants (
    key TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    approved_at INTEGER NOT NULL
)";

/// Handle to the per-device grant file. Cheap to clone; every operation
/// opens its own short-lived connection, so readers always see the
/// latest recorded state.
#[derive(Debug, Clone)]
pub struct PersistentGrantStore {
    path: PathBuf,
}

impl PersistentGrantStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every remembered grant key. A missing file or table is an empty
    /// store, not an error.
    pub fn keys(&self) -> Result<HashSet<String>> {
        let Some(conn) = self.open_read_only()? else {
            return Ok(HashSet::new());
        };
        let mut stmt = match conn.prepare("SELECT key FROM permission_grants") {
            Ok(stmt) => stmt,
            Err(rusqlite::Error::SqliteFailure(_, Some(message)))
                if message.contains("no such table") =>
            {
                return Ok(HashSet::new());
            }
            Err(err) => return Err(err).context("query permission grants"),
        };
        let keys = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<HashSet<String>>>()?;
        Ok(keys)
    }

    /// Number of remembered grants.
    pub fn count(&self) -> Result<usize> {
        Ok(self.keys()?.len())
    }

    /// Remember `keys` (approved for `tool`), refreshing the approval time
    /// of keys already present, then evict the oldest rows beyond the cap.
    pub fn record(&self, tool: &str, keys: &[String]) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let mut conn = self.open_read_write()?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default();
        let tx = conn.transaction()?;
        // Approval times are strictly increasing so eviction order is
        // exact even when several approvals land in the same millisecond.
        let latest: i64 = tx.query_row(
            "SELECT COALESCE(MAX(approved_at), 0) FROM permission_grants",
            [],
            |row| row.get(0),
        )?;
        let now = now.max(latest + 1);
        for (offset, key) in keys.iter().enumerate() {
            tx.execute(
                "INSERT INTO permission_grants (key, tool, approved_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(key) DO UPDATE SET tool = excluded.tool,
                                                approved_at = excluded.approved_at",
                params![key, tool, now + offset as i64],
            )?;
        }
        tx.execute(
            "DELETE FROM permission_grants WHERE key NOT IN (
                 SELECT key FROM permission_grants
                 ORDER BY approved_at DESC, key DESC LIMIT ?1
             )",
            params![MAX_GRANTS as i64],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Forget every remembered grant. Returns how many were removed.
    pub fn clear(&self) -> Result<usize> {
        if !self.path.exists() {
            return Ok(0);
        }
        let conn = self.open_read_write()?;
        Ok(conn.execute("DELETE FROM permission_grants", [])?)
    }

    fn open_read_only(&self) -> Result<Option<Connection>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let conn = Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("open {} read-only", self.path.display()))?;
        Ok(Some(conn))
    }

    fn open_read_write(&self) -> Result<Connection> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&self.path)
            .with_context(|| format!("open {}", self.path.display()))?;
        conn.execute(SCHEMA, [])?;
        Ok(conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (PersistentGrantStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (
            PersistentGrantStore::new(dir.path().join(STORE_FILE_NAME)),
            dir,
        )
    }

    #[test]
    fn lookups_never_create_the_file() {
        let (store, _dir) = store();
        assert!(store.keys().unwrap().is_empty());
        assert_eq!(store.count().unwrap(), 0);
        assert_eq!(store.clear().unwrap(), 0);
        assert!(!store.path().exists());
    }

    #[test]
    fn recorded_grants_survive_a_new_handle() {
        let (store, _dir) = store();
        store
            .record("bash", &["bash:git status".into(), "bash:ls".into()])
            .unwrap();
        let reopened = PersistentGrantStore::new(store.path());
        let keys = reopened.keys().unwrap();
        assert!(keys.contains("bash:git status"));
        assert!(keys.contains("bash:ls"));
        assert_eq!(reopened.count().unwrap(), 2);
    }

    #[test]
    fn clear_empties_the_table() {
        let (store, _dir) = store();
        store.record("edit", &["tool:edit".into()]).unwrap();
        assert_eq!(store.clear().unwrap(), 1);
        assert!(store.keys().unwrap().is_empty());
    }

    #[test]
    fn oldest_grant_is_evicted_beyond_the_cap() {
        let (store, _dir) = store();
        let keys: Vec<String> = (0..MAX_GRANTS).map(|i| format!("bash:cmd{i}")).collect();
        store.record("bash", &keys).unwrap();
        assert_eq!(store.count().unwrap(), MAX_GRANTS);
        store.record("bash", &["bash:newest".into()]).unwrap();
        let kept = store.keys().unwrap();
        assert_eq!(kept.len(), MAX_GRANTS);
        assert!(kept.contains("bash:newest"));
        assert!(!kept.contains("bash:cmd0"), "oldest row should be evicted");
        assert!(kept.contains("bash:cmd1"));
    }

    #[test]
    fn re_recording_refreshes_instead_of_duplicating() {
        let (store, _dir) = store();
        store.record("bash", &["bash:ls".into()]).unwrap();
        store.record("bash", &["bash:ls".into()]).unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }
}
