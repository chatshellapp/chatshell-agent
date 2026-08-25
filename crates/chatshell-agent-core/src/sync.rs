//! Snapshot-sync engine (ADR 01: file transport + atomic snapshot publish +
//! record-level LWW merge). Graduated from the Phase-0 spike crate with the
//! production table registry.

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use std::path::{Path, PathBuf};

/// Meta keys for in-DB snapshot versioning.
pub const META_SYNC_VERSION: &str = "sync_version";
pub const META_DEVICE_ID: &str = "device_id";
pub const META_BASE_VERSION: &str = "base_version";

/// One table in the merge registry.
#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub name: &'static str,
    /// Primary-key column (default `id`; overridden by key-value tables).
    pub pk: &'static str,
    /// Mutable tables carry `updated_at` and merge with last-writer-wins.
    /// Immutable (append-only) tables merge with INSERT OR IGNORE.
    pub mutable: bool,
    /// Natural UNIQUE conflict target columns (e.g. `"name"`, `"name, source"`).
    /// Two devices may independently create the same logical row under
    /// different UUIDs (seeded defaults are the classic case); a second
    /// conflict clause drops the remote copy instead of failing the merge.
    /// First-seen wins; later edits still propagate through the pk clause.
    pub natural_key: Option<&'static str>,
}

const fn t(name: &'static str) -> TableSpec {
    TableSpec {
        name,
        pk: "id",
        mutable: true,
        natural_key: None,
    }
}

const fn immut(name: &'static str) -> TableSpec {
    TableSpec {
        name,
        pk: "id",
        mutable: false,
        natural_key: None,
    }
}

const fn tpk(name: &'static str, pk: &'static str) -> TableSpec {
    TableSpec {
        name,
        pk,
        mutable: true,
        natural_key: None,
    }
}

/// Mutable table with a natural UNIQUE key (conflict target expression).
const fn tnk(name: &'static str, natural_key: &'static str) -> TableSpec {
    TableSpec {
        natural_key: Some(natural_key),
        ..t(name)
    }
}

/// Production merge registry, ordered parents-first so FK constraints hold
/// during the merge with `PRAGMA foreign_keys = ON`.
///
/// Mutable = rows updated in place after insert (needs `updated_at`, which the
/// desktop schema maintains since v11). Immutable = insert-only rows; natural
/// UNIQUE constraints (junction tables) make cross-device re-creation dedup.
/// `meta` and the FTS virtual table are not merged.
pub const PRODUCTION_MERGE_TABLES: &[TableSpec] = &[
    t("conversations"),
    tnk("users", "username"),
    t("user_relationships"),
    t("providers"),
    t("models"),
    tnk("model_parameter_presets", "name"),
    t("assistants"),
    t("tools"),
    tnk("skills", "name, source"),
    t("knowledge_bases"),
    t("prompts"),
    tpk("settings", "key"),
    t("fetch_results"),
    t("files"),
    t("messages"),
    t("search_results"),
    immut("search_decisions"),
    immut("thinking_steps"),
    t("tool_calls"),
    t("code_executions"),
    t("compactions"),
    immut("content_blocks"),
    immut("message_attachments"),
    immut("message_contexts"),
    immut("message_prompts"),
    immut("message_knowledge_bases"),
    immut("message_tools"),
    immut("assistant_tools"),
    immut("assistant_skills"),
    immut("assistant_knowledge_bases"),
    tpk("conversation_settings", "conversation_id"),
    immut("conversation_participants"),
];

/// Per-table outcome of one merge pass.
#[derive(Debug, Default, PartialEq)]
pub struct TableReport {
    pub inserted: usize,
    pub updated: usize,
}

/// Outcome of a full merge.
#[derive(Debug, Default)]
pub struct MergeReport {
    pub tables: Vec<(&'static str, TableReport)>,
}

impl MergeReport {
    pub fn total_inserted(&self) -> usize {
        self.tables.iter().map(|(_, r)| r.inserted).sum()
    }
    pub fn total_updated(&self) -> usize {
        self.tables.iter().map(|(_, r)| r.updated).sum()
    }
}

/// Validate that a database satisfies the registry's merge preconditions:
/// every table exists with its primary key, and mutable tables carry
/// `updated_at`. Call before the first sync to fail fast on schema drift.
pub fn validate_registry(conn: &Connection, tables: &[TableSpec]) -> Result<()> {
    for spec in tables {
        let columns = table_columns(conn, spec.name)
            .with_context(|| format!("registry validation: table {}", spec.name))?;
        if !columns.iter().any(|c| c == spec.pk) {
            bail!(
                "table {} lacks primary-key column {} (has {:?})",
                spec.name,
                spec.pk,
                columns
            );
        }
        if spec.mutable && !columns.iter().any(|c| c == "updated_at") {
            bail!(
                "mutable table {} lacks updated_at (has {:?})",
                spec.name,
                columns
            );
        }
    }
    Ok(())
}

/// Merge a remote snapshot (a published database file) into `local` under the
/// given table registry.
///
/// Mutable tables upsert with last-writer-wins on `updated_at` (strictly
/// greater — same-timestamp writes never flap). Tombstones (`deleted_at`) are
/// ordinary column values: an edit newer than a tombstone resurrects the row.
/// Immutable tables merge with INSERT OR IGNORE (PK + natural UNIQUE dedup).
pub fn merge_remote(
    local: &Connection,
    remote_path: &Path,
    tables: &[TableSpec],
) -> Result<MergeReport> {
    if !remote_path.exists() {
        bail!("remote snapshot not found: {}", remote_path.display());
    }
    local
        .pragma_update(None, "foreign_keys", "ON")
        .context("enable foreign_keys")?;

    local.execute(
        "ATTACH DATABASE ?1 AS remote_db",
        [remote_path.to_string_lossy().as_ref()],
    )?;
    let report = (|| {
        let mut report = MergeReport::default();
        local.execute_batch("BEGIN IMMEDIATE")?;
        for spec in tables {
            // Skip tables the remote snapshot predates (devices upgrade at
            // different times; a fresh local table must not abort the merge
            // against an older remote).
            let remote_has_table: bool = local
                .query_row(
                    "SELECT COUNT(*) FROM remote_db.sqlite_master                      WHERE type = 'table' AND name = ?1",
                    [spec.name],
                    |row| row.get::<_, i64>(0),
                )
                .is_ok_and(|count| count > 0);
            if !remote_has_table {
                continue;
            }
            let cols = table_columns(local, spec.name)?;
            let col_list = cols.join(", ");
            let src = format!("SELECT {col_list} FROM remote_db.{}", spec.name);
            let sql = if spec.mutable {
                let assignments = cols
                    .iter()
                    .filter(|c| c.as_str() != spec.pk)
                    .map(|c| format!("{c} = excluded.{c}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                // Second conflict clause for natural-UNIQUE tables: the
                // remote row loses to the local natural-key twin instead of
                // aborting the merge (SQLite >= 3.35 multi-clause upsert).
                let natural_clause = spec
                    .natural_key
                    .map(|nk| format!(" ON CONFLICT({nk}) DO NOTHING"))
                    .unwrap_or_default();
                format!(
                    "INSERT INTO main.{name} ({col_list}) {src} WHERE true \
                     ON CONFLICT({pk}) DO UPDATE SET {assignments} \
                     WHERE excluded.updated_at > main.{name}.updated_at{natural_clause}",
                    name = spec.name,
                    pk = spec.pk
                )
            } else {
                format!(
                    "INSERT OR IGNORE INTO main.{name} ({col_list}) {src}",
                    name = spec.name
                )
            };

            let overlap_before: usize = local.query_row(
                &format!(
                    "SELECT COUNT(*) FROM main.{name} m \
                     JOIN remote_db.{name} r ON r.{pk} = m.{pk}",
                    name = spec.name,
                    pk = spec.pk
                ),
                [],
                |row| row.get::<_, i64>(0),
            )? as usize;

            let changed = local.execute(&sql, [])?;

            let overlap_after: usize = local.query_row(
                &format!(
                    "SELECT COUNT(*) FROM main.{name} m \
                     JOIN remote_db.{name} r ON r.{pk} = m.{pk}",
                    name = spec.name,
                    pk = spec.pk
                ),
                [],
                |row| row.get::<_, i64>(0),
            )? as usize;

            let inserted = overlap_after - overlap_before;
            let updated = changed - inserted;
            report
                .tables
                .push((spec.name, TableReport { inserted, updated }));
        }
        local.execute_batch("COMMIT")?;
        Ok(report)
    })();
    let _ = local.execute("DETACH DATABASE remote_db", []);
    report
}

/// Publish a consistent standalone snapshot via `VACUUM INTO` a temp path +
/// atomic rename. WAL content is folded in, no side files land next to the
/// target, and the source connection stays usable.
pub fn publish_snapshot(conn: &Connection, target: &Path) -> Result<()> {
    if target.exists() {
        std::fs::remove_file(target).context("remove stale snapshot")?;
    }
    let tmp = target.with_extension("tmp-publishing");
    if tmp.exists() {
        std::fs::remove_file(&tmp).context("remove stale temp snapshot")?;
    }
    conn.execute("VACUUM INTO ?1", [tmp.to_string_lossy().as_ref()])
        .context("VACUUM INTO temp snapshot")?;
    std::fs::rename(&tmp, target).context("atomic rename of snapshot")?;
    Ok(())
}

/// What the local device should do with a remote snapshot.
#[derive(Debug, PartialEq)]
pub enum SyncAction {
    UpToDate,
    Adopt,
    Merge,
}

/// Three-way version compare. Version ids are `<counter>:<device_id>` —
/// unique per publish (a bare counter collides when two devices branch from
/// one base and both mint the same number). Only the counter orders lineage.
pub fn plan_sync(local: &Connection, remote: &Connection) -> Result<SyncAction> {
    let local_version = get_meta(local, META_SYNC_VERSION)?.unwrap_or_else(|| "0:none".into());
    let local_base = get_meta(local, META_BASE_VERSION)?.unwrap_or_else(|| "0:none".into());
    let remote_version = get_meta(remote, META_SYNC_VERSION)?.unwrap_or_else(|| "0:none".into());

    if remote_version == local_version || remote_version == local_base {
        return Ok(SyncAction::UpToDate);
    }

    let local_counter = counter_of(&local_version);
    let remote_counter = counter_of(&remote_version);
    let local_clean = local_version == local_base;

    if local_clean {
        if remote_counter > local_counter {
            return Ok(SyncAction::Adopt);
        }
        if remote_counter == local_counter {
            return Ok(SyncAction::Merge);
        }
        return Ok(SyncAction::UpToDate); // stale remote
    }
    Ok(SyncAction::Merge)
}

fn now_rfc3339() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("unix:{}", d.as_secs()))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn counter_of(version: &str) -> u64 {
    version
        .split(':')
        .next()
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}

pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
    let mut rows = stmt.query([key])?;
    Ok(rows.next()?.map(|r| r.get::<_, String>(0)).transpose()?)
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value, updated_at) VALUES(?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        rusqlite::params![key, value, now_rfc3339()],
    )?;
    Ok(())
}

fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let cols = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if cols.is_empty() {
        bail!("table {table} has no columns (missing?)");
    }
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mini-schema mirroring production shapes + sync columns.
    const SCHEMA_SQL: &str = r#"
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE conversations (
    id TEXT PRIMARY KEY, title TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_type TEXT NOT NULL, content TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE files (
    id TEXT PRIMARY KEY, file_name TEXT NOT NULL, content_hash TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE message_attachments (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    attachment_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    UNIQUE(message_id, attachment_id));
"#;

    const TEST_TABLES: &[TableSpec] = &[
        t("conversations"),
        t("files"),
        t("messages"),
        immut("message_attachments"),
    ];

    fn init(conn: &Connection) {
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        conn.execute_batch(SCHEMA_SQL).unwrap();
        set_meta(conn, META_SYNC_VERSION, "1:base").unwrap();
        set_meta(conn, META_BASE_VERSION, "1:base").unwrap();
    }

    fn conv(c: &Connection, id: &str, title: &str, updated_at: &str) {
        c.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at, deleted_at)
             VALUES(?1, ?2, ?3, ?3, NULL)
             ON CONFLICT(id) DO UPDATE SET title = excluded.title, updated_at = excluded.updated_at",
            [id, title, updated_at],
        )
        .unwrap();
    }

    fn msg(c: &Connection, id: &str, conversation_id: &str, content: &str, updated_at: &str) {
        c.execute(
            "INSERT INTO messages(id, conversation_id, sender_type, content, created_at, updated_at, deleted_at)
             VALUES(?1, ?2, 'user', ?3, ?4, ?4, NULL)
             ON CONFLICT(id) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
            rusqlite::params![id, conversation_id, content, updated_at],
        )
        .unwrap();
    }

    fn count(c: &Connection, sql: &str) -> i64 {
        c.query_row(sql, [], |r| r.get::<_, i64>(0)).unwrap()
    }

    fn scalar(c: &Connection, sql: &str) -> String {
        c.query_row(sql, [], |r| r.get::<_, String>(0)).unwrap()
    }

    struct Env {
        dir: tempfile::TempDir,
        b_path: std::path::PathBuf,
    }

    fn env_with(seed: fn(&Connection)) -> (Env, Connection, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.db");
        let base = Connection::open(&base_path).unwrap();
        init(&base);
        seed(&base);
        let a_path = dir.path().join("a.db");
        let b_path = dir.path().join("b.db");
        publish_snapshot(&base, &a_path).unwrap();
        publish_snapshot(&base, &b_path).unwrap();
        let a = Connection::open(&a_path).unwrap();
        let b = Connection::open(&b_path).unwrap();
        set_meta(&a, META_DEVICE_ID, "a").unwrap();
        set_meta(&b, META_DEVICE_ID, "b").unwrap();
        (Env { dir, b_path }, a, b)
    }

    #[test]
    fn registry_invariants() {
        // Unique names, pk non-empty, mutable tables... structural checks.
        let mut seen = std::collections::HashSet::new();
        for spec in PRODUCTION_MERGE_TABLES {
            assert!(!spec.pk.is_empty(), "{}", spec.name);
            assert!(seen.insert(spec.name), "duplicate table {}", spec.name);
        }
        // Parents before children for the FK-critical pairs.
        let pos = |name: &str| {
            PRODUCTION_MERGE_TABLES
                .iter()
                .position(|s| s.name == name)
                .unwrap()
        };
        for (parent, child) in [
            ("conversations", "messages"),
            ("messages", "message_attachments"),
            ("files", "message_attachments"),
            ("fetch_results", "message_contexts"),
            ("prompts", "message_prompts"),
            ("tools", "message_tools"),
            ("knowledge_bases", "message_knowledge_bases"),
            ("assistants", "assistant_tools"),
            ("messages", "tool_calls"),
            ("messages", "thinking_steps"),
            ("search_results", "search_decisions"),
            ("providers", "models"),
            ("conversations", "conversation_settings"),
        ] {
            assert!(pos(parent) < pos(child), "{parent} must precede {child}");
        }
        // Key-value tables use their natural pk.
        assert_eq!(
            PRODUCTION_MERGE_TABLES
                .iter()
                .find(|s| s.name == "settings")
                .unwrap()
                .pk,
            "key"
        );
        assert_eq!(
            PRODUCTION_MERGE_TABLES
                .iter()
                .find(|s| s.name == "conversation_settings")
                .unwrap()
                .pk,
            "conversation_id"
        );
    }

    #[test]
    fn validate_registry_checks_columns() {
        let conn = Connection::open_in_memory().unwrap();
        init(&conn);
        assert!(validate_registry(&conn, TEST_TABLES).is_ok());
        // Mutable table missing updated_at must fail.
        conn.execute("CREATE TABLE bad(id TEXT PRIMARY KEY, title TEXT)", [])
            .unwrap();
        assert!(
            validate_registry(
                &conn,
                &[TableSpec {
                    name: "bad",
                    pk: "id",
                    mutable: true,
                    natural_key: None,
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn disjoint_appends_merge_union() {
        let (env, a, b) = env_with(|base| {
            conv(base, "c1", "base", "2026-08-21T10:00:00+00:00");
        });
        msg(&a, "a1", "c1", "from A", "2026-08-21T11:00:00+00:00");
        msg(&b, "b1", "c1", "from B", "2026-08-21T11:30:00+00:00");
        let report = merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(count(&a, "SELECT COUNT(*) FROM messages"), 2);
        assert_eq!(report.total_inserted(), 1);
    }

    #[test]
    fn lww_both_directions() {
        let (env, a, b) = env_with(|base| {
            conv(base, "c1", "original", "2026-08-21T10:00:00+00:00");
        });
        conv(&a, "c1", "from A", "2026-08-21T11:00:00+00:00");
        conv(&b, "c1", "from B", "2026-08-21T12:00:00+00:00");
        merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(
            scalar(&a, "SELECT title FROM conversations WHERE id='c1'"),
            "from B"
        );
        conv(&a, "c1", "from A again", "2026-08-21T13:00:00+00:00");
        merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(
            scalar(&a, "SELECT title FROM conversations WHERE id='c1'"),
            "from A again"
        );
    }

    #[test]
    fn tombstone_and_resurrect() {
        let (env, a, b) = env_with(|base| {
            conv(base, "c1", "keep", "2026-08-21T10:00:00+00:00");
            conv(base, "c2", "delete", "2026-08-21T10:00:00+00:00");
        });
        b.execute(
            "UPDATE conversations SET deleted_at='2026-08-21T12:00:00+00:00', updated_at='2026-08-21T12:00:00+00:00' WHERE id='c1'",
        []
        )
        .unwrap();
        conv(&a, "c1", "A edit", "2026-08-21T11:00:00+00:00");
        merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(
            scalar(&a, "SELECT deleted_at FROM conversations WHERE id='c1'"),
            "2026-08-21T12:00:00+00:00"
        );
        // Newer edit on A resurrects c2 even though B tombstoned it.
        b.execute(
            "UPDATE conversations SET deleted_at='2026-08-21T12:00:00+00:00', updated_at='2026-08-21T12:00:00+00:00' WHERE id='c2'",
        []
        )
        .unwrap();
        conv(&a, "c2", "resurrect", "2026-08-21T13:00:00+00:00");
        merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(
            count(
                &a,
                "SELECT COUNT(*) FROM conversations WHERE id='c2' AND deleted_at IS NULL"
            ),
            1
        );
    }

    #[test]
    fn merge_is_idempotent() {
        let (env, a, b) = env_with(|base| {
            conv(base, "c1", "base", "2026-08-21T10:00:00+00:00");
        });
        msg(&b, "b1", "c1", "from B", "2026-08-21T11:00:00+00:00");
        conv(&b, "c1", "B title", "2026-08-21T11:30:00+00:00");
        let first = merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!((first.total_inserted(), first.total_updated()), (1, 1));
        let second = merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!((second.total_inserted(), second.total_updated()), (0, 0));
    }

    #[test]
    fn fk_parents_first() {
        let (env, a, _b) = env_with(|_| {});
        let b = Connection::open(&env.b_path).unwrap();
        conv(&b, "c-new", "B only", "2026-08-21T11:00:00+00:00");
        msg(&b, "m-new", "c-new", "child", "2026-08-21T11:01:00+00:00");
        drop(b);
        merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(
            count(&a, "SELECT COUNT(*) FROM messages WHERE id='m-new'"),
            1
        );
    }

    #[test]
    fn junction_natural_key_dedup() {
        let (env, a, b) = env_with(|base| {
            conv(base, "c1", "c", "2026-08-21T10:00:00+00:00");
            msg(base, "m1", "c1", "attach", "2026-08-21T10:00:00+00:00");
            base.execute(
                "INSERT INTO files(id, file_name, content_hash, created_at, updated_at)
                 VALUES('f1', 'a.png', 'h1', '2026-08-21T10:00:00+00:00', '2026-08-21T10:00:00+00:00')",
                [],
            )
            .unwrap();
        });
        for (c, id) in [(&a, "att-a"), (&b, "att-b")] {
            c.execute(
                "INSERT INTO message_attachments(id, message_id, attachment_id, created_at)
                 VALUES(?1, 'm1', 'f1', '2026-08-21T11:00:00+00:00')",
                [id],
            )
            .unwrap();
        }
        merge_remote(&a, &env.b_path, TEST_TABLES).unwrap();
        assert_eq!(
            count(
                &a,
                "SELECT COUNT(*) FROM message_attachments WHERE message_id='m1' AND attachment_id='f1'"
            ),
            1
        );
    }

    #[test]
    fn publish_snapshot_standalone() {
        let dir = tempfile::tempdir().unwrap();
        let live = Connection::open(dir.path().join("live.db")).unwrap();
        init(&live);
        conv(&live, "c1", "with wal", "2026-08-21T10:00:00+00:00");
        live.pragma_update(None, "journal_mode", "WAL").unwrap();
        msg(&live, "m1", "c1", "via wal", "2026-08-21T10:05:00+00:00");
        let snap = dir.path().join("snapshot.db");
        publish_snapshot(&live, &snap).unwrap();
        assert!(!dir.path().join("snapshot.db-wal").exists());
        let opened = Connection::open(&snap).unwrap();
        assert_eq!(
            scalar(&opened, "SELECT content FROM messages WHERE id='m1'"),
            "via wal"
        );
        msg(&live, "m2", "c1", "after", "2026-08-21T10:06:00+00:00");
        assert_eq!(count(&live, "SELECT COUNT(*) FROM messages"), 2);
    }

    #[test]
    fn full_divergence_converges() {
        let (env, a, b) = env_with(|base| {
            conv(base, "c1", "base", "2026-08-21T10:00:00+00:00");
        });
        msg(&a, "a1", "c1", "A only", "2026-08-21T11:00:00+00:00");
        set_meta(&a, META_SYNC_VERSION, "2:a").unwrap();
        let a_snap = env.dir.path().join("a-snapshot.db");
        publish_snapshot(&a, &a_snap).unwrap();

        msg(&b, "b1", "c1", "B only", "2026-08-21T11:30:00+00:00");
        conv(&b, "c1", "B title", "2026-08-21T11:45:00+00:00");
        set_meta(&b, META_SYNC_VERSION, "2:b").unwrap();
        let b_snap = env.dir.path().join("b-snapshot.db");
        publish_snapshot(&b, &b_snap).unwrap();

        let remote = Connection::open(&b_snap).unwrap();
        assert_eq!(plan_sync(&a, &remote).unwrap(), SyncAction::Merge);
        merge_remote(&a, &b_snap, TEST_TABLES).unwrap();
        set_meta(&a, META_SYNC_VERSION, "3:a").unwrap();
        set_meta(&a, META_BASE_VERSION, "3:a").unwrap();
        publish_snapshot(&a, &a_snap).unwrap();
        assert_eq!(count(&a, "SELECT COUNT(*) FROM messages"), 2);
        assert_eq!(
            scalar(&a, "SELECT title FROM conversations WHERE id='c1'"),
            "B title"
        );

        let a_remote = Connection::open(&a_snap).unwrap();
        assert_eq!(plan_sync(&b, &a_remote).unwrap(), SyncAction::Merge);
        merge_remote(&b, &a_snap, TEST_TABLES).unwrap();
        set_meta(&b, META_BASE_VERSION, "3:a").unwrap();
        set_meta(&b, META_SYNC_VERSION, "3:a").unwrap();
        assert_eq!(count(&b, "SELECT COUNT(*) FROM messages"), 2);

        assert_eq!(plan_sync(&a, &a_remote).unwrap(), SyncAction::UpToDate);
        let old_b = Connection::open(&b_snap).unwrap();
        assert_eq!(plan_sync(&b, &old_b).unwrap(), SyncAction::UpToDate);
    }
}

#[cfg(test)]
mod natural_key_tests {
    use super::*;

    #[test]
    fn same_natural_key_different_pk_does_not_abort_merge() {
        // Two devices seed the same preset name under different UUIDs.
        let dir = tempfile::tempdir().unwrap();
        let a_path = dir.path().join("a.db");
        let b_path = dir.path().join("b.db");
        let mk = |path: &std::path::Path, id: &str| {
            let conn = Connection::open(path).unwrap();
            conn.pragma_update(None, "foreign_keys", "ON").unwrap();
            conn.execute_batch(
                "CREATE TABLE presets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    updated_at TEXT NOT NULL
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO presets(id, name, updated_at) VALUES(?1, 'Default', '2026-08-21T10:00:00+00:00')",
                [id],
            )
            .unwrap();
            conn
        };
        let a = mk(&a_path, "id-a");
        let _b = mk(&b_path, "id-b");
        drop(_b);

        let tables = [tnk("presets", "name")];
        let report =
            merge_remote(&a, &b_path, &tables).expect("natural-key collision must not abort");
        // Remote's twin lost to the local natural key; no rows inserted.
        assert_eq!(report.total_inserted(), 0);
        let count: i64 = a
            .query_row("SELECT COUNT(*) FROM presets", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
        let id: String = a
            .query_row("SELECT id FROM presets", [], |r| r.get(0))
            .unwrap();
        assert_eq!(id, "id-a", "local natural-key twin wins");
    }
}

// ---------------------------------------------------------------------------
// SyncEngine: publish/merge orchestration (lifted from desktop Phase 3a)
// ---------------------------------------------------------------------------

/// Filename of the published snapshot inside the cloud directory.
pub const SNAPSHOT_FILE: &str = "snapshot.db";

/// What one `sync_now` pass did.
#[derive(Debug, Default, PartialEq)]
pub struct SyncOutcome {
    pub action: String,
    pub rows_merged: usize,
    pub republished: bool,
}

/// Drives [`merge_remote`]/[`publish_snapshot`]/[`plan_sync`] against a live
/// database file through a dedicated side connection (the app's own async
/// pool keeps running; WAL allows the mix).
///
/// - **publish**: `VACUUM INTO` temp + atomic rename; the version id bumps
///   only when another connection committed since the last publish (dirty
///   detection via `PRAGMA data_version`).
/// - **sync_now**: merge is the only mutating path (adopt collapses into
///   merge - same result for a clean local, no file swap under the live
///   pool). An empty merge report means the local content already equals the
///   remote, so the local version id adopts the remote id and two devices
///   converge instead of ping-ponging counters. A non-empty merge mints a
///   fresh version and republishes.
pub struct SyncEngine {
    conn: Connection,
    cloud_dir: PathBuf,
    tables: &'static [TableSpec],
    last_published_data_version: Option<i64>,
}

impl SyncEngine {
    /// Open a side connection against the live database and validate the
    /// production registry. The cloud directory must already exist (the
    /// device that owns the iCloud container provisions it).
    pub fn new(db_path: &Path, cloud_dir: PathBuf) -> Result<Self> {
        Self::new_with_tables(db_path, cloud_dir, PRODUCTION_MERGE_TABLES)
    }

    /// Same as [`SyncEngine::new`] with an explicit table registry (tests,
    /// schema variants).
    pub fn new_with_tables(
        db_path: &Path,
        cloud_dir: PathBuf,
        tables: &'static [TableSpec],
    ) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("open sync connection {}", db_path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let mut engine = Self {
            conn,
            cloud_dir,
            tables,
            last_published_data_version: None,
        };
        engine.ensure_device_id()?;
        validate_registry(&engine.conn, engine.tables)
            .context("sync registry validation against live schema")?;
        Ok(engine)
    }

    /// Access the underlying connection (tests, meta inspection).
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    pub fn cloud_dir(&self) -> &Path {
        &self.cloud_dir
    }

    fn ensure_device_id(&mut self) -> Result<()> {
        if get_meta(&self.conn, META_DEVICE_ID)?.is_none() {
            let id = uuid::Uuid::now_v7().to_string();
            set_meta(&self.conn, META_DEVICE_ID, &id)?;
        }
        if get_meta(&self.conn, META_SYNC_VERSION)?.is_none() {
            set_meta(&self.conn, META_SYNC_VERSION, "0:local")?;
            set_meta(&self.conn, META_BASE_VERSION, "0:local")?;
        }
        Ok(())
    }

    fn device_id(&self) -> String {
        get_meta(&self.conn, META_DEVICE_ID)
            .ok()
            .flatten()
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn current_version(&self) -> String {
        get_meta(&self.conn, META_SYNC_VERSION)
            .ok()
            .flatten()
            .unwrap_or_else(|| "0:local".to_string())
    }

    fn data_version(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))?)
    }

    fn snapshot_path(&self) -> PathBuf {
        self.cloud_dir.join(SNAPSHOT_FILE)
    }

    /// True when another connection committed since the last publish.
    fn dirty_since_last_publish(&self) -> Result<bool> {
        match self.last_published_data_version {
            None => Ok(true),
            Some(seen) => Ok(self.data_version()? != seen),
        }
    }

    /// Publish the current database state to the cloud dir. Bumps the
    /// version id only when the database changed since the last publish;
    /// otherwise the snapshot is republished under the same id.
    pub fn publish(&mut self) -> Result<()> {
        if self.dirty_since_last_publish()? {
            let counter = self
                .current_version()
                .split(':')
                .next()
                .and_then(|c| c.parse::<u64>().ok())
                .unwrap_or(0);
            let new_version = format!("{}:{}", counter + 1, self.device_id());
            set_meta(&self.conn, META_SYNC_VERSION, &new_version)?;
            set_meta(&self.conn, META_BASE_VERSION, &new_version)?;
        }
        publish_snapshot(&self.conn, &self.snapshot_path())?;
        self.last_published_data_version = Some(self.data_version()?);
        Ok(())
    }

    /// One sync pass against the cloud snapshot, if one exists.
    pub fn sync_now(&mut self) -> Result<SyncOutcome> {
        let snapshot = self.snapshot_path();
        if !snapshot.exists() {
            // First run on this device: publish so the other side can see us.
            self.publish()?;
            return Ok(SyncOutcome {
                action: "published-initial".into(),
                ..Default::default()
            });
        }

        let remote = Connection::open(&snapshot)?;
        let action = plan_sync(&self.conn, &remote)?;
        drop(remote);

        match action {
            SyncAction::UpToDate => Ok(SyncOutcome {
                action: "up-to-date".into(),
                ..Default::default()
            }),
            SyncAction::Adopt | SyncAction::Merge => {
                let report = merge_remote(&self.conn, &snapshot, self.tables)?;
                let rows_merged = report.total_inserted() + report.total_updated();
                let remote_version = Connection::open(&snapshot)
                    .ok()
                    .and_then(|c| get_meta(&c, META_SYNC_VERSION).ok().flatten())
                    .unwrap_or_else(|| "0:none".to_string());

                if rows_merged == 0 {
                    // Content already equal: adopt the remote version id and
                    // converge - republishing under a new id would ping-pong.
                    set_meta(&self.conn, META_SYNC_VERSION, &remote_version)?;
                    set_meta(&self.conn, META_BASE_VERSION, &remote_version)?;
                    self.last_published_data_version = Some(self.data_version()?);
                    Ok(SyncOutcome {
                        action: "converged".into(),
                        rows_merged: 0,
                        republished: false,
                    })
                } else {
                    // Fresh local state: mint a new version and publish it.
                    let local_counter = self
                        .current_version()
                        .split(':')
                        .next()
                        .and_then(|c| c.parse::<u64>().ok())
                        .unwrap_or(0);
                    let remote_counter = remote_version
                        .split(':')
                        .next()
                        .and_then(|c| c.parse::<u64>().ok())
                        .unwrap_or(0);
                    let new_version = format!(
                        "{}:{}",
                        local_counter.max(remote_counter) + 1,
                        self.device_id()
                    );
                    set_meta(&self.conn, META_SYNC_VERSION, &new_version)?;
                    set_meta(&self.conn, META_BASE_VERSION, &new_version)?;
                    self.publish()?;
                    Ok(SyncOutcome {
                        action: "merged".into(),
                        rows_merged,
                        republished: true,
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod engine_tests {
    use super::*;

    const ENGINE_SCHEMA: &str = r#"
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE conversations (
    id TEXT PRIMARY KEY, title TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
"#;
    const ENGINE_TABLES: &[TableSpec] = &[t("conversations")];

    fn seed(path: &Path, id: &str, title: &str) -> Connection {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(ENGINE_SCHEMA).unwrap();
        conn.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at)
             VALUES(?1, ?2, '2026-08-21T10:00:00+00:00', '2026-08-21T10:00:00+00:00')",
            rusqlite::params![id, title],
        )
        .unwrap();
        conn
    }

    fn dir(_tag: &str) -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn two_engines_converge_via_cloud_snapshot() {
        let space = dir("engine");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        seed(&b_path, "b1", "from B");

        let mut a = SyncEngine::new_with_tables(&a_path, cloud.clone(), ENGINE_TABLES).unwrap();
        assert_eq!(a.sync_now().unwrap().action, "published-initial");

        let mut b = SyncEngine::new_with_tables(&b_path, cloud.clone(), ENGINE_TABLES).unwrap();
        let out_b = b.sync_now().unwrap();
        assert!(out_b.rows_merged > 0 && out_b.republished);

        let out_a2 = a.sync_now().unwrap();
        assert!(out_a2.rows_merged > 0);

        let out_b2 = b.sync_now().unwrap();
        assert_eq!(out_b2.action, "converged");
        assert!(!out_b2.republished);

        let va = get_meta(a.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
        let vb = get_meta(b.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
        assert_eq!(va, vb, "version ids converge");
        assert_eq!(a.sync_now().unwrap().action, "up-to-date");
        assert_eq!(b.sync_now().unwrap().action, "up-to-date");

        // Both hold both rows.
        for conn in [&a, &b] {
            let n: i64 = conn
                .connection()
                .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
                .unwrap();
            assert_eq!(n, 2);
        }
    }

    #[test]
    fn republish_without_writes_keeps_version() {
        let space = dir("stable");
        let db = space.path().join("a.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();
        seed(&db, "x", "t");
        let mut engine = SyncEngine::new_with_tables(&db, cloud, ENGINE_TABLES).unwrap();
        engine.publish().unwrap();
        let v1 = get_meta(engine.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
        engine.publish().unwrap();
        let v2 = get_meta(engine.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
        assert_eq!(v1, v2);
    }
}
