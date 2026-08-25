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
pub const SNAPSHOT_FILE: &str = "snapshot.db";

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
            // Column intersection: only columns present on BOTH sides enter
            // the INSERT/UPDATE. This makes the merge tolerant of deployment
            // skew (one device upgraded ahead of the other) instead of
            // aborting with "no such column" on the first divergent pass.
            // Column intersection: only columns present on BOTH sides enter
            // the INSERT/UPDATE. This makes the merge tolerant of deployment
            // skew (one device upgraded ahead of the other).
            let local_cols = table_columns(local, spec.name)?;
            let remote_cols: Vec<String> = {
                let mut stmt = local.prepare(&format!(
                    "SELECT name FROM pragma_table_info('{}') WHERE schema = 'remote_db'",
                    spec.name
                ))?;
                stmt.query_map([], |row| row.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?
            };
            let cols: Vec<String> = local_cols
                .iter()
                .filter(|c| remote_cols.contains(c))
                .cloned()
                .collect();
            if cols.is_empty() {
                bail!(
                    "table {} has no common columns between local and remote",
                    spec.name
                );
            }
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

/// Mini-schema mirroring production shapes + sync columns (shared by tests
/// and the column-intersection test module).
#[cfg(test)]
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

/// Mini merge registry for tests.
#[cfg(test)]
const TEST_TABLES: &[TableSpec] = &[
    t("conversations"),
    t("files"),
    t("messages"),
    immut("message_attachments"),
];

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
///   fresh version and republishes. When the remote is already up to date,
///   pending local writes still ship via a throttled flush (`maybe_flush`),
///   so timer-driven ticks both pull and push without fixed-cadence churn.
pub struct SyncEngine {
    conn: Connection,
    cloud_dir: PathBuf,
    tables: &'static [TableSpec],
    /// `PRAGMA data_version` observed at the last publish (or adoption).
    /// `None` until this process publishes/adopts; `None` makes
    /// [`Self::dirty_since_last_publish`] report dirty so the first mint of
    /// a real version id (`0:local` -> `<counter>:<device>`) always happens.
    last_published_data_version: Option<i64>,
    /// `PRAGMA data_version` captured at construction and refreshed at each
    /// publish. Pool commits since then mean unshipped local writes.
    pool_baseline_data_version: i64,
    /// Wall-clock guard for tick-driven flushes so an idle app never
    /// republishes more often than this interval. Exit-time publishes bypass
    /// it. Defaults to [`PUBLISH_MIN_INTERVAL`]; tests shrink it to zero.
    min_publish_interval: std::time::Duration,
    /// Instant of the last publish; `None` until this process publishes.
    last_publish_at: Option<std::time::Instant>,
    /// Armed at construction when the previous process may have died with
    /// unshipped writes (no clean-exit marker); consumed by the first
    /// `maybe_flush`, which then forces a publish regardless of the pool
    /// baseline.
    force_first_flush: bool,
}

/// Meta key recording whether the previous process ended with a successful
/// publish. Absent/`0` at construction arms [`SyncEngine::force_first_flush`].
const META_CLEAN_EXIT: &str = "sync_clean_exit";

/// Minimum wall-clock spacing between tick-driven publishes. Each publish is
/// a full `VACUUM INTO` of the database plus a cloud upload, so an idle or
/// lightly-used app must not republish on every timer tick.
const PUBLISH_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Placeholder version id before a device mints its first real version.
const INITIAL_VERSION: &str = "0:local";

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
        let pool_baseline_data_version =
            conn.query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))?;
        let mut engine = Self {
            conn,
            cloud_dir,
            tables,
            last_published_data_version: None,
            pool_baseline_data_version,
            min_publish_interval: PUBLISH_MIN_INTERVAL,
            last_publish_at: None,
            force_first_flush: false,
        };
        engine.ensure_device_id()?;
        // If the previous process did not end with a successful publish
        // (killed, crashed, or pre-dating the marker), writes from that
        // session may never have shipped - arm one forced flush. The first
        // publish in this process re-arms the "clean" state.
        let prior_clean = get_meta(&engine.conn, META_CLEAN_EXIT)?
            .map(|v| v == "1")
            .unwrap_or(false);
        engine.force_first_flush =
            !prior_clean && engine.current_version() != INITIAL_VERSION;
        set_meta(&engine.conn, META_CLEAN_EXIT, "0")?;
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
            set_meta(&self.conn, META_SYNC_VERSION, INITIAL_VERSION)?;
            set_meta(&self.conn, META_BASE_VERSION, INITIAL_VERSION)?;
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
            .unwrap_or_else(|| INITIAL_VERSION.to_string())
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
    /// otherwise the snapshot is republished under the same id. Unthrottled:
    /// used for exit/background flushes where terminal state must land.
    pub fn publish(&mut self) -> Result<()> {
        self.publish_inner(false)
    }

    fn publish_inner(&mut self, force_bump: bool) -> Result<()> {
        if force_bump || self.dirty_since_last_publish()? {
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
        let seen = self.data_version()?;
        self.last_published_data_version = Some(seen);
        self.pool_baseline_data_version = seen;
        self.last_publish_at = Some(std::time::Instant::now());
        // Any successful publish ships everything this process knew about;
        // the construction-time suspicion is resolved either way.
        self.force_first_flush = false;
        set_meta(&self.conn, META_CLEAN_EXIT, "1")?;
        Ok(())
    }

    /// True when the app pool committed since this process last published
    /// (or since construction, if it has not published yet). The placeholder
    /// version [`INITIAL_VERSION`] means this device never minted a real
    /// version - the published-initial / adopt paths own that first
    /// transition, so a tick flush stays out of their way.
    fn pool_has_unshipped_writes(&self) -> Result<bool> {
        if self.current_version() == INITIAL_VERSION {
            return Ok(false);
        }
        Ok(self.data_version()? != self.pool_baseline_data_version)
    }

    /// Throttled flush for timer-driven ticks: publish pending local writes
    /// only when the app pool actually committed since the last publish AND
    /// the last publish is older than the configured interval. Returns true
    /// when a snapshot was published.
    fn maybe_flush(&mut self) -> Result<bool> {
        // Armed at construction: the previous process may have died with
        // unshipped writes that the pool baseline cannot see. Consume the
        // flag once and force the publish (bypassing the throttle - this is
        // the startup pass, before any publish in this process).
        if self.force_first_flush {
            self.force_first_flush = false;
            self.publish_inner(true)?;
            return Ok(true);
        }
        if !self.pool_has_unshipped_writes()? {
            return Ok(false);
        }
        if let Some(at) = self.last_publish_at
            && at.elapsed() < self.min_publish_interval
        {
            return Ok(false);
        }
        // Forced bump: the DB may carry an id just adopted from the remote
        // while content has moved on - publishing under an unchanged id
        // would make peers treat the new rows as already-synced.
        self.publish_inner(true)?;
        Ok(true)
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
            SyncAction::UpToDate => {
                // Remote has nothing new, but local writes since the last
                // publish must still ship (throttled - see `maybe_flush`).
                let flushed = self.maybe_flush()?;
                Ok(SyncOutcome {
                    action: if flushed { "flushed" } else { "up-to-date" }.into(),
                    ..Default::default()
                })
            }
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
                    // `pool_baseline_data_version` is intentionally NOT
                    // refreshed here: pool writes since construction are
                    // still unshipped and must stay detectable below.
                    set_meta(&self.conn, META_SYNC_VERSION, &remote_version)?;
                    set_meta(&self.conn, META_BASE_VERSION, &remote_version)?;
                    let seen = self.data_version()?;
                    self.last_published_data_version = Some(seen);
                    // Adoption alone must not strand local writes made while
                    // the app was running (e.g. a conversation created since
                    // the last publish): ship them via the throttled flush.
                    let flushed = self.maybe_flush()?;
                    Ok(SyncOutcome {
                        action: if flushed { "flushed" } else { "converged" }.into(),
                        rows_merged: 0,
                        republished: flushed,
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
    fn tick_flush_ships_pending_pool_writes_and_throttles() {
        let space = dir("flush");
        let db = space.path().join("a.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();
        seed(&db, "c1", "first");

        let mut engine = SyncEngine::new_with_tables(&db, cloud, ENGINE_TABLES).unwrap();
        engine.min_publish_interval = std::time::Duration::ZERO;
        assert_eq!(engine.sync_now().unwrap().action, "published-initial");

        // No pool writes since construction: the tick must stay a cheap
        // no-op (no republish, no version churn for peers to react to).
        let out = engine.sync_now().unwrap();
        assert_eq!(out.action, "up-to-date");
        assert!(!out.republished);

        // A pool commit since construction ships on the next tick under a
        // freshly minted version id.
        let version_before =
            get_meta(engine.connection(), META_SYNC_VERSION).unwrap().unwrap();
        {
            let pool = Connection::open(&db).unwrap();
            pool.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at)
                 VALUES('c2', 'second', '2026-08-21T11:00:00+00:00', '2026-08-21T11:00:00+00:00')",
                [],
            )
            .unwrap();
        }
        let out = engine.sync_now().unwrap();
        assert_eq!(out.action, "flushed");
        let version_after =
            get_meta(engine.connection(), META_SYNC_VERSION).unwrap().unwrap();
        assert_ne!(version_before, version_after, "flush mints a new id");

        // Throttle: fresh pool writes inside the window are deferred - the
        // tick still reports up-to-date instead of publishing.
        engine.min_publish_interval = std::time::Duration::from_secs(3600);
        {
            let pool = Connection::open(&db).unwrap();
            pool.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at)
                 VALUES('c3', 'third', '2026-08-21T12:00:00+00:00', '2026-08-21T12:00:00+00:00')",
                [],
            )
            .unwrap();
        }
        let out = engine.sync_now().unwrap();
        assert_eq!(out.action, "up-to-date");
    }

    #[test]
    fn converged_adoption_flushes_pending_pool_writes() {
        let space = dir("converged-flush");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        // A publishes one conversation.
        seed(&a_path, "c1", "shared");
        let mut a = SyncEngine::new_with_tables(&a_path, cloud.clone(), ENGINE_TABLES).unwrap();
        assert_eq!(a.sync_now().unwrap().action, "published-initial");

        // B already holds that same row, so its first pass converges with an
        // empty merge report - but it also has a pending local write made
        // after engine construction (the app was running).
        seed(&b_path, "c1", "shared");
        let mut b = SyncEngine::new_with_tables(&b_path, cloud.clone(), ENGINE_TABLES).unwrap();
        b.min_publish_interval = std::time::Duration::ZERO;
        {
            let pool = Connection::open(&b_path).unwrap();
            pool.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at)
                 VALUES('c2', 'from B', '2026-08-21T11:00:00+00:00', '2026-08-21T11:00:00+00:00')",
                [],
            )
            .unwrap();
        }

        let out_b = b.sync_now().unwrap();
        assert_eq!(out_b.action, "flushed", "adoption must not strand writes");
        assert!(out_b.republished);

        // A pulls B's pending row on its next pass.
        let out_a = a.sync_now().unwrap();
        assert!(out_a.rows_merged > 0);
        let n: i64 = a
            .connection()
            .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
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

#[cfg(test)]
mod column_intersection_tests {
    use super::*;

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

    fn scalar(c: &Connection, sql: &str) -> String {
        c.query_row(sql, [], |r| r.get::<_, String>(0)).unwrap()
    }

    const SCHEMA: &str = r#"
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE conversations (
    id TEXT PRIMARY KEY, title TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT
);
"#;

    #[test]
    fn merge_tolerates_column_skew() {
        // Local (desktop v15): has new_col. Remote (iOS, old schema): doesn't.
        let dir = tempfile::tempdir().unwrap();
        let local_path = dir.path().join("local.db");
        let remote_path = dir.path().join("remote.db");

        let local = Connection::open(&local_path).unwrap();
        init(&local);
        conv(&local, "c1", "local title", "2026-08-21T10:00:00+00:00");

        let remote = Connection::open(&remote_path).unwrap();
        remote.execute_batch(SCHEMA).unwrap();
        remote
            .execute(
                "INSERT INTO meta(key, value, updated_at) VALUES('sync_version', '1:ios', '2026-08-21T10:00:00+00:00')",
                [],
            )
            .unwrap();
        remote
            .execute(
                "INSERT INTO conversations(id, title, created_at, updated_at)
                 VALUES('c2', 'from ios', '2026-08-21T10:00:00+00:00', '2026-08-21T10:00:00+00:00')",
                [],
            )
            .unwrap();
        drop(remote);

        // Add new_col to LOCAL only — simulating desktop upgraded ahead of iOS.
        local
            .execute("ALTER TABLE conversations ADD COLUMN new_col TEXT", [])
            .unwrap();

        // Merge must NOT abort with "no such column" — the intersection skips it.
        let report = merge_remote(&local, &remote_path, TEST_TABLES).unwrap();
        assert_eq!(report.total_inserted(), 1, "B's row must arrive");

        // Verify data integrity on both common columns.
        let title: String = local
            .query_row("SELECT title FROM conversations WHERE id='c2'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(title, "from ios");
    }

    #[test]
    fn reverse_direction_also_tolerates_skew() {
        // Remote has extra_col that local doesn't — SELECT must not fail.
        let dir = tempfile::tempdir().unwrap();
        let a_path = dir.path().join("a.db");
        let b_path = dir.path().join("b.db");
        let a = Connection::open(&a_path).unwrap();
        init(&a);
        conv(&a, "c1", "A", "2026-08-21T10:00:00+00:00");
        drop(a);

        let b = Connection::open(&b_path).unwrap();
        b.execute_batch(SCHEMA).unwrap();
        set_meta(&b, META_SYNC_VERSION, "1:b").unwrap();
        set_meta(&b, META_BASE_VERSION, "1:b").unwrap();
        conv(&b, "c1", "B with extra col", "2026-08-21T11:00:00+00:00");
        b.execute("ALTER TABLE conversations ADD COLUMN extra_col TEXT", [])
            .unwrap();
        drop(b);

        let a2 = Connection::open(&a_path).unwrap();
        let report = merge_remote(&a2, &b_path, TEST_TABLES).unwrap();
        assert_eq!(report.total_updated(), 1, "LWW update applied");
        assert_eq!(
            scalar(&a2, "SELECT title FROM conversations WHERE id='c1'"),
            "B with extra col"
        );
    }
}
