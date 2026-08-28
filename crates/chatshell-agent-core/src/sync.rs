//! Snapshot-sync engine (ADR 01: file transport + atomic snapshot publish +
//! record-level LWW merge). Graduated from the Phase-0 spike crate with the
//! production table registry.

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use std::path::{Path, PathBuf};

use crate::sync_crypto::{self, ArtifactHeader, SyncCrypto};

/// Meta keys for in-DB snapshot versioning.
pub const META_SYNC_VERSION: &str = "sync_version";
pub const META_DEVICE_ID: &str = "device_id";

/// Name of the plain snapshot file this engine publishes from.
pub const SNAPSHOT_FILE: &str = "snapshot.db";

/// Encrypted transport artifact (ADR 04). The plaintext `.zst` form never
/// shipped beyond developer devices, so the encrypted artifact takes its own
/// filename and a stale plaintext copy in a dev container is never read. On
/// any artifact the engine does open, an unexpected magic means corrupt or
/// foreign: error, never a plaintext parse.
pub const ENCRYPTED_SNAPSHOT_FILE: &str = "snapshot.db.enc";

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
    t("providers"),
    t("models"),
    tnk("model_parameter_presets", "name"),
    t("assistants"),
    t("tools"),
    tnk("skills", "name, source"),
    t("prompts"),
    tpk("settings", "key"),
    t("fetch_results"),
    t("files"),
    t("messages"),
    t("search_results"),
    // Tables whose rows can be removed on one device are mutable-with-
    // natural-key: whole-row LWW carries `deleted_at` tombstones to peers.
    // Immutable (INSERT OR IGNORE) specs would swallow the tombstone update
    // on any peer that already has the row, so removals could not propagate.
    t("search_decisions"),
    t("thinking_steps"),
    t("tool_calls"),
    t("code_executions"),
    t("compactions"),
    t("content_blocks"),
    tnk("message_attachments", "message_id, attachment_id"),
    tnk("message_contexts", "message_id, context_type, context_id"),
    immut("message_prompts"),
    immut("message_tools"),
    tnk("assistant_tools", "assistant_id, tool_id"),
    tnk("assistant_skills", "assistant_id, skill_id"),
    tpk("conversation_settings", "conversation_id"),
    tnk(
        "conversation_participants",
        "conversation_id, participant_type, participant_id",
    ),
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
    let merge_pass = || -> Result<MergeReport> {
        let mut report = MergeReport::default();
        // Schema-shape provenance: log (don't reject) when the snapshot was
        // written under a different canonical-schema version. Column
        // intersection below keeps mismatched shapes merging safely for
        // additive changes; the warning makes silent column freeze visible
        // and marks where a future hard gate would go.
        if let (Some(local_v), Some(remote_v)) = (
            crate::sync_schema::read_schema_version(local)?,
            local
                .query_row(
                    "SELECT value FROM remote_db.meta WHERE key = ?1",
                    [crate::sync_schema::META_SCHEMA_VERSION],
                    |row| row.get::<_, String>(0),
                )
                .ok()
                .and_then(|v| v.parse::<i64>().ok()),
        ) && local_v != remote_v
        {
            tracing::warn!(
                local_schema_version = local_v,
                remote_schema_version = remote_v,
                "merging snapshot written under a different canonical schema \
                 version; columns absent on either side are not exchanged"
            );
        }
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
                // Compare instants, not byte order: iOS stamps RFC3339 with a
                // "Z" suffix while desktop writes chrono's "+00:00" form with
                // variable fraction width, and lexicographic order across
                // those shapes does not match time order ('Z' and '+'
                // terminate the fraction, digits continue it). julianday()
                // normalizes every notation to one comparable value at
                // millisecond resolution - matching what the iOS stamp
                // carries anyway. Sub-millisecond cross-device races on one
                // row resolve as ties (strict > keeps the merged-in row),
                // which is deterministic and flap-free; unparseable values
                // (e.g. the '' schema default) coalesce to 0.0 so any real
                // timestamp beats them.
                format!(
                    "INSERT INTO main.{name} ({col_list}) {src} WHERE true \
                     ON CONFLICT({pk}) DO UPDATE SET {assignments} \
                     WHERE COALESCE(julianday(excluded.updated_at), 0.0) > \
                     COALESCE(julianday(main.{name}.updated_at), 0.0){natural_clause}",
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
        Ok(report)
    };
    local.execute_batch("BEGIN IMMEDIATE")?;
    let report = match merge_pass() {
        Ok(report) => {
            local.execute_batch("COMMIT")?;
            Ok(report)
        }
        Err(err) => {
            // A failed pass must not strand the side connection's write lock
            // (it would stall every app-pool commit until busy timeout) nor
            // leave remote_db attached (the next ATTACH would fail for the
            // whole process lifetime).
            let _ = local.execute_batch("ROLLBACK");
            Err(err)
        }
    };
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

/// Name of the compressed cloud artifact for snapshot `snapshot.db`.
pub fn compressed_snapshot_name(target: &Path) -> PathBuf {
    let mut os = target.as_os_str().to_os_string();
    os.push(".zst");
    PathBuf::from(os)
}

/// Name of the encrypted cloud artifact for snapshot `snapshot.db`.
pub fn encrypted_snapshot_name(target: &Path) -> PathBuf {
    let mut os = target.as_os_str().to_os_string();
    os.push(".enc");
    PathBuf::from(os)
}

/// Compress `src` into `dst` (zstd level 3). Level 3 measured ~70% size
/// reduction on the real database with compression time comparable to the
/// `VACUUM INTO` itself; higher levels cost 8x CPU for a few more percent.
fn compress_file_zst(src: &Path, dst: &Path) -> Result<()> {
    let input = std::fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
    let output = std::fs::File::create(dst).with_context(|| format!("create {}", dst.display()))?;
    zstd::stream::copy_encode(input, output, 3)
        .with_context(|| format!("zstd compress {}", src.display()))
}

/// Decompress a [`compress_file_zst`] artifact into `dst`.
fn decompress_file_zst(src: &Path, dst: &Path) -> Result<()> {
    let input = std::fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
    let output = std::fs::File::create(dst).with_context(|| format!("create {}", dst.display()))?;
    zstd::stream::copy_decode(input, output)
        .with_context(|| format!("zstd decompress {}", src.display()))
}

/// Publish the snapshot in its transport form: plain `VACUUM INTO` temp,
/// zstd-compress to a temp artifact, atomic rename over `<target>.zst`, then
/// drop both the plain temp and any legacy uncompressed cloud copy.
///
/// Peers either see the previous `.zst` or the new one - never a partial
/// file. Devices running pre-compression engines are out of scope by design:
/// this lands inside the migration-free window before any sync-enabled
/// release ships.
pub fn publish_snapshot_compressed(conn: &Connection, target: &Path, staging: &Path) -> Result<()> {
    std::fs::create_dir_all(staging)
        .with_context(|| format!("provision staging dir {}", staging.display()))?;
    let zst_target = compressed_snapshot_name(target);
    // Plaintext temps stage app-private (see publish_snapshot_encrypted):
    // the cloud provider uploads whatever appears in the container.
    // Unique per publish: a fixed name lets two engines (diagnostics, a
    // future second process) sharing a staging dir delete each other's
    // VACUUM output mid-write — the read then races a half-written or
    // zero-extended file and publishes garbage (real-device finding).
    let tmp_plain = staging.join(format!("snapshot-plain-{}.tmp", uuid::Uuid::now_v7()));
    if tmp_plain.exists() {
        std::fs::remove_file(&tmp_plain).context("remove stale temp snapshot")?;
    }
    conn.execute("VACUUM INTO ?1", [tmp_plain.to_string_lossy().as_ref()])
        .context("VACUUM INTO temp snapshot")?;
    let tmp_zst = zst_target.with_extension(format!("tmp-{}", uuid::Uuid::now_v7()));
    compress_file_zst(&tmp_plain, &tmp_zst)?;
    std::fs::rename(&tmp_zst, &zst_target).context("atomic rename of compressed snapshot")?;
    let _ = std::fs::remove_file(&tmp_plain);
    let _ = std::fs::remove_file(target);
    Ok(())
}

/// The header's `sync_version`: `<counter>:<blake3_hex(device_id)>`. The DB
/// meta table keeps the full device id; the header hashes it so the
/// provider sees no device census or per-device activity timeline
/// (ADR 04 §4).
fn hashed_sync_version(conn: &Connection) -> Result<String> {
    let version = get_meta(conn, META_SYNC_VERSION)?.unwrap_or_else(|| "0:none".into());
    let device = get_meta(conn, META_DEVICE_ID)?.unwrap_or_else(|| "none".into());
    Ok(format!(
        "{}:{}",
        counter_of(&version),
        crate::crypto::blake3_hex(device.as_bytes())
    ))
}

/// Canonical schema version for the header; falls back to the built-in
/// constant when the meta stamp is missing.
fn canonical_schema_version(conn: &Connection) -> u64 {
    crate::sync_schema::read_schema_version(conn)
        .ok()
        .flatten()
        .unwrap_or(crate::sync_schema::CANONICAL_SCHEMA_VERSION)
        .max(0) as u64
}

/// Publish the snapshot in its encrypted transport form (ADR 04 §1):
/// plain `VACUUM INTO` temp → zstd → AEAD under the content key → atomic
/// rename over `snapshot.db.enc`. Compression is strictly before
/// encryption. Legacy plaintext artifacts are dropped so the container
/// converges on the encrypted form.
///
/// `staging` is an app-private directory for the PLAINTEXT temp: the
/// cloud container must never hold plaintext bytes, not even transiently —
/// the sync provider uploads whatever appears in the container, so a
/// plaintext `VACUUM INTO` temp written there can travel (real-device
/// finding on iOS: bird grabs new container files within seconds). Only
/// the encrypted temp (harmless if uploaded) and the final artifact are
/// written into the container; the encrypted temp's rename stays
/// same-filesystem and therefore atomic.
pub fn publish_snapshot_encrypted(
    conn: &Connection,
    target: &Path,
    staging: &Path,
    crypto: &SyncCrypto,
) -> Result<()> {
    std::fs::create_dir_all(staging)
        .with_context(|| format!("provision staging dir {}", staging.display()))?;
    let enc_target = encrypted_snapshot_name(target);
    // Unique per publish: a fixed name lets two engines (diagnostics, a
    // future second process) sharing a staging dir delete each other's
    // VACUUM output mid-write — the read then races a half-written or
    // zero-extended file and publishes garbage (real-device finding).
    let tmp_plain = staging.join(format!("snapshot-plain-{}.tmp", uuid::Uuid::now_v7()));
    if tmp_plain.exists() {
        std::fs::remove_file(&tmp_plain).context("remove stale temp snapshot")?;
    }
    conn.execute("VACUUM INTO ?1", [tmp_plain.to_string_lossy().as_ref()])
        .context("VACUUM INTO temp snapshot")?;
    let plain_bytes = std::fs::read(&tmp_plain).context("read temp snapshot")?;
    let mut zst = Vec::with_capacity(plain_bytes.len() / 2);
    zstd::stream::copy_encode(&plain_bytes[..], &mut zst, 3).context("zstd compress snapshot")?;
    let (header, salt) = crypto
        .build_header(&hashed_sync_version(conn)?, canonical_schema_version(conn))
        .map_err(anyhow::Error::from)?;
    let payload = sync_crypto::encrypt_snapshot_payload(crypto.current_key(), &salt, &zst)
        .map_err(anyhow::Error::from)?;
    let artifact = sync_crypto::build_artifact(&header, &payload);
    // Unique per publish — same rationale as the staging temp: concurrent
    // engines in one process (or processes sharing the container) must not
    // rename each other's temp away mid-publish.
    let tmp_enc = enc_target.with_extension(format!("tmp-{}", uuid::Uuid::now_v7()));
    std::fs::write(&tmp_enc, &artifact).context("write temp encrypted snapshot")?;
    std::fs::rename(&tmp_enc, &enc_target).context("atomic rename of encrypted snapshot")?;
    // The plaintext temp lived in app-private storage and never traveled.
    let _ = std::fs::remove_file(&tmp_plain);
    Ok(())
}

/// What the local device should do with a remote snapshot.
#[derive(Debug, PartialEq)]
pub enum SyncAction {
    UpToDate,
    Adopt,
    Merge,
}

/// Header-first version compare (ADR 04 §1): decides from the parsed
/// artifact header alone — no decryption, no staging, so idle ticks skip
/// the payload entirely. The local meta keeps the full `<counter>:<device>`
/// form; the header carries `<counter>:<blake3_hex(device)>`, so the local
/// device id is hashed before the equality check. Only the counter orders
/// lineage; a counter tie with a different device hash means divergent
/// snapshots → Merge (the lossless union).
pub fn plan_sync_header(local: &Connection, header: &ArtifactHeader) -> Result<SyncAction> {
    let local_version = get_meta(local, META_SYNC_VERSION)?.unwrap_or_else(|| "0:none".into());
    // The id comes from the LOCAL VERSION STRING, not the device meta: a
    // device that adopted a peer's version id carries the publisher's id in
    // its meta, and that adoption is exactly what must compare equal here.
    let local_id = local_version
        .split_once(':')
        .map(|(_, id)| id)
        .unwrap_or("");
    let local_hash = crate::crypto::blake3_hex(local_id.as_bytes());
    let remote_id = header
        .sync_version
        .split_once(':')
        .map(|(_, id)| id)
        .unwrap_or("");
    let local_counter = counter_of(&local_version);
    let remote_counter = counter_of(&header.sync_version);

    if remote_counter == local_counter && remote_id == local_hash {
        return Ok(SyncAction::UpToDate);
    }
    if remote_counter > local_counter {
        return Ok(SyncAction::Adopt);
    }
    if remote_counter == local_counter {
        return Ok(SyncAction::Merge);
    }
    Ok(SyncAction::UpToDate) // stale remote
}

/// Version compare. Version ids are `<counter>:<device_id>` — unique per
/// publish (a bare counter collides when two devices branch from one base
/// and both mint the same number). Only the counter orders lineage; an id
/// tie means the snapshot was published by us (or already adopted).
pub fn plan_sync(local: &Connection, remote: &Connection) -> Result<SyncAction> {
    let local_version = get_meta(local, META_SYNC_VERSION)?.unwrap_or_else(|| "0:none".into());
    let remote_version = get_meta(remote, META_SYNC_VERSION)?.unwrap_or_else(|| "0:none".into());

    if remote_version == local_version {
        return Ok(SyncAction::UpToDate);
    }

    let local_counter = counter_of(&local_version);
    let remote_counter = counter_of(&remote_version);

    if remote_counter > local_counter {
        return Ok(SyncAction::Adopt);
    }
    if remote_counter == local_counter {
        return Ok(SyncAction::Merge);
    }
    Ok(SyncAction::UpToDate) // stale remote
}

/// Wall-clock marker for the meta table's own `updated_at` (never merged, so
/// the compact unix form is fine despite what an RFC3339 name would suggest).
fn now_unix_marker() -> String {
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
        rusqlite::params![key, value, now_unix_marker()],
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
    /// App-private staging directory for decrypted remote snapshots —
    /// never the shared OS temp dir: per-user temp is readable by
    /// same-user processes and swept by backup/forensic scans, which would
    /// undercut the threat model transport encryption exists to close
    /// (ADR 04 §1).
    cache_dir: PathBuf,
    /// Transport encryption state (ADR 04). `None` keeps the legacy
    /// plaintext `.zst` artifacts — a debug-only staging mode now that
    /// release wiring always encrypts.
    crypto: Option<SyncCrypto>,
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
/// publish (diagnostic only; the forced-flush decision uses
/// [`META_SHIPPED_ROW_WATERMARK`]).
const META_CLEAN_EXIT: &str = "sync_clean_exit";
/// Meta key carrying the row-timestamp watermark (max `updated_at` over
/// the merge registry) of the last publish; a fresh engine arms a forced
/// flush when the current watermark differs (audit F2 — data_version is
/// unusable for this: fresh WAL connections don't observe prior
/// processes' commits).
const META_SHIPPED_ROW_WATERMARK: &str = "sync_shipped_row_watermark";

/// Max `updated_at` across the registry's tables ("" when all empty).
/// RFC3339 stamps compare correctly as strings; the migration guaranteed
/// `updated_at` on every synced table.
fn row_watermark(conn: &Connection, tables: &[TableSpec]) -> String {
    tables
        .iter()
        .filter_map(|spec| {
            conn.query_row(
                &format!(
                    "SELECT COALESCE(MAX(updated_at), '') FROM {} WHERE deleted_at IS NULL",
                    spec.name
                ),
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
        })
        .max()
        .unwrap_or_default()
}

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
    pub fn new(
        db_path: &Path,
        cloud_dir: PathBuf,
        cache_dir: PathBuf,
        crypto: Option<SyncCrypto>,
    ) -> Result<Self> {
        Self::new_with_tables(
            db_path,
            cloud_dir,
            cache_dir,
            crypto,
            PRODUCTION_MERGE_TABLES,
        )
    }

    /// Same as [`SyncEngine::new`] with an explicit table registry (tests,
    /// schema variants).
    pub fn new_with_tables(
        db_path: &Path,
        cloud_dir: PathBuf,
        cache_dir: PathBuf,
        crypto: Option<SyncCrypto>,
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
            cache_dir,
            crypto,
            tables,
            last_published_data_version: None,
            pool_baseline_data_version,
            min_publish_interval: PUBLISH_MIN_INTERVAL,
            last_publish_at: None,
            force_first_flush: false,
        };
        engine.ensure_device_id()?;
        // Cross-process unshipped-write detection (audit F2, fixed
        // 2026-08-27): `PRAGMA data_version` cannot serve here — a fresh
        // WAL connection does not observe prior processes' commits (the
        // header change counter only advances at checkpoints; verified
        // empirically). Instead every publish persists a row-timestamp
        // watermark (max updated_at over the registry); a fresh engine
        // arms one forced flush when the current watermark is newer than
        // the shipped one — some row changed that no publish carried.
        // This replaces the old clean-exit marker, which missed the
        // "published cleanly, THEN wrote while the engine was gone" case
        // (confirmed live twice in real-device testing).
        let shipped = get_meta(&engine.conn, META_SHIPPED_ROW_WATERMARK)?;
        let watermark = row_watermark(&engine.conn, engine.tables);
        engine.force_first_flush = engine.current_version() != INITIAL_VERSION
            && shipped.as_deref().is_none_or(|seen| seen != watermark);
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

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Transport-encryption state, when this engine encrypts (ADR 04).
    /// Platforms read the (possibly slot-refreshed) chain back from here
    /// to persist their cache after passes.
    pub fn crypto(&self) -> Option<&SyncCrypto> {
        self.crypto.as_ref()
    }

    /// Replace the crypto state (e.g. after the platform unlocked the full
    /// chain via the passphrase ladder mid-process).
    pub fn set_crypto(&mut self, crypto: SyncCrypto) {
        self.crypto = Some(crypto);
    }

    fn ensure_device_id(&mut self) -> Result<()> {
        if get_meta(&self.conn, META_DEVICE_ID)?.is_none() {
            let id = uuid::Uuid::now_v7().to_string();
            set_meta(&self.conn, META_DEVICE_ID, &id)?;
        }
        if get_meta(&self.conn, META_SYNC_VERSION)?.is_none() {
            set_meta(&self.conn, META_SYNC_VERSION, INITIAL_VERSION)?;
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

    /// Manual escape hatch: publish unconditionally WITH a version bump,
    /// even when the engine believes everything is up-to-date. The lineage
    /// counter is the only merge ordering signal, and it can lie under
    /// concurrent publishers (real-device counter-strand finding
    /// 2026-08-28: a peer's rows became "stale remote" forever while every
    /// pass reported up-to-date). A forced bump re-ships this device's full
    /// state so stranded peers see it as newer. No data is destroyed — the
    /// merge is the same lossless union as every other publish.
    pub fn force_publish(&mut self) -> Result<()> {
        self.publish_inner(true)
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
        }
        match self.crypto.as_ref() {
            Some(crypto) => publish_snapshot_encrypted(
                &self.conn,
                &self.snapshot_path(),
                &self.cache_dir,
                crypto,
            )?,
            None => {
                publish_snapshot_compressed(&self.conn, &self.snapshot_path(), &self.cache_dir)?
            }
        }
        let seen = self.data_version()?;
        self.last_published_data_version = Some(seen);
        self.pool_baseline_data_version = seen;
        self.last_publish_at = Some(std::time::Instant::now());
        // Any successful publish ships everything this process knew about;
        // the construction-time suspicion is resolved either way. Persist
        // the shipped data_version so the NEXT engine process can detect
        // writes committed after this publish (audit F2).
        self.force_first_flush = false;
        set_meta(&self.conn, META_CLEAN_EXIT, "1")?;
        set_meta(
            &self.conn,
            META_SHIPPED_ROW_WATERMARK,
            &row_watermark(&self.conn, self.tables),
        )?;
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

    /// The UpToDate tail: nothing to pull, but local writes since the last
    /// publish must still ship (throttled — see `maybe_flush`).
    fn up_to_date_tail(&mut self) -> Result<SyncOutcome> {
        let flushed = self.maybe_flush()?;
        Ok(SyncOutcome {
            action: if flushed { "flushed" } else { "up-to-date" }.into(),
            rows_merged: 0,
            republished: flushed,
        })
    }

    pub fn sync_now(&mut self) -> Result<SyncOutcome> {
        if self.crypto.is_some() {
            let enc = encrypted_snapshot_name(&self.snapshot_path());
            if !enc.exists() {
                // First run in this cloud dir: publish so the other side
                // can see us.
                self.publish()?;
                return Ok(SyncOutcome {
                    action: "published-initial".into(),
                    ..Default::default()
                });
            }
            let header = sync_crypto::read_artifact_header(&enc).map_err(anyhow::Error::from)?;
            if let Some(crypto) = self.crypto.as_mut() {
                crypto.adopt_slot_from(&header);
            }
            let action = plan_sync_header(&self.conn, &header)?;
            match action {
                SyncAction::UpToDate => self.up_to_date_tail(),
                SyncAction::Adopt | SyncAction::Merge => {
                    let staged = self.stage_encrypted(&enc)?;
                    let outcome = self.merge_against_staged(&staged, Some(&header), action);
                    let _ = std::fs::remove_file(&staged);
                    outcome
                }
            }
        } else {
            let staged = self.stage_remote_snapshot()?;
            match staged {
                Some((snapshot, is_temp)) => {
                    let result = self.sync_against(&snapshot);
                    if is_temp {
                        let _ = std::fs::remove_file(&snapshot);
                    }
                    result
                }
                None => {
                    // First run on this device: publish so the other side
                    // can see us.
                    self.publish()?;
                    Ok(SyncOutcome {
                        action: "published-initial".into(),
                        ..Default::default()
                    })
                }
            }
        }
    }

    /// Decrypt + decompress the encrypted artifact into the app-private
    /// cache dir (never the OS temp dir). The staged file is removed by the
    /// caller after the pass.
    fn stage_encrypted(&self, enc: &Path) -> Result<PathBuf> {
        let Some(crypto) = self.crypto.as_ref() else {
            bail!("stage_encrypted without crypto state");
        };
        let bytes = std::fs::read(enc).context("read encrypted artifact")?;
        let (parsed, payload) = sync_crypto::split_artifact(&bytes).map_err(anyhow::Error::from)?;
        let zst = sync_crypto::decrypt_snapshot_payload(
            crypto.current_key(),
            &parsed.payload_salt,
            payload,
        )
        .map_err(anyhow::Error::from)?;
        std::fs::create_dir_all(&self.cache_dir)
            .with_context(|| format!("provision cache dir {}", self.cache_dir.display()))?;
        let plain = self.cache_dir.join(format!(
            "chatshell-staged-{}-{}.db",
            std::process::id(),
            uuid::Uuid::now_v7()
        ));
        let mut out = std::fs::File::create(&plain)
            .with_context(|| format!("create staged snapshot {}", plain.display()))?;
        zstd::stream::copy_decode(&zst[..], &mut out)
            .with_context(|| format!("zstd decompress into {}", plain.display()))?;
        Ok(plain)
    }

    /// Resolve the cloud artifact to a readable plain SQLite path. Prefers
    /// the compressed `snapshot.db.zst`; falls back to a legacy uncompressed
    /// `snapshot.db`. Decompressed copies land in the app-private cache dir
    /// (never the OS temp dir, ADR 04 §1) and are flagged so the caller
    /// removes them after the pass. `None` = no snapshot in the cloud dir
    /// yet. Debug/staging path — the encrypted transport plans from the
    /// header instead and stages via [`Self::stage_encrypted`].
    fn stage_remote_snapshot(&self) -> Result<Option<(PathBuf, bool)>> {
        let zst = compressed_snapshot_name(&self.snapshot_path());
        if zst.exists() {
            let plain = self.cache_dir.join(format!(
                "chatshell-snapshot-{}-{}.db",
                std::process::id(),
                uuid::Uuid::now_v7()
            ));
            std::fs::create_dir_all(&self.cache_dir).ok();
            decompress_file_zst(&zst, &plain)?;
            Ok(Some((plain, true)))
        } else if self.snapshot_path().exists() {
            Ok(Some((self.snapshot_path(), false)))
        } else {
            Ok(None)
        }
    }

    fn sync_against(&mut self, snapshot: &Path) -> Result<SyncOutcome> {
        let remote = Connection::open(snapshot)?;
        let action = plan_sync(&self.conn, &remote)?;
        drop(remote);

        match action {
            SyncAction::UpToDate => self.up_to_date_tail(),
            SyncAction::Adopt | SyncAction::Merge => {
                self.merge_against_staged(snapshot, None, action)
            }
        }
    }

    /// Adopt/Merge tail over an already-staged plaintext snapshot. In
    /// encrypted mode `header` is the parsed remote header and the staged
    /// DB's meta `sync_version` is asserted against it: a mismatch (bug,
    /// torn write) degrades to Merge — the lossless union — with a warning,
    /// never silence, because the post-merge version arithmetic below reads
    /// the meta value and a quiet disagreement would skew it (ADR 04 §1).
    fn merge_against_staged(
        &mut self,
        snapshot: &Path,
        header: Option<&ArtifactHeader>,
        _action: SyncAction,
    ) -> Result<SyncOutcome> {
        if let Some(header) = header {
            let staged = Connection::open(snapshot).ok();
            let staged_version = staged
                .as_ref()
                .and_then(|c| get_meta(c, META_SYNC_VERSION).ok().flatten())
                .unwrap_or_default();
            let staged_device = staged
                .as_ref()
                .and_then(|c| get_meta(c, META_DEVICE_ID).ok().flatten())
                .unwrap_or_default();
            let staged_hash = crate::crypto::blake3_hex(staged_device.as_bytes());
            let header_id = header
                .sync_version
                .split_once(':')
                .map(|(_, id)| id)
                .unwrap_or("");
            if counter_of(&staged_version) != counter_of(&header.sync_version)
                || staged_hash != header_id
            {
                tracing::warn!(
                    staged = %staged_version,
                    header = %header.sync_version,
                    "staged snapshot meta disagrees with the artifact header; \
                     merging losslessly (never silently)"
                );
            }
        }
        // Both verdicts share this body: adopt collapses into merge — the
        // same lossless union a degraded Merge needs.
        let report = merge_remote(&self.conn, snapshot, self.tables)?;
        let rows_merged = report.total_inserted() + report.total_updated();
        let remote_version = Connection::open(snapshot)
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
            // The mint above IS the version bump; mark the current state
            // as published-so-far so publish() republishes under THIS id
            // instead of bumping a second time (audit F4: a fresh process
            // merged once and shipped +2 counters).
            let seen = self.data_version()?;
            self.last_published_data_version = Some(seen);
            self.publish()?;
            Ok(SyncOutcome {
                action: "merged".into(),
                rows_merged,
                republished: true,
            })
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

    /// Plaintext (debug-staging) engine: no content key, legacy `.zst`
    /// artifacts, staged under the test space's own cache dir.
    fn plain_engine(db: &Path, cloud: PathBuf, cache: PathBuf) -> SyncEngine {
        SyncEngine::new_with_tables(db, cloud, cache, None, ENGINE_TABLES).unwrap()
    }

    /// The manual escape hatch for the counter-strand failure mode: a
    /// device whose rows were outranked by a peer's higher lineage counter
    /// reports up-to-date forever; `force_publish` bumps and re-ships so
    /// the stranded peer merges them (real-device finding 2026-08-28).
    #[test]
    fn force_publish_rescues_stranded_peer() {
        let space = dir("force-publish");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        // A publishes; B pulls and converges.
        seed(&a_path, "a1", "from A");
        let mut a = plain_engine(&a_path, cloud.clone(), space.path().join("cache-a"));
        a.sync_now().unwrap();
        seed(&b_path, "a1", "from A");
        let mut b = plain_engine(&b_path, cloud.clone(), space.path().join("cache-b"));
        assert_eq!(b.sync_now().unwrap().action, "converged");

        // A ships a second row in v2:a...
        a.connection()
            .execute(
                "INSERT INTO conversations(id, title, created_at, updated_at)
                 VALUES('a2', 'stranded', '2026-08-28T10:00:00+00:00',
                        '2026-08-28T10:00:00+00:00')",
                [],
            )
            .unwrap();
        a.force_publish().unwrap();

        // ...but B, unaware under propagation lag, flushes twice and its
        // higher counters (v2:b, v3:b) swallow A's lineage file-level.
        b.force_publish().unwrap();
        b.force_publish().unwrap();

        // Now A itself reports converged/up-to-date with a2 stranded: the
        // zero-row adopt branch takes B's version id without republishing,
        // and nothing B does ever pulls a2 in.
        assert_eq!(a.sync_now().unwrap().action, "converged");
        assert_eq!(b.sync_now().unwrap().action, "up-to-date");

        // The escape hatch: A force-publishes, B merges the stranded row.
        a.force_publish().unwrap();
        let out = b.sync_now().unwrap();
        assert_eq!(out.action, "merged");
        assert_eq!(out.rows_merged, 1, "stranded row must land: {out:?}");
        let n: i64 = b
            .connection()
            .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
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

        let mut a = plain_engine(&a_path, cloud.clone(), space.path().join("cache-a"));
        assert_eq!(a.sync_now().unwrap().action, "published-initial");

        let mut b = plain_engine(&b_path, cloud.clone(), space.path().join("cache-b"));
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

    /// Audit F2: writes committed while no engine existed (previous process
    /// after its last publish, or a harness) must ship on the next engine's
    /// first pass — the shipped-data-version marker detects them across
    /// process boundaries.
    #[test]
    fn cross_process_writes_ship_on_next_engine() {
        let space = dir("f2");
        let db = space.path().join("a.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();
        seed(&db, "f1", "first");

        // Process 1: publish (records the shipped data_version), exit.
        {
            let mut e = plain_engine(&db, cloud.clone(), space.path().join("c1"));
            assert_eq!(e.sync_now().unwrap().action, "published-initial");
        }
        let v1 = get_meta(&Connection::open(&db).unwrap(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();

        // Writes happen with NO engine alive (another connection commits).
        let pool = Connection::open(&db).unwrap();
        pool.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at)
             VALUES('f2', 'second', '2026-08-27T12:00:00+00:00', '2026-08-27T12:00:00+00:00')",
            [],
        )
        .unwrap();
        drop(pool);

        // Process 2: the first pass must force-publish the stranded write.
        let mut e2 = plain_engine(&db, cloud.clone(), space.path().join("c2"));
        let out = e2.sync_now().unwrap();
        assert!(
            out.republished,
            "stranded cross-process write must ship: {out:?}"
        );
        let v2 = get_meta(e2.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
        assert_ne!(v1, v2, "the forced flush must mint a new version");

        // The write is in the published artifact: a fresh reader merges it.
        let reader_db = space.path().join("reader.db");
        seed(&reader_db, "r1", "reader");
        let mut reader = plain_engine(&reader_db, cloud.clone(), space.path().join("c3"));
        let out_r = reader.sync_now().unwrap();
        assert!(out_r.rows_merged >= 1, "{out_r:?}");
        let n: i64 = reader
            .connection()
            .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
            .unwrap();
        assert!(n >= 3, "reader must see the stranded row: {n}");
    }

    /// Audit F4: a fresh process's first MERGED publish bumps the counter
    /// exactly once (the old behavior minted max+1 and then bumped again).
    #[test]
    fn merged_publish_bumps_counter_once() {
        let space = dir("f4");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();
        seed(&a_path, "a1", "from A");
        seed(&b_path, "b1", "from B");

        let mut a = plain_engine(&a_path, cloud.clone(), space.path().join("ca"));
        assert_eq!(a.sync_now().unwrap().action, "published-initial");
        let v_after_initial = counter_of(
            &get_meta(a.connection(), META_SYNC_VERSION)
                .unwrap()
                .unwrap(),
        );

        // B's first pass merges A's snapshot and republishes: exactly one
        // increment over the max it saw (was +2 before the fix).
        let mut b = plain_engine(&b_path, cloud.clone(), space.path().join("cb"));
        let out = b.sync_now().unwrap();
        assert!(out.rows_merged > 0 && out.republished);
        let v_b = counter_of(
            &get_meta(b.connection(), META_SYNC_VERSION)
                .unwrap()
                .unwrap(),
        );
        assert_eq!(
            v_b,
            v_after_initial + 1,
            "merged publish must bump the counter exactly once"
        );
    }

    #[test]
    fn cloud_artifact_is_compressed_and_peers_still_converge() {
        let space = dir("zst");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        seed(&b_path, "b1", "from B");

        let mut a = plain_engine(&a_path, cloud.clone(), space.path().join("cache-a"));
        assert_eq!(a.sync_now().unwrap().action, "published-initial");

        // The container carries only the compressed artifact.
        assert!(compressed_snapshot_name(&cloud.join(SNAPSHOT_FILE)).exists());
        assert!(!cloud.join(SNAPSHOT_FILE).exists());

        // A peer stages the .zst and merges it like any snapshot.
        let mut b = plain_engine(&b_path, cloud.clone(), space.path().join("cache-b"));
        let out = b.sync_now().unwrap();
        assert!(out.rows_merged > 0);
        let n: i64 = b
            .connection()
            .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
        // (No global "no staged leftovers" assertion here: staged copies are
        // UUID-named per pass, so a shared-temp-dir check races with parallel
        // engine tests.)
    }

    #[test]
    fn tick_flush_ships_pending_pool_writes_and_throttles() {
        let space = dir("flush");
        let db = space.path().join("a.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();
        seed(&db, "c1", "first");

        let mut engine = plain_engine(&db, cloud, space.path().join("cache"));
        engine.min_publish_interval = std::time::Duration::ZERO;
        assert_eq!(engine.sync_now().unwrap().action, "published-initial");

        // No pool writes since construction: the tick must stay a cheap
        // no-op (no republish, no version churn for peers to react to).
        let out = engine.sync_now().unwrap();
        assert_eq!(out.action, "up-to-date");
        assert!(!out.republished);

        // A pool commit since construction ships on the next tick under a
        // freshly minted version id.
        let version_before = get_meta(engine.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
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
        let version_after = get_meta(engine.connection(), META_SYNC_VERSION)
            .unwrap()
            .unwrap();
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
        let mut a = plain_engine(&a_path, cloud.clone(), space.path().join("cache-a"));
        assert_eq!(a.sync_now().unwrap().action, "published-initial");

        // B already holds that same row, so its first pass converges with an
        // empty merge report - but it also has a pending local write made
        // after engine construction (the app was running).
        seed(&b_path, "c1", "shared");
        let mut b = plain_engine(&b_path, cloud.clone(), space.path().join("cache-b"));
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
        let mut engine = plain_engine(&db, cloud, space.path().join("cache"));
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

    fn weak_crypto() -> sync_crypto::SyncCrypto {
        sync_crypto::SyncCrypto::bootstrap_with_params(
            "engine test passphrase",
            sync_crypto::ArgonParams::test(),
        )
        .unwrap()
    }

    fn encrypted_engine(
        db: &Path,
        cloud: PathBuf,
        cache: PathBuf,
        crypto: sync_crypto::SyncCrypto,
    ) -> SyncEngine {
        SyncEngine::new_with_tables(db, cloud, cache, Some(crypto), ENGINE_TABLES).unwrap()
    }

    #[test]
    fn encrypted_two_engines_converge() {
        let space = dir("enc");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        seed(&b_path, "b1", "from B");

        let group = weak_crypto();
        let mut a = encrypted_engine(
            &a_path,
            cloud.clone(),
            space.path().join("cache-a"),
            group.clone(),
        );
        assert_eq!(a.sync_now().unwrap().action, "published-initial");

        // Only the encrypted artifact exists in the container; the payload
        // is opaque ciphertext.
        let enc = encrypted_snapshot_name(&cloud.join(SNAPSHOT_FILE));
        assert!(enc.exists());
        assert!(!compressed_snapshot_name(&cloud.join(SNAPSHOT_FILE)).exists());
        assert!(!cloud.join(SNAPSHOT_FILE).exists());
        let raw = std::fs::read(&enc).unwrap();
        assert_eq!(&raw[..8], sync_crypto::ARTIFACT_MAGIC);
        assert!(
            !raw.windows(6).any(|w| w == b"from A"),
            "plaintext must not leak"
        );

        // A peer holding the same content key merges and republishes.
        let mut b = encrypted_engine(
            &b_path,
            cloud.clone(),
            space.path().join("cache-b"),
            group.clone(),
        );
        let out_b = b.sync_now().unwrap();
        assert!(out_b.rows_merged > 0 && out_b.republished);

        let out_a2 = a.sync_now().unwrap();
        assert!(out_a2.rows_merged > 0);
        let out_b2 = b.sync_now().unwrap();
        assert_eq!(out_b2.action, "converged");
        assert_eq!(a.sync_now().unwrap().action, "up-to-date");
        assert_eq!(b.sync_now().unwrap().action, "up-to-date");

        for conn in [&a, &b] {
            let n: i64 = conn
                .connection()
                .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
                .unwrap();
            assert_eq!(n, 2);
        }
        // No staged plaintext survives the pass in the cache dirs.
        for cache in ["cache-a", "cache-b"] {
            let leftovers: Vec<_> = std::fs::read_dir(space.path().join(cache))
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().contains("chatshell"))
                .collect();
            assert!(leftovers.is_empty(), "staged snapshots must be removed");
        }
    }

    #[test]
    fn encrypted_idle_tick_skips_staging_entirely() {
        // After convergence, an up-to-date tick plans from the header alone:
        // corrupting the payload must not affect it (decryption never runs).
        let space = dir("enc-idle");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        let group = weak_crypto();
        let mut a = encrypted_engine(
            &a_path,
            cloud.clone(),
            space.path().join("cache-a"),
            group.clone(),
        );
        a.sync_now().unwrap();

        seed(&b_path, "a1", "from A");
        let mut b = encrypted_engine(
            &b_path,
            cloud.clone(),
            space.path().join("cache-b"),
            group.clone(),
        );
        let out = b.sync_now().unwrap();
        assert_eq!(out.action, "converged");
        assert_eq!(b.sync_now().unwrap().action, "up-to-date");

        // Trash the payload bytes (keep the header region intact).
        let enc = encrypted_snapshot_name(&cloud.join(SNAPSHOT_FILE));
        let mut raw = std::fs::read(&enc).unwrap();
        let (_, offset) = sync_crypto::ArtifactHeader::parse(&raw).unwrap();
        for byte in &mut raw[offset..] {
            *byte ^= 0x5A;
        }
        std::fs::write(&enc, &raw).unwrap();

        // The idle tick still succeeds — it never touched the payload.
        assert_eq!(b.sync_now().unwrap().action, "up-to-date");
    }

    #[test]
    fn encrypted_wrong_key_is_an_error_never_plaintext() {
        let space = dir("enc-wrongkey");
        let a_path = space.path().join("a.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        let mut a = encrypted_engine(
            &a_path,
            cloud.clone(),
            space.path().join("cache-a"),
            weak_crypto(),
        );
        a.sync_now().unwrap();

        // A device holding a DIFFERENT content key plans from the header
        // (fine) but cannot decrypt the payload — deterministic error, no
        // silent fallback to treating the artifact as plaintext.
        let b_path = space.path().join("b.db");
        seed(&b_path, "b1", "from B");
        let mut b = encrypted_engine(
            &b_path,
            cloud.clone(),
            space.path().join("cache-b"),
            weak_crypto(), // same passphrase, independent mint → different CK
        );
        let err = b.sync_now().unwrap_err().to_string();
        assert!(
            err.contains("decryption failed"),
            "expected decrypt failure, got: {err}"
        );
    }

    #[test]
    fn foreign_magic_never_parses_as_plaintext() {
        let space = dir("enc-magic");
        let a_path = space.path().join("a.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        let mut a = encrypted_engine(
            &a_path,
            cloud.clone(),
            space.path().join("cache-a"),
            weak_crypto(),
        );
        a.sync_now().unwrap();

        // Overwrite with a plaintext-looking SQLite file under the
        // encrypted artifact name: the header parse must reject it.
        let enc = encrypted_snapshot_name(&cloud.join(SNAPSHOT_FILE));
        std::fs::write(&enc, b"SQLite format 3\0...junk...").unwrap();
        let err = a.sync_now().unwrap_err().to_string();
        assert!(
            err.contains("not an encrypted snapshot artifact"),
            "expected header failure, got: {err}"
        );
    }

    #[test]
    fn staged_meta_mismatch_degrades_to_lossless_merge() {
        // Simulate a torn write: the header's counter disagrees with the
        // staged DB's meta. The pass must still merge losslessly (rows from
        // both sides survive) instead of adopting skewed version math.
        let space = dir("enc-torn");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        seed(&a_path, "a1", "from A");
        let group = weak_crypto();
        let mut a = encrypted_engine(
            &a_path,
            cloud.clone(),
            space.path().join("cache-a"),
            group.clone(),
        );
        a.sync_now().unwrap();

        // Tamper ONLY the header's sync_version counter (same length).
        let enc = encrypted_snapshot_name(&cloud.join(SNAPSHOT_FILE));
        let raw = std::fs::read(&enc).unwrap();
        // The header carries the hashed form; rebuild it from A's meta.
        let version = hashed_sync_version(a.connection()).unwrap();
        let (counter, id) = version.split_once(':').unwrap();
        let bumped = format!("{}:{id}", counter.parse::<u64>().unwrap() + 8);
        let needle = version.as_bytes();
        let pos = raw
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("sync_version bytes in header");
        let mut tampered = raw.clone();
        tampered[pos..pos + needle.len()].copy_from_slice(bumped.as_bytes());
        std::fs::write(&enc, &tampered).unwrap();

        seed(&b_path, "b1", "from B");
        let mut b = encrypted_engine(
            &b_path,
            cloud.clone(),
            space.path().join("cache-b"),
            group.clone(),
        );
        let out = b.sync_now().unwrap();
        assert!(out.rows_merged > 0, "mismatch must still merge losslessly");
        let n: i64 = b
            .connection()
            .query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn engine_adopts_slot_for_keychain_adopter() {
        // Device B holds ONLY the CK (Keychain path), no wrapped slot; its
        // pass parses the remote header, adopts the slot, converges — and
        // can afterwards publish a complete header itself (the 2026-08-27
        // liveness finding, at engine level).
        let space = dir("enc-adopt");
        let a_path = space.path().join("a.db");
        let b_path = space.path().join("b.db");
        let cloud = space.path().join("cloud");
        std::fs::create_dir_all(&cloud).unwrap();

        // A bootstraps the group and publishes the first artifact.
        let crypto_a = weak_crypto();
        // (cloned below for B — B adopts A's CK via the Keychain path)
        seed(&a_path, "a1", "from A");
        let mut a = encrypted_engine(
            &a_path,
            cloud.clone(),
            space.path().join("cache-a"),
            crypto_a.clone(),
        );
        a.publish().unwrap();

        // B adopts ONLY A's CK (Keychain path), zero slots, and converges.
        let crypto_b =
            sync_crypto::SyncCrypto::from_keys(*crypto_a.current_key(), None, true).unwrap();
        seed(&b_path, "a1", "from A");
        let mut b = encrypted_engine(
            &b_path,
            cloud.clone(),
            space.path().join("cache-b"),
            crypto_b,
        );
        assert_eq!(b.sync_now().unwrap().action, "converged");

        let enc = encrypted_snapshot_name(&cloud.join(SNAPSHOT_FILE));
        let header = sync_crypto::read_artifact_header(&enc).unwrap();
        assert_eq!(header.key_version, 1);
        assert_eq!(header.slots.len(), 1);
        // The adopter's engine picked up the slot during the pass and can
        // now publish a complete header (B's row ships under a bumped id).
        b.connection()
            .execute(
                "INSERT INTO conversations(id, title, created_at, updated_at)
                 VALUES('b1', 'from B', '2026-08-28T11:00:00+00:00',
                        '2026-08-28T11:00:00+00:00')",
                [],
            )
            .unwrap();
        b.force_publish().unwrap();
        let header_after = sync_crypto::read_artifact_header(&enc).unwrap();
        assert_eq!(header_after.slots.len(), 1, "adopter publishes the slot");
    }

    #[test]
    fn failed_merge_rolls_back_and_detaches() {
        let space = dir("lockleak");
        let local_path = space.path().join("local.db");
        let remote_path = space.path().join("remote.db");

        let schema = r#"
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE owners (id TEXT PRIMARY KEY, updated_at TEXT NOT NULL);
CREATE TABLE items (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES owners(id),
    label TEXT NOT NULL, updated_at TEXT NOT NULL);
"#;
        const FK_TABLES: &[TableSpec] = &[t("owners"), t("items")];

        let local = Connection::open(&local_path).unwrap();
        local.pragma_update(None, "foreign_keys", "ON").unwrap();
        local.execute_batch(schema).unwrap();
        set_meta(&local, META_SYNC_VERSION, "1:a").unwrap();
        let remote = Connection::open(&remote_path).unwrap();
        // Bundled SQLite compiles with SQLITE_DEFAULT_FOREIGN_KEYS=1; the
        // authoring side must not enforce the constraint the merge violates.
        remote.pragma_update(None, "foreign_keys", "OFF").unwrap();
        remote.execute_batch(schema).unwrap();
        remote
            .execute(
                "INSERT INTO owners(id, updated_at) VALUES('o1', '2026-08-21T10:00:00+00:00')",
                [],
            )
            .unwrap();
        // Child row references an owner the snapshot never carries: the
        // items pass aborts mid-transaction on the immediate FK check.
        remote
            .execute(
                "INSERT INTO items(id, owner_id, label, updated_at)
                 VALUES('i1', 'missing', 'orphan', '2026-08-21T10:00:00+00:00')",
                [],
            )
            .unwrap();
        drop(remote);

        assert!(merge_remote(&local, &remote_path, FK_TABLES).is_err());

        // No partial state survived (the write transaction rolled back)...
        let n: i64 = local
            .query_row("SELECT COUNT(*) FROM owners", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0, "rollback must discard the parents-first inserts");
        // ...remote_db is gone (a second ATTACH would otherwise fail for the
        // process lifetime)...
        let attached: i64 = local
            .query_row(
                "SELECT COUNT(*) FROM pragma_database_list WHERE name = 'remote_db'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(attached, 0, "DETACH must succeed after the rollback");
        // ...and the connection's write lock is free again.
        local.execute_batch("BEGIN IMMEDIATE").unwrap();
        local.execute_batch("COMMIT").unwrap();
    }
}

#[cfg(test)]
mod column_intersection_tests {
    use super::*;

    fn init(conn: &Connection) {
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        conn.execute_batch(SCHEMA_SQL).unwrap();
        set_meta(conn, META_SYNC_VERSION, "1:base").unwrap();
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

    #[test]
    fn lww_compares_instants_across_timestamp_formats() {
        let dir = tempfile::tempdir().unwrap();
        let local_path = dir.path().join("local.db");
        let remote_path = dir.path().join("remote.db");

        let local = Connection::open(&local_path).unwrap();
        init(&local);
        conv(
            &local,
            "c1",
            "lexically larger, temporally older",
            "2026-08-21T10:00:05+00:00",
        );

        let remote = Connection::open(&remote_path).unwrap();
        remote.execute_batch(SCHEMA).unwrap();
        set_meta(&remote, META_SYNC_VERSION, "1:b").unwrap();
        conv(
            &remote,
            "c1",
            "lexically smaller, temporally newer",
            "2026-08-21T10:00:05.100Z",
        );
        drop(remote);

        let report = merge_remote(&local, &remote_path, TEST_TABLES).unwrap();
        assert_eq!(report.total_updated(), 1, "the later instant must win");
        assert_eq!(
            scalar(&local, "SELECT title FROM conversations WHERE id='c1'"),
            "lexically smaller, temporally newer"
        );
    }
}
