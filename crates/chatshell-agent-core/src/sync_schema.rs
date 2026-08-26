//! Canonical sync schema DDL (ADR 01 snapshot sync; roadmap Phase 1c's
//! deferred "shared DDL as data in core" item).
//!
//! Why a shared constant: `sync::merge_remote` builds its `SELECT` list from
//! the **local** table's columns and applies it to the remote snapshot, and
//! every device merges snapshots produced by every other device. Bidirectional
//! merging therefore requires the same column-name set per table on all
//! devices — an extra local column breaks the next device's merge with "no
//! such column", a missing one silently drops data. This module is the single
//! source of that shape.
//!
//! Consumers:
//! - **chatshell-desktop** carries the authoritative migration ladder. Its
//!   drift-guard test asserts that a freshly migrated desktop database and
//!   [`ensure_sync_tables`] produce identical table/column sets - extending
//!   the desktop schema without updating this constant fails that test.
//! - **chatshell-ios** collapsed its historical V1-V8 GRDB ladder into a
//!   single migration that executes this DDL verbatim over FFI
//!   (`currentSyncSchemaSql()`); databases predating the collapse already
//!   hold the final shape and skip the migration untouched.
//!
//! The `meta` table is included (both sides need it for version bookkeeping);
//! the FTS5 virtual table and its shadow tables are not (never merged,
//! desktop-local search infra).
//!
//! Statement order is parents-first, mirroring `PRODUCTION_MERGE_TABLES`, so
//! the batch also works with `PRAGMA foreign_keys = ON`. Every statement is
//! idempotent (`IF NOT EXISTS`).

use anyhow::{Context, Result};
use rusqlite::Connection;

/// The canonical sync schema as one idempotent SQL batch.
///
/// Shape = the consolidated desktop ladder endpoint (user_version 11, which
/// collapsed the previously shipped v11-v15 steps): every synced table
/// carries `updated_at` LWW stamps and `deleted_at` tombstones,
/// mutable-with-natural-key tables carry their UNIQUE indexes, and the
/// formerly dual-representation columns (`messages.images`,
/// `messages.tool_calls`, `messages.reasoning_content`,
/// `providers.api_key_account`) are gone everywhere — tool calls, user
/// attachments, and reasoning persist only in their normalized tables
/// (`tool_calls`, `files` + `message_attachments`, `thinking_steps`).
///
/// Evolution: bump [`CANONICAL_SCHEMA_VERSION`] whenever this shape changes.
/// The stamp is diagnostic provenance, not a merge gate:
/// `crate::sync::merge_remote` logs a warning on mismatch but proceeds, so
/// additive changes keep degrading gracefully through column intersection
/// (the documented design: drift-guard tests are the primary contract,
/// intersection the runtime safety net). A future destructive migration can
/// promote the stamp to a hard check deliberately.
pub const SYNC_SCHEMA_SQL: &str = include_str!("sync_schema.sql");

/// Monotonic version of [`SYNC_SCHEMA_SQL`]'s shape, recorded in the `meta`
/// table under [`META_SCHEMA_VERSION`]. Bump on every shape change;
/// [`ensure_sync_tables`] stamps it into fresh databases.
pub const CANONICAL_SCHEMA_VERSION: i64 = 1;

/// `meta` key holding [`CANONICAL_SCHEMA_VERSION`].
pub const META_SCHEMA_VERSION: &str = "schema_version";

/// Apply [`SYNC_SCHEMA_SQL`] to `conn` and stamp
/// [`CANONICAL_SCHEMA_VERSION`] into `meta` when absent. Safe to call on a
/// database that already has any subset of the tables (`IF NOT EXISTS`
/// throughout); an existing stamp is never overwritten here - version
/// changes go through real migrations, not re-running this batch.
pub fn ensure_sync_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(SYNC_SCHEMA_SQL)
        .context("apply canonical sync schema")?;
    if crate::sync::get_meta(conn, META_SCHEMA_VERSION)?.is_none() {
        crate::sync::set_meta(
            conn,
            META_SCHEMA_VERSION,
            &CANONICAL_SCHEMA_VERSION.to_string(),
        )?;
    }
    Ok(())
}

/// Read the canonical-schema version stamped in `meta`, if any. `None`
/// means the database predates the stamping gate and is treated as
/// compatible by the merge path.
pub fn read_schema_version(conn: &Connection) -> Result<Option<i64>> {
    Ok(crate::sync::get_meta(conn, META_SCHEMA_VERSION)?.and_then(|v| v.parse().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{PRODUCTION_MERGE_TABLES, validate_registry};

    fn fresh() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory");
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        conn
    }

    #[test]
    fn fresh_schema_satisfies_the_merge_registry() {
        let conn = fresh();
        ensure_sync_tables(&conn).expect("apply schema");
        validate_registry(&conn, PRODUCTION_MERGE_TABLES)
            .expect("registry validation against the canonical schema");
    }

    #[test]
    fn schema_application_is_idempotent() {
        let conn = fresh();
        ensure_sync_tables(&conn).expect("first apply");
        ensure_sync_tables(&conn).expect("second apply");
        validate_registry(&conn, PRODUCTION_MERGE_TABLES).unwrap();
    }

    #[test]
    fn ensure_sync_tables_stamps_the_canonical_version() {
        let conn = fresh();
        ensure_sync_tables(&conn).unwrap();
        assert_eq!(
            read_schema_version(&conn).unwrap(),
            Some(CANONICAL_SCHEMA_VERSION),
            "fresh application must stamp CANONICAL_SCHEMA_VERSION into meta"
        );
    }

    #[test]
    fn stamping_never_clobbers_an_existing_version() {
        let conn = fresh();
        ensure_sync_tables(&conn).unwrap();
        crate::sync::set_meta(&conn, META_SCHEMA_VERSION, "99").unwrap();
        ensure_sync_tables(&conn).unwrap();
        assert_eq!(
            read_schema_version(&conn).unwrap(),
            Some(99),
            "re-running the batch is not a migration; existing stamps stay"
        );
    }

    #[test]
    fn schema_covers_every_registry_table_plus_meta() {
        let conn = fresh();
        ensure_sync_tables(&conn).unwrap();
        for spec in PRODUCTION_MERGE_TABLES {
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [spec.name],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1, "registry table {} missing", spec.name);
        }
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='meta'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "meta table missing");
    }

    /// The shape this module promises: a desktop-built snapshot merging into
    /// an `ensure_sync_tables` database must exercise every column the
    /// registry knows about. Seed one row per registry table here and merge a
    /// copy into a second canonical database — proves the SELECT lists line up
    /// (equal column sets) without any desktop involvement.
    #[test]
    fn canonical_schema_merges_into_itself() {
        let dir = std::env::temp_dir().join(format!(
            "chatshell-sync-schema-merge-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let snapshot = dir.join("snapshot.db");
        let _ = std::fs::remove_file(&snapshot);

        {
            let src = fresh();
            ensure_sync_tables(&src).unwrap();
            crate::sync::set_meta(&src, "sync_version", "5:desktop").unwrap();
            // Insert one row into every registry table with all columns
            // non-defaulted where cheap; ids are fixed UUIDv7-shaped strings.
            seed_row(
                &src,
                "conversations",
                "('id','t','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z',NULL)",
            );
            seed_row(
                &src,
                "messages",
                "('id','id','user',NULL,'c',NULL,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z',NULL)",
            );
            seed_row(
                &src,
                "providers",
                "('id','p','openai',NULL,NULL,NULL,NULL,1,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z',NULL)",
            );
            seed_row(
                &src,
                "models",
                "('id','m','id','m',NULL,0,0,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z',NULL)",
            );
            seed_row(
                &src,
                "conversation_participants",
                "('id','id','user','u',NULL,'member','active','2026-01-01T00:00:00Z',NULL,NULL,NULL,'2026-01-01T00:00:00Z',NULL)",
            );
            crate::sync::publish_snapshot(&src, &snapshot).unwrap();
        }

        let dst = fresh();
        ensure_sync_tables(&dst).unwrap();
        crate::sync::set_meta(&dst, "sync_version", "1:ios").unwrap();
        let report =
            crate::sync::merge_remote(&dst, &snapshot, PRODUCTION_MERGE_TABLES).expect("merge");
        assert_eq!(report.total_inserted(), 5, "seeded rows must arrive");
        let _ = std::fs::remove_file(&snapshot);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A payload-wiping tombstone written on one device (desktop's
    /// soft_delete module shape) must propagate to a peer through the
    /// mutable natural-key specs — including the flipped junction/participant
    /// tables — or deletions resurrect from peer snapshots.
    #[test]
    fn wiped_tombstones_propagate_to_peers() {
        let dir =
            std::env::temp_dir().join(format!("chatshell-sync-tombstone-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let snapshot = dir.join("snapshot.db");
        let _ = std::fs::remove_file(&snapshot);

        {
            let src = fresh();
            ensure_sync_tables(&src).unwrap();
            crate::sync::set_meta(&src, "sync_version", "1:a").unwrap();
            seed_row(
                &src,
                "conversations",
                "('id','t','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z',NULL)",
            );
            seed_row(
                &src,
                "messages",
                "('id','id','user',NULL,'secret content',NULL,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z',NULL)",
            );
            seed_row(
                &src,
                "conversation_participants",
                "('id','id','user','u',NULL,'member','active','2026-01-01T00:00:00Z',NULL,NULL,NULL,'2026-01-01T00:00:00Z',NULL)",
            );
            crate::sync::publish_snapshot(&src, &snapshot).unwrap();

            let dst = fresh();
            ensure_sync_tables(&dst).unwrap();
            crate::sync::set_meta(&dst, "sync_version", "1:b").unwrap();
            crate::sync::merge_remote(&dst, &snapshot, PRODUCTION_MERGE_TABLES).unwrap();

            // On the second device, remove the participant with a wiping
            // tombstone and republish.
            dst.execute(
                "UPDATE conversation_participants \
                 SET display_name = NULL, role = NULL, status = NULL, joined_at = '', \
                     left_at = NULL, last_read_at = NULL, metadata = NULL, \
                     deleted_at = '2026-08-24T00:00:00Z', updated_at = '2026-08-24T00:00:00Z' \
                 WHERE id = 'id'",
                [],
            )
            .unwrap();
            crate::sync::set_meta(&dst, "sync_version", "2:b").unwrap();
            crate::sync::publish_snapshot(&dst, &snapshot).unwrap();

            // Back on the first device the removal must arrive, not resurrect.
            crate::sync::merge_remote(&src, &snapshot, PRODUCTION_MERGE_TABLES).unwrap();
            let (deleted, role): (Option<String>, Option<String>) = src
                .query_row(
                    "SELECT deleted_at, role FROM conversation_participants WHERE id = 'id'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(
                deleted.as_deref(),
                Some("2026-08-24T00:00:00Z"),
                "tombstone must propagate to the peer"
            );
            assert!(role.is_none(), "wiped payload must not resurrect content");
        }

        let _ = std::fs::remove_file(&snapshot);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn seed_row(conn: &Connection, table: &str, values: &str) {
        conn.execute(&format!("INSERT INTO {table} VALUES {values}"), [])
            .unwrap_or_else(|e| panic!("seed {table}: {e}"));
    }
}
