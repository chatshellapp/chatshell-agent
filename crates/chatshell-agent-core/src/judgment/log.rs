//! Local auto-mode decision log (AR-11).
//!
//! One row per judgment: what was judged, the verdict axes, where the
//! outcome came from, how long it took, which model and ruleset decided,
//! and the per-execution jail delta an allow authorized. The log exists
//! to measure the wrong-allow rate; the human-override column (set when
//! the user taps run-anyway on a denial) makes the wrong-deny rate
//! derivable too.
//!
//! Device-local, like the remembered-grant store: its own SQLite file,
//! never the canonical database, so the whole-database sync snapshot can
//! never carry it. Capped at [`MAX_DECISIONS`] rows; recording beyond the
//! cap evicts the oldest.

use super::JudgmentRequest;
use super::engine::{Review, ReviewOutcome};
use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum rows kept per device.
pub const MAX_DECISIONS: usize = 5_000;

/// File name of the log inside the app data directory.
pub const LOG_FILE_NAME: &str = "auto_review_decisions.db";

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    action TEXT NOT NULL,
    source TEXT NOT NULL,
    risk_level TEXT,
    user_authorization TEXT,
    outcome TEXT NOT NULL,
    outcome_source TEXT NOT NULL,
    rationale TEXT,
    duration_ms INTEGER NOT NULL,
    model TEXT NOT NULL,
    ruleset_version TEXT NOT NULL,
    jail_delta TEXT,
    overridden_at INTEGER
)";

/// Where a recorded outcome came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeSource {
    /// A verdict on the first attempt.
    Judgment,
    /// A verdict after the single transient-error retry.
    Retry,
    /// No verdict (provider unavailable or unparseable output).
    ReviewFailure,
    /// The stream ended while the review was in flight.
    Cancelled,
    /// The user ran a denied action anyway (set by [`DecisionLog::mark_override`]).
    HumanOverride,
}

impl OutcomeSource {
    pub fn as_str(self) -> &'static str {
        match self {
            OutcomeSource::Judgment => "judgment",
            OutcomeSource::Retry => "retry",
            OutcomeSource::ReviewFailure => "review_failure",
            OutcomeSource::Cancelled => "cancelled",
            OutcomeSource::HumanOverride => "human_override",
        }
    }
}

/// One decision row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRecord {
    /// UUID v7: time-ordered, so it doubles as the eviction tiebreak.
    pub id: String,
    pub created_at: i64,
    pub conversation_id: String,
    pub tool: String,
    /// The rendered evidence (compact action and facts).
    pub action: String,
    pub source: String,
    pub risk_level: Option<String>,
    pub user_authorization: Option<String>,
    /// `allow`, `deny`, `failed`, or `cancelled`.
    pub outcome: String,
    pub outcome_source: String,
    pub rationale: Option<String>,
    pub duration_ms: i64,
    pub model: String,
    pub ruleset_version: String,
    /// The per-execution jail delta an allow authorized (AR-14), if any.
    pub jail_delta: Option<String>,
    pub overridden_at: Option<i64>,
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

fn wire<T: serde::Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
}

impl DecisionRecord {
    /// The record for a completed review.
    pub fn from_review(
        conversation_id: &str,
        request: &JudgmentRequest,
        review: &Review,
        jail_delta: Option<String>,
    ) -> Self {
        let verdict = review.outcome.verdict();
        let (outcome, outcome_source) = match &review.outcome {
            ReviewOutcome::Allow { .. } | ReviewOutcome::Block { .. } => (
                if matches!(review.outcome, ReviewOutcome::Allow { .. }) {
                    "allow"
                } else {
                    "deny"
                },
                if review.attempts > 1 {
                    OutcomeSource::Retry
                } else {
                    OutcomeSource::Judgment
                },
            ),
            ReviewOutcome::Failed { .. } => ("failed", OutcomeSource::ReviewFailure),
            ReviewOutcome::Cancelled => ("cancelled", OutcomeSource::Cancelled),
        };
        let rationale = match &review.outcome {
            ReviewOutcome::Failed { .. } => review.outcome.model_reason(),
            _ => verdict.map(|v| v.rationale.clone()),
        };
        DecisionRecord {
            id: uuid::Uuid::now_v7().to_string(),
            created_at: now_millis(),
            conversation_id: conversation_id.to_string(),
            tool: request.evidence.tool.clone(),
            action: request.evidence.render(),
            source: wire(&request.source).unwrap_or_default(),
            risk_level: verdict.and_then(|v| wire(&v.risk_level)),
            user_authorization: verdict.and_then(|v| wire(&v.user_authorization)),
            outcome: outcome.to_string(),
            outcome_source: outcome_source.as_str().to_string(),
            rationale,
            duration_ms: review.duration.as_millis() as i64,
            model: review.model.clone(),
            ruleset_version: super::ruleset::RULESET_VERSION.to_string(),
            jail_delta,
            overridden_at: None,
        }
    }
}

/// Handle to the device-local decision log. Cheap to clone; every
/// operation opens its own short-lived connection.
#[derive(Debug, Clone)]
pub struct DecisionLog {
    path: PathBuf,
}

impl DecisionLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a decision, then evict the oldest rows beyond the cap.
    pub fn record(&self, record: &DecisionRecord) -> Result<()> {
        let mut conn = self.open_read_write()?;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO decisions (id, created_at, conversation_id, tool, action, source,
                 risk_level, user_authorization, outcome, outcome_source, rationale,
                 duration_ms, model, ruleset_version, jail_delta, overridden_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                record.id,
                record.created_at,
                record.conversation_id,
                record.tool,
                record.action,
                record.source,
                record.risk_level,
                record.user_authorization,
                record.outcome,
                record.outcome_source,
                record.rationale,
                record.duration_ms,
                record.model,
                record.ruleset_version,
                record.jail_delta,
                record.overridden_at,
            ],
        )?;
        tx.execute(
            "DELETE FROM decisions WHERE id NOT IN (
                 SELECT id FROM decisions ORDER BY created_at DESC, id DESC LIMIT ?1
             )",
            params![MAX_DECISIONS as i64],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Record that the user ran the decision's action anyway. Returns
    /// whether the row exists (it may have been evicted).
    pub fn mark_override(&self, id: &str) -> Result<bool> {
        if !self.path.exists() {
            return Ok(false);
        }
        let conn = self.open_read_write()?;
        let changed = conn.execute(
            "UPDATE decisions SET outcome_source = ?1, overridden_at = ?2 WHERE id = ?3",
            params![OutcomeSource::HumanOverride.as_str(), now_millis(), id],
        )?;
        Ok(changed > 0)
    }

    /// Number of rows.
    pub fn count(&self) -> Result<usize> {
        let Some(conn) = self.open_read_only()? else {
            return Ok(0);
        };
        let count: Option<i64> = conn
            .query_row("SELECT COUNT(*) FROM decisions", [], |row| row.get(0))
            .optional()
            .or_else(|err| no_table(err).map(|()| None))?;
        Ok(count.unwrap_or(0) as usize)
    }

    /// One row by id.
    pub fn get(&self, id: &str) -> Result<Option<DecisionRecord>> {
        let Some(conn) = self.open_read_only()? else {
            return Ok(None);
        };
        conn.query_row(
            "SELECT id, created_at, conversation_id, tool, action, source, risk_level,
                    user_authorization, outcome, outcome_source, rationale, duration_ms,
                    model, ruleset_version, jail_delta, overridden_at
             FROM decisions WHERE id = ?1",
            params![id],
            |row| {
                Ok(DecisionRecord {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    conversation_id: row.get(2)?,
                    tool: row.get(3)?,
                    action: row.get(4)?,
                    source: row.get(5)?,
                    risk_level: row.get(6)?,
                    user_authorization: row.get(7)?,
                    outcome: row.get(8)?,
                    outcome_source: row.get(9)?,
                    rationale: row.get(10)?,
                    duration_ms: row.get(11)?,
                    model: row.get(12)?,
                    ruleset_version: row.get(13)?,
                    jail_delta: row.get(14)?,
                    overridden_at: row.get(15)?,
                })
            },
        )
        .optional()
        .or_else(|err| no_table(err).map(|()| None))
        .context("query decision")
    }

    /// Forget every decision. Returns how many were removed.
    pub fn clear(&self) -> Result<usize> {
        if !self.path.exists() {
            return Ok(0);
        }
        let conn = self.open_read_write()?;
        Ok(conn.execute("DELETE FROM decisions", [])?)
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

/// A missing table reads as empty, not as an error.
fn no_table(err: rusqlite::Error) -> rusqlite::Result<()> {
    match &err {
        rusqlite::Error::SqliteFailure(_, Some(message)) if message.contains("no such table") => {
            Ok(())
        }
        _ => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judgment::engine::ReviewFailure;
    use crate::judgment::verdict::parse_verdict;
    use crate::judgment::{Evidence, JudgmentSource};
    use crate::permission::GateContext;
    use std::time::Duration;

    fn request() -> JudgmentRequest {
        JudgmentRequest {
            subject: "curl https://example.com".into(),
            source: JudgmentSource::Network,
            forced: None,
            evidence: Evidence::collect(
                "bash",
                &serde_json::json!({ "command": "curl https://example.com" }),
                &GateContext::default(),
            ),
        }
    }

    fn review(outcome: ReviewOutcome, attempts: u32) -> Review {
        Review {
            outcome,
            attempts,
            duration: Duration::from_millis(420),
            model: "mock-model".into(),
        }
    }

    fn deny() -> ReviewOutcome {
        ReviewOutcome::Block {
            verdict: parse_verdict(
                r#"{"risk_level":"high","user_authorization":"low","outcome":"deny","rationale":"Unrequested upload."}"#,
            )
            .unwrap(),
        }
    }

    #[test]
    fn records_every_field() {
        let dir = tempfile::tempdir().unwrap();
        let log = DecisionLog::new(dir.path().join(LOG_FILE_NAME));
        let record = DecisionRecord::from_review(
            "conv-1",
            &request(),
            &review(deny(), 1),
            Some("network for one execution".into()),
        );
        log.record(&record).unwrap();
        let stored = log.get(&record.id).unwrap().unwrap();
        assert_eq!(stored, record);
        assert_eq!(stored.tool, "bash");
        assert_eq!(stored.source, "network");
        assert_eq!(stored.risk_level.as_deref(), Some("high"));
        assert_eq!(stored.user_authorization.as_deref(), Some("low"));
        assert_eq!(stored.outcome, "deny");
        assert_eq!(stored.outcome_source, "judgment");
        assert_eq!(stored.duration_ms, 420);
        assert_eq!(stored.model, "mock-model");
        assert_eq!(
            stored.ruleset_version,
            super::super::ruleset::RULESET_VERSION
        );
        assert!(uuid::Uuid::parse_str(&stored.id).unwrap().get_version_num() == 7);
    }

    #[test]
    fn outcome_sources() {
        let r = request();
        let retry = DecisionRecord::from_review("c", &r, &review(deny(), 2), None);
        assert_eq!(retry.outcome_source, "retry");
        let failed = DecisionRecord::from_review(
            "c",
            &r,
            &review(
                ReviewOutcome::Failed {
                    failure: ReviewFailure::Unavailable("offline".into()),
                },
                2,
            ),
            None,
        );
        assert_eq!(failed.outcome, "failed");
        assert_eq!(failed.outcome_source, "review_failure");
        assert_eq!(failed.risk_level, None);
        assert!(failed.rationale.unwrap().contains("review failure"));
    }

    #[test]
    fn run_anyway_marks_the_correlated_row() {
        let dir = tempfile::tempdir().unwrap();
        let log = DecisionLog::new(dir.path().join(LOG_FILE_NAME));
        let record = DecisionRecord::from_review("c", &request(), &review(deny(), 1), None);
        log.record(&record).unwrap();
        assert!(log.mark_override(&record.id).unwrap());
        let stored = log.get(&record.id).unwrap().unwrap();
        assert_eq!(stored.outcome_source, "human_override");
        assert!(stored.overridden_at.is_some());
        // The original outcome stays: the row is now a measured wrong-deny
        // candidate.
        assert_eq!(stored.outcome, "deny");
        assert!(!log.mark_override("missing").unwrap());
    }

    #[test]
    fn the_row_beyond_the_cap_evicts_the_oldest() {
        let dir = tempfile::tempdir().unwrap();
        let log = DecisionLog::new(dir.path().join(LOG_FILE_NAME));
        let base = DecisionRecord::from_review("c", &request(), &review(deny(), 1), None);
        // Seed MAX rows in one transaction for speed.
        {
            let mut conn = log.open_read_write().unwrap();
            let tx = conn.transaction().unwrap();
            for n in 0..MAX_DECISIONS {
                tx.execute(
                    "INSERT INTO decisions (id, created_at, conversation_id, tool, action,
                         source, outcome, outcome_source, duration_ms, model, ruleset_version)
                     VALUES (?1, ?2, 'c', 'bash', '{}', 'network', 'deny', 'judgment', 1,
                             'm', 'v')",
                    params![format!("row-{n:05}"), n as i64],
                )
                .unwrap();
            }
            tx.commit().unwrap();
        }
        assert_eq!(log.count().unwrap(), MAX_DECISIONS);
        let newest = DecisionRecord {
            created_at: MAX_DECISIONS as i64 + 10,
            ..base
        };
        log.record(&newest).unwrap();
        assert_eq!(log.count().unwrap(), MAX_DECISIONS);
        assert!(log.get("row-00000").unwrap().is_none(), "oldest evicted");
        assert!(log.get("row-00001").unwrap().is_some());
        assert!(log.get(&newest.id).unwrap().is_some());
    }

    #[test]
    fn missing_file_reads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let log = DecisionLog::new(dir.path().join(LOG_FILE_NAME));
        assert_eq!(log.count().unwrap(), 0);
        assert_eq!(log.get("x").unwrap(), None);
        assert_eq!(log.clear().unwrap(), 0);
        assert!(!dir.path().join(LOG_FILE_NAME).exists());
    }
}
