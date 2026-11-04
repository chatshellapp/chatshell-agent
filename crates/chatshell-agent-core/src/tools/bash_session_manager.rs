//! Manages per-conversation bash sessions.
//!
//! Each conversation gets its own persistent bash session handle so that
//! environment variables and working directory survive across agent turns
//! within the same conversation. Idle sessions are automatically swept
//! by a background task to prevent resource leaks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::bash::SharedBashSession;

pub(crate) type BashAbortHandle = Arc<tokio::sync::Notify>;

struct SessionEntry {
    session: SharedBashSession,
    abort_handle: BashAbortHandle,
    /// "Allow network this session" sandbox escalation flag.
    network_allowed: Arc<AtomicBool>,
    last_used_at: Instant,
}

impl SessionEntry {
    fn new() -> Self {
        Self {
            session: Arc::new(tokio::sync::Mutex::new(None)),
            abort_handle: Arc::new(tokio::sync::Notify::new()),
            network_allowed: Arc::new(AtomicBool::new(false)),
            last_used_at: Instant::now(),
        }
    }
}

#[allow(dead_code)]
pub struct BashSessionManager {
    sessions: Mutex<HashMap<String, SessionEntry>>,
}

impl Default for BashSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BashSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Get an existing session handle or create a new (empty) one for the given
    /// conversation. The actual `BashSession` inside is lazily created by `BashTool`
    /// on first command execution. Refreshes `last_used_at` on every access.
    ///
    /// Returns `(session, abort_handle)` so the caller can wire the abort signal
    /// into `BashTool` for cooperative cancellation.
    pub fn get_or_create(&self, conversation_id: &str) -> (SharedBashSession, BashAbortHandle) {
        let mut map = self.sessions.lock().unwrap();
        let entry = map
            .entry(conversation_id.to_string())
            .or_insert_with(SessionEntry::new);
        entry.last_used_at = Instant::now();
        (entry.session.clone(), entry.abort_handle.clone())
    }

    /// The conversation's "allow network this session" sandbox flag
    /// (created with the session entry, dropped with it).
    pub fn network_flag(&self, conversation_id: &str) -> Arc<AtomicBool> {
        let mut map = self.sessions.lock().unwrap();
        map.entry(conversation_id.to_string())
            .or_insert_with(SessionEntry::new)
            .network_allowed
            .clone()
    }

    /// Withdraw a network escalation (permission mode switched). The next
    /// command respawns the session without network.
    pub fn reset_network(&self, conversation_id: &str) {
        if let Some(entry) = self.sessions.lock().unwrap().get(conversation_id) {
            entry.network_allowed.store(false, Ordering::SeqCst);
        }
    }

    /// Signal the abort handle for the given conversation so that a running
    /// `BashTool::call()` can break out of `execute()` and release the mutex.
    pub fn abort_running(&self, conversation_id: &str) {
        let map = self.sessions.lock().unwrap();
        if let Some(entry) = map.get(conversation_id) {
            tracing::info!(
                "🛑 [bash] Aborting running command for conversation {}",
                conversation_id
            );
            entry.abort_handle.notify_waiters();
        }
    }

    /// Remove the session handle for a conversation. Any running bash process
    /// will be killed when the last `Arc` reference is dropped (via `BashSession::Drop`).
    pub fn remove(&self, conversation_id: &str) {
        let mut map = self.sessions.lock().unwrap();
        map.remove(conversation_id);
    }

    /// Remove sessions that have been idle longer than `max_idle` and kill their
    /// bash processes. Called periodically by a background sweep task.
    pub async fn sweep_idle(&self, max_idle: Duration) {
        let stale: Vec<(String, SharedBashSession)> = {
            let mut map = self.sessions.lock().unwrap();
            let now = Instant::now();
            let keys: Vec<String> = map
                .iter()
                .filter(|(_, e)| now.duration_since(e.last_used_at) > max_idle)
                .map(|(k, _)| k.clone())
                .collect();
            keys.into_iter()
                .filter_map(|k| map.remove(&k).map(|e| (k, e.session)))
                .collect()
        };

        for (conv_id, session) in stale {
            let mut guard = session.lock().await;
            if let Some(mut s) = guard.take() {
                s.kill().await;
                tracing::info!("🖥️ [bash] Swept idle session for conversation {}", conv_id);
            }
        }
    }

    /// Synchronous kill of all sessions. Called on app exit where async is unavailable.
    /// Uses `try_lock` to avoid blocking; sessions that are currently in use will be
    /// cleaned up by `BashSession::Drop` when the process exits.
    pub fn kill_all_sync(&self) {
        let mut map = self.sessions.lock().unwrap();
        for (conv_id, entry) in map.drain() {
            if let Ok(mut guard) = entry.session.try_lock()
                && let Some(ref mut s) = *guard
            {
                s.kill_sync();
                tracing::info!(
                    "🖥️ [bash] Killed session for conversation {} on shutdown",
                    conv_id
                );
            }
        }
    }

    /// Kill all sessions (async). Waits for each process to exit.
    #[allow(dead_code)]
    pub async fn kill_all(&self) {
        let all: Vec<(String, SharedBashSession)> = {
            let mut map = self.sessions.lock().unwrap();
            map.drain().map(|(k, e)| (k, e.session)).collect()
        };

        for (conv_id, session) in all {
            let mut guard = session.lock().await;
            if let Some(mut s) = guard.take() {
                s.kill().await;
                tracing::info!(
                    "🖥️ [bash] Killed session for conversation {} on shutdown",
                    conv_id
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::bash::{BashArgs, BashTool};
    use rig::agent::tool::{Tool as _, ToolContext};
    use std::sync::Arc;
    use std::time::Duration;

    /// Put a real live bash session into `handle` via the public path: one
    /// BashTool command lazily spawns the session in place.
    async fn put_live_session(handle: &SharedBashSession) {
        let cwd = std::env::temp_dir().to_string_lossy().to_string();
        let tool: BashTool = BashTool::with_session(handle.clone(), Some(cwd));
        let mut context = ToolContext::new();
        tool.call(
            &mut context,
            BashArgs {
                command: Some("true".to_string()),
                timeout: None,
                description: None,
                workdir: None,
            },
        )
        .await
        .unwrap();
    }

    #[test]
    fn same_conversation_shares_one_handle_pair() {
        let manager = BashSessionManager::new();
        let (session_a, abort_a) = manager.get_or_create("conv-1");
        let (session_b, abort_b) = manager.get_or_create("conv-1");
        assert!(Arc::ptr_eq(&session_a, &session_b));
        assert!(Arc::ptr_eq(&abort_a, &abort_b));

        let (session_c, _) = manager.get_or_create("conv-2");
        assert!(
            !Arc::ptr_eq(&session_a, &session_c),
            "conversations must not share sessions"
        );
    }

    #[test]
    fn network_flag_is_per_conversation_and_resettable() {
        let manager = BashSessionManager::new();
        let flag = manager.network_flag("conv-1");
        flag.store(true, Ordering::SeqCst);
        assert!(manager.network_flag("conv-1").load(Ordering::SeqCst));
        assert!(!manager.network_flag("conv-2").load(Ordering::SeqCst));
        manager.reset_network("conv-1");
        assert!(!flag.load(Ordering::SeqCst));
        flag.store(true, Ordering::SeqCst);
        manager.remove("conv-1");
        assert!(!manager.network_flag("conv-1").load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn abort_running_wakes_a_registered_waiter() {
        let manager = BashSessionManager::new();
        let (_, abort) = manager.get_or_create("conv-1");

        let notified = abort.notified();
        tokio::pin!(notified);
        // Enroll the future with the Notify before signalling: one poll inside
        // select registers interest, yield_now completes immediately after.
        tokio::select! {
            _ = &mut notified => panic!("woken before abort_running"),
            _ = tokio::task::yield_now() => {}
        }

        manager.abort_running("conv-1");

        tokio::time::timeout(Duration::from_millis(100), &mut notified)
            .await
            .expect("abort_running must wake registered waiters");
    }

    #[tokio::test]
    async fn remove_detaches_so_the_next_call_gets_a_fresh_session() {
        let manager = BashSessionManager::new();
        let (session, _) = manager.get_or_create("conv-1");

        manager.remove("conv-1");

        let (next, _) = manager.get_or_create("conv-1");
        assert!(!Arc::ptr_eq(&session, &next));
    }

    #[tokio::test]
    async fn sweep_idle_kills_stale_sessions_and_spares_fresh_ones() {
        let manager = BashSessionManager::new();
        let (stale, _) = manager.get_or_create("stale");
        put_live_session(&stale).await;
        // Age the stale entry past the idle budget; spawning the live fresh
        // session takes time too, so refresh its timestamp right before the
        // sweep via another get_or_create.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let (fresh, _) = manager.get_or_create("fresh");
        put_live_session(&fresh).await;
        let _ = manager.get_or_create("fresh");

        manager.sweep_idle(Duration::from_millis(10)).await;

        assert!(stale.lock().await.is_none(), "stale session must be killed");
        assert!(fresh.lock().await.is_some(), "fresh session must survive");
        let (after, _) = manager.get_or_create("stale");
        assert!(
            !Arc::ptr_eq(&stale, &after),
            "evicted entry is gone for good"
        );
    }

    #[tokio::test]
    async fn kill_all_terminates_every_tracked_session() {
        let manager = BashSessionManager::new();
        let handles: Vec<_> = ["a", "b"]
            .iter()
            .map(|c| manager.get_or_create(c).0)
            .collect();
        for handle in &handles {
            put_live_session(handle).await;
        }

        manager.kill_all().await;

        for handle in &handles {
            assert!(handle.lock().await.is_none());
        }
    }

    #[tokio::test]
    async fn kill_all_sync_drains_sessions_without_await() {
        let manager = BashSessionManager::new();
        let handles: Vec<_> = ["a", "b"]
            .iter()
            .map(|c| manager.get_or_create(c).0)
            .collect();
        for handle in &handles {
            put_live_session(handle).await;
        }

        manager.kill_all_sync();

        // kill_sync terminates the process but leaves the shell struct in the
        // handle (the process exit cleans it up); the manager's map itself
        // must be drained, so every follow-up get_or_create spawns fresh.
        for (index, conversation) in ["a", "b"].iter().enumerate() {
            let (next, _) = manager.get_or_create(conversation);
            assert!(
                !Arc::ptr_eq(&handles[index], &next),
                "entry for {conversation} must be gone from the map"
            );
        }
    }
}
