//! Bounded, cursor-following list operations.
//!
//! MCP list requests (`tools/list`, `resources/list`, ...) are paginated with
//! an opaque `nextCursor`. A server is untrusted input, so every loop here is
//! bounded: a page cap, an item cap, a cursor-size cap, and cursor dedup (a
//! repeated cursor ends the loop instead of spinning forever).

use std::collections::HashSet;
use std::future::Future;

use anyhow::{Context, Result};
use rmcp::RoleClient;
use rmcp::model::{PaginatedRequestParams, Resource, ResourceTemplate, Tool};
use rmcp::service::Peer;

/// Maximum number of pages fetched for one listing.
pub const MAX_LIST_PAGES: usize = 100;
/// Maximum number of items kept for one listing (per server).
pub const MAX_LIST_ITEMS: usize = 2048;
/// Maximum accepted cursor length in bytes.
pub const MAX_CURSOR_BYTES: usize = 64 * 1024;

/// Why a paginated listing stopped before the server said it was done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListTruncation {
    PageLimit,
    ItemLimit,
    RepeatedCursor,
    OversizedCursor,
}

/// Items assembled from every page, plus the reason the loop stopped early
/// (if it did).
#[derive(Debug)]
pub struct Paginated<T> {
    pub items: Vec<T>,
    pub truncated: Option<ListTruncation>,
}

/// Follow `nextCursor` until the server stops paging or a guard trips.
///
/// `fetch` receives the cursor for the next page (`None` for the first one)
/// and returns that page's items and its `nextCursor`. A guard tripping is not
/// an error: the items gathered so far are returned with the reason.
pub async fn paginate<T, F, Fut>(mut fetch: F) -> Result<Paginated<T>>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: Future<Output = Result<(Vec<T>, Option<String>)>>,
{
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor: Option<String> = None;

    for _ in 0..MAX_LIST_PAGES {
        let (page, next) = fetch(cursor.take()).await?;
        items.extend(page);
        if items.len() >= MAX_LIST_ITEMS {
            let overflowed = items.len() > MAX_LIST_ITEMS || next.is_some();
            items.truncate(MAX_LIST_ITEMS);
            return Ok(Paginated {
                items,
                truncated: overflowed.then_some(ListTruncation::ItemLimit),
            });
        }
        let Some(next) = next else {
            return Ok(Paginated {
                items,
                truncated: None,
            });
        };
        if next.len() > MAX_CURSOR_BYTES {
            return Ok(Paginated {
                items,
                truncated: Some(ListTruncation::OversizedCursor),
            });
        }
        if !seen.insert(next.clone()) {
            return Ok(Paginated {
                items,
                truncated: Some(ListTruncation::RepeatedCursor),
            });
        }
        cursor = Some(next);
    }

    Ok(Paginated {
        items,
        truncated: Some(ListTruncation::PageLimit),
    })
}

fn log_truncation(what: &str, server: &str, truncated: Option<ListTruncation>, count: usize) {
    if let Some(reason) = truncated {
        tracing::warn!(
            "MCP server '{}' {} listing stopped early ({:?}); keeping {} item(s)",
            server,
            what,
            reason,
            count
        );
    }
}

/// List every tool of a server, following pagination with the guards above.
/// `server` only labels log lines.
pub async fn list_all_tools_bounded(peer: &Peer<RoleClient>, server: &str) -> Result<Vec<Tool>> {
    let result = paginate(|cursor| async move {
        let page = peer
            .list_tools(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .context("Failed to list tools from MCP server")?;
        Ok((page.tools, page.next_cursor))
    })
    .await?;
    log_truncation("tool", server, result.truncated, result.items.len());
    Ok(result.items)
}

/// Whether the server advertised the `resources` capability at
/// initialization (servers without it answer resource requests with
/// "method not found").
pub fn supports_resources(peer: &Peer<RoleClient>) -> bool {
    peer.peer_info()
        .is_some_and(|info| info.capabilities.resources.is_some())
}

/// List every resource of a server with the same guards as tools (INV-3).
pub async fn list_all_resources_bounded(
    peer: &Peer<RoleClient>,
    server: &str,
) -> Result<Vec<Resource>> {
    let result = paginate(|cursor| async move {
        let page = peer
            .list_resources(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .context("Failed to list resources from MCP server")?;
        Ok((page.resources, page.next_cursor))
    })
    .await?;
    log_truncation("resource", server, result.truncated, result.items.len());
    Ok(result.items)
}

/// List every resource template of a server with the same guards.
pub async fn list_all_resource_templates_bounded(
    peer: &Peer<RoleClient>,
    server: &str,
) -> Result<Vec<ResourceTemplate>> {
    let result = paginate(|cursor| async move {
        let page = peer
            .list_resource_templates(Some(PaginatedRequestParams::default().with_cursor(cursor)))
            .await
            .context("Failed to list resource templates from MCP server")?;
        Ok((page.resource_templates, page.next_cursor))
    })
    .await?;
    log_truncation(
        "resource template",
        server,
        result.truncated,
        result.items.len(),
    );
    Ok(result.items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn assembles_every_page_in_order() {
        let pages = [
            (vec![1, 2], Some("a".to_string())),
            (vec![3], Some("b".to_string())),
            (vec![4, 5], None),
        ];
        let calls = Arc::new(AtomicUsize::new(0));
        let result = paginate(|cursor| {
            let i = calls.fetch_add(1, Ordering::SeqCst);
            let expected = [None, Some("a"), Some("b")][i].map(str::to_string);
            let page = pages[i].clone();
            async move {
                assert_eq!(cursor, expected);
                Ok(page)
            }
        })
        .await
        .unwrap();
        assert_eq!(result.items, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.truncated, None);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn repeated_cursor_terminates() {
        let calls = Arc::new(AtomicUsize::new(0));
        let result = paginate(|_| {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Ok((vec![0u8], Some("same".to_string()))) }
        })
        .await
        .unwrap();
        assert_eq!(result.truncated, Some(ListTruncation::RepeatedCursor));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(result.items.len(), 2);
    }

    #[tokio::test]
    async fn page_limit_terminates_endless_distinct_cursors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let result = paginate(|_| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            async move { Ok((Vec::<u8>::new(), Some(format!("c{n}")))) }
        })
        .await
        .unwrap();
        assert_eq!(result.truncated, Some(ListTruncation::PageLimit));
        assert_eq!(calls.load(Ordering::SeqCst), MAX_LIST_PAGES);
    }

    #[tokio::test]
    async fn item_limit_truncates() {
        let calls = Arc::new(AtomicUsize::new(0));
        let result = paginate(|_| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            async move { Ok((vec![0u8; 1000], Some(format!("c{n}")))) }
        })
        .await
        .unwrap();
        assert_eq!(result.items.len(), MAX_LIST_ITEMS);
        assert_eq!(result.truncated, Some(ListTruncation::ItemLimit));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn exactly_item_limit_on_last_page_is_not_truncated() {
        let result = paginate(|_| async { Ok((vec![0u8; MAX_LIST_ITEMS], None)) })
            .await
            .unwrap();
        assert_eq!(result.items.len(), MAX_LIST_ITEMS);
        assert_eq!(result.truncated, None);
    }

    #[tokio::test]
    async fn oversized_cursor_terminates() {
        let result =
            paginate(|_| async { Ok((vec![1u8], Some("x".repeat(MAX_CURSOR_BYTES + 1)))) })
                .await
                .unwrap();
        assert_eq!(result.items, vec![1]);
        assert_eq!(result.truncated, Some(ListTruncation::OversizedCursor));
    }

    #[tokio::test]
    async fn fetch_error_propagates() {
        let result: Result<Paginated<u8>> =
            paginate(|_| async { Err(anyhow::anyhow!("boom")) }).await;
        assert!(result.is_err());
    }
}
