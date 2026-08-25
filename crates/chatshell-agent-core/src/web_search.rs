//! Web search - Brave API + Yahoo/Baidu HTML parsing, shared across platforms.
//!
//! The fetching strategy differs per platform (desktop uses headless Chrome,
//! iOS uses WKWebView for bot-blocked sites, URLSession for APIs). This module
//! shares what can be shared:
//!
//! - **Brave Search API**: pure HTTP + JSON, fully shareable.
//! - **Yahoo/Baidu result parsing**: CSS-selector-based extraction from HTML
//!   strings. Each platform fetches the HTML its own way, then calls the
//!   shared parser. This ensures both platforms extract results identically.
//! - **Yahoo redirect URL resolution**: shared pure function.
//!
//! DuckDuckGo stays entirely platform-specific (WKWebView/headless Chrome for
//! bot-detection bypass, no shareable parsing).

use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// A single search result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Search via the Brave Search API. Requires an API key (free tier:
/// 2000 queries/month). Endpoint: `https://api.search.brave.com/res/v1/web/search`.
pub async fn search_brave(
    query: &str,
    api_key: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>, String> {
    let client = reqwest::Client::new();
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
        urlencoding::encode(query),
        max_results
    );

    let response = client
        .get(&url)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| format!("Brave request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Brave API returned status: {}", response.status()));
    }

    let json: BraveResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Brave response: {e}"))?;

    let results = json
        .web
        .and_then(|w| w.results)
        .unwrap_or_default()
        .into_iter()
        .take(max_results)
        .filter_map(|r| {
            if r.url.is_empty() {
                None
            } else {
                Some(SearchResult {
                    title: r.title.unwrap_or_default(),
                    url: r.url,
                    snippet: r.description.unwrap_or_default(),
                })
            }
        })
        .collect();

    Ok(results)
}

/// Parse Yahoo HTML search results. Pure function - the caller is responsible
/// for fetching the HTML (WKWebView on iOS, headless Chrome on desktop).
///
/// Yahoo serves several layouts, so we try a list of container selectors and
/// use the first that yields results.
pub fn parse_yahoo_results(html: &str, max: usize) -> Vec<SearchResult> {
    let document = Html::parse_document(html);

    let container_selectors = [
        "div.algo-sr",
        "div.dd.algo",
        "li.ov-a",
        "div.Sr",
        "div[data-uuid]",
        "div.algo",
        "li.reg.searchCenterMiddle",
    ];

    let title_selector = "h3 a, a.ac-algo, h3.title a, a[href]";
    let snippet_selector = "p, div.compText, span.fc-falcon";

    for container_str in &container_selectors {
        let Ok(container_sel) = Selector::parse(container_str) else {
            continue;
        };
        let containers: Vec<_> = document.select(&container_sel).collect();
        if containers.is_empty() {
            continue;
        }

        let Ok(title_sel) = Selector::parse(title_selector) else {
            continue;
        };
        let Ok(snippet_sel) = Selector::parse(snippet_selector) else {
            continue;
        };

        let mut results = Vec::new();
        for container in containers.into_iter().take(max * 2) {
            // Find the first link whose href looks usable.
            let title_el = container.select(&title_sel).find(|el| {
                let href = el.value().attr("href").unwrap_or("");
                href.starts_with("http") || href.contains("r.search.yahoo")
            });

            let Some(title_el) = title_el else { continue };

            let title = title_el.text().collect::<String>().trim().to_string();
            let href = title_el.value().attr("href").unwrap_or("").to_string();
            let resolved = resolve_yahoo_redirect_url(&href);

            if title.is_empty()
                || !resolved.starts_with("http")
                || resolved.contains("yahoo.com/search")
            {
                continue;
            }

            let snippet = container
                .select(&snippet_sel)
                .next()
                .map(|el| el.text().collect::<String>().trim().to_string())
                .unwrap_or_default();

            results.push(SearchResult {
                title,
                url: resolved,
                snippet,
            });
            if results.len() >= max {
                break;
            }
        }

        if !results.is_empty() {
            return results;
        }
    }

    Vec::new()
}

/// Parse Baidu HTML search results. Pure function.
///
/// Baidu's result containers are `div.result` or `div.c-container`; each has
/// an `h3 a` link (title + Baidu redirect URL) and a snippet. The `href` is a
/// `baidu.com/link?url=...` redirect - returned as-is (would need an extra
/// HTTP hop to resolve).
pub fn parse_baidu_results(html: &str, max: usize) -> Vec<SearchResult> {
    let document = Html::parse_document(html);

    let Ok(container_sel) = Selector::parse("div.result, div.c-container") else {
        return Vec::new();
    };
    let Ok(title_sel) = Selector::parse("h3 a, h3.c-title a") else {
        return Vec::new();
    };
    let Ok(snippet_sel) =
        Selector::parse("div.c-abstract, span.content-right_8Zs40, div.c-span-last")
    else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for container in document.select(&container_sel).take(max * 2) {
        let Some(title_el) = container.select(&title_sel).next() else {
            continue;
        };

        let title = title_el.text().collect::<String>().trim().to_string();
        let href = title_el.value().attr("href").unwrap_or("").to_string();

        if title.is_empty()
            || href.is_empty()
            || href.contains("baidu.com/sf")
            || href.contains("baike.baidu.com")
            || href.contains("tieba.baidu.com")
        {
            continue;
        }

        let snippet = container
            .select(&snippet_sel)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        results.push(SearchResult {
            title,
            url: href,
            snippet,
        });
        if results.len() >= max {
            break;
        }
    }

    results
}

/// Decode Yahoo's redirect wrapper:
/// `https://r.search.yahoo.com/_ylt=.../RU=https%3A%2F%2Fexample.com/RK=...`
/// -> `https://example.com`.
pub fn resolve_yahoo_redirect_url(href: &str) -> String {
    if href.contains("r.search.yahoo.com") || href.contains("/RU=") {
        if let Some(ru_start) = href.find("/RU=") {
            let after_ru = &href[ru_start + 4..];
            let encoded = match after_ru.find("/RK=") {
                Some(end) => &after_ru[..end],
                None => after_ru,
            };
            if let Ok(decoded) = urlencoding::decode(encoded) {
                if decoded.starts_with("http") {
                    return decoded.into_owned();
                }
            }
        }
    }
    href.to_string()
}

// ---------------------------------------------------------------------------
// Brave API response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWeb>,
}

#[derive(Deserialize)]
struct BraveWeb {
    results: Option<Vec<BraveResult>>,
}

#[derive(Deserialize)]
struct BraveResult {
    title: Option<String>,
    url: String,
    description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_yahoo_direct_url() {
        assert_eq!(
            resolve_yahoo_redirect_url("https://example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn resolve_yahoo_redirect_url_decodes() {
        let href = "https://r.search.yahoo.com/_ylt=abc/RU=https%3A%2F%2Fexample.com/RK=def";
        assert_eq!(resolve_yahoo_redirect_url(href), "https://example.com");
    }

    #[test]
    fn parse_yahoo_empty_html() {
        assert!(parse_yahoo_results("", 5).is_empty());
    }

    #[test]
    fn parse_baidu_empty_html() {
        assert!(parse_baidu_results("", 5).is_empty());
    }

    #[test]
    fn parse_baidu_extracts_results() {
        let html = r#"
        <div class="result">
            <h3><a href="https://example.com/test">Test Title</a></h3>
            <div class="c-abstract">Test snippet text</div>
        </div>
        "#;
        let results = parse_baidu_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Test Title");
        assert_eq!(results[0].url, "https://example.com/test");
        assert_eq!(results[0].snippet, "Test snippet text");
    }

    #[test]
    fn parse_baidu_skips_baike() {
        let html = r#"
        <div class="result">
            <h3><a href="https://baike.baidu.com/item/test">Baike</a></h3>
            <div class="c-abstract">Should be skipped</div>
        </div>
        "#;
        let results = parse_baidu_results(html, 5);
        assert!(results.is_empty());
    }
}
