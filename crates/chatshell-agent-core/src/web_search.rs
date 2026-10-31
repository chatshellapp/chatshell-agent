//! Web search - Brave API + Yahoo/Baidu/Bing HTML parsing, shared across platforms.
//!
//! The fetching strategy differs per platform (desktop uses headless Chrome,
//! iOS uses WKWebView for bot-blocked sites, URLSession for APIs). This module
//! shares what can be shared:
//!
//! - **Brave Search API**: pure HTTP + JSON, fully shareable.
//! - **Yahoo/Baidu/Bing result parsing**: CSS-selector-based extraction from HTML
//!   strings. Each platform fetches the HTML its own way, then calls the
//!   shared parser. This ensures both platforms extract results identically.
//! - **Yahoo/Bing redirect URL resolution**: shared pure functions.
//!
//! DuckDuckGo's *fetching* stays platform-specific (WKWebView / headless
//! Chrome for bot-detection bypass), but its result parsing and redirect
//! resolution are shared here like Yahoo's.

use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use url::form_urlencoded;

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
    if (href.contains("r.search.yahoo.com") || href.contains("/RU="))
        && let Some(ru_start) = href.find("/RU=")
    {
        let after_ru = &href[ru_start + 4..];
        let encoded = match after_ru.find("/RK=") {
            Some(end) => &after_ru[..end],
            None => after_ru,
        };
        if let Ok(decoded) = urlencoding::decode(encoded)
            && decoded.starts_with("http")
        {
            return decoded.into_owned();
        }
    }
    href.to_string()
}

/// Parse DuckDuckGo HTML search results. Pure function - the caller fetches
/// the HTML (WKWebView on iOS, headless Chrome on desktop).
///
/// DuckDuckGo's HTML endpoint wraps each result in a `.result` div with a
/// `.result__a` title link and a `.result__snippet` anchor.
pub fn parse_duckduckgo_results(html: &str, max: usize) -> Vec<SearchResult> {
    let document = Html::parse_document(html);

    let Ok(result_sel) = Selector::parse(".result") else {
        return Vec::new();
    };
    let Ok(title_sel) = Selector::parse(".result__a") else {
        return Vec::new();
    };
    let Ok(snippet_sel) = Selector::parse(".result__snippet") else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for result_el in document.select(&result_sel).take(max) {
        let Some(title_el) = result_el.select(&title_sel).next() else {
            continue;
        };

        let title = title_el.text().collect::<String>().trim().to_string();
        let href = title_el.value().attr("href").unwrap_or_default();
        let url = resolve_duckduckgo_redirect_url(href);

        let snippet = result_el
            .select(&snippet_sel)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        if !title.is_empty() && !url.is_empty() && url.starts_with("http") {
            results.push(SearchResult {
                title,
                url,
                snippet,
            });
        }
    }

    results
}

/// Resolve DuckDuckGo's redirect wrapper to the actual URL:
/// `//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...` ->
/// `https://example.com`. Protocol-relative URLs gain an `https:` scheme.
pub fn resolve_duckduckgo_redirect_url(href: &str) -> String {
    if (href.contains("duckduckgo.com/l/") || href.contains("uddg="))
        && let Some(uddg_start) = href.find("uddg=")
    {
        let url_part = &href[uddg_start + 5..];
        let url_end = url_part.find('&').unwrap_or(url_part.len());
        let encoded_url = &url_part[..url_end];

        // Decode via form-urlencoded pairs, then fall back to direct
        // percent decoding for values that are not k=v shaped.
        let decoded: String = form_urlencoded::parse(encoded_url.as_bytes())
            .map(|(k, v)| {
                if v.is_empty() {
                    k.to_string()
                } else {
                    format!("{}={}", k, v)
                }
            })
            .collect::<Vec<_>>()
            .join("");
        if decoded.starts_with("http") {
            return decoded;
        }
        if let Ok(parsed) = url::Url::parse(&format!("http://x?x={}", encoded_url))
            && let Some(value) = parsed
                .query_pairs()
                .find(|(k, _)| k == "x")
                .map(|(_, v)| v.to_string())
            && value.starts_with("http")
        {
            return value;
        }
        let decoded = percent_decode_str(encoded_url);
        if decoded.starts_with("http") {
            return decoded;
        }
    }

    if href.starts_with("//") {
        return format!("https:{}", href);
    }

    href.to_string()
}

/// Bing's decorative slug-icon class (`span.algoSlug_icon`, the "Web" label
/// next to captions); its text is not part of the source's summary.
const BING_SLUG_ICON_CLASS: &str = "algoSlug_icon";

/// Parse Bing HTML search results. Pure function - the caller fetches the
/// HTML (system WebView or headless Chrome on desktop, WKWebView on iOS).
///
/// Bing serves organic results as `ol#b_results > li.b_algo`; each has an
/// `h2 > a` title link (usually a `bing.com/ck/a?...&u=a1<base64url>`
/// redirect) and a caption that can span several paragraphs. Query params
/// like `mkt`/`cc` deliberately stay off the request URL: they make Bing
/// return unrelated results (see SearXNG's Bing traits).
pub fn parse_bing_results(html: &str, max: usize) -> Vec<SearchResult> {
    let document = Html::parse_document(html);

    let Ok(container_sel) = Selector::parse("ol#b_results > li.b_algo") else {
        return Vec::new();
    };
    let Ok(title_sel) = Selector::parse("h2 > a") else {
        return Vec::new();
    };
    let Ok(paragraph_sel) = Selector::parse("p") else {
        return Vec::new();
    };
    let Ok(slug_sel) = Selector::parse(".b_algoSlug") else {
        return Vec::new();
    };
    let Ok(icon_sel) = Selector::parse(&format!("span.{BING_SLUG_ICON_CLASS}")) else {
        return Vec::new();
    };

    let mut seen_urls = std::collections::HashSet::new();
    let mut results = Vec::new();
    for container in document.select(&container_sel).take(max * 2) {
        let Some(title_el) = container.select(&title_sel).next() else {
            continue;
        };

        let title = title_el.text().collect::<String>().trim().to_string();
        let href = title_el.value().attr("href").unwrap_or_default();
        let url = resolve_bing_redirect_url(href);

        // Bing repeats links across re-ranked slots; keep the first.
        if title.is_empty() || !url.starts_with("http") || !seen_urls.insert(url.clone()) {
            continue;
        }

        // Bing can split a caption across several paragraphs; slug-icon
        // labels ("Web") are decorative and excluded from the summary text.
        let mut snippet_parts: Vec<String> = container
            .select(&paragraph_sel)
            .map(|p| {
                if p.select(&icon_sel).next().is_none() {
                    p.text().collect::<String>()
                } else {
                    bing_text_excluding_slug_icons(p)
                }
            })
            .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|text| !text.is_empty())
            .collect();
        if snippet_parts.is_empty() {
            snippet_parts = container
                .select(&slug_sel)
                .next()
                .map(|el| bing_text_excluding_slug_icons(el))
                .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
                .into_iter()
                .collect();
        }

        results.push(SearchResult {
            title,
            url,
            snippet: snippet_parts.join(" "),
        });
        if results.len() >= max {
            break;
        }
    }

    results
}

/// Resolve Bing's `/ck/a` tracking redirect to the actual URL:
/// `https://www.bing.com/ck/a?!&&p=...&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbQ&ntb=1`
/// -> `https://example.com`, where `u` is `a1` + base64url-encoded target.
/// Malformed wrappers fall through to the input so one bad link cannot
/// discard the surrounding results. Protocol-relative URLs gain an
/// `https:` scheme.
pub fn resolve_bing_redirect_url(href: &str) -> String {
    let trimmed = href.trim();
    let parsed = if trimmed.starts_with("http") {
        url::Url::parse(trimmed).ok()
    } else {
        // Bing also emits absolute-path wrappers (`/ck/a?...`).
        url::Url::parse("https://www.bing.com")
            .ok()
            .and_then(|base| base.join(trimmed).ok())
    };

    if let Some(url) = parsed
        && let Some(host) = url.host_str()
        && (host == "bing.com" || host.ends_with(".bing.com"))
        && matches!(url.path(), "/ck/a" | "/ck/a/")
        && let Some((_, u)) = url.query_pairs().find(|(k, _)| k == "u")
        && let Some(decoded) = u.strip_prefix("a1").and_then(decode_base64url_utf8)
        && decoded.starts_with("http")
    {
        return decoded;
    }

    if trimmed.starts_with("//") {
        return format!("https:{}", trimmed);
    }

    trimmed.to_string()
}

/// Decode an unpadded base64url string into UTF-8 text. Bing omits padding.
fn decode_base64url_utf8(input: &str) -> Option<String> {
    use base64::Engine as _;

    let mut padded = input.to_string();
    match padded.len() % 4 {
        0 => {}
        2 => padded.push_str("=="),
        3 => padded.push('='),
        // A length of 1 modulo 4 cannot be valid base64.
        _ => return None,
    }
    base64::engine::general_purpose::URL_SAFE
        .decode(padded.as_bytes())
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
}

/// Collect an element's text, skipping Bing slug-icon subtrees. Mirrors
/// kelivo's DOM removal of `span.algoSlug_icon` before reading captions.
fn bing_text_excluding_slug_icons(element: scraper::ElementRef) -> String {
    fn collect(element: scraper::ElementRef, out: &mut String) {
        for child in element.children() {
            match child.value() {
                scraper::Node::Text(text) => out.push_str(text),
                scraper::Node::Element(el) => {
                    let is_icon = el.name() == "span"
                        && el.attr("class").is_some_and(|classes| {
                            classes
                                .split_whitespace()
                                .any(|c| c == BING_SLUG_ICON_CLASS)
                        });
                    if !is_icon && let Some(child_el) = scraper::ElementRef::wrap(child) {
                        collect(child_el, out);
                    }
                }
                _ => {}
            }
        }
    }

    let mut out = String::new();
    collect(element, &mut out);
    out
}

/// Manual percent decoding for URL strings.
fn percent_decode_str(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if hex.len() == 2
                && let Ok(byte) = u8::from_str_radix(&hex, 16)
            {
                result.push(byte as char);
                continue;
            }
            result.push('%');
            result.push_str(&hex);
        } else {
            result.push(c);
        }
    }

    result
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
    #[test]
    fn resolve_duckduckgo_redirect_url_decodes() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc";
        assert_eq!(
            resolve_duckduckgo_redirect_url(href),
            "https://example.com/page"
        );
    }

    #[test]
    fn resolve_duckduckgo_redirect_url_protocol_relative() {
        assert_eq!(
            resolve_duckduckgo_redirect_url("//example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn resolve_duckduckgo_redirect_url_direct() {
        assert_eq!(
            resolve_duckduckgo_redirect_url("https://example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn parse_duckduckgo_extracts_results() {
        let html = r#"
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftest&rut=x">Test Title</a>
            <a class="result__snippet">Test snippet</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://example.org/direct">Direct Link</a>
            <a class="result__snippet">Org snippet</a>
        </div>
        "#;
        let results = parse_duckduckgo_results(html, 5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Test Title");
        assert_eq!(results[0].url, "https://example.com/test");
        assert_eq!(results[0].snippet, "Test snippet");
        assert_eq!(results[1].url, "https://example.org/direct");
    }

    #[test]
    fn parse_duckduckgo_respects_max() {
        let html = r#"
        <div class="result"><a class="result__a" href="https://a.com">A</a></div>
        <div class="result"><a class="result__a" href="https://b.com">B</a></div>
        <div class="result"><a class="result__a" href="https://c.com">C</a></div>
        "#;
        assert_eq!(parse_duckduckgo_results(html, 2).len(), 2);
    }

    #[test]
    fn parse_bing_empty_html() {
        assert!(parse_bing_results("", 5).is_empty());
    }

    #[test]
    fn parse_bing_extracts_results_and_decodes_redirects() {
        let html = r#"
        <ol id="b_results">
            <li class="b_algo">
                <h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=abc&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9wYWdl&amp;ntb=1">Example Title</a></h2>
                <div class="b_caption"><p>First part of the caption.</p><p>Second part.</p></div>
            </li>
            <li class="b_algo">
                <h2><a href="https://www.rust-lang.org/learn">Direct Link</a></h2>
                <p class="b_lineclamp2 b_algoSlug"><span class="algoSlug_icon">Web</span><svg></svg>Learn Rust with example projects.</p>
            </li>
            <li class="b_ans">Not an organic result</li>
        </ol>
        "#;
        let results = parse_bing_results(html, 5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Example Title");
        assert_eq!(results[0].url, "https://example.com/page");
        assert_eq!(
            results[0].snippet,
            "First part of the caption. Second part."
        );
        assert_eq!(results[1].title, "Direct Link");
        assert_eq!(results[1].url, "https://www.rust-lang.org/learn");
        assert_eq!(results[1].snippet, "Learn Rust with example projects.");
    }

    #[test]
    fn parse_bing_respects_max_and_dedupes() {
        let item = r#"<li class="b_algo"><h2><a href="https://a.com">A</a></h2><p>sa</p></li>"#;
        let html = format!(
            r#"<ol id="b_results">{0}{0}<li class="b_algo"><h2><a href="https://b.com">B</a></h2><p>sb</p></li></ol>"#,
            item
        );
        assert_eq!(parse_bing_results(&html, 5).len(), 2);
        assert_eq!(parse_bing_results(&html, 1).len(), 1);
    }

    #[test]
    fn parse_bing_skips_links_without_usable_url() {
        let html = r#"
        <ol id="b_results">
            <li class="b_algo"><h2><a href="javascript:void(0)">No href</a></h2><p>x</p></li>
            <li class="b_algo"><h2><a href="https://example.com/real">Real</a></h2><p>y</p></li>
        </ol>
        "#;
        let results = parse_bing_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://example.com/real");
    }

    #[test]
    fn resolve_bing_redirect_url_decodes_absolute_path_wrapper() {
        let href = "/ck/a?!&&p=abc&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9wYWdl&ntb=1";
        assert_eq!(resolve_bing_redirect_url(href), "https://example.com/page");
    }

    #[test]
    fn resolve_bing_redirect_url_keeps_malformed_wrapper() {
        // Undecodable `u` falls through to the raw href, not an empty URL.
        let href = "https://www.bing.com/ck/a?!&&p=abc&u=a1%%%&ntb=1";
        assert_eq!(resolve_bing_redirect_url(href), href);
    }

    #[test]
    fn resolve_bing_redirect_url_direct_and_protocol_relative() {
        assert_eq!(
            resolve_bing_redirect_url("https://example.com/page"),
            "https://example.com/page"
        );
        assert_eq!(
            resolve_bing_redirect_url("//example.com/page"),
            "https://example.com/page"
        );
    }
}
