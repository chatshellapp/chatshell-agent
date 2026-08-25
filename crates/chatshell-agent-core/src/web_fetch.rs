//! Web fetch - HTTP + Jina Reader, shared across desktop and iOS.
//!
//! Extracted from desktop's `src-tauri/src/web_fetch/`. Only the HTTP and
//! Jina paths are shared; the headless-browser / WebView path stays
//! platform-specific (desktop uses headless Chrome, iOS uses WKWebView).
//!
//! The extraction pipeline uses the same crates as desktop:
//! - `scraper` for HTML parsing (CSS selectors)
//! - `readability` for article extraction (Mozilla Readability algorithm)
//! - `htmd` for HTML-to-markdown conversion

use std::io::Cursor;
use std::sync::LazyLock;
use std::time::Duration;

use readability::extractor::extract;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use url::Url;

/// A fetched web resource with extracted content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchedResource {
    pub url: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub content: String,
    pub mime_type: String,
    pub extraction_error: Option<String>,
}

impl FetchedResource {
    fn error(url: &str, mime_type: &str, error: &str) -> Self {
        Self {
            url: url.to_string(),
            title: None,
            description: None,
            content: String::new(),
            mime_type: mime_type.to_string(),
            extraction_error: Some(error.to_string()),
        }
    }
}

/// Shared HTTP client with connection pooling.
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(10))
        .user_agent("Mozilla/5.0 (compatible; ChatShell/1.0)")
        .build()
        .expect("Failed to create HTTP client")
});

static IMG_TAG_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<img\s[^>]*?/?>").unwrap());

static IMG_SRC_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)src\s*=\s*["']([^"']+)["']"#).unwrap());

static IMG_ALT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)alt\s*=\s*["']([^"']*)["']"#).unwrap());

const JINA_READER_BASE_URL: &str = "https://r.jina.ai/";

/// Fetch a web resource via HTTP and extract its main content.
///
/// For HTML content, uses the Readability algorithm to extract the article
/// and converts it to markdown. For JSON/XML/text, formats as a code block.
/// Returns an error result if the fetch fails.
pub async fn fetch_http(url: &str, max_chars: Option<usize>) -> FetchedResource {
    tracing::info!("Fetching via HTTP: {}", url);

    if Url::parse(url).is_err() {
        return FetchedResource::error(url, "", &format!("Invalid URL: {url}"));
    }

    let response = match HTTP_CLIENT
        .get(url)
        .header("Accept", "text/markdown, text/html, */*")
        .header("Accept-Encoding", "gzip, deflate, br, zstd")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return FetchedResource::error(url, "", &format!("HTTP request failed: {e}"));
        }
    };

    if !response.status().is_success() {
        return FetchedResource::error(url, "", &format!("HTTP error: {}", response.status()));
    }

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    let mime_type = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();

    let body = match response.text().await {
        Ok(c) => c,
        Err(e) => {
            return FetchedResource::error(url, &mime_type, &format!("Failed to read body: {e}"));
        }
    };

    if mime_type.contains("html") {
        process_html(url, &body, &mime_type, max_chars)
    } else if mime_type.contains("json") {
        process_text(url, &body, &mime_type, max_chars, true)
    } else if mime_type.contains("xml") {
        process_text(url, &body, &mime_type, max_chars, false)
    } else {
        process_text(url, &body, &mime_type, max_chars, false)
    }
}

/// Fetch a web resource using the Jina Reader API. Returns markdown content.
pub async fn fetch_jina(url: &str, api_key: Option<&str>) -> FetchedResource {
    tracing::info!("Fetching via Jina Reader: {}", url);

    let jina_url = format!("{JINA_READER_BASE_URL}{url}");
    let mut request = HTTP_CLIENT.get(&jina_url).header("Accept", "text/markdown");

    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        request = request.header("Authorization", format!("Bearer {key}"));
    }

    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => {
            return FetchedResource::error(url, "", &format!("Jina request failed: {e}"));
        }
    };

    if !response.status().is_success() {
        return FetchedResource::error(
            url,
            "",
            &format!("Jina Reader returned status: {}", response.status()),
        );
    }

    let content = match response.text().await {
        Ok(c) => c,
        Err(e) => {
            return FetchedResource::error(url, "", &format!("Failed to read Jina response: {e}"));
        }
    };

    let title = extract_title_from_markdown(&content);

    FetchedResource {
        url: url.to_string(),
        title,
        description: None,
        content,
        mime_type: "text/markdown".to_string(),
        extraction_error: None,
    }
}

// ---------------------------------------------------------------------------
// Content processing
// ---------------------------------------------------------------------------

fn process_html(
    url: &str,
    html: &str,
    mime_type: &str,
    max_chars: Option<usize>,
) -> FetchedResource {
    let document = Html::parse_document(html);
    let description = extract_meta_description(&document);

    let parsed_url = match Url::parse(url) {
        Ok(u) => u,
        Err(e) => return FetchedResource::error(url, mime_type, &format!("Invalid URL: {e}")),
    };

    let mut cursor = Cursor::new(html.as_bytes());

    let (title, content_html) = match extract(&mut cursor, &parsed_url) {
        Ok(product) => (Some(product.title), product.content),
        Err(e) => {
            tracing::warn!("Readability extraction failed: {e}, using fallback");
            let title = Selector::parse("title")
                .ok()
                .and_then(|sel| document.select(&sel).next())
                .map(|el| el.text().collect::<String>().trim().to_string())
                .filter(|s| !s.is_empty());
            let body_html = Selector::parse("body")
                .ok()
                .and_then(|sel| document.select(&sel).next())
                .map(|el| el.html())
                .unwrap_or_else(|| html.to_string());
            (title, body_html)
        }
    };

    let normalized_html = normalize_html_images(&content_html);
    let markdown = htmd::convert(&normalized_html).unwrap_or_default();
    let (content, _truncated) = truncate_by_chars(&markdown, max_chars);

    let cleaned = content
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    FetchedResource {
        url: url.to_string(),
        title,
        description,
        content: cleaned,
        mime_type: mime_type.to_string(),
        extraction_error: None,
    }
}

fn process_text(
    url: &str,
    text: &str,
    mime_type: &str,
    max_chars: Option<usize>,
    is_code: bool,
) -> FetchedResource {
    let (content, _truncated) = truncate_by_chars(text, max_chars);
    let formatted = if is_code {
        format!("```\n{content}\n```")
    } else {
        content
    };

    FetchedResource {
        url: url.to_string(),
        title: None,
        description: None,
        content: formatted,
        mime_type: mime_type.to_string(),
        extraction_error: None,
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

fn extract_meta_description(document: &Html) -> Option<String> {
    Selector::parse(r#"meta[name="description"]"#)
        .ok()
        .and_then(|sel| document.select(&sel).next())
        .and_then(|el| el.value().attr("content"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Selector::parse(r#"meta[property="og:description"]"#)
                .ok()
                .and_then(|sel| document.select(&sel).next())
                .and_then(|el| el.value().attr("content"))
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
}

fn extract_title_from_markdown(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            let title = title.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

fn truncate_by_chars(s: &str, max_chars: Option<usize>) -> (String, bool) {
    match max_chars {
        Some(limit) => {
            let char_count = s.chars().count();
            if char_count <= limit {
                (s.to_string(), false)
            } else {
                let truncated: String = s.chars().take(limit).collect();
                (
                    format!(
                        "{truncated}\n\n[Content truncated - original length: {char_count} characters]"
                    ),
                    true,
                )
            }
        }
        None => (s.to_string(), false),
    }
}

fn normalize_html_images(html: &str) -> String {
    IMG_TAG_REGEX
        .replace_all(html, |caps: &regex::Captures| {
            let img_tag = &caps[0];
            let src = IMG_SRC_REGEX
                .captures(img_tag)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str())
                .unwrap_or("");
            let alt = IMG_ALT_REGEX
                .captures(img_tag)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str())
                .unwrap_or("");
            format!(r#"<img src="{src}" alt="{alt}">"#)
        })
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_title_from_markdown_heading() {
        assert_eq!(
            extract_title_from_markdown("# Hello World\n\nContent"),
            Some("Hello World".to_string())
        );
    }

    #[test]
    fn extract_title_from_markdown_no_heading() {
        assert_eq!(extract_title_from_markdown("Just text"), None);
    }

    #[test]
    fn truncate_short_string() {
        let (s, truncated) = truncate_by_chars("hello", Some(10));
        assert_eq!(s, "hello");
        assert!(!truncated);
    }

    #[test]
    fn truncate_long_string() {
        let (s, truncated) = truncate_by_chars("hello world", Some(5));
        assert!(s.starts_with("hello"));
        assert!(s.contains("truncated"));
        assert!(truncated);
    }

    #[test]
    fn normalize_simple_img_tag() {
        let html = r#"<p>Before</p><img src="/test.png" alt="Test" class="foo"><p>After</p>"#;
        let result = normalize_html_images(html);
        assert!(result.contains(r#"<img src="/test.png" alt="Test">"#));
        assert!(!result.contains(r#"class="foo""#));
    }

    #[test]
    fn fetched_resource_error() {
        let r = FetchedResource::error("https://example.com", "text/html", "Failed");
        assert_eq!(r.url, "https://example.com");
        assert_eq!(r.extraction_error, Some("Failed".to_string()));
        assert!(r.content.is_empty());
    }
}
