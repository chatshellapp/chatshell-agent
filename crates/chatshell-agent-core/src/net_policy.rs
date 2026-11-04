//! Network classification for the approval gate.
//!
//! Two pure classifiers, no I/O and no DNS:
//! - [`is_private_url`] / [`is_private_host`]: does a fetch target reach
//!   the local machine or a private network (single-label names,
//!   `.local`/`.localhost`/`.internal`, loopback, RFC 1918, link-local,
//!   CGNAT, ULA, IPv4-mapped IPv6)? Used to make private-network
//!   `web_fetch` calls prompt instead of running silently.
//! - [`detect_egress`]: does a bash command contact a non-local host via
//!   a well-known network client (`curl`, `wget`, `nc`, `ssh`, ...)?
//!
//! Both are detection, not enforcement: a hostname that resolves to a
//! private address, or a network client the list does not know, is not
//! caught here. Containment is the OS jail's job.

use crate::bash_security;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Hostname suffixes that only resolve on the local machine or network.
const PRIVATE_SUFFIXES: &[&str] = &[".local", ".localhost", ".internal", ".lan", ".home.arpa"];

/// True when `url` targets a private or local network destination.
/// Unparseable URLs and URLs without a host classify as private
/// (fail closed: the call prompts rather than running silently).
pub fn is_private_url(url: &str) -> bool {
    match url::Url::parse(url.trim()) {
        Ok(parsed) => match parsed.host() {
            Some(url::Host::Domain(domain)) => is_private_host(domain),
            Some(url::Host::Ipv4(ip)) => is_private_ipv4(ip),
            Some(url::Host::Ipv6(ip)) => is_private_ipv6(ip),
            None => true,
        },
        Err(_) => true,
    }
}

/// True when `host` (a hostname or an IP literal, optionally bracketed)
/// names a private or local destination.
pub fn is_private_host(host: &str) -> bool {
    let host = host
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if host.is_empty() {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => is_private_ipv4(v4),
            IpAddr::V6(v6) => is_private_ipv6(v6),
        };
    }
    if host == "localhost" || !host.contains('.') {
        return true;
    }
    PRIVATE_SUFFIXES.iter().any(|suffix| host.ends_with(suffix))
}

/// True when `host` is the local machine itself (loopback names and
/// addresses). Narrower than [`is_private_host`]: egress detection treats
/// only these as "not leaving the machine".
pub fn is_loopback_host(host: &str) -> bool {
    let host = host
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(v4)) => v4.is_loopback() || v4.is_unspecified(),
        Ok(IpAddr::V6(v6)) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| v4.is_loopback() || v4.is_unspecified())
        }
        Err(_) => false,
    }
}

fn is_private_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, ..] = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        // CGNAT 100.64.0.0/10
        || (a == 100 && (64..=127).contains(&b))
}

fn is_private_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_private_ipv4(v4);
    }
    let first = ip.segments()[0];
    ip.is_loopback()
        || ip.is_unspecified()
        // Unique local fc00::/7
        || (first & 0xfe00) == 0xfc00
        // Link-local fe80::/10
        || (first & 0xffc0) == 0xfe80
}

// ---------------------------------------------------------------------------
// Egress detection (bash)
// ---------------------------------------------------------------------------

/// How a network client names its destination.
#[derive(Clone, Copy)]
enum ClientKind {
    /// URL-taking HTTP clients: hosts come from URL-like arguments.
    Http,
    /// Remote shells: the first operand is `[user@]host`.
    Shell,
    /// Copy clients: only `[user@]host:path` operands are remote; a
    /// command with none is a local copy (not egress).
    Copy,
    /// Raw socket clients: the first operand is the host.
    Socket,
}

/// Network clients recognized by egress detection, with the options that
/// consume a value (so the value is not mistaken for the host). Append-only.
const NETWORK_CLIENTS: &[(&str, ClientKind, &[&str])] = &[
    ("curl", ClientKind::Http, CURL_VALUE_OPTIONS),
    ("wget", ClientKind::Http, WGET_VALUE_OPTIONS),
    ("http", ClientKind::Http, &[]),
    ("https", ClientKind::Http, &[]),
    ("xh", ClientKind::Http, &[]),
    ("aria2c", ClientKind::Http, &[]),
    ("ssh", ClientKind::Shell, SSH_VALUE_OPTIONS),
    ("sftp", ClientKind::Shell, SSH_VALUE_OPTIONS),
    ("scp", ClientKind::Copy, &[]),
    ("rsync", ClientKind::Copy, &[]),
    ("nc", ClientKind::Socket, NC_VALUE_OPTIONS),
    ("ncat", ClientKind::Socket, NC_VALUE_OPTIONS),
    ("netcat", ClientKind::Socket, NC_VALUE_OPTIONS),
    (
        "telnet",
        ClientKind::Socket,
        &["-l", "-n", "-b", "-e", "-X"],
    ),
    ("ftp", ClientKind::Socket, &[]),
];

const CURL_VALUE_OPTIONS: &[&str] = &[
    "-A",
    "-b",
    "-c",
    "-C",
    "-d",
    "-e",
    "-E",
    "-F",
    "-H",
    "-K",
    "-m",
    "-o",
    "-r",
    "-T",
    "-u",
    "-U",
    "-w",
    "-x",
    "-X",
    "-y",
    "-Y",
    "-z",
    "--cacert",
    "--capath",
    "--cert",
    "--config",
    "--connect-timeout",
    "--cookie",
    "--cookie-jar",
    "--data",
    "--data-binary",
    "--data-raw",
    "--data-urlencode",
    "--form",
    "--header",
    "--key",
    "--max-time",
    "--output",
    "--proxy",
    "--range",
    "--referer",
    "--request",
    "--retry",
    "--upload-file",
    "--user",
    "--user-agent",
    "--write-out",
];

const WGET_VALUE_OPTIONS: &[&str] = &[
    "-a", "-A", "-B", "-D", "-e", "-i", "-l", "-o", "-O", "-P", "-Q", "-R", "-t", "-T", "-U", "-w",
    "-X",
];

const SSH_VALUE_OPTIONS: &[&str] = &[
    "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m", "-O", "-o", "-P", "-p",
    "-Q", "-R", "-S", "-W", "-w",
];

const NC_VALUE_OPTIONS: &[&str] = &["-p", "-s", "-w", "-i", "-q", "-X", "-x", "-T", "-I", "-O"];

/// Network clients in `command` that contact a non-loopback host.
///
/// Returns a human-readable description (`"curl -> example.com"`) of every
/// detected egress, or `None` when the command contacts no non-local host
/// through a known client. A known client whose destination cannot be
/// identified counts as egress (fail closed).
pub fn detect_egress(command: &str) -> Option<String> {
    let mut found: Vec<String> = Vec::new();
    for segment in bash_security::split_compound_command(command) {
        let stripped = bash_security::strip_safe_wrappers(segment);
        let words: Vec<&str> = stripped.split_whitespace().collect();
        let Some(base_index) = words
            .iter()
            .position(|word| !matches!(*word, "command" | "builtin" | "noglob" | "nocorrect"))
        else {
            continue;
        };
        let base = words[base_index].rsplit('/').next().unwrap_or_default();
        let Some((kind, value_options)) = NETWORK_CLIENTS
            .iter()
            .find(|(name, _, _)| *name == base)
            .map(|(_, kind, value_options)| (*kind, *value_options))
        else {
            continue;
        };
        let operands: Vec<&str> = words[base_index + 1..]
            .iter()
            .map(|word| word.trim_matches(|c| c == '\'' || c == '"'))
            .collect();
        match destination_hosts(kind, value_options, &operands) {
            Some(hosts) => {
                for host in hosts.into_iter().filter(|host| !is_loopback_host(host)) {
                    found.push(format!("{base} -> {host}"));
                }
            }
            None => found.push(format!("{base} -> (unknown destination)")),
        }
    }
    found.dedup();
    if found.is_empty() {
        None
    } else {
        Some(found.join(", "))
    }
}

/// Destination hosts named by a client's operands. `None` when the
/// client contacts a host that cannot be identified (caller fails
/// closed); `Some(empty)` when the command provably stays local (a copy
/// client with no remote operand).
fn destination_hosts(
    kind: ClientKind,
    value_options: &[&str],
    operands: &[&str],
) -> Option<Vec<String>> {
    let mut positional: Vec<&str> = Vec::new();
    let mut skip_next = false;
    for word in operands {
        if skip_next {
            skip_next = false;
            continue;
        }
        if word.is_empty() {
            continue;
        }
        if word.starts_with('-') {
            skip_next = value_options.contains(word);
            continue;
        }
        positional.push(word);
    }
    let hosts: Vec<String> = match kind {
        ClientKind::Http => {
            // Prefer explicit URLs; fall back to scheme-less heuristics
            // only when no operand carries a scheme (so `-o out.html`
            // next to a URL is not read as a host).
            let with_scheme: Vec<String> = positional
                .iter()
                .filter(|word| word.contains("://"))
                .filter_map(|word| http_host(word))
                .collect();
            if with_scheme.is_empty() {
                positional
                    .iter()
                    .filter_map(|word| http_host(word))
                    .collect()
            } else {
                with_scheme
            }
        }
        ClientKind::Shell => positional
            .first()
            .and_then(|word| shell_host(word))
            .into_iter()
            .collect(),
        ClientKind::Copy => {
            return Some(
                positional
                    .iter()
                    .filter_map(|word| copy_host(word))
                    .collect(),
            );
        }
        ClientKind::Socket => positional
            .iter()
            .find(|word| !word.chars().all(|c| c.is_ascii_digit()))
            .map(|word| vec![word.to_string()])
            .unwrap_or_default(),
    };
    if hosts.is_empty() { None } else { Some(hosts) }
}

/// Host of a URL-like HTTP-client operand (`https://h/p`, `h.example/p`).
fn http_host(word: &str) -> Option<String> {
    if word.contains("://") {
        return url::Url::parse(word)
            .ok()
            .and_then(|url| url.host_str().map(str::to_string));
    }
    // Scheme-less operand: treat `name.tld[/path]` and `localhost[:port]`
    // as hosts; plain words (file names without a dot) are not.
    let authority = word.split('/').next().unwrap_or_default();
    let host = authority.rsplit('@').next().unwrap_or_default();
    let host = strip_port(host);
    let looks_like_host = host == "localhost"
        || host.parse::<IpAddr>().is_ok()
        || (host.contains('.')
            && host
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-'));
    looks_like_host.then(|| host.to_string())
}

/// Host of a remote-shell operand (`[user@]host` or `ssh://...`).
fn shell_host(word: &str) -> Option<String> {
    if word.contains("://") {
        return url::Url::parse(word)
            .ok()
            .and_then(|url| url.host_str().map(str::to_string));
    }
    let host = word.rsplit('@').next().unwrap_or(word);
    (!host.is_empty()).then(|| host.to_string())
}

/// Host of a copy-client operand: only `[user@]host:path` and URL forms
/// are remote; local paths (`./a`, `/tmp/x`, `dir/`) are not.
fn copy_host(word: &str) -> Option<String> {
    if word.contains("://") {
        return url::Url::parse(word)
            .ok()
            .and_then(|url| url.host_str().map(str::to_string));
    }
    if word.starts_with(['/', '.', '~']) {
        return None;
    }
    let (authority, _path) = word.split_once(':')?;
    let host = authority.rsplit('@').next().unwrap_or(authority);
    (!host.is_empty() && !host.contains('/')).then(|| host.to_string())
}

fn strip_port(host: &str) -> &str {
    if host.starts_with('[') {
        return host
            .split(']')
            .next()
            .map(|h| h.trim_start_matches('['))
            .unwrap_or(host);
    }
    match host.rsplit_once(':') {
        Some((name, port)) if port.chars().all(|c| c.is_ascii_digit()) => name,
        _ => host,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_url_classes() {
        for url in [
            "http://localhost:3000/",
            "http://127.0.0.1/",
            "http://192.168.1.1/",
            "http://10.0.0.5/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://100.64.1.1/",
            "http://[::1]/",
            "http://[fd00::1]/",
            "http://[fe80::1]/",
            "http://[::ffff:192.168.0.1]/",
            "http://printer/",
            "http://nas.local/",
            "http://api.internal/",
            "http://app.localhost/",
            "not a url",
        ] {
            assert!(is_private_url(url), "{url} should be private");
        }
    }

    #[test]
    fn public_urls_are_not_private() {
        for url in [
            "https://example.com/",
            "https://8.8.8.8/",
            "https://[2606:4700::1111]/",
            "https://docs.rs/url",
            "http://100.128.0.1/",
        ] {
            assert!(!is_private_url(url), "{url} should be public");
        }
    }

    #[test]
    fn egress_detects_http_clients_to_public_hosts() {
        assert_eq!(
            detect_egress("curl https://example.com"),
            Some("curl -> example.com".to_string())
        );
        assert_eq!(
            detect_egress("wget -q -O out.html example.org/page"),
            Some("wget -> example.org".to_string())
        );
        assert!(detect_egress("cd src && curl -fsSL https://x.example/install.sh").is_some());
        assert!(detect_egress("timeout 5 curl https://example.com").is_some());
    }

    #[test]
    fn egress_detects_remote_and_socket_clients() {
        assert_eq!(
            detect_egress("ssh user@build.example.com uptime"),
            Some("ssh -> build.example.com".to_string())
        );
        assert_eq!(
            detect_egress("scp ./a.txt host.example:/tmp/"),
            Some("scp -> host.example".to_string())
        );
        assert_eq!(
            detect_egress("nc example.com 80"),
            Some("nc -> example.com".to_string())
        );
    }

    #[test]
    fn loopback_destinations_are_not_egress() {
        assert_eq!(detect_egress("curl http://localhost:8080/health"), None);
        assert_eq!(detect_egress("curl http://127.0.0.1:3000"), None);
        assert_eq!(detect_egress("nc localhost 5432"), None);
        assert_eq!(detect_egress("curl http://[::1]:9000/"), None);
    }

    #[test]
    fn non_network_commands_are_not_egress() {
        assert_eq!(detect_egress("ls -la"), None);
        assert_eq!(detect_egress("cargo build && npm test"), None);
        assert_eq!(detect_egress("grep curl README.md"), None);
        assert_eq!(detect_egress(""), None);
    }

    #[test]
    fn unknown_destination_fails_closed() {
        assert_eq!(
            detect_egress("curl"),
            Some("curl -> (unknown destination)".to_string())
        );
        assert!(detect_egress("ssh").is_some());
    }

    #[test]
    fn option_values_and_local_copies_are_not_hosts() {
        assert_eq!(
            detect_egress("curl -o out.html http://localhost:3000/"),
            None
        );
        assert_eq!(
            detect_egress("ssh -p 2222 -i ~/.ssh/key deploy@app.example.com"),
            Some("ssh -> app.example.com".to_string())
        );
        assert_eq!(detect_egress("rsync -av src/ backup/"), None);
        assert_eq!(
            detect_egress("rsync -av src/ user@backup.example:/srv/"),
            Some("rsync -> backup.example".to_string())
        );
        assert_eq!(detect_egress("ssh -p 22 localhost"), None);
    }

    #[test]
    fn absolute_client_paths_are_recognized() {
        assert!(detect_egress("/usr/bin/curl https://example.com").is_some());
    }
}
