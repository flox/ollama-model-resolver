use std::io::{self, BufRead, BufReader, IsTerminal};
use std::net::IpAddr;

use bytesize::ByteSize;
use reqwest::blocking::{Client, Response};
use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::error::{ResolverError, Result};
use crate::sanitize::{capped_terminal_line, terminal_line, ERROR_DETAIL_LIMIT_CHARS, RAW_STREAM_LINE_LIMIT_CHARS};

#[derive(Debug, Clone)]
pub struct LocalModel {
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    name: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Serialize)]
struct PullRequest<'a> {
    model: &'a str,
    stream: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct PullEvent {
    status: Option<String>,
    error: Option<String>,
    digest: Option<String>,
    total: Option<u64>,
    completed: Option<u64>,
}

pub fn base_url(host: &str, port: u16) -> String {
    let trimmed = host.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.trim_end_matches('/').to_string()
    } else {
        format!("http://{}:{port}", host_for_url_authority(trimmed))
    }
}

fn host_for_url_authority(host: &str) -> String {
    let trimmed = host.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed.to_string();
    }

    match trimmed.parse::<IpAddr>() {
        Ok(IpAddr::V6(_)) => format!("[{trimmed}]"),
        _ => trimmed.to_string(),
    }
}

pub fn validate_ollama_host(host: &str, allow_remote: bool) -> Result<()> {
    let trimmed = host.trim();
    if trimmed.is_empty() {
        return Err(ResolverError::InvalidInput("--ollama-host cannot be empty".into()));
    }

    if allow_remote || ollama_host_is_loopback(trimmed) {
        Ok(())
    } else {
        Err(ResolverError::InvalidInput(format!(
            "--ollama-host '{}' is not loopback; pass --allow-remote-ollama to target a remote Ollama endpoint",
            terminal_line(trimmed)
        )))
    }
}

pub fn ollama_host_is_loopback(host: &str) -> bool {
    if host.starts_with("http://") || host.starts_with("https://") {
        return Url::parse(host)
            .ok()
            .and_then(|url| url.host_str().map(host_label_is_loopback))
            .unwrap_or(false);
    }

    host_label_is_loopback(host)
}

fn host_label_is_loopback(host: &str) -> bool {
    let normalized = host
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.');

    if normalized.eq_ignore_ascii_case("localhost") {
        return true;
    }

    normalized
        .parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

pub fn is_reachable(client: &Client, host: &str, port: u16) -> bool {
    let url = format!("{}/api/tags", base_url(host, port));
    client.get(url).send().map(|resp| resp.status().is_success()).unwrap_or(false)
}

pub fn list_local_models(client: &Client, host: &str, port: u16) -> Result<Vec<LocalModel>> {
    let base = base_url(host, port);
    let url = format!("{base}/api/tags");
    let resp = client.get(&url).send().map_err(|err| ResolverError::OllamaUnreachable {
        base_url: base.clone(),
        detail: err.to_string(),
    })?;

    if !resp.status().is_success() {
        return Err(ResolverError::OllamaUnreachable {
            base_url: base,
            detail: format!("HTTP {}", resp.status()),
        });
    }

    let tags: TagsResponse = resp.json()?;
    Ok(tags
        .models
        .into_iter()
        .map(|model| LocalModel {
            name: model.name,
            size: model.size,
        })
        .collect())
}

pub fn pull_model(
    client: &Client,
    host: &str,
    port: u16,
    model: &str,
    show_progress: bool,
) -> Result<()> {
    let base = base_url(host, port);
    let url = format!("{base}/api/pull");
    let resp = client
        .post(&url)
        .json(&PullRequest { model, stream: true })
        .send()
        .map_err(|err| ResolverError::OllamaUnreachable {
            base_url: base.clone(),
            detail: err.to_string(),
        })?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().unwrap_or_default();
        let body = capped_terminal_line(&body, ERROR_DETAIL_LIMIT_CHARS);
        return Err(ResolverError::PullFailed {
            model: model.to_string(),
            detail: format!("HTTP {status}: {body}"),
        });
    }

    let progress_mode = PullProgressMode::for_stderr(show_progress);
    consume_pull_stream(resp, model, progress_mode)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PullProgressMode {
    Quiet,
    Animated,
    LineDelimited,
}

impl PullProgressMode {
    fn for_stderr(show_progress: bool) -> Self {
        if !show_progress {
            Self::Quiet
        } else if io::stderr().is_terminal() {
            Self::Animated
        } else {
            Self::LineDelimited
        }
    }

    fn is_enabled(self) -> bool {
        !matches!(self, Self::Quiet)
    }
}

fn consume_pull_stream(resp: Response, model: &str, progress_mode: PullProgressMode) -> Result<()> {
    let reader = BufReader::new(resp);
    let mut last_rendered = String::new();

    for line in reader.lines() {
        let line = line.map_err(|err| ResolverError::PullFailed {
            model: model.to_string(),
            detail: format!("failed reading pull stream: {err}"),
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let event = parse_pull_event(&line, model)?;
        if let Some(error) = event.error.as_deref() {
            finish_pull_progress_line(progress_mode, &last_rendered);
            return Err(ResolverError::PullFailed {
                model: model.to_string(),
                detail: capped_terminal_line(error, ERROR_DETAIL_LIMIT_CHARS),
            });
        }

        print_pull_progress(&event, progress_mode, &mut last_rendered);

        if event.status.as_deref() == Some("success") {
            finish_pull_progress_line(progress_mode, &last_rendered);
            return Ok(());
        }
    }

    finish_pull_progress_line(progress_mode, &last_rendered);

    Err(ResolverError::PullFailed {
        model: model.to_string(),
        detail: "pull stream ended before success".into(),
    })
}

fn parse_pull_event(line: &str, model: &str) -> Result<PullEvent> {
    serde_json::from_str(line).map_err(|err| ResolverError::PullFailed {
        model: model.to_string(),
        detail: format!(
            "invalid pull stream event: {err}: {}",
            capped_terminal_line(line, RAW_STREAM_LINE_LIMIT_CHARS)
        ),
    })
}

fn print_pull_progress(event: &PullEvent, mode: PullProgressMode, last_rendered: &mut String) {
    if !mode.is_enabled() {
        return;
    }

    let Some(message) = format_pull_progress(event) else {
        return;
    };

    if *last_rendered == message {
        return;
    }

    match mode {
        PullProgressMode::Quiet => {}
        PullProgressMode::Animated => eprint!("\r\x1b[2K{message}"),
        PullProgressMode::LineDelimited => eprintln!("{message}"),
    }
    *last_rendered = message;
}

fn finish_pull_progress_line(mode: PullProgressMode, last_rendered: &str) {
    if mode == PullProgressMode::Animated && !last_rendered.is_empty() {
        eprintln!();
    }
}

fn format_pull_progress(event: &PullEvent) -> Option<String> {
    let status = event.status.as_deref()?;
    let mut message = terminal_line(status);

    if let Some(digest) = event.digest.as_deref() {
        message.push(' ');
        message.push_str(&short_digest(digest));
    }

    match (event.completed, event.total) {
        (Some(completed), Some(total)) if total > 0 => {
            let pct = (completed as f64 / total as f64) * 100.0;
            message.push_str(&format!(
                " {pct:.1}% ({}/{})",
                ByteSize(completed),
                ByteSize(total)
            ));
        }
        (Some(completed), _) => {
            message.push_str(&format!(" ({})", ByteSize(completed)));
        }
        _ => {}
    }

    Some(message)
}

fn short_digest(digest: &str) -> String {
    let sanitized = terminal_line(digest);
    let stripped = sanitized.strip_prefix("sha256:").unwrap_or(&sanitized);
    stripped.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_loopback_ollama_hosts() {
        assert!(ollama_host_is_loopback("127.0.0.1"));
        assert!(ollama_host_is_loopback("localhost"));
        assert!(ollama_host_is_loopback("localhost."));
        assert!(ollama_host_is_loopback("::1"));
        assert!(ollama_host_is_loopback("[::1]"));
        assert!(ollama_host_is_loopback("http://127.0.0.1:11434"));
        assert!(ollama_host_is_loopback("https://localhost:11434/"));
    }

    #[test]
    fn rejects_non_loopback_ollama_hosts_without_explicit_allow() {
        assert!(!ollama_host_is_loopback("192.0.2.10"));
        assert!(!ollama_host_is_loopback("example.com"));
        assert!(!ollama_host_is_loopback("http://192.0.2.10:11434"));

        let err = validate_ollama_host("http://192.0.2.10:11434", false).unwrap_err();
        assert!(err.to_string().contains("--allow-remote-ollama"));
    }

    #[test]
    fn allows_non_loopback_ollama_hosts_with_explicit_allow() {
        validate_ollama_host("http://192.0.2.10:11434", true).unwrap();
        validate_ollama_host("ollama.internal", true).unwrap();
    }

    #[test]
    fn rejects_empty_ollama_host() {
        let err = validate_ollama_host("   ", false).unwrap_err();
        assert!(err.to_string().contains("cannot be empty"));
    }

    #[test]
    fn builds_base_urls() {
        assert_eq!(base_url("127.0.0.1", 11434), "http://127.0.0.1:11434");
        assert_eq!(base_url("http://host:123/", 11434), "http://host:123");
    }

    #[test]
    fn builds_base_urls_for_bare_ipv6_hosts() {
        assert_eq!(base_url("::1", 11434), "http://[::1]:11434");
        assert_eq!(base_url("[::1]", 11434), "http://[::1]:11434");
        assert_eq!(base_url("  ::1  ", 11434), "http://[::1]:11434");
        assert_eq!(base_url("2001:db8::1", 11434), "http://[2001:db8::1]:11434");
        assert_eq!(base_url("http://[::1]:11434/", 11434), "http://[::1]:11434");
    }

    #[test]
    fn parses_streaming_pull_events() {
        let event = parse_pull_event(
            r#"{"status":"pulling manifest","digest":"sha256:abcdef1234567890","total":100,"completed":25}"#,
            "qwen",
        )
        .unwrap();

        assert_eq!(event.status.as_deref(), Some("pulling manifest"));
        assert_eq!(event.total, Some(100));
        assert_eq!(event.completed, Some(25));
    }

    #[test]
    fn detects_streaming_pull_errors() {
        let event = parse_pull_event(r#"{"error":"not found"}"#, "qwen").unwrap();
        assert_eq!(event.error.as_deref(), Some("not found"));
    }

    #[test]
    fn rejects_invalid_pull_stream_json() {
        let err = parse_pull_event("not-json", "qwen").unwrap_err();
        assert!(err.to_string().contains("invalid pull stream event"));
    }

    #[test]
    fn shortens_sha256_digests() {
        assert_eq!(short_digest("sha256:abcdef1234567890"), "abcdef123456");
        assert_eq!(short_digest("short"), "short");
    }


    #[test]
    fn selects_quiet_progress_when_disabled() {
        assert_eq!(PullProgressMode::for_stderr(false), PullProgressMode::Quiet);
    }

    #[test]
    fn line_delimited_progress_is_distinct_from_animated() {
        assert_ne!(PullProgressMode::LineDelimited, PullProgressMode::Animated);
        assert!(PullProgressMode::LineDelimited.is_enabled());
    }

    #[test]
    fn quiet_progress_is_disabled() {
        assert!(!PullProgressMode::Quiet.is_enabled());
    }

    #[test]
    fn formats_pull_progress_with_digest_percent_and_bytes() {
        let event = PullEvent {
            status: Some("pulling layer".into()),
            error: None,
            digest: Some("sha256:abcdef1234567890".into()),
            total: Some(100_000_000),
            completed: Some(25_000_000),
        };

        let rendered = format_pull_progress(&event).unwrap();
        assert!(rendered.contains("pulling layer"));
        assert!(rendered.contains("abcdef123456"));
        assert!(rendered.contains("25.0%"));
        // ByteSize 2.x uses IEC units
        assert!(rendered.contains(&format!(
            "{}/{}",
            ByteSize(25_000_000),
            ByteSize(100_000_000)
        )));
    }

    #[test]
    fn formats_pull_progress_without_total() {
        let event = PullEvent {
            status: Some("verifying sha256 digest".into()),
            error: None,
            digest: None,
            total: None,
            completed: Some(42),
        };

        let rendered = format_pull_progress(&event).unwrap();
        assert!(rendered.contains("verifying sha256 digest"));
        assert!(rendered.contains("42 B"));
    }


    #[test]
    fn pull_progress_sanitizes_status_and_digest() {
        let event = PullEvent {
            status: Some("pulling\x1b[2K layer".into()),
            error: None,
            digest: Some("sha256:abc\x1bdef123456".into()),
            total: None,
            completed: None,
        };

        let rendered = format_pull_progress(&event).unwrap();
        assert!(!rendered.contains('\x1b'));
        assert!(rendered.contains('�'));
    }

    #[test]
    fn invalid_pull_stream_error_sanitizes_and_caps_raw_line() {
        let raw = format!("{{bad}}{}", "x".repeat(RAW_STREAM_LINE_LIMIT_CHARS + 100));
        let err = parse_pull_event(&raw, "qwen").unwrap_err().to_string();
        assert!(err.contains("invalid pull stream event"));
        assert!(err.contains('…'));
        assert!(err.len() < RAW_STREAM_LINE_LIMIT_CHARS + 300);
    }

    #[test]
    fn does_not_render_events_without_status() {
        let event = PullEvent {
            status: None,
            error: None,
            digest: None,
            total: None,
            completed: None,
        };

        assert_eq!(format_pull_progress(&event), None);
    }
}
