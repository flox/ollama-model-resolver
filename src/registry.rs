use std::collections::{HashMap, HashSet};

use reqwest::blocking::Client;
use scraper::{Html, Selector};
use serde::Deserialize;

use crate::error::{ResolverError, Result};
use crate::types::{tag_info_from_str, SearchResult, TagInfo};

const OLLAMA_BASE: &str = "https://ollama.com";
const REGISTRY_BASE: &str = "https://registry.ollama.ai/v2";
const USER_AGENT: &str = concat!("ollama-model-resolver/", env!("CARGO_PKG_VERSION"));

pub trait Registry {
    fn list_tags(&mut self, model: &str) -> Result<Vec<TagInfo>>;
    fn get_manifest_size(&mut self, model: &str, tag: &str) -> Result<(u64, u64)>;
}

pub struct HttpRegistry<'a> {
    client: &'a Client,
    manifest_cache: HashMap<(String, String), (u64, u64)>,
}

impl<'a> HttpRegistry<'a> {
    pub fn new(client: &'a Client) -> Self {
        Self {
            client,
            manifest_cache: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub fn cached_manifest_count(&self) -> usize {
        self.manifest_cache.len()
    }
}

impl Registry for HttpRegistry<'_> {
    fn list_tags(&mut self, model: &str) -> Result<Vec<TagInfo>> {
        list_tags(self.client, model)
    }

    fn get_manifest_size(&mut self, model: &str, tag: &str) -> Result<(u64, u64)> {
        let key = (model.to_string(), tag.to_string());
        if let Some(cached) = self.manifest_cache.get(&key).copied() {
            return Ok(cached);
        }

        let sizes = get_manifest_size(self.client, model, tag)?;
        self.manifest_cache.insert(key, sizes);
        Ok(sizes)
    }
}

pub fn search_models(client: &Client, query: &str) -> Result<Vec<SearchResult>> {
    let url = format!("{OLLAMA_BASE}/search?q={}", urlencoded(query));
    let html = fetch_html(client, &url)?;
    let results = parse_search_html(&html);

    if results.is_empty() {
        return Err(ResolverError::NoSearchResults {
            query: query.to_string(),
        });
    }

    Ok(results)
}

pub fn list_tags(client: &Client, model: &str) -> Result<Vec<TagInfo>> {
    let url = format!("{OLLAMA_BASE}/library/{}/tags", urlencoded_path_component(model));
    let html = fetch_html(client, &url)?;
    let mut tags = parse_tags_html(model, &html);

    if tags.is_empty() {
        return Err(ResolverError::NoTags {
            model: model.to_string(),
        });
    }

    tags.sort_by(|a, b| natural_tag_order_key(&b.tag).cmp(&natural_tag_order_key(&a.tag)));
    Ok(tags)
}

#[derive(Debug, Deserialize)]
struct Manifest {
    layers: Vec<Layer>,
}

#[derive(Debug, Deserialize)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    size: u64,
}

pub fn get_manifest_size(client: &Client, model: &str, tag: &str) -> Result<(u64, u64)> {
    let repo = registry_repo_path(model);
    let url = format!(
        "{REGISTRY_BASE}/{repo}/manifests/{}",
        urlencoded_path_component(tag)
    );

    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.docker.distribution.manifest.v2+json")
        .header("User-Agent", USER_AGENT)
        .send()
        .map_err(|err| ResolverError::ManifestUnavailable {
            model: model.to_string(),
            tag: tag.to_string(),
            detail: err.to_string(),
        })?;

    let status = resp.status();
    if status.as_u16() == 404 {
        return Err(ResolverError::ManifestMissing {
            model: model.to_string(),
            tag: tag.to_string(),
            detail: format!("registry returned {status}"),
        });
    }
    if !status.is_success() {
        return Err(ResolverError::ManifestUnavailable {
            model: model.to_string(),
            tag: tag.to_string(),
            detail: format!("registry returned {status}"),
        });
    }

    let manifest: Manifest = resp.json().map_err(|err| ResolverError::ManifestInvalid {
        model: model.to_string(),
        tag: tag.to_string(),
        detail: err.to_string(),
    })?;
    manifest_sizes(model, tag, &manifest)
}

fn manifest_sizes(model: &str, tag: &str, manifest: &Manifest) -> Result<(u64, u64)> {
    let mut weights_bytes = 0_u64;
    let mut model_layer_count = 0_usize;
    let mut total_bytes = 0_u64;

    for layer in &manifest.layers {
        if layer.media_type.contains("application/vnd.ollama.image.model") {
            model_layer_count += 1;
            weights_bytes = weights_bytes.checked_add(layer.size).ok_or_else(|| {
                ResolverError::ManifestInvalid {
                    model: model.to_string(),
                    tag: tag.to_string(),
                    detail: "ollama model layer sizes overflow u64 while summing weights".to_string(),
                }
            })?;
        }

        total_bytes = total_bytes.checked_add(layer.size).ok_or_else(|| {
            ResolverError::ManifestInvalid {
                model: model.to_string(),
                tag: tag.to_string(),
                detail: "manifest layer sizes overflow u64 while summing total bytes".to_string(),
            }
        })?;
    }

    if model_layer_count == 0 {
        return Err(ResolverError::ManifestInvalid {
            model: model.to_string(),
            tag: tag.to_string(),
            detail: "manifest does not contain an ollama model layer".to_string(),
        });
    }

    Ok((weights_bytes, total_bytes))
}

fn fetch_html(client: &Client, url: &str) -> Result<String> {
    let resp = client.get(url).header("User-Agent", USER_AGENT).send()?;
    let status = resp.status();
    if !status.is_success() {
        return Err(ResolverError::HtmlParse {
            url: url.to_string(),
            detail: format!("HTTP {status}"),
        });
    }
    Ok(resp.text()?)
}

fn parse_search_html(html: &str) -> Vec<SearchResult> {
    let document = Html::parse_document(html);
    let item_sel = Selector::parse("[x-test-model]").unwrap();
    let anchor_sel = Selector::parse("a[href]").unwrap();
    let title_sel = Selector::parse("[x-test-search-response-title]").unwrap();
    let desc_sel = Selector::parse("p").unwrap();
    let pulls_sel = Selector::parse("[x-test-pull-count]").unwrap();
    let tags_sel = Selector::parse("[x-test-tag-count]").unwrap();
    let updated_sel = Selector::parse("[x-test-updated]").unwrap();

    let mut results = Vec::new();

    for item in document.select(&item_sel) {
        let linked_name = item
            .select(&anchor_sel)
            .filter_map(|el| el.value().attr("href"))
            .filter_map(model_name_from_library_href)
            .next();

        let title_name = item
            .select(&title_sel)
            .next()
            .map(|el| normalized_text(el.text()))
            .filter(|value| !value.is_empty());

        let Some(name) = title_name.or(linked_name) else {
            continue;
        };

        results.push(SearchResult {
            name,
            description: item
                .select(&desc_sel)
                .next()
                .map(|el| normalized_text(el.text()))
                .unwrap_or_default(),
            pulls: item
                .select(&pulls_sel)
                .next()
                .map(|el| normalized_text(el.text()))
                .unwrap_or_default(),
            tag_count: item
                .select(&tags_sel)
                .next()
                .map(|el| normalized_text(el.text()))
                .unwrap_or_default(),
            updated: item
                .select(&updated_sel)
                .next()
                .map(|el| normalized_text(el.text()))
                .unwrap_or_default(),
        });
    }

    if results.is_empty() {
        for anchor in document.select(&anchor_sel) {
            if let Some(name) = anchor.value().attr("href").and_then(model_name_from_library_href) {
                results.push(SearchResult {
                    name,
                    description: String::new(),
                    pulls: String::new(),
                    tag_count: String::new(),
                    updated: String::new(),
                });
            }
        }
    }

    dedup_search_results(results)
}

fn parse_tags_html(model: &str, html: &str) -> Vec<TagInfo> {
    let document = Html::parse_document(html);
    let anchor_sel = Selector::parse("a[href]").unwrap();
    let mut tags = Vec::new();
    let mut seen = HashSet::new();

    for anchor in document.select(&anchor_sel) {
        let Some(href) = anchor.value().attr("href") else {
            continue;
        };
        let Some(tag) = tag_from_library_href(model, href) else {
            continue;
        };
        if !seen.insert(tag.clone()) {
            continue;
        }

        let mut info = tag_info_from_str(model, &tag);
        let anchor_text = normalized_text(anchor.text());
        info.approx_size = extract_size_from_text(&anchor_text);
        tags.push(info);
    }

    tags
}

fn dedup_search_results(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();

    for result in results {
        if seen.insert(result.name.clone()) {
            deduped.push(result);
        }
    }

    deduped
}

fn model_name_from_library_href(href: &str) -> Option<String> {
    let rest = library_href_rest(href)?;
    let name = rest
        .split(['?', '#'])
        .next()?
        .trim_matches('/')
        .trim();

    if name.is_empty()
        || name.contains(':')
        || name.contains('/')
        || name.chars().any(|ch| ch.is_ascii_control())
    {
        return None;
    }

    Some(name.to_string())
}

fn library_href_rest(href: &str) -> Option<&str> {
    let href = href.trim();
    if let Some(rest) = href.strip_prefix("/library/") {
        return Some(rest);
    }

    for prefix in [
        "https://ollama.com/library/",
        "http://ollama.com/library/",
        "https://www.ollama.com/library/",
        "http://www.ollama.com/library/",
    ] {
        if let Some(rest) = href.strip_prefix(prefix) {
            return Some(rest);
        }
    }

    None
}

fn tag_from_library_href(model: &str, href: &str) -> Option<String> {
    let rest = library_href_rest(href)?;
    let tag_prefix = format!("{model}:");
    let rest = rest.strip_prefix(&tag_prefix)?;
    let tag = rest.split(['?', '#']).next()?.trim_matches('/');

    if tag.is_empty()
        || !tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return None;
    }

    Some(tag.to_string())
}

fn registry_repo_path(model: &str) -> String {
    if model.contains('/') {
        model
            .split('/')
            .map(urlencoded_path_component)
            .collect::<Vec<_>>()
            .join("/")
    } else {
        format!("library/{}", urlencoded_path_component(model))
    }
}

fn urlencoded(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    for byte in query.bytes() {
        match byte {
            b' ' => result.push('+'),
            _ if is_unreserved(byte) => result.push(byte as char),
            _ => result.push_str(&format!("%{byte:02X}")),
        }
    }
    result
}

fn urlencoded_path_component(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    for byte in value.bytes() {
        if is_unreserved(byte) {
            result.push(byte as char);
        } else {
            result.push_str(&format!("%{byte:02X}"));
        }
    }
    result
}

fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

fn normalized_text<'a>(text: impl Iterator<Item = &'a str>) -> String {
    text.collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .map(|ch| if ch.is_ascii_control() { '\u{FFFD}' } else { ch })
        .collect()
}

fn extract_size_from_text(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    for start in 0..chars.len() {
        if !chars[start].is_ascii_digit() {
            continue;
        }
        let mut end = start;
        while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
            end += 1;
        }
        let mut unit_start = end;
        while unit_start < chars.len() && chars[unit_start].is_whitespace() {
            unit_start += 1;
        }
        if unit_start + 1 >= chars.len() {
            continue;
        }
        let unit = chars[unit_start..unit_start + 2]
            .iter()
            .collect::<String>()
            .to_ascii_uppercase();
        if matches!(unit.as_str(), "KB" | "MB" | "GB" | "TB") {
            return Some(chars[start..unit_start + 2].iter().collect());
        }
    }
    None
}

fn natural_tag_order_key(tag: &str) -> String {
    tag.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_search_fallback_links() {
        let html = r#"<a href="/library/qwen2.5-coder">qwen2.5-coder</a><a href="/library/qwen2.5-coder:7b">tag</a>"#;
        let results = parse_search_html(html);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "qwen2.5-coder");
    }

    #[test]
    fn parses_absolute_ollama_library_links() {
        assert_eq!(
            model_name_from_library_href("https://ollama.com/library/qwen3?x=1").as_deref(),
            Some("qwen3")
        );
        assert_eq!(
            model_name_from_library_href("https://www.ollama.com/library/qwen3#tags").as_deref(),
            Some("qwen3")
        );
        assert_eq!(
            tag_from_library_href("qwen3", "https://ollama.com/library/qwen3:14b-instruct-q4_K_M").as_deref(),
            Some("14b-instruct-q4_K_M")
        );
        assert_eq!(model_name_from_library_href("https://example.com/library/qwen3"), None);
    }

    #[test]
    fn parses_structured_search_cards() {
        let html = r#"
            <div x-test-model>
              <a href="/library/qwen3"><span x-test-search-response-title>qwen3</span></a>
              <p>general model</p>
              <span x-test-pull-count>12M Pulls</span>
              <span x-test-tag-count>99 Tags</span>
              <span x-test-updated>2 weeks ago</span>
            </div>
        "#;
        let results = parse_search_html(html);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "qwen3");
        assert_eq!(results[0].description, "general model");
        assert_eq!(results[0].pulls, "12M Pulls");
        assert_eq!(results[0].tag_count, "99 Tags");
        assert_eq!(results[0].updated, "2 weeks ago");
    }

    #[test]
    fn parses_tag_links_without_dynamic_selectors() {
        let html = r#"<a href="/library/qwen2.5-coder:7b-instruct-q4_K_M">7b 4.7GB</a>"#;
        let tags = parse_tags_html("qwen2.5-coder", html);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].tag, "7b-instruct-q4_K_M");
        assert_eq!(tags[0].approx_size.as_deref(), Some("4.7GB"));
    }

    #[test]
    fn ignores_invalid_library_links() {
        assert_eq!(model_name_from_library_href("/library/foo:7b"), None);
        assert_eq!(model_name_from_library_href("/library/ns/foo"), None);
        assert_eq!(tag_from_library_href("foo", "/library/foo:bad/tag"), None);
    }


    #[test]
    fn normalized_text_replaces_terminal_controls() {
        let text = normalized_text(["safe\x1b[2K", "name"].into_iter());
        assert!(!text.contains('\x1b'));
        assert!(text.contains('�'));
    }

    #[test]
    fn fallback_model_href_rejects_control_characters() {
        assert_eq!(model_name_from_library_href("/library/qwen\x1b[2K"), None);
    }

    #[test]
    fn builds_registry_paths() {
        assert_eq!(registry_repo_path("qwen2.5-coder"), "library/qwen2.5-coder");
        assert_eq!(registry_repo_path("acme/model"), "acme/model");
    }

    #[test]
    fn sums_manifest_layers_and_model_layers() {
        let manifest = Manifest {
            layers: vec![
                Layer { media_type: "application/vnd.ollama.image.system".into(), size: 10 },
                Layer { media_type: "application/vnd.ollama.image.model".into(), size: 40 },
                Layer { media_type: "application/vnd.ollama.image.model".into(), size: 50 },
            ],
        };
        assert_eq!(manifest_sizes("m", "t", &manifest).unwrap(), (90, 100));
    }

    #[test]
    fn manifest_size_total_overflow_is_invalid_with_context() {
        let manifest = Manifest {
            layers: vec![
                Layer { media_type: "application/vnd.ollama.image.model".into(), size: u64::MAX },
                Layer { media_type: "application/vnd.ollama.image.system".into(), size: 1 },
            ],
        };
        let err = manifest_sizes("m", "t", &manifest).unwrap_err();
        assert!(matches!(
            err,
            ResolverError::ManifestInvalid { model, tag, detail }
                if model == "m" && tag == "t" && detail.contains("total bytes")
        ));
    }

    #[test]
    fn manifest_size_model_layer_overflow_is_invalid_with_context() {
        let manifest = Manifest {
            layers: vec![
                Layer { media_type: "application/vnd.ollama.image.model".into(), size: u64::MAX },
                Layer { media_type: "application/vnd.ollama.image.model".into(), size: 1 },
            ],
        };
        let err = manifest_sizes("m", "t", &manifest).unwrap_err();
        assert!(matches!(
            err,
            ResolverError::ManifestInvalid { model, tag, detail }
                if model == "m" && tag == "t" && detail.contains("weights")
        ));
    }

    #[test]
    fn missing_model_layer_is_manifest_invalid_with_context() {
        let manifest = Manifest {
            layers: vec![Layer { media_type: "application/vnd.ollama.image.system".into(), size: 10 }],
        };
        let err = manifest_sizes("m", "t", &manifest).unwrap_err();
        assert!(matches!(
            err,
            ResolverError::ManifestInvalid { model, tag, .. }
                if model == "m" && tag == "t"
        ));
    }
}
