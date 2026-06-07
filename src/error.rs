use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResolverError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("failed to parse HTML from {url}: {detail}")]
    HtmlParse { url: String, detail: String },

    #[error("no models found matching '{query}'")]
    NoSearchResults { query: String },

    #[error("model search for '{query}' did not produce an exact match; candidates: {candidates}")]
    AmbiguousModel { query: String, candidates: String },

    #[error("no tags found for model '{model}'")]
    NoTags { model: String },

    #[error("manifest not found for '{model}:{tag}': {detail}")]
    ManifestMissing { model: String, tag: String, detail: String },

    #[error("registry manifest unavailable for '{model}:{tag}': {detail}")]
    ManifestUnavailable { model: String, tag: String, detail: String },

    #[error("cloud-only model — no local weights for '{model}:{tag}'")]
    ManifestCloudOnly { model: String, tag: String },

    #[error("platform-restricted model for '{model}:{tag}' (registry returned {status})")]
    ManifestPlatformRestricted { model: String, tag: String, status: u16 },

    #[error("invalid registry manifest for '{model}:{tag}': {detail}")]
    ManifestInvalid { model: String, tag: String, detail: String },

    #[error("no usable manifest found for model '{model}'. Tried manifests:
{attempts}")]
    NoUsableManifest { model: String, attempts: String },

    #[error("failed to detect system RAM: {0}")]
    RamDetection(String),

    #[error("failed to detect disk space: {0}")]
    DiskDetection(String),

    #[error("ollama is not reachable at {base_url}: {detail}")]
    OllamaUnreachable { base_url: String, detail: String },

    #[error("ollama pull failed for '{model}': {detail}")]
    PullFailed { model: String, detail: String },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ResolverError>;

