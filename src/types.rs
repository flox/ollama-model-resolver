use std::fmt;
use std::path::PathBuf;

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GpuFitPolicy {
    /// Use the single visible NVIDIA GPU with the most free VRAM.
    Best,
    /// Sum free VRAM across CUDA-visible NVIDIA GPUs for multi-GPU fit estimates.
    VisibleSum,
    /// Sum free VRAM across all detected NVIDIA GPUs, ignoring CUDA_VISIBLE_DEVICES.
    AllSum,
}

impl fmt::Display for GpuFitPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            GpuFitPolicy::Best => "best",
            GpuFitPolicy::VisibleSum => "visible-sum",
            GpuFitPolicy::AllSum => "all-sum",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    pub index: u32,
    pub uuid: Option<String>,
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
    pub visible: bool,
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub gpu_name: Option<String>,
    pub vram_total: u64,
    pub vram_free: u64,
    pub ram_total: u64,
    pub ram_available: u64,
    pub disk_free: u64,
    pub models_dir: PathBuf,
    pub gpus: Vec<GpuInfo>,
    pub selected_gpu_indices: Vec<u32>,
    pub cuda_visible_devices: Option<String>,
    pub gpu_fit_policy: GpuFitPolicy,
}

impl HardwareProfile {
    pub fn has_gpu(&self) -> bool {
        self.vram_total > 0 && self.vram_free > 0 && !self.selected_gpu_indices.is_empty()
    }

    pub fn visible_gpu_count(&self) -> usize {
        self.gpus.iter().filter(|gpu| gpu.visible).count()
    }

    pub fn selected_gpu_count(&self) -> usize {
        self.selected_gpu_indices.len()
    }

    pub fn available_runtime_bytes(&self, allow_split: bool) -> u64 {
        if self.has_gpu() {
            if allow_split {
                self.vram_free.saturating_add(self.ram_available)
            } else {
                self.vram_free
            }
        } else {
            self.ram_available
        }
    }

    pub fn gpu_fit_basis(&self) -> String {
        if !self.has_gpu() {
            if self.gpus.is_empty() {
                return "system RAM (CPU-only; no NVIDIA GPU detected)".to_string();
            }
            return "system RAM (no CUDA-visible NVIDIA GPU selected)".to_string();
        }

        match self.gpu_fit_policy {
            GpuFitPolicy::Best => "single CUDA-visible NVIDIA GPU with the most free VRAM".to_string(),
            GpuFitPolicy::VisibleSum => "sum of CUDA-visible NVIDIA GPU free VRAM".to_string(),
            GpuFitPolicy::AllSum => "sum of all detected NVIDIA GPU free VRAM; ignores CUDA_VISIBLE_DEVICES".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelVariant {
    pub name: String,
    pub tag: String,
    pub full_ref: String,
    pub weights_bytes: u64,
    pub total_bytes: u64,
    pub estimated_runtime_bytes: u64,
    pub runtime_margin_pct: u32,
    pub context_tokens: u32,
    pub param_billions: Option<f64>,
    pub quantization: Option<String>,
    pub is_instruct: bool,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub name: String,
    pub description: String,
    pub pulls: String,
    pub tag_count: String,
    pub updated: String,
}

#[derive(Debug, Clone)]
pub struct AnnotatedSearchResult {
    pub result: SearchResult,
    pub variant: Option<ModelVariant>,
    pub fit: Option<FitResult>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TagInfo {
    pub tag: String,
    pub full_ref: String,
    pub approx_size: Option<String>,
    pub param_billions: Option<f64>,
    pub quantization: Option<String>,
    pub is_instruct: bool,
}

impl TagInfo {
    pub fn has_default_quantization(&self) -> bool {
        self.tag == "latest"
            || self
                .quantization
                .as_deref()
                .map(|q| q.eq_ignore_ascii_case("q4_k_m"))
                .unwrap_or(true)
    }

    pub fn approx_size_bytes(&self) -> Option<u64> {
        self.approx_size.as_deref().and_then(parse_human_size_bytes)
    }
}

#[derive(Debug, Clone)]
pub enum FitResult {
    FitsVram,
    FitsWithSplit { gpu_pct: f64 },
    FitsRamOnly,
    DoesNotFit { need: u64, have: u64 },
    InsufficientDisk { need: u64, have: u64 },
}

impl FitResult {
    pub fn fits(&self) -> bool {
        matches!(
            self,
            FitResult::FitsVram | FitResult::FitsWithSplit { .. } | FitResult::FitsRamOnly
        )
    }

    pub fn summary(&self) -> String {
        match self {
            FitResult::FitsVram => "fits GPU VRAM basis".to_string(),
            FitResult::FitsWithSplit { gpu_pct } => format!("fits with split ({gpu_pct:.0}% GPU)"),
            FitResult::FitsRamOnly => "fits RAM / CPU".to_string(),
            FitResult::DoesNotFit { .. } => "does not fit".to_string(),
            FitResult::InsufficientDisk { .. } => "estimated disk shortfall".to_string(),
        }
    }
}

/// Parse a parameter count from a tag string.
/// Examples: "7b" -> 7.0, "0.5b" -> 0.5, "8x7b" -> 56.0.
///
/// Ollama tags are not standardized. This parser is intentionally best-effort;
/// callers must not treat a missing value as an error.
pub fn parse_param_billions(tag: &str) -> Option<f64> {
    for raw_segment in tag.split(['-', ':']) {
        let segment = raw_segment.to_ascii_lowercase();
        if let Some(value) = parse_b_suffix(&segment) {
            return Some(value);
        }
        if let Some((left, right)) = segment.split_once('x') {
            let experts = left.parse::<f64>().ok()?;
            let each = parse_b_suffix(right)?;
            return Some(experts * each);
        }
    }
    None
}

fn parse_b_suffix(segment: &str) -> Option<f64> {
    let number = segment.strip_suffix('b')?;
    if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    number.parse::<f64>().ok()
}

/// Parse common Ollama quantization labels from a tag string.
///
/// Ollama tags are not standardized. This parser is intentionally best-effort;
/// callers must not treat a missing value as an error.
pub fn parse_quantization(tag: &str) -> Option<String> {
    for part in tag.split('-') {
        let lower = part.to_ascii_lowercase();
        let bytes = lower.as_bytes();
        if bytes.len() > 2 && bytes[0] == b'q' && bytes[1].is_ascii_digit() {
            return Some(part.to_string());
        }
        if matches!(lower.as_str(), "fp16" | "fp32" | "f16" | "f32") {
            return Some(part.to_string());
        }
    }
    None
}


/// Parse an Ollama tag-page size label such as "4.7GB" or "512 MB".
///
/// Ollama displays these values as human-readable decimal units. The result is
/// only an upstream page hint; registry manifests remain the source used for
/// final fit math.
pub fn parse_human_size_bytes(value: &str) -> Option<u64> {
    let compact = value.trim().replace(' ', "");
    if compact.is_empty() {
        return None;
    }

    let split_at = compact
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_ascii_digit() && ch != '.').then_some(idx))?;
    let (number, unit) = compact.split_at(split_at);
    if number.is_empty() || number.matches('.').count() > 1 {
        return None;
    }

    let multiplier: u128 = match unit.to_ascii_lowercase().as_str() {
        "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "tb" => 1_000_000_000_000,
        _ => return None,
    };

    let (whole, frac) = number.split_once('.').unwrap_or((number, ""));
    let whole = whole.parse::<u128>().ok()?;
    let frac_digits = frac.chars().take_while(|ch| ch.is_ascii_digit()).collect::<String>();
    if frac_digits.len() != frac.len() {
        return None;
    }

    let frac_value = if frac_digits.is_empty() {
        0
    } else {
        let scale = 10_u128.checked_pow(frac_digits.len() as u32)?;
        frac_digits.parse::<u128>().ok()?.saturating_mul(multiplier) / scale
    };

    whole
        .saturating_mul(multiplier)
        .saturating_add(frac_value)
        .try_into()
        .ok()
}

pub fn is_instruct_tag(tag: &str) -> bool {
    let lower = tag.to_ascii_lowercase();
    !lower.contains("base")
}

pub fn tag_info_from_str(model: &str, tag: &str) -> TagInfo {
    TagInfo {
        tag: tag.to_string(),
        full_ref: format!("{model}:{tag}"),
        approx_size: None,
        param_billions: parse_param_billions(tag),
        quantization: parse_quantization(tag),
        is_instruct: is_instruct_tag(tag),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_parameter_counts() {
        assert_eq!(parse_param_billions("7b"), Some(7.0));
        assert_eq!(parse_param_billions("0.5b-instruct-q4_K_M"), Some(0.5));
        assert_eq!(parse_param_billions("8x7b-q4_0"), Some(56.0));
        assert_eq!(parse_param_billions("latest"), None);
    }

    #[test]
    fn parses_quantization_labels() {
        assert_eq!(parse_quantization("14b-instruct-q4_K_M"), Some("q4_K_M".into()));
        assert_eq!(parse_quantization("70b-fp16"), Some("fp16".into()));
        assert_eq!(parse_quantization("7b"), None);
    }

    #[test]
    fn parses_tag_page_size_hints() {
        assert_eq!(parse_human_size_bytes("4.7GB"), Some(4_700_000_000));
        assert_eq!(parse_human_size_bytes("512 MB"), Some(512_000_000));
        assert_eq!(parse_human_size_bytes("n/a"), None);
    }

    #[test]
    fn exposes_approx_size_bytes_on_tags() {
        let mut tag = tag_info_from_str("m", "7b");
        tag.approx_size = Some("1.5GB".into());
        assert_eq!(tag.approx_size_bytes(), Some(1_500_000_000));
    }

    #[test]
    fn marks_base_tags_lower_priority() {
        assert!(is_instruct_tag("7b-instruct-q4_K_M"));
        assert!(!is_instruct_tag("7b-base-q4_K_M"));
    }
}
