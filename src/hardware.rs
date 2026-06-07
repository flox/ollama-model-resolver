use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{ResolverError, Result};
use crate::types::{GpuFitPolicy, GpuInfo, HardwareProfile};

pub fn detect(models_dir: Option<PathBuf>) -> Result<HardwareProfile> {
    detect_with_policy(models_dir, GpuFitPolicy::Best)
}

pub fn detect_with_policy(
    models_dir: Option<PathBuf>,
    gpu_fit_policy: GpuFitPolicy,
) -> Result<HardwareProfile> {
    let models_dir = models_dir.unwrap_or_else(default_models_dir);
    let gpus = detect_gpus();
    let cuda_visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let (gpu_name, vram_total, vram_free, selected_gpu_indices) =
        select_gpu_fit_basis(&gpus, gpu_fit_policy);
    let (ram_total, ram_available) = detect_ram()?;
    let disk_free = detect_disk(&models_dir)?;

    Ok(HardwareProfile {
        gpu_name,
        vram_total,
        vram_free,
        ram_total,
        ram_available,
        disk_free,
        models_dir,
        gpus,
        selected_gpu_indices,
        cuda_visible_devices,
        gpu_fit_policy,
    })
}

fn default_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        PathBuf::from(dir)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".ollama").join("models")
    } else {
        PathBuf::from("/tmp")
    }
}

fn detect_gpus() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };

    parse_nvidia_smi_csv(&String::from_utf8_lossy(&output.stdout))
}

fn parse_nvidia_smi_csv(stdout: &str) -> Vec<GpuInfo> {
    let visibility = CudaVisibility::from_env();
    parse_nvidia_smi_csv_with_visibility(stdout, &visibility)
}

fn parse_nvidia_smi_csv_with_visibility(stdout: &str, visibility: &CudaVisibility) -> Vec<GpuInfo> {
    stdout
        .lines()
        .filter_map(|line| parse_gpu_line(line, visibility))
        .collect()
}

fn parse_gpu_line(line: &str, visibility: &CudaVisibility) -> Option<GpuInfo> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 5 {
        return None;
    }

    let index = parts[0].parse::<u32>().ok()?;
    let uuid = normalize_uuid(parts[1]);
    let name = parts[2].to_string();
    let total_mib = parts[3].parse::<u64>().ok()?;
    let free_mib = parts[4].parse::<u64>().ok()?;
    let visible = visibility.includes(index, uuid.as_deref());

    Some(GpuInfo {
        index,
        uuid,
        name,
        vram_total: total_mib.saturating_mul(1024 * 1024),
        vram_free: free_mib.saturating_mul(1024 * 1024),
        visible,
    })
}

fn normalize_uuid(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("[not supported]") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CudaVisibility {
    All,
    None,
    Tokens(Vec<String>),
}

impl CudaVisibility {
    fn from_env() -> Self {
        match std::env::var("CUDA_VISIBLE_DEVICES") {
            Ok(value) => Self::from_value(&value),
            Err(_) => Self::All,
        }
    }

    fn from_value(value: &str) -> Self {
        let trimmed = value.trim();
        if trimmed.is_empty()
            || trimmed == "-1"
            || trimmed.eq_ignore_ascii_case("none")
            || trimmed.eq_ignore_ascii_case("NoDevFiles")
        {
            return Self::None;
        }

        let tokens = trimmed
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();

        if tokens.is_empty() {
            Self::None
        } else {
            Self::Tokens(tokens)
        }
    }

    fn includes(&self, index: u32, uuid: Option<&str>) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Tokens(tokens) => tokens.iter().any(|token| {
                token == &index.to_string()
                    || uuid
                        .map(|gpu_uuid| uuid_matches_token(gpu_uuid, token))
                        .unwrap_or(false)
            }),
        }
    }
}

fn uuid_matches_token(uuid: &str, token: &str) -> bool {
    uuid == token || uuid.strip_prefix("GPU-").map(|short| short.starts_with(token)).unwrap_or(false)
}

fn select_gpu_fit_basis(
    gpus: &[GpuInfo],
    policy: GpuFitPolicy,
) -> (Option<String>, u64, u64, Vec<u32>) {
    match policy {
        GpuFitPolicy::Best => select_best_visible_gpu(gpus),
        GpuFitPolicy::VisibleSum => select_sum(gpus.iter().filter(|gpu| gpu.visible), false),
        GpuFitPolicy::AllSum => select_sum(gpus.iter(), true),
    }
}

fn select_best_visible_gpu(gpus: &[GpuInfo]) -> (Option<String>, u64, u64, Vec<u32>) {
    let best = gpus
        .iter()
        .filter(|gpu| gpu.visible)
        .max_by_key(|gpu| (gpu.vram_free, gpu.vram_total));

    best.map(|gpu| {
        (
            Some(format!("{} (GPU {})", gpu.name, gpu.index)),
            gpu.vram_total,
            gpu.vram_free,
            vec![gpu.index],
        )
    })
    .unwrap_or((None, 0, 0, Vec::new()))
}

fn select_sum<'a, I>(gpus: I, include_hidden: bool) -> (Option<String>, u64, u64, Vec<u32>)
where
    I: Iterator<Item = &'a GpuInfo>,
{
    let selected = gpus.collect::<Vec<_>>();
    if selected.is_empty() {
        return (None, 0, 0, Vec::new());
    }

    let total = selected
        .iter()
        .fold(0_u64, |acc, gpu| acc.saturating_add(gpu.vram_total));
    let free = selected
        .iter()
        .fold(0_u64, |acc, gpu| acc.saturating_add(gpu.vram_free));
    let indices = selected.iter().map(|gpu| gpu.index).collect::<Vec<_>>();
    let label = if selected.len() == 1 {
        format!("{} (GPU {})", selected[0].name, selected[0].index)
    } else if include_hidden {
        format!("{} detected NVIDIA GPUs (aggregate)", selected.len())
    } else {
        format!("{} CUDA-visible NVIDIA GPUs (aggregate)", selected.len())
    };

    (Some(label), total, free, indices)
}

#[cfg(target_os = "linux")]
fn detect_ram() -> Result<(u64, u64)> {
    let contents = std::fs::read_to_string("/proc/meminfo")
        .map_err(|err| ResolverError::RamDetection(err.to_string()))?;

    let mut total: Option<u64> = None;
    let mut available: Option<u64> = None;

    for line in contents.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            available = parse_meminfo_kb(line);
        }
    }

    let total = total.ok_or_else(|| ResolverError::RamDetection("MemTotal not found".into()))?;
    let available =
        available.ok_or_else(|| ResolverError::RamDetection("MemAvailable not found".into()))?;

    Ok((total.saturating_mul(1024), available.saturating_mul(1024)))
}

#[cfg(not(target_os = "linux"))]
fn detect_ram() -> Result<(u64, u64)> {
    Err(ResolverError::RamDetection(
        "only Linux /proc/meminfo RAM detection is implemented in v0.1.0".into(),
    ))
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

fn detect_disk(path: &Path) -> Result<u64> {
    let check_path = existing_ancestor(path).ok_or_else(|| {
        ResolverError::DiskDetection(format!("no existing ancestor for {}", path.display()))
    })?;

    let c_path = std::ffi::CString::new(check_path.to_string_lossy().as_bytes())
        .map_err(|err| ResolverError::DiskDetection(err.to_string()))?;

    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return Err(ResolverError::DiskDetection(format!(
                "statvfs failed on {}",
                check_path.display()
            )));
        }

        let bytes = (stat.f_bavail as u128).saturating_mul(stat.f_frsize as u128);
        Ok(bytes.min(u64::MAX as u128) as u64)
    }
}

fn existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = path.to_path_buf();
    loop {
        if current.exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_meminfo_line() {
        assert_eq!(parse_meminfo_kb("MemAvailable:   12345 kB"), Some(12345));
    }

    #[test]
    fn parses_nvidia_smi_inventory_and_visibility_by_index() {
        let visibility = CudaVisibility::from_value("1");
        let gpus = parse_nvidia_smi_csv_with_visibility(
            "0, GPU-aaaa, NVIDIA A, 24576, 12000\n1, GPU-bbbb, NVIDIA B, 49152, 48000\n",
            &visibility,
        );

        assert_eq!(gpus.len(), 2);
        assert!(!gpus[0].visible);
        assert!(gpus[1].visible);
    }

    #[test]
    fn cuda_visible_devices_empty_hides_all_gpus() {
        let visibility = CudaVisibility::from_value("");
        assert!(!visibility.includes(0, Some("GPU-abc")));
    }

    #[test]
    fn cuda_visible_devices_matches_uuid_prefix() {
        let visibility = CudaVisibility::from_value("abc");
        assert!(visibility.includes(3, Some("GPU-abcdef")));
    }

    #[test]
    fn best_policy_uses_most_free_visible_gpu() {
        let gpus = vec![
            GpuInfo {
                index: 0,
                uuid: None,
                name: "A".into(),
                vram_total: 100,
                vram_free: 50,
                visible: true,
            },
            GpuInfo {
                index: 1,
                uuid: None,
                name: "B".into(),
                vram_total: 200,
                vram_free: 80,
                visible: true,
            },
        ];

        let (_, total, free, indices) = select_gpu_fit_basis(&gpus, GpuFitPolicy::Best);
        assert_eq!((total, free, indices), (200, 80, vec![1]));
    }

    #[test]
    fn visible_sum_policy_sums_visible_gpus_only() {
        let gpus = vec![
            GpuInfo {
                index: 0,
                uuid: None,
                name: "A".into(),
                vram_total: 100,
                vram_free: 50,
                visible: true,
            },
            GpuInfo {
                index: 1,
                uuid: None,
                name: "B".into(),
                vram_total: 200,
                vram_free: 80,
                visible: false,
            },
        ];

        let (_, total, free, indices) = select_gpu_fit_basis(&gpus, GpuFitPolicy::VisibleSum);
        assert_eq!((total, free, indices), (100, 50, vec![0]));
    }

    #[test]
    fn all_sum_policy_ignores_cuda_visibility() {
        let gpus = vec![
            GpuInfo {
                index: 0,
                uuid: None,
                name: "A".into(),
                vram_total: 100,
                vram_free: 50,
                visible: true,
            },
            GpuInfo {
                index: 1,
                uuid: None,
                name: "B".into(),
                vram_total: 200,
                vram_free: 80,
                visible: false,
            },
        ];

        let (_, total, free, indices) = select_gpu_fit_basis(&gpus, GpuFitPolicy::AllSum);
        assert_eq!((total, free, indices), (300, 130, vec![0, 1]));
    }
}
