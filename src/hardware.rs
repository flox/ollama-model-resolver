use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{ResolverError, Result};
use crate::types::{GpuFitPolicy, GpuInfo, HardwareProfile};

pub fn detect(models_dir: Option<PathBuf>) -> Result<HardwareProfile> {
    detect_with_policy(models_dir, GpuFitPolicy::Best)
}

#[allow(deprecated)]
pub fn detect_with_policy(
    models_dir: Option<PathBuf>,
    gpu_fit_policy: GpuFitPolicy,
) -> Result<HardwareProfile> {
    let models_dir = models_dir.unwrap_or_else(default_models_dir);
    let unified_total = detect_unified_mem();
    let gpus = detect_gpus();
    let cuda_visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();

    let (gpu_name, vram_total, vram_free, selected_gpu_indices) = if let Some(total) = unified_total {
        select_unified_fit_basis(total)
    } else {
        select_gpu_fit_basis(&gpus, gpu_fit_policy)
    };

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
        unified_mem_total: unified_total.unwrap_or(0),
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
    // Ollama serves only one model at a time so vram_free is irrelevant.
    // Pick the CUDA-visible GPU with the most total VRAM.
    let best = gpus
        .iter()
        .filter(|gpu| gpu.visible)
        .max_by_key(|gpu| (gpu.vram_total, gpu.index));

    best.map(|gpu| {
        (
            Some(format!("{} (GPU {})", gpu.name, gpu.index)),
            gpu.vram_total,
            gpu.vram_total,
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
        .fold(0_u64, |acc, gpu| acc.saturating_add(gpu.vram_total));
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

#[cfg(target_os = "macos")]
fn detect_ram() -> Result<(u64, u64)> {
    // Read hw.memsize directly for the RAM total rather than threading it back
    // out of detect_unified_mem(). Both read the same sysctl; reading it here
    // keeps ram_total and the unified-pool total from drifting if the
    // Apple-Silicon detection in detect_unified_mem ever changes.
    let ram_total = sysctl_value("hw.memsize")
        .ok_or_else(|| ResolverError::RamDetection("hw.memsize unavailable".into()))?;
    let ram_available = darwin_free_ram_bytes().unwrap_or(0);
    Ok((ram_total, ram_available))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_ram() -> Result<(u64, u64)> {
    Err(ResolverError::RamDetection(
        "RAM detection is not implemented on this platform".into(),
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

/// Select GPU fit basis from Apple Silicon unified memory. The fit ceiling comes
/// from ram_available on the profile (see HardwareProfile::unified_fit_ceiling);
/// vram_free is the deprecated, always-equal-to-vram_total field, so it mirrors
/// the pool total here rather than carrying a separate "free" value.
fn select_unified_fit_basis(total: u64) -> (Option<String>, u64, u64, Vec<u32>) {
    (
        Some("Apple Silicon (Unified)".into()),
        total,
        total,
        vec![0],
    )
}

/// Detect Apple Silicon unified memory via sysctl.
/// Returns Some(total_bytes) on Apple Silicon; None on Intel Macs and Linux.
#[cfg(target_os = "macos")]
fn detect_unified_mem() -> Option<u64> {
    // hw.optional.arm64 == 1 identifies Apple Silicon (all M-series SoCs).
    // We deliberately do NOT key off hw.cpufamily: that value differs per chip
    // generation (M1, M2, M3, M4 each report a distinct family), so an equality
    // check against any single constant only matches one generation. Intel Macs
    // report 0 / absent here and fall through to the discrete-GPU path.
    if sysctl_value("hw.optional.arm64").unwrap_or(0) != 1 {
        return None;
    }
    sysctl_value("hw.memsize")
}

/// Non-macos stub: no unified memory.
#[cfg(not(target_os = "macos"))]
fn detect_unified_mem() -> Option<u64> {
    None
}

/// Read an integer sysctl value by name. Handles both 4-byte (int/uint32, e.g.
/// `hw.cpufamily`, `hw.ncpu`, `hw.optional.arm64`) and 8-byte (e.g. `hw.memsize`,
/// and `hw.pagesize`, which is a 64-bit long on 64-bit macOS) values.
#[cfg(target_os = "macos")]
fn sysctl_value(name: &str) -> Option<u64> {
    use std::ffi::CString;
    let c = CString::new(name).ok()?;
    let mut len: usize = 0;
    // sysctlbyname returns 0 on success, -1 on error (errno set).
    let rc = unsafe {
        libc::sysctlbyname(
            c.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    let rc = unsafe {
        libc::sysctlbyname(
            c.as_ptr(),
            buf.as_mut_ptr() as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    match len {
        8 => Some(u64::from_ne_bytes(buf[..8].try_into().ok()?)),
        4 => Some(u32::from_ne_bytes(buf[..4].try_into().ok()?) as u64),
        _ => None,
    }
}

/// Get available RAM on darwin. Returns available bytes, or None on failure.
///
/// This is the darwin analogue of Linux's /proc/meminfo MemAvailable. Counting
/// only free pages drastically understates available memory on macOS, which
/// keeps most RAM mapped as active/inactive/cached and reclaims it on demand.
/// Instead we mirror the kernel's own "memory free" notion (what Activity
/// Monitor and `memory_pressure` report): total RAM minus the pages that cannot
/// be reclaimed for a new allocation — wired pages and pages already held by the
/// compressor.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc::mach_host_self is the standard accessor; mach2 is not a dependency.
fn darwin_free_ram_bytes() -> Option<u64> {
    use std::mem;

    // Page size in bytes via sysconf(_SC_PAGESIZE).
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    let page_size = page_size as u64;

    let total = sysctl_value("hw.memsize")?;

    // host_statistics64 with HOST_VM_INFO64 retrieves vm_statistics64_data_t.
    let mut vm_stat: libc::vm_statistics64_data_t = unsafe { mem::zeroed() };
    let mut count: libc::mach_msg_type_number_t =
        (mem::size_of::<libc::vm_statistics64_data_t>() / mem::size_of::<i32>()) as libc::mach_msg_type_number_t;

    let host = unsafe { libc::mach_host_self() };
    let ret = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64 as i32,
            &mut vm_stat as *mut libc::vm_statistics64_data_t as *mut libc::integer_t,
            &mut count,
        )
    };

    // KERN_SUCCESS == 0 on darwin
    if ret != 0 {
        return None;
    }

    // Non-reclaimable pages: wired (kernel/locked) plus pages occupied by the
    // compressor. Everything else (free, active, inactive, speculative,
    // purgeable, file-backed) can be made available under memory pressure.
    let unavailable_pages =
        (vm_stat.wire_count as u64).saturating_add(vm_stat.compressor_page_count as u64);
    let unavailable_bytes = unavailable_pages.saturating_mul(page_size);
    Some(total.saturating_sub(unavailable_bytes))
}

#[allow(deprecated)]
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_meminfo_line() {
        assert_eq!(parse_meminfo_kb("MemAvailable:   12345 kB"), Some(12345));
    }

    // hw.memsize is a uint64 sysctl, exercising the 8-byte path of sysctl_value.
    #[cfg(target_os = "macos")]
    #[test]
    fn sysctl_value_reads_8_byte_memsize() {
        let memsize = sysctl_value("hw.memsize").expect("hw.memsize should be readable");
        assert!(memsize >= 1 << 30, "memsize unexpectedly small: {memsize}");
    }

    // hw.ncpu is a 4-byte int sysctl, exercising the 4-byte path of sysctl_value.
    // (hw.pagesize is NOT 4-byte on 64-bit macOS — it returns 8 bytes — so it
    // would exercise the wrong arm.) This is the same arm hw.optional.arm64 and
    // hw.cpufamily use, which the Apple-Silicon detection depends on.
    #[cfg(target_os = "macos")]
    #[test]
    fn sysctl_value_reads_4_byte_ncpu() {
        let ncpu = sysctl_value("hw.ncpu").expect("hw.ncpu should be readable");
        assert!(ncpu > 0, "hw.ncpu should be positive");
        assert!(ncpu <= 4096, "hw.ncpu implausibly large: {ncpu}");
    }

    // Unknown name: sysctlbyname fails on the first (sizing) call, so we get None.
    #[cfg(target_os = "macos")]
    #[test]
    fn sysctl_value_returns_none_for_unknown_name() {
        assert_eq!(sysctl_value("definitely.not.a.real.sysctl.xyzzy"), None);
    }

    // An interior NUL makes CString::new fail; sysctl_value must return None, not panic.
    #[cfg(target_os = "macos")]
    #[test]
    fn sysctl_value_returns_none_for_name_with_interior_nul() {
        assert_eq!(sysctl_value("hw.mem\0size"), None);
    }

    // Available RAM must be positive and never exceed installed RAM.
    #[cfg(target_os = "macos")]
    #[test]
    fn darwin_free_ram_is_positive_and_within_total() {
        let total = sysctl_value("hw.memsize").expect("hw.memsize should be readable");
        let available =
            darwin_free_ram_bytes().expect("darwin_free_ram_bytes should succeed on macOS");
        assert!(available > 0, "available RAM should be positive");
        assert!(
            available <= total,
            "available {available} should not exceed total {total}"
        );
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
    fn best_policy_uses_most_total_visible_gpu() {
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

        let (_, total, fit_capacity, indices) = select_gpu_fit_basis(&gpus, GpuFitPolicy::Best);
        // Ollama serves one model at a time, so fit basis is total VRAM
        assert_eq!((total, fit_capacity, indices), (200, 200, vec![1]));
    }

    #[test]
    fn visible_sum_policy_sums_total_visible_gpus() {
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

        let (_, total, fit_capacity, indices) = select_gpu_fit_basis(&gpus, GpuFitPolicy::VisibleSum);
        assert_eq!((total, fit_capacity, indices), (100, 100, vec![0]));
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

        let (_, total, fit_capacity, indices) = select_gpu_fit_basis(&gpus, GpuFitPolicy::AllSum);
        assert_eq!((total, fit_capacity, indices), (300, 300, vec![0, 1]));
    }
}
