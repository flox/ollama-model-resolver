# Plan: macOS / Apple Silicon Unified Memory Detection

## Current state on darwin (broken)

1. `detect_gpus()` calls `nvidia-smi` → returns empty vec on darwin (no NVIDIA hardware).
2. `detect_ram()` is `#[cfg(not(target_os = "linux"))]` → returns `Err` on darwin.
3. `detect_disk()` uses `statvfs` via `libc` → works on darwin (POSIX). No change needed.
4. `select_gpu_fit_basis()` → `(None, 0, 0, [])` on darwin.
5. `check_fit()` → `hw.has_gpu()` is false (both `vram_total == 0` and `selected_gpu_indices` empty) → falls to `FitsRamOnly` check against `hw.ram_available`.
6. **But** `detect_ram()` already errored, so `cmd_search` / `cmd_resolve` / `cmd_info` all fail early with `RamDetection` error.

**Bottom line: darwin is completely non-functional. `info` shows nothing. `resolve` and `search --fit` both error.**

## What needs to change

### 1. `HardwareProfile` fields (types.rs)

Add:
```rust
pub unified_mem_total: u64,   // Apple Silicon total unified memory (bytes)
pub unified_mem_free: u64,    // Apple Silicon free unified memory (bytes)
```

The plan to add these is correct. But note: we should **also** repurpose `vram_total` from unified memory — not add a separate path in `check_fit()`. The existing `hw.vram_total` is the canonical "GPU resource" in all downstream code (`has_gpu()`, `check_fit()`, `print_hardware()`, `available_runtime_bytes()`). Setting it from unified memory makes all downstream code work unchanged.

### 2. `HardwareProfile` methods (types.rs)

The plan to add `has_unified_mem()` is unnecessary and over-engineered. Instead:

- Don't add `has_unified_mem()` at all.
- Set `vram_total = unified_mem_total` and `selected_gpu_indices = [0]` from unified memory. Then `has_gpu()` (types.rs:56-60) returns true on darwin, which is correct — unified memory IS the GPU resource on Apple Silicon.
- `has_gpu()` currently requires `!selected_gpu_indices.is_empty()`. Setting it to `[0]` on darwin is semantically misleading (it's not a CUDA GPU index), but harmless in practice. A cleaner approach: change `has_gpu()` to check `vram_total > 0` AND one of (`!selected_gpu_indices.is_empty()` OR `unified_mem_total > 0`). But that requires the new field first.

**Revised**: Add the two fields but keep `has_gpu()` as-is, and set `vram_total` from unified memory. On darwin, `has_gpu()` will return true (because `vram_total > 0` and `selected_gpu_indices = [0]`). This means the existing `check_fit()` NVIDIA path works unchanged.

### 3. `detect_ram()` — darwin path (hardware.rs)

The plan's approach is **wrong**. Instead of a separate `detect_unified_mem()` function, **replace** the existing `#[cfg(not(target_os = "linux"))]` fallback:

```rust
#[cfg(target_os = "macos")]
fn detect_ram() -> Result<(u64, u64)> {
    // On darwin, return (unified_total, 0) since unified pool is the primary resource
    // ram_total = unified total, ram_available = 0 (conservative)
    let unified = detect_unified_mem().unwrap_or((0, 0));
    Ok((unified.0, 0))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_ram() -> Result<(u64, u64)> {
    Err(ResolverError::RamDetection(
        "RAM detection is not implemented on this platform".into(),
    ))
}
```

For `ram_available` on darwin, the plan suggests `sysctl vm.pageable_purgeable_page_count × PAGE_SIZE`. **This is wrong** — purgable pages are cache that can be evicted, not available memory. A better v1 approach: use a conservative heuristic like `hw.memsize * 0.3` (30% of total) since there's no clean way to get "available" memory from a Rust binary on darwin without `mach` imports.

Actually, there's a better approach: use `sysctl hw.memsize` for total, and report `ram_available` as **unknown** (0) — meaning "don't use --split, rely on unified total only". Or use `sysctl vm.loadavg` to estimate, but that's imprecise. For v1, conservative is better than overconfident.

**Best approach for v1**:
- `ram_total = unified_mem_total` (from `hw.memsize`)
- `ram_available = 0` (not estimated — user must rely on unified total for fit)
- Document in `info` that "RAM available" is not estimated on macOS

Wait, but that means `--split` won't work on darwin at all, because `vram_total + ram_available = vram_total + 0 = vram_total`. That's fine for v1 — `--split` is unnecessary on darwin since the unified total IS the resource. The split path would only be relevant if a model exceeds the unified pool.

### 4. Unified memory detection in `detect_with_policy()` (hardware.rs)

The plan's approach is correct in spirit. Revised implementation:

```rust
#[cfg(target_os = "macos")]
fn detect_with_policy(
    models_dir: Option<PathBuf>,
    gpu_fit_policy: GpuFitPolicy,
) -> Result<HardwareProfile> {
    let unified = detect_unified_mem(); // Option<(u64, u64)>
    let models_dir = models_dir.unwrap_or_else(default_models_dir);
    let gpus = detect_gpus(); // empty on darwin
    let cuda_visible_devices = None; // not applicable on darwin
    let (gpu_name, vram_total, vram_free, selected_gpu_indices) =
        select_unified_fit_basis(&unified);
    let (ram_total, ram_available) = detect_ram()?; // darwin branch
    let disk_free = detect_disk(&models_dir)?;

    Ok(HardwareProfile {
        gpu_name,
        vram_total,
        vram_free,
        ram_total,
        ram_available,
        disk_free,
        unified_mem_total: unified.map_or(0, |u| u.0),
        unified_mem_free: unified.map_or(0, |u| u.1),
        models_dir,
        gpus,
        selected_gpu_indices,
        cuda_visible_devices,
        gpu_fit_policy,
    })
}
```

Wait — this duplicates the entire `detect_with_policy` function with a cfg guard. That's ugly. **Better approach**: use a single `detect_with_policy` with platform-specific helpers:

```rust
// In detect_with_policy():
let unified = detect_unified_mem(); // Option<(u64, u64)>
let has_unified = unified.is_some();

if has_unified {
    let (total, free) = unified.unwrap();
    // Set vram_total/Free from unified
    // Set gpu_name = "Apple Silicon (Unified)"
    // Set selected_gpu_indices = [0]
} else {
    // existing logic: gpus + select_gpu_fit_basis
}
```

**Actually, the cleanest approach**: Don't duplicate. Just add platform-specific detection that feeds into the existing flow:

```rust
pub fn detect_with_policy(
    models_dir: Option<PathBuf>,
    gpu_fit_policy: GpuFitPolicy,
) -> Result<HardwareProfile> {
    let models_dir = models_dir.unwrap_or_else(default_models_dir);
    
    // Platform-specific unified memory detection
    let unified = detect_unified_mem(); // Option<(u64, u64)>
    
    let gpus = detect_gpus();
    let cuda_visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    
    // Determine GPU fit basis: unified memory OR CUDA GPUs
    let (gpu_name, vram_total, vram_free, selected_gpu_indices) = if let Some((total, free)) = unified {
        select_unified_fit_basis(total, free)
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
        unified_mem_total: unified.map_or(0, |u| u.0),
        unified_mem_free: unified.map_or(0, |u| u.1),
        models_dir,
        gpus,
        selected_gpu_indices,
        cuda_visible_devices,
        gpu_fit_policy,
    })
}
```

This keeps `detect_with_policy` as a single function. The `detect_unified_mem()` helper returns `Option<(u64, u64)>` — `Some` on Apple Silicon, `None` on Intel/mac or Linux.

### 5. `check_fit()` — does it need changes?

The plan says to add a `has_unified_mem()` check in `check_fit()`. **This is unnecessary** if we set `vram_total` from unified memory. Since `vram_total > 0`, `has_gpu()` returns true, and the existing `check_fit()` path uses `hw.vram_total` which is the unified pool. **No changes to `check_fit()` needed.**

The one case where it matters: when a model exceeds the unified pool, `combined = vram_total + ram_available` uses `ram_available`. If `ram_available = 0` on darwin (conservative v1), the combined path is effectively dead — which is correct for v1 since we don't estimate free RAM on darwin.

**But**: if we later implement accurate `ram_available` on darwin, the `--split` path will automatically work because `hw.vram_total` already points to unified memory.

### 6. `select_gpu_fit_basis()` — darwin path

The plan calls for a new `select_unified_fit_basis()` function. This is correct:

```rust
fn select_unified_fit_basis(total: u64, free: u64) -> (Option<String>, u64, u64, Vec<u32>) {
    (
        Some("Apple Silicon (Unified)".into()),
        total,
        free,
        vec![0], // placeholder index
    )
}
```

For `--gpu-fit-policy`, the only relevant policy on darwin is `best` (single SoC). `visible-sum` and `all-sum` are meaningless since there's always exactly one "GPU" — the SoC itself.

### 7. `detect_unified_mem()` (hardware.rs, darwin-only)

```rust
#[cfg(target_os = "macos")]
fn detect_unified_mem() -> Option<(u64, u64)> {
    // 1. sysctl hw.cpufamily → if != Apple Silicon family (0x97a2f6cf), return None
    // 2. sysctl hw.memsize → total bytes
    // 3. Conservative: return (total, 0) for v1 (can't reliably estimate free)
    let cpufamily = sysctl_value("hw.cpufamily")?;
    if cpufamily != 0x97a2f6cf {
        return None; // Intel Mac — not unified memory
    }
    let total = sysctl_value("hw.memsize")?;
    Some((total, 0)) // v1: conservative, no free estimate
}

fn sysctl_value(name: &str) -> Option<u64> {
    use std::ffi::CString;
    let c = CString::new(name).ok()?;
    let mut len: usize = 0;
    unsafe { libc::sysctlbyname(c.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null(), 0) }.ok()?;
    let mut buf = vec![0u8; len];
    unsafe { libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null(), 0) }.ok()?;
    // parse buf as u64 (native endian)
    u64::from_ne_bytes(buf.try_into().ok()?)
}
```

For Apple Silicon detection, `hw.cpufamily = 0x97a2f6cf` is correct. But T2 Intel Macs also have this value. A more robust check would add `hw.machine` — Apple Silicon Macs have `Mac-xxxx` model identifiers that differ from Intel. For v1, `hw.cpufamily` alone is acceptable — the risk is showing unified memory info on a T2 Intel Mac, where the iGPU shares system RAM anyway.

### 8. `print_hardware()` — display (display.rs)

The plan is correct. After setting `vram_total` from unified memory:
- `hw.has_gpu()` → true → VRAM basis line prints (green, correct)
- `gpu_name` → "Apple Silicon (Unified)" → GPU line shows this
- `gpus.is_empty()` → true (no NVIDIA) → NVIDIA GPU count block doesn't print
- `selected_gpu_count()` → 1 (from `[0]`) — slightly misleading but harmless

The plan to update `gpu_fit_basis()` (types.rs:82-95) is **correct**. It needs a darwin branch:

```rust
pub fn gpu_fit_basis(&self) -> String {
    if !self.has_gpu() {
        if self.gpus.is_empty() {
            return "system RAM (CPU-only; no NVIDIA GPU detected)".to_string();
        }
        return "system RAM (no CUDA-visible NVIDIA GPU selected)".to_string();
    }

    if self.unified_mem_total > 0 {
        return "Apple Silicon unified memory".to_string();
    }

    match self.gpu_fit_policy {
        GpuFitPolicy::Best => "single CUDA-visible NVIDIA GPU with the most total VRAM".to_string(),
        GpuFitPolicy::VisibleSum => "sum of CUDA-visible NVIDIA GPU total VRAM".to_string(),
        GpuFitPolicy::AllSum => "sum of all detected NVIDIA GPU total VRAM; ignores CUDA_VISIBLE_DEVICES".to_string(),
    }
}
```

### 9. `detect_disk()` — works on darwin?

Uses `libc::statvfs` via FFI. Darwin supports `statvfs` via POSIX — the `libc` crate provides it on darwin. **No changes needed.**

### 10. `ram_available` on darwin — v1 conservative approach

The plan says to report `0` for `ram_available` on darwin. This is correct for v1 because:
- No clean way to get "available" memory from Rust without `mach` imports.
- Overestimating would show models that fail at runtime.
- Underestimating (zero) is safe — the unified pool total is what matters for fit.

If we later add accurate darwin RAM detection (using `sysctl hw.memsize` minus `mach_task_basic_info` or `vm_pressure_level`), the `--split` path will automatically work because `hw.vram_total` already points to unified memory.

### 11. Platform detection in `detect_ram()` (hardware.rs)

Current code (lines 223-251) has:
```rust
#[cfg(target_os = "linux")]
fn detect_ram() -> Result<(u64, u64)> { /proc/meminfo/ }

#[cfg(not(target_os = "linux"))]
fn detect_ram() -> Result<(u64, u64)> { Err("only Linux ...") }
```

Needs to become:
```rust
#[cfg(target_os = "linux")]
fn detect_ram() -> Result<(u64, u64)> { /proc/meminfo/ }

#[cfg(target_os = "macos")]
fn detect_ram() -> Result<(u64, u64)> {
    // On darwin, return (unified_total, 0) since unified pool is the primary resource
    // ram_total = unified total, ram_available = 0 (conservative)
    let unified = detect_unified_mem().unwrap_or((0, 0));
    Ok((unified.0, 0))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_ram() -> Result<(u64, u64)> { Err("not implemented") }
```

Wait — `detect_ram()` calls `detect_unified_mem()`. But `detect_unified_mem()` is `#[cfg(target_os = "macos")]` and `detect_ram()` is also `#[cfg(target_os = "macos")]`. This works because they're both darwin-only. But it creates a circular dependency issue: `detect_ram()` needs unified mem, which is only available on darwin. This is fine since both are cfg-gated the same way.

Actually — there's a subtlety. `detect_ram()` on darwin returns `(unified_total, 0)` where `unified_total` is from `hw.memsize`. This is **the same number** that `unified_mem_total` will be. The plan needs to make sure these two sources of truth are consistent (they are — same sysctl).

## Revised file-by-file change summary

| File | Change |
|------|--------|
| `src/types.rs` | Add `unified_mem_total: u64`, `unified_mem_free: u64` to `HardwareProfile`. Update `gpu_fit_basis()` to return "Apple Silicon unified memory" when `unified_mem_total > 0`. |
| `src/hardware.rs` | Add `detect_unified_mem() → Option<(u64, u64)>` (darwin-only `#[cfg]`). Add `select_unified_fit_basis()` (darwin-only). Update `detect_ram()` to add darwin branch. Update `detect_with_policy()` to check unified memory first. |
| `src/resolver.rs` | **No changes needed.** Setting `vram_total` from unified memory makes the existing NVIDIA path work unchanged. |
| `src/display.rs` | **No changes needed.** `has_gpu()` returns true (vram_total > 0), VRAM basis prints correctly, gpu_name shows "Apple Silicon (Unified)". |
| `src/resolver.rs` tests | Update `hw()` helper to include the two new fields. |

## What this does NOT do (future work)

- Intel Mac detection (iGPU VRAM separate from system RAM).
- Multi-GPU support on NVIDIA (separate task).
- Accurate `ram_available` on darwin (uses conservative 0 for v1).
- `mach_task_basic_info` precise free memory — initial version uses conservative estimate.
- Dynamic VRAM tracking (uses static boot-time total, same as NVIDIA path).
- `--gpu-fit-policy` validation on darwin (always single SoC, policies are meaningless).
