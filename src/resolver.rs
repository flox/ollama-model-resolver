use std::cmp::Ordering;

use reqwest::blocking::Client;

use crate::error::{ResolverError, Result};
use crate::registry::{HttpRegistry, Registry};
use crate::types::{FitResult, HardwareProfile, ModelVariant, TagInfo};

#[derive(Debug, Clone, Copy)]
pub struct ResolveOpts {
    pub allow_split: bool,
    pub margin_pct: u32,
    pub context_tokens: u32,
    pub max_manifest_lookups: Option<usize>,
    /// Hide models that do not fit VRAM in search output.
    pub fit_filter: bool,
    /// Show all models including cloud-only, platform-restricted, and non-fitting.
    pub all: bool,
}

#[derive(Debug, Clone)]
pub struct ResolutionOutcome {
    pub variant: ModelVariant,
    pub fit: FitResult,
    pub diagnostics: ResolutionDiagnostics,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolutionDiagnostics {
    pub approx_deferred: Vec<CandidateTrace>,
    pub manifest_checked: Vec<CandidateTrace>,
    pub manifest_skipped_by_cap: Vec<CandidateTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateTrace {
    pub full_ref: String,
    pub approx_size: Option<String>,
    pub decision: String,
}

pub fn resolve(
    client: &Client,
    model: &str,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
) -> Result<(ModelVariant, FitResult)> {
    let mut registry = HttpRegistry::new(client);
    resolve_with_registry(&mut registry, model, hw, opts)
}

pub fn resolve_with_registry<R: Registry>(
    registry: &mut R,
    model: &str,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
) -> Result<(ModelVariant, FitResult)> {
    let outcome = resolve_with_registry_diagnostics(registry, model, hw, opts)?;
    Ok((outcome.variant, outcome.fit))
}

pub fn resolve_with_registry_diagnostics<R: Registry>(
    registry: &mut R,
    model: &str,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
) -> Result<ResolutionOutcome> {
    let mut candidates = registry.list_tags(model)?;
    rank_candidates(&mut candidates);

    let plan = manifest_evaluation_plan(&candidates, hw, opts);
    let mut diagnostics = diagnostics_from_plan(&candidates, &plan);
    let mut smallest_evaluated: Option<(ModelVariant, FitResult)> = None;
    // Highest-ranked candidate that the registry gated behind a platform
    // precondition (HTTP 412, "requires macOS"). Used to report a macOS-only
    // model when no sizable variant exists, so search can surface it on macOS.
    let mut platform_restricted: Option<(String, u16)> = None;
    // Highest-ranked candidate that is cloud-only (no local weights). Used to
    // report a cloud-only model when no sizable variant exists, so search can
    // hide it cleanly instead of leaking a generic "no usable manifest" error.
    let mut cloud_only: Option<String> = None;
    let mut manifest_lookups = 0_usize;

    for candidate_index in plan.primary {
        let tag_info = &candidates[candidate_index];
        let Some(evaluation) = maybe_evaluate_candidate(
            registry,
            model,
            tag_info,
            hw,
            opts,
            &mut manifest_lookups,
            &mut diagnostics,
        )? else {
            continue;
        };

        match evaluation {
            CandidateEvaluation::MissingManifest(detail) => {
                diagnostics.manifest_checked.push(candidate_trace(
                    tag_info,
                    format!("manifest missing; skipped ({detail})"),
                ));
            }
            CandidateEvaluation::Evaluated(variant, fit) => {
                let decision = if fit.fits() {
                    "selected".to_string()
                } else {
                    fit.summary()
                };
                diagnostics
                    .manifest_checked
                    .push(candidate_trace(tag_info, decision));

                if fit.fits() {
                    return Ok(ResolutionOutcome {
                        variant,
                        fit,
                        diagnostics,
                    });
                }

                record_smallest_nonfit(&mut smallest_evaluated, variant, fit);
            }
            CandidateEvaluation::PlatformRestricted(status) => {
                if platform_restricted.is_none() {
                    platform_restricted = Some((tag_info.tag.clone(), status));
                }
                diagnostics.manifest_checked.push(candidate_trace(
                    tag_info,
                    format!("platform-restricted (HTTP {status}); requires macOS"),
                ));
            }
            CandidateEvaluation::CloudOnly => {
                if cloud_only.is_none() {
                    cloud_only = Some(tag_info.tag.clone());
                }
                diagnostics.manifest_checked.push(candidate_trace(
                    tag_info,
                    "cloud-only model; no local weights".to_string(),
                ));
            }
        }
    }

    // Tag-page sizes are hints, not final sizing data. They decide which
    // candidates are checked after plausible/unknown-size candidates, not
    // whether a ranked candidate can ever win. This prevents a pessimistic or
    // stale page hint from hiding a model whose exact manifest fits.
    for candidate_index in plan.approx_rejected {
        let tag_info = &candidates[candidate_index];
        let Some(evaluation) = maybe_evaluate_candidate(
            registry,
            model,
            tag_info,
            hw,
            opts,
            &mut manifest_lookups,
            &mut diagnostics,
        )? else {
            continue;
        };

        match evaluation {
            CandidateEvaluation::MissingManifest(detail) => {
                diagnostics.manifest_checked.push(candidate_trace(
                    tag_info,
                    format!("manifest missing; skipped ({detail})"),
                ));
            }
            CandidateEvaluation::Evaluated(variant, fit) => {
                let decision = if fit.fits() {
                    "selected after exact manifest overrode page-size hint".to_string()
                } else {
                    fit.summary()
                };
                diagnostics
                    .manifest_checked
                    .push(candidate_trace(tag_info, decision));

                if fit.fits() {
                    return Ok(ResolutionOutcome {
                        variant,
                        fit,
                        diagnostics,
                    });
                }

                record_smallest_nonfit(&mut smallest_evaluated, variant, fit);
            }
            CandidateEvaluation::PlatformRestricted(status) => {
                if platform_restricted.is_none() {
                    platform_restricted = Some((tag_info.tag.clone(), status));
                }
                diagnostics.manifest_checked.push(candidate_trace(
                    tag_info,
                    format!("platform-restricted (HTTP {status}); requires macOS"),
                ));
            }
            CandidateEvaluation::CloudOnly => {
                if cloud_only.is_none() {
                    cloud_only = Some(tag_info.tag.clone());
                }
                diagnostics.manifest_checked.push(candidate_trace(
                    tag_info,
                    "cloud-only model; no local weights".to_string(),
                ));
            }
        }
    }

    // A sized-but-non-fitting variant is a more specific verdict than
    // platform-restriction, so it takes precedence in the fallback ordering.
    if let Some((variant, fit)) = smallest_evaluated {
        return Ok(ResolutionOutcome {
            variant,
            fit,
            diagnostics,
        });
    }

    // No variant was sizable. Report the most actionable reason distinctly so
    // search can classify the model rather than leaking a generic unusable
    // manifest. Platform-restricted (locally runnable on macOS) takes precedence
    // over cloud-only (never runnable locally).
    if let Some((tag, status)) = platform_restricted {
        return Err(ResolverError::ManifestPlatformRestricted {
            model: model.to_string(),
            tag,
            status,
        });
    }

    if let Some(tag) = cloud_only {
        return Err(ResolverError::ManifestCloudOnly {
            model: model.to_string(),
            tag,
        });
    }

    Err(ResolverError::NoUsableManifest {
        model: model.to_string(),
        attempts: manifest_failure_summary(&diagnostics),
    })
}

fn manifest_failure_summary(diagnostics: &ResolutionDiagnostics) -> String {
    let mut lines = Vec::new();

    for trace in &diagnostics.manifest_checked {
        lines.push(format!("- {}: {}", trace.full_ref, trace.decision));
    }

    for trace in &diagnostics.manifest_skipped_by_cap {
        lines.push(format!("- {}: {}", trace.full_ref, trace.decision));
    }

    if lines.is_empty() {
        "- no manifest lookups were attempted".to_string()
    } else {
        lines.join("\n")
    }
}

fn diagnostics_from_plan(candidates: &[TagInfo], plan: &ManifestEvaluationPlan) -> ResolutionDiagnostics {
    ResolutionDiagnostics {
        approx_deferred: plan
            .approx_rejected
            .iter()
            .map(|idx| {
                candidate_trace(
                    &candidates[*idx],
                    "page-size hint exceeded capacity; checked only if primary candidates do not fit",
                )
            })
            .collect(),
        manifest_checked: Vec::new(),
        manifest_skipped_by_cap: Vec::new(),
    }
}

fn candidate_trace(tag_info: &TagInfo, decision: impl Into<String>) -> CandidateTrace {
    CandidateTrace {
        full_ref: tag_info.full_ref.clone(),
        approx_size: tag_info.approx_size.clone(),
        decision: decision.into(),
    }
}

enum CandidateEvaluation {
    Evaluated(ModelVariant, FitResult),
    MissingManifest(String),
    /// Registry gated this candidate behind a platform precondition (HTTP 412,
    /// "requires macOS"). Carries the status for diagnostics.
    PlatformRestricted(u16),
    /// Candidate is cloud-only (manifest has no local weight layers).
    CloudOnly,
}

fn maybe_evaluate_candidate<R: Registry>(
    registry: &mut R,
    model: &str,
    tag_info: &TagInfo,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
    manifest_lookups: &mut usize,
    diagnostics: &mut ResolutionDiagnostics,
) -> Result<Option<CandidateEvaluation>> {
    if let Some(max_manifest_lookups) = opts.max_manifest_lookups {
        if *manifest_lookups >= max_manifest_lookups {
            diagnostics.manifest_skipped_by_cap.push(candidate_trace(
                tag_info,
                "not checked; --max-manifest-lookups reached",
            ));
            return Ok(None);
        }
    }

    *manifest_lookups = (*manifest_lookups).saturating_add(1);
    evaluate_candidate(registry, model, tag_info, hw, opts).map(Some)
}

fn evaluate_candidate<R: Registry>(
    registry: &mut R,
    model: &str,
    tag_info: &TagInfo,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
) -> Result<CandidateEvaluation> {
    let (weights_bytes, total_bytes) = match registry.get_manifest_size(model, &tag_info.tag) {
        Ok(sizes) => sizes,
        Err(ResolverError::ManifestMissing { detail, .. }) => {
            return Ok(CandidateEvaluation::MissingManifest(detail));
        }
        Err(ResolverError::ManifestCloudOnly { .. }) => {
            return Ok(CandidateEvaluation::CloudOnly);
        }
        Err(ResolverError::ManifestPlatformRestricted { status, .. }) => {
            return Ok(CandidateEvaluation::PlatformRestricted(status));
        }
        Err(err) => return Err(err),
    };
    let runtime_margin_pct = effective_margin_pct(opts);
    let estimated_runtime_bytes = estimate_runtime_bytes(weights_bytes, runtime_margin_pct);
    let variant = ModelVariant {
        name: model.to_string(),
        tag: tag_info.tag.clone(),
        full_ref: tag_info.full_ref.clone(),
        weights_bytes,
        total_bytes,
        estimated_runtime_bytes,
        runtime_margin_pct,
        context_tokens: opts.context_tokens,
        param_billions: tag_info.param_billions,
        quantization: tag_info.quantization.clone(),
        is_instruct: tag_info.is_instruct,
    };
    let fit = check_fit(&variant, hw, opts);
    Ok(CandidateEvaluation::Evaluated(variant, fit))
}

fn record_smallest_nonfit(
    smallest_evaluated: &mut Option<(ModelVariant, FitResult)>,
    variant: ModelVariant,
    fit: FitResult,
) {
    match smallest_evaluated {
        Some((current, _)) if current.weights_bytes <= variant.weights_bytes => {}
        _ => *smallest_evaluated = Some((variant, fit)),
    }
}

fn rank_candidates(candidates: &mut [TagInfo]) {
    let has_instructish = candidates.iter().any(|tag| tag.is_instruct);
    let has_default_quant = candidates.iter().any(TagInfo::has_default_quantization);

    candidates.sort_by(|a, b| {
        candidate_group(a, has_instructish, has_default_quant)
            .cmp(&candidate_group(b, has_instructish, has_default_quant))
            .then_with(|| compare_params_desc(a.param_billions, b.param_billions))
            .then_with(|| a.tag.cmp(&b.tag))
    });
}

fn candidate_group(tag: &TagInfo, has_instructish: bool, has_default_quant: bool) -> u8 {
    let instruct_penalty = if has_instructish && !tag.is_instruct { 2 } else { 0 };
    let quant_penalty = if has_default_quant && !tag.has_default_quantization() {
        1
    } else {
        0
    };
    instruct_penalty + quant_penalty
}

fn compare_params_desc(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(a), Some(b)) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ManifestEvaluationPlan {
    primary: Vec<usize>,
    approx_rejected: Vec<usize>,
}

fn manifest_evaluation_plan(
    candidates: &[TagInfo],
    hw: &HardwareProfile,
    opts: &ResolveOpts,
) -> ManifestEvaluationPlan {
    let mut plan = ManifestEvaluationPlan::default();

    for (idx, candidate) in candidates.iter().enumerate() {
        if approx_rejects_candidate(candidate, hw, opts) {
            plan.approx_rejected.push(idx);
        } else {
            plan.primary.push(idx);
        }
    }

    plan
}

fn approx_rejects_candidate(candidate: &TagInfo, hw: &HardwareProfile, opts: &ResolveOpts) -> bool {
    let Some(approx_bytes) = candidate.approx_size_bytes() else {
        return false;
    };

    if approx_bytes > hw.disk_free {
        return true;
    }

    estimate_runtime_bytes(approx_bytes, effective_margin_pct(opts))
        > hw.available_runtime_bytes(opts.allow_split)
}

pub fn effective_margin_pct(opts: &ResolveOpts) -> u32 {
    opts.margin_pct.saturating_add(context_margin_pct(opts.context_tokens))
}

pub fn context_margin_pct(context_tokens: u32) -> u32 {
    const BASE_CONTEXT_TOKENS: u32 = 8_192;
    if context_tokens <= BASE_CONTEXT_TOKENS {
        return 0;
    }

    // Context memory is model-architecture dependent. Without hidden size and
    // layer count from authoritative metadata, the resolver adds a conservative
    // page-fit margin: every additional 8K tokens contributes 5 percentage
    // points, rounded up. Users can still raise --margin for more cautious
    // deployments.
    let extra_tokens = context_tokens.saturating_sub(BASE_CONTEXT_TOKENS);
    let extra_blocks = (extra_tokens as u64 + BASE_CONTEXT_TOKENS as u64 - 1)
        / BASE_CONTEXT_TOKENS as u64;
    extra_blocks.saturating_mul(5).min(u32::MAX as u64) as u32
}

pub fn estimate_runtime_bytes(weights_bytes: u64, margin_pct: u32) -> u64 {
    let numerator = (weights_bytes as u128).saturating_mul(100_u128 + margin_pct as u128);
    let estimated = numerator.saturating_add(99) / 100;
    estimated.min(u64::MAX as u128) as u64
}

pub fn check_fit(variant: &ModelVariant, hw: &HardwareProfile, opts: &ResolveOpts) -> FitResult {
    if variant.total_bytes > hw.disk_free {
        return FitResult::InsufficientDisk {
            need: variant.total_bytes,
            have: hw.disk_free,
        };
    }

    let estimated = variant.estimated_runtime_bytes;

    // Apple Silicon unified memory: VRAM and system RAM are one physical pool,
    // so there is no split to consider. Fit against the memory available right
    // now (ram_available) rather than the full installed total, so the decision
    // reflects current system load.
    if hw.unified_mem_total > 0 {
        let ceiling = hw.unified_fit_ceiling();
        return if estimated <= ceiling {
            FitResult::FitsVram
        } else {
            FitResult::DoesNotFit {
                need: estimated,
                have: ceiling,
            }
        };
    }

    if hw.has_gpu() {
        if estimated <= hw.vram_total {
            return FitResult::FitsVram;
        }

        let combined = hw.vram_total.saturating_add(hw.ram_available);
        if opts.allow_split && estimated <= combined {
            let gpu_pct = if estimated == 0 {
                100.0
            } else {
                (hw.vram_total as f64 / estimated as f64) * 100.0
            };
            return FitResult::FitsWithSplit { gpu_pct };
        }

        FitResult::DoesNotFit {
            need: estimated,
            have: if opts.allow_split { combined } else { hw.vram_total },
        }
    } else if estimated <= hw.ram_available {
        FitResult::FitsRamOnly
    } else {
        FitResult::DoesNotFit {
            need: estimated,
            have: hw.ram_available,
        }
    }
}

#[allow(deprecated)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{tag_info_from_str, HardwareProfile};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn hw(vram: u64, ram: u64, disk: u64) -> HardwareProfile {
        HardwareProfile {
            gpu_name: if vram > 0 { Some("test-gpu".into()) } else { None },
            vram_total: vram,
            vram_free: vram, // deprecated: always equal to vram_total
            ram_total: ram,
            ram_available: ram,
            disk_free: disk,
            models_dir: PathBuf::from("/tmp"),
            gpus: Vec::new(),
            selected_gpu_indices: if vram > 0 { vec![0] } else { Vec::new() },
            cuda_visible_devices: None,
            gpu_fit_policy: crate::types::GpuFitPolicy::Best,
            unified_mem_total: 0,
        }
    }

    fn opts() -> ResolveOpts {
        ResolveOpts {
            allow_split: false,
            margin_pct: 20,
            context_tokens: 8_192,
            max_manifest_lookups: None,
            fit_filter: false,
            all: false,
        }
    }

    fn split_opts() -> ResolveOpts {
        ResolveOpts {
            allow_split: true,
            margin_pct: 20,
            context_tokens: 8_192,
            max_manifest_lookups: None,
            fit_filter: false,
            all: false,
        }
    }

    fn capped_opts(max_manifest_lookups: usize) -> ResolveOpts {
        ResolveOpts {
            allow_split: false,
            margin_pct: 20,
            context_tokens: 8_192,
            max_manifest_lookups: Some(max_manifest_lookups),
            fit_filter: false,
            all: false,
        }
    }

    fn variant(weights: u64, total: u64, estimated: u64) -> ModelVariant {
        ModelVariant {
            name: "m".into(),
            tag: "7b".into(),
            full_ref: "m:7b".into(),
            weights_bytes: weights,
            total_bytes: total,
            estimated_runtime_bytes: estimated,
            runtime_margin_pct: 20,
            context_tokens: 8_192,
            param_billions: Some(7.0),
            quantization: None,
            is_instruct: true,
        }
    }

    fn tag(name: &str, approx: Option<&str>) -> TagInfo {
        let mut info = tag_info_from_str("m", name);
        info.approx_size = approx.map(str::to_string);
        info
    }

    struct FakeRegistry {
        tags: Vec<TagInfo>,
        manifests: HashMap<String, (u64, u64)>,
        fatal_manifest_errors: HashMap<String, String>,
        manifest_calls: Vec<String>,
    }

    impl Registry for FakeRegistry {
        fn list_tags(&mut self, _model: &str) -> Result<Vec<TagInfo>> {
            Ok(self.tags.clone())
        }

        fn get_manifest_size(&mut self, model: &str, tag: &str) -> Result<(u64, u64)> {
            self.manifest_calls.push(tag.to_string());
            if let Some(detail) = self.fatal_manifest_errors.get(tag) {
                return Err(ResolverError::ManifestUnavailable {
                    model: model.to_string(),
                    tag: tag.to_string(),
                    detail: detail.clone(),
                });
            }
            // Mirror the real registry: nvfp4 quant tags are macOS-only and
            // return HTTP 412 ("requires macOS").
            if tag.contains("nvfp4") {
                return Err(ResolverError::ManifestPlatformRestricted {
                    model: model.to_string(),
                    tag: tag.to_string(),
                    status: 412,
                });
            }
            // Mirror the real registry: cloud-only tags have no local weights.
            if tag.contains("cloud") {
                return Err(ResolverError::ManifestCloudOnly {
                    model: model.to_string(),
                    tag: tag.to_string(),
                });
            }
            self.manifests.get(tag).copied().ok_or_else(|| ResolverError::ManifestMissing {
                model: model.to_string(),
                tag: tag.to_string(),
                detail: "fake registry has no such manifest".to_string(),
            })
        }
    }

    #[test]
    fn estimates_runtime_with_integer_ceiling() {
        assert_eq!(estimate_runtime_bytes(10, 20), 12);
        assert_eq!(estimate_runtime_bytes(11, 20), 14);
    }


    #[test]
    fn context_tokens_raise_effective_margin_above_8k() {
        let mut options = opts();
        options.context_tokens = 32_768;
        assert_eq!(context_margin_pct(options.context_tokens), 15);
        assert_eq!(effective_margin_pct(&options), 35);
    }

    #[test]
    fn detects_vram_fit_before_split() {
        let fit = check_fit(&variant(10, 10, 12), &hw(12, 100, 100), &split_opts());
        assert!(matches!(fit, FitResult::FitsVram));
    }

    #[test]
    fn detects_split_fit() {
        let fit = check_fit(&variant(10, 10, 50), &hw(20, 40, 100), &split_opts());
        assert!(matches!(fit, FitResult::FitsWithSplit { .. }));
    }

    #[test]
    fn disk_check_takes_precedence() {
        let fit = check_fit(&variant(10, 200, 12), &hw(100, 100, 100), &opts());
        assert!(matches!(fit, FitResult::InsufficientDisk { .. }));
    }

    // Apple Silicon unified memory: pool total is `total`, free-right-now is
    // `free`. has_gpu() must stay true, so vram_total mirrors the pool total.
    fn unified_hw(total: u64, free: u64, disk: u64) -> HardwareProfile {
        HardwareProfile {
            gpu_name: Some("Apple Silicon (Unified)".into()),
            vram_total: total,
            vram_free: total, // deprecated: always equal to vram_total
            ram_total: total,
            ram_available: free,
            disk_free: disk,
            models_dir: PathBuf::from("/tmp"),
            gpus: Vec::new(),
            selected_gpu_indices: vec![0],
            cuda_visible_devices: None,
            gpu_fit_policy: crate::types::GpuFitPolicy::Best,
            unified_mem_total: total,
        }
    }

    #[test]
    fn unified_fits_against_available_memory() {
        // estimated 10 <= free 12 -> fits, even though pool total is 16.
        let fit = check_fit(&variant(8, 8, 10), &unified_hw(16, 12, 100), &opts());
        assert!(matches!(fit, FitResult::FitsVram));
    }

    #[test]
    fn unified_rejects_above_available_even_when_below_pool_total() {
        // estimated 14 > free 12 but < pool total 16: conservative path rejects.
        let fit = check_fit(&variant(11, 11, 14), &unified_hw(16, 12, 100), &opts());
        match fit {
            FitResult::DoesNotFit { need, have } => {
                assert_eq!(need, 14);
                assert_eq!(have, 12); // reports free-right-now, not the 16 pool
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn unified_ignores_split() {
        // --split must not rescue a model that exceeds the unified pool's free
        // memory; there is no separate VRAM/RAM to split across.
        let fit = check_fit(&variant(11, 11, 14), &unified_hw(16, 12, 100), &split_opts());
        assert!(matches!(fit, FitResult::DoesNotFit { .. }));
    }

    #[test]
    fn unified_falls_back_to_pool_total_when_available_reads_zero() {
        // ram_available == 0 signals a failed vm_statistics read; fall back to
        // the pool total so a transient failure does not reject every model.
        let fit = check_fit(&variant(11, 11, 14), &unified_hw(16, 0, 100), &opts());
        assert!(matches!(fit, FitResult::FitsVram));
    }

    #[test]
    fn approx_sizes_split_manifest_plan_into_primary_and_rejected() {
        let candidates = vec![
            tag("70b-q4_K_M", Some("40GB")),
            tag("14b-q4_K_M", Some("9GB")),
            tag("7b-q4_K_M", None),
        ];
        let plan = manifest_evaluation_plan(
            &candidates,
            &hw(12_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        );
        assert_eq!(plan.primary, vec![1, 2]);
        assert_eq!(plan.approx_rejected, vec![0]);
    }

    #[test]
    fn rejected_candidates_keep_ranked_order_for_fallback() {
        let candidates = vec![
            tag("70b-q4_K_M", Some("40GB")),
            tag("32b-q4_K_M", Some("20GB")),
            tag("14b-q4_K_M", Some("9GB")),
        ];
        let plan = manifest_evaluation_plan(
            &candidates,
            &hw(8_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        );
        assert!(plan.primary.is_empty());
        assert_eq!(plan.approx_rejected, vec![0, 1, 2]);
    }

    #[test]
    fn resolve_skips_approx_rejected_candidate_when_plausible_candidate_fits() {
        let mut registry = FakeRegistry {
            tags: vec![
                tag("70b-q4_K_M", Some("40GB")),
                tag("14b-q4_K_M", Some("9GB")),
            ],
            manifests: HashMap::from([
                ("70b-q4_K_M".into(), (40_000_000_000, 40_000_000_000)),
                ("14b-q4_K_M".into(), (9_000_000_000, 9_000_000_000)),
            ]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let (variant, fit) = resolve_with_registry(
            &mut registry,
            "m",
            &hw(12_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap();
        assert_eq!(variant.tag, "14b-q4_K_M");
        assert!(fit.fits());
        assert_eq!(registry.manifest_calls, vec!["14b-q4_K_M"]);
    }

    #[test]
    fn resolve_checks_all_rejected_candidates_in_ranked_order_when_primary_does_not_fit() {
        let mut registry = FakeRegistry {
            tags: vec![
                tag("70b-q4_K_M", Some("40GB")),
                tag("32b-q4_K_M", Some("20GB")),
                tag("14b-q4_K_M", Some("9GB")),
            ],
            manifests: HashMap::from([
                ("70b-q4_K_M".into(), (40_000_000_000, 40_000_000_000)),
                ("32b-q4_K_M".into(), (20_000_000_000, 20_000_000_000)),
                ("14b-q4_K_M".into(), (9_000_000_000, 9_000_000_000)),
            ]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let (variant, fit) = resolve_with_registry(
            &mut registry,
            "m",
            &hw(8_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap();
        assert_eq!(variant.tag, "14b-q4_K_M");
        assert!(!fit.fits());
        assert_eq!(
            registry.manifest_calls,
            vec!["70b-q4_K_M", "32b-q4_K_M", "14b-q4_K_M"]
        );
    }

    #[test]
    fn all_platform_restricted_candidates_report_platform_restricted() {
        // A model published only as macOS-only nvfp4 quants: every manifest 412s.
        let mut registry = FakeRegistry {
            tags: vec![tag("9b-nvfp4", None), tag("35b-a3b-nvfp4", None)],
            manifests: HashMap::new(),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let err = resolve_with_registry(
            &mut registry,
            "qwen3.5",
            &hw(0, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap_err();

        assert!(
            matches!(err, ResolverError::ManifestPlatformRestricted { .. }),
            "expected ManifestPlatformRestricted, got {err:?}"
        );
    }

    #[test]
    fn platform_restricted_candidate_ignored_when_a_variant_fits() {
        // A normal tag that fits wins; the macOS-only nvfp4 tag is never reached.
        let mut registry = FakeRegistry {
            tags: vec![tag("9b-nvfp4", None), tag("7b", None)],
            manifests: HashMap::from([("7b".into(), (4_000_000_000, 4_000_000_000))]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let (variant, fit) = resolve_with_registry(
            &mut registry,
            "m",
            &hw(12_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap();

        // A fitting variant is selected despite a 412 macOS-only candidate in
        // the tag list (regardless of which is probed first).
        assert_eq!(variant.tag, "7b");
        assert!(fit.fits());
    }

    #[test]
    fn all_cloud_only_candidates_report_cloud_only() {
        let mut registry = FakeRegistry {
            tags: vec![tag("cloud", None), tag("cloud-large", None)],
            manifests: HashMap::new(),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let err = resolve_with_registry(
            &mut registry,
            "m",
            &hw(0, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap_err();

        assert!(
            matches!(err, ResolverError::ManifestCloudOnly { .. }),
            "expected ManifestCloudOnly, got {err:?}"
        );
    }

    #[test]
    fn platform_restricted_takes_precedence_over_cloud_only() {
        // A model with both a cloud-only tag and a macOS-only tag (no sizable
        // variant) is reported as macOS-only — it is at least locally runnable
        // on macOS, unlike cloud-only.
        let mut registry = FakeRegistry {
            tags: vec![tag("cloud", None), tag("9b-nvfp4", None)],
            manifests: HashMap::new(),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let err = resolve_with_registry(
            &mut registry,
            "m",
            &hw(0, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap_err();

        assert!(
            matches!(err, ResolverError::ManifestPlatformRestricted { .. }),
            "expected ManifestPlatformRestricted, got {err:?}"
        );
    }

    #[test]
    fn sized_non_fitting_variant_takes_precedence_over_platform_restricted() {
        // A sized-but-too-big tag is a more specific verdict than 412, so the
        // resolver reports DoesNotFit rather than platform-restricted.
        let mut registry = FakeRegistry {
            tags: vec![tag("9b-nvfp4", None), tag("70b", None)],
            manifests: HashMap::from([("70b".into(), (40_000_000_000, 40_000_000_000))]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let (variant, fit) = resolve_with_registry(
            &mut registry,
            "m",
            &hw(8_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap();

        assert_eq!(variant.tag, "70b");
        assert!(!fit.fits());
    }

    #[test]
    fn exact_manifest_can_rescue_pessimistic_hint_after_larger_rejected_candidates_fail() {
        let mut registry = FakeRegistry {
            tags: vec![
                tag("70b-q4_K_M", Some("80GB")),
                tag("14b-q4_K_M", Some("20GB")),
            ],
            manifests: HashMap::from([
                ("70b-q4_K_M".into(), (40_000_000_000, 40_000_000_000)),
                ("14b-q4_K_M".into(), (9_000_000_000, 9_000_000_000)),
            ]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let (variant, fit) = resolve_with_registry(
            &mut registry,
            "m",
            &hw(12_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap();
        assert_eq!(variant.tag, "14b-q4_K_M");
        assert!(fit.fits());
        assert_eq!(registry.manifest_calls, vec!["70b-q4_K_M", "14b-q4_K_M"]);
    }

    #[test]
    fn manifest_lookup_cap_records_unchecked_candidates() {
        let mut registry = FakeRegistry {
            tags: vec![
                tag("70b-q4_K_M", Some("80GB")),
                tag("32b-q4_K_M", Some("40GB")),
                tag("14b-q4_K_M", Some("20GB")),
            ],
            manifests: HashMap::from([
                ("70b-q4_K_M".into(), (40_000_000_000, 40_000_000_000)),
                ("32b-q4_K_M".into(), (20_000_000_000, 20_000_000_000)),
                ("14b-q4_K_M".into(), (9_000_000_000, 9_000_000_000)),
            ]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let outcome = resolve_with_registry_diagnostics(
            &mut registry,
            "m",
            &hw(8_000_000_000, 64_000_000_000, 100_000_000_000),
            &capped_opts(1),
        )
        .unwrap();

        assert_eq!(registry.manifest_calls, vec!["70b-q4_K_M"]);
        assert_eq!(outcome.diagnostics.manifest_skipped_by_cap.len(), 2);
        assert_eq!(outcome.variant.tag, "70b-q4_K_M");
    }

    #[test]
    fn resolve_skips_missing_manifest_but_continues_to_next_candidate() {
        // Use large enough VRAM (20GB) so neither candidate is approx-deferred.
        // 14b has no manifest → skipped, 7b has manifest → fits → selected.
        let mut registry = FakeRegistry {
            tags: vec![
                tag("14b-q4_K_M", Some("9GB")),
                tag("7b-q4_K_M", Some("4GB")),
            ],
            manifests: HashMap::from([("7b-q4_K_M".into(), (4_000_000_000, 4_000_000_000))]),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let (variant, fit) = resolve_with_registry(
            &mut registry,
            "m",
            &hw(20_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap();

        assert_eq!(variant.tag, "7b-q4_K_M");
        assert!(fit.fits());
        // rank_candidates sorts 14b before 7b (descending); both are primary
        assert_eq!(registry.manifest_calls, vec!["14b-q4_K_M", "7b-q4_K_M"]);
    }

    #[test]
    fn resolve_fails_fast_on_manifest_unavailable() {
        // Use large enough VRAM so 14b is primary (not approx-deferred)
        let mut registry = FakeRegistry {
            tags: vec![
                tag("14b-q4_K_M", Some("9GB")),
                tag("7b-q4_K_M", Some("4GB")),
            ],
            manifests: HashMap::from([("7b-q4_K_M".into(), (4_000_000_000, 4_000_000_000))]),
            fatal_manifest_errors: HashMap::from([("14b-q4_K_M".into(), "timeout".into())]),
            manifest_calls: Vec::new(),
        };

        let err = resolve_with_registry(
            &mut registry,
            "m",
            &hw(20_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ResolverError::ManifestUnavailable { model, tag, detail }
                if model == "m" && tag == "14b-q4_K_M" && detail == "timeout"
        ));
        assert_eq!(registry.manifest_calls, vec!["14b-q4_K_M"]);
    }


    #[test]
    fn resolve_reports_all_missing_manifests_with_candidate_context() {
        // Use large enough VRAM so both candidates are primary
        let mut registry = FakeRegistry {
            tags: vec![
                tag("14b-q4_K_M", Some("9GB")),
                tag("7b-q4_K_M", Some("4GB")),
            ],
            manifests: HashMap::new(),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let err = resolve_with_registry(
            &mut registry,
            "m",
            &hw(20_000_000_000, 64_000_000_000, 100_000_000_000),
            &opts(),
        )
        .unwrap_err();

        let ResolverError::NoUsableManifest { model, attempts } = err else {
            panic!("expected NoUsableManifest");
        };
        assert_eq!(model, "m");
        assert!(attempts.contains("- m:14b-q4_K_M: manifest missing; skipped"));
        assert!(attempts.contains("- m:7b-q4_K_M: manifest missing; skipped"));
        assert!(attempts.contains("fake registry has no such manifest"));
        // rank_candidates sorts 14b before 7b (descending)
        assert_eq!(registry.manifest_calls, vec!["14b-q4_K_M", "7b-q4_K_M"]);
    }

    #[test]
    fn resolve_reports_manifest_lookup_cap_in_no_usable_manifest_error() {
        // Use large enough VRAM so 14b is primary (checked first)
        let mut registry = FakeRegistry {
            tags: vec![
                tag("14b-q4_K_M", Some("9GB")),
                tag("7b-q4_K_M", Some("4GB")),
            ],
            manifests: HashMap::new(),
            fatal_manifest_errors: HashMap::new(),
            manifest_calls: Vec::new(),
        };

        let err = resolve_with_registry(
            &mut registry,
            "m",
            &hw(20_000_000_000, 64_000_000_000, 100_000_000_000),
            &capped_opts(1),
        )
        .unwrap_err();

        let ResolverError::NoUsableManifest { attempts, .. } = err else {
            panic!("expected NoUsableManifest");
        };
        assert!(attempts.contains("- m:14b-q4_K_M: manifest missing; skipped"));
        assert!(attempts.contains("- m:7b-q4_K_M: not checked; --max-manifest-lookups reached"));
        assert_eq!(registry.manifest_calls, vec!["14b-q4_K_M"]);
    }

}
