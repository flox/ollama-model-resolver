use dialoguer::{Confirm, Select};
use reqwest::blocking::Client;
use std::io::{self, IsTerminal};
use std::time::Duration;

use crate::cli::{Cli, Commands};
use crate::display;
use crate::error::{ResolverError, Result};
use crate::hardware;
use crate::local;
use crate::registry::{self, HttpRegistry, Registry};
use crate::resolver::{self, ResolveOpts};
use crate::sanitize::terminal_line;
use crate::types::{
    AnnotatedSearchResult, FilteredReason, FitResult, HardwareProfile, ModelVariant, SearchResult,
};

const METADATA_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const PULL_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

pub fn run(cli: Cli) -> Result<()> {
    if cli.margin > 500 {
        return Err(ResolverError::InvalidInput(
            "--margin must be between 0 and 500".into(),
        ));
    }
    if matches!(cli.max_manifest_lookups, Some(0)) {
        return Err(ResolverError::InvalidInput(
            "--max-manifest-lookups must be greater than 0".into(),
        ));
    }
    if cli.pull_stall_timeout == 0 {
        return Err(ResolverError::InvalidInput(
            "--pull-stall-timeout must be greater than 0".into(),
        ));
    }
    if cli.context_tokens == 0 {
        return Err(ResolverError::InvalidInput(
            "--context-tokens must be greater than 0".into(),
        ));
    }

    let metadata_client = metadata_client()?;
    let pull_client = pull_client(Duration::from_secs(cli.pull_stall_timeout))?;

    let opts = ResolveOpts {
        allow_split: cli.allow_split,
        margin_pct: cli.margin,
        context_tokens: cli.context_tokens,
        max_manifest_lookups: cli.max_manifest_lookups,
        fit_filter: false,
        all: false,
    };

    match cli.command {
        Commands::Search { query, limit, fit, no_fit, all, wide, macos } => {
            if no_fit {
                // --quick/--fast: the basic browse — approximate tag-page sizes,
                // no manifest lookups or hardware fit.
                return cmd_search_library(&metadata_client, &query, limit, wide);
            }
            // Default and --fit both use the manifest-sourced annotated view
            // (exact model:tag, exact size, fit verdict) for every match.
            // --fit additionally keeps only models that fit; default shows all.
            let hw = hardware::detect_with_policy(None, cli.gpu_fit_policy)?;
            warn_if_split_ignored(&hw, cli.allow_split);
            let search_opts = ResolveOpts {
                fit_filter: fit,
                all,
                ..opts
            };
            cmd_search(&metadata_client, &query, limit, &hw, &search_opts, all, wide, macos)
        }
        Commands::Resolve {
            model,
            quiet,
            select,
            first,
            fail_on_ambiguous,
            yes,
        } => {
            local::validate_ollama_host(&cli.ollama_host, cli.allow_remote_ollama)?;
            let selection = ModelSelectionOptions::new(quiet, select, first, fail_on_ambiguous)?;
            if model.ends_with('?') {
                let hw = hardware::detect_with_policy(None, cli.gpu_fit_policy)?;
                warn_if_split_ignored(&hw, cli.allow_split);
                cmd_resolve(
                    &metadata_client,
                    &pull_client,
                    &model,
                    &hw,
                    &opts,
                    selection,
                    yes,
                    &cli.ollama_host,
                    cli.ollama_port,
                )
            } else {
                cmd_pull_exact(&pull_client, &model, quiet, &cli.ollama_host, cli.ollama_port)
            }
        }
        Commands::Info => {
            local::validate_ollama_host(&cli.ollama_host, cli.allow_remote_ollama)?;
            let hw = hardware::detect_with_policy(None, cli.gpu_fit_policy)?;
            cmd_info(&metadata_client, &cli.ollama_host, cli.ollama_port, &hw)
        }
    }
}

/// On Apple Silicon unified memory there is no separate VRAM and system RAM to
/// split across, so --split has no effect. Surface that rather than silently
/// accepting the flag.
fn warn_if_split_ignored(hw: &HardwareProfile, allow_split: bool) {
    if allow_split && hw.unified_mem_total > 0 {
        eprintln!(
            "Note: --split has no effect on Apple Silicon unified memory; VRAM and system RAM are a single pool."
        );
    }
}

fn metadata_client() -> Result<Client> {
    Client::builder()
        .timeout(METADATA_REQUEST_TIMEOUT)
        .build()
        .map_err(ResolverError::from)
}

fn pull_client(_pull_stall_timeout: Duration) -> Result<Client> {
    // reqwest::blocking does not expose a per-read idle timeout. Using .timeout()
    // would set a total request deadline that kills large downloads. The pull
    // client uses only a connect timeout; stall detection during streaming pull
    // is left to the OS TCP keepalive and ollama's server-side behavior.
    Client::builder()
        .connect_timeout(PULL_CONNECT_TIMEOUT)
        .build()
        .map_err(ResolverError::from)
}

#[allow(clippy::too_many_arguments)]
fn cmd_search(
    client: &Client,
    query: &str,
    limit: usize,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
    all: bool,
    wide: bool,
    macos_only: bool,
) -> Result<()> {
    let mut results = registry::search_models(client, query)?;
    // ollama.com ranks by popularity, which buries name matches under popular
    // unrelated models; re-rank by name relevance so matches come first.
    registry::rank_search_results(&mut results, query);

    let mut registry_client = HttpRegistry::new(client);

    if macos_only {
        return cmd_search_macos_only(&mut registry_client, query, limit, hw, results, wide);
    }

    // Annotate the most-relevant results; a few extra absorb hidden ones.
    results.truncate(limit.saturating_add(5));

    // macOS-only variants are surfaced as their own rows on macOS (and under
    // --all anywhere); on Linux they can't run, so they're hidden from the
    // default view and the footer hints to use --macos.
    let show_macos = cfg!(target_os = "macos") || all;
    let mut normal_rows: Vec<AnnotatedSearchResult> = Vec::new();
    let mut macos_rows: Vec<AnnotatedSearchResult> = Vec::new();
    let mut macos_hidden = 0u64;

    for result in results {
        // Only surface a macOS-only variant for a name-relevant model, so a
        // popular-but-unrelated padding result (e.g. gemma4 for "glm") doesn't
        // contribute a macOS-only row. list_tags is cached → shares the fetch
        // with resolve below.
        let macos_row = if registry::relevance_score(query, &result.name) > 0 {
            annotate_macos_only(&mut registry_client, result.clone())
        } else {
            None
        };
        let normal = annotate_search_result(&mut registry_client, result, hw, opts);

        // An entirely-macOS-only model resolves to platform-restricted; its
        // macOS-only row represents it, so drop the redundant normal row.
        if !matches!(normal.filtered, Some(FilteredReason::PlatformRestricted)) {
            normal_rows.push(normal);
        }
        match macos_row {
            Some(row) if show_macos => macos_rows.push(row),
            Some(_) => macos_hidden += 1,
            None => {}
        }
    }

    let (cloud_count, fit_count) = filter_search_rows(&mut normal_rows, opts.fit_filter, all);

    // Runnable models first, macOS-only rows after. Reserve room so the (sparse)
    // macOS-only rows aren't starved by runnable ones; total stays within limit.
    let rows = cap_rows(normal_rows, macos_rows, limit);

    if wide {
        display::print_search_results(&rows, hw, cloud_count, macos_hidden, fit_count);
    } else {
        display::print_search_results_compact(&rows, hw, cloud_count, macos_hidden, fit_count);
    }
    display::print_macos_only_note(&rows);
    Ok(())
}

/// Combine runnable rows and macOS-only rows into a single capped list: runnable
/// first, macOS-only after, total `<= limit`. macOS-only rows get reserved slots
/// (capped at `limit`) so a screenful of runnable models can't crowd them out.
fn cap_rows(
    mut normal: Vec<AnnotatedSearchResult>,
    mut macos: Vec<AnnotatedSearchResult>,
    limit: usize,
) -> Vec<AnnotatedSearchResult> {
    let macos_keep = macos.len().min(limit);
    normal.truncate(limit.saturating_sub(macos_keep));
    macos.truncate(macos_keep);
    normal.extend(macos);
    normal
}

/// `--macos`: list models that offer a macOS-only (Apple-Silicon-optimized)
/// variant, showing that variant. Detection is by tag name (see
/// `TagInfo::is_macos_only`) read from each model's tag list — no manifest
/// probing, since those manifests are gated and unsizable anyway. Works on any
/// host; on Linux it's discovery-only (the models can't run there).
fn cmd_search_macos_only<R: Registry>(
    registry: &mut R,
    query: &str,
    limit: usize,
    hw: &HardwareProfile,
    mut results: Vec<SearchResult>,
    wide: bool,
) -> Result<()> {
    // Drop name-irrelevant padding (e.g. gemma4 for "glm") so --macos doesn't
    // surface a popular-but-unrelated model just because it has an nvfp4 tag.
    results.retain(|result| registry::relevance_score(query, &result.name) > 0);

    let mut rows: Vec<AnnotatedSearchResult> = results
        .into_iter()
        .filter_map(|result| annotate_macos_only(registry, result))
        .collect();
    rows.truncate(limit);

    if rows.is_empty() {
        display::print_hardware(hw);
        println!(
            "No macOS-optimized models found for '{}'. (Looked for macOS-gated variants such as nvfp4 among the matches.)",
            terminal_line(query)
        );
        return Ok(());
    }

    if wide {
        display::print_search_results(&rows, hw, 0, 0, 0);
    } else {
        display::print_search_results_compact(&rows, hw, 0, 0, 0);
    }
    display::print_macos_only_note(&rows);
    display::print_search_ranking_note();
    Ok(())
}

/// Build a macOS-only search row for a model, or None if it offers no macOS-only
/// variant. Picks the largest such variant as the representative; its size/fit
/// is unknown (the manifest is gated), so the row carries `FitResult::MacosOnly`.
fn annotate_macos_only<R: Registry>(
    registry: &mut R,
    result: SearchResult,
) -> Option<AnnotatedSearchResult> {
    let tags = registry.list_tags(&result.name).ok()?;
    let best = tags
        .into_iter()
        .filter(|tag| tag.is_macos_only())
        .max_by(|a, b| {
            a.param_billions
                .unwrap_or(0.0)
                .partial_cmp(&b.param_billions.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.tag.cmp(&b.tag))
        })?;

    let variant = ModelVariant {
        name: result.name.clone(),
        tag: best.tag.clone(),
        full_ref: best.full_ref.clone(),
        weights_bytes: 0,
        total_bytes: 0,
        estimated_runtime_bytes: 0,
        runtime_margin_pct: 0,
        context_tokens: 0,
        param_billions: best.param_billions,
        quantization: best.quantization.clone(),
        is_instruct: best.is_instruct,
    };

    Some(AnnotatedSearchResult {
        result,
        variant: Some(variant),
        fit: Some(FitResult::MacosOnly),
        error: None,
        filtered: None,
    })
}

/// Hide runnable-path rows that can't run locally: cloud-only always (no local
/// weights), and non-fitting under `--fit`. `--all` keeps everything. macOS-only
/// rows are produced and capped separately by the caller, so they don't pass
/// through here. Returns `(cloud_hidden, fit_hidden)`.
fn filter_search_rows(
    rows: &mut Vec<AnnotatedSearchResult>,
    fit_filter: bool,
    all: bool,
) -> (u64, u64) {
    if all {
        return (0, 0);
    }

    let mut cloud = 0u64;
    let mut fit = 0u64;

    rows.retain(|row| {
        if matches!(row.filtered, Some(FilteredReason::CloudOnly)) {
            cloud += 1;
            return false;
        }
        if fit_filter {
            if let Some(fit_status) = &row.fit {
                if !fit_status.fits() {
                    fit += 1;
                    return false;
                }
            }
        }
        true
    });

    (cloud, fit)
}

/// `--quick` basic browse: list matches with an approximate download-size range
/// from each model's tag page (one fetch per result, no manifest lookups, no
/// hardware fit). The fast/cheap counterpart to the default annotated view.
fn cmd_search_library(client: &Client, query: &str, limit: usize, wide: bool) -> Result<()> {
    let mut results = registry::search_models(client, query)?;
    registry::rank_search_results(&mut results, query);
    results.truncate(limit);

    let mut registry_client = HttpRegistry::new(client);
    let rows: Vec<(SearchResult, Option<String>)> = results
        .into_iter()
        .map(|result| {
            let size = registry_client
                .list_tags(&result.name)
                .ok()
                .and_then(|tags| display::size_range_label(&tags));
            (result, size)
        })
        .collect();

    if wide {
        display::print_library_results_with_size(&rows);
    } else {
        display::print_library_results_with_size_compact(&rows);
    }
    Ok(())
}

fn annotate_search_result<R: Registry>(
    registry_client: &mut R,
    result: SearchResult,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
) -> AnnotatedSearchResult {
    match resolver::resolve_with_registry(registry_client, &result.name, hw, opts) {
        Ok((variant, fit)) => AnnotatedSearchResult {
            result,
            variant: Some(variant),
            fit: Some(fit),
            error: None,
            filtered: None,
        },
        Err(err) => {
            let filtered = match &err {
                ResolverError::ManifestCloudOnly { .. } => Some(FilteredReason::CloudOnly),
                ResolverError::ManifestPlatformRestricted { .. } => {
                    Some(FilteredReason::PlatformRestricted)
                }
                _ => None,
            };
            AnnotatedSearchResult {
                result,
                variant: None,
                fit: None,
                error: Some(err.to_string()),
                filtered,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_resolve(
    metadata_client: &Client,
    pull_client: &Client,
    model_input: &str,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
    selection: ModelSelectionOptions,
    yes: bool,
    ollama_host: &str,
    ollama_port: u16,
) -> Result<()> {
    let model_name = model_input
        .strip_suffix('?')
        .ok_or_else(|| ResolverError::InvalidInput("resolve input must end with '?'".into()))?;

    if model_name.is_empty() {
        return Err(ResolverError::InvalidInput("model name cannot be empty".into()));
    }

    let search_results = registry::search_models(metadata_client, model_name)?;
    let resolved_name = choose_model_name(model_name, &search_results, selection)?;

    if !selection.quiet && resolved_name != model_name {
        eprintln!("Using selected model: {}", terminal_line(&resolved_name));
    }

    let mut registry_client = HttpRegistry::new(metadata_client);
    let outcome = resolver::resolve_with_registry_diagnostics(&mut registry_client, &resolved_name, hw, opts)?;
    let variant = outcome.variant;
    let fit = outcome.fit;
    let diagnostics = outcome.diagnostics;

    if !fit.fits() {
        if selection.quiet {
            return Err(ResolverError::Other(format!(
                "no variant of '{resolved_name}' fits the detected hardware"
            )));
        }

        display::print_resolve_result(&variant, &fit, hw);
        display::print_resolution_diagnostics(&diagnostics);
        if !yes {
            let proceed = Confirm::new()
                .with_prompt(format!("{} does not fit the detected hardware. Pull anyway?", terminal_line(&variant.full_ref)))
                .default(false)
                .interact()
                .unwrap_or(false);
            if !proceed {
                return Err(ResolverError::Other("aborted by user".into()));
            }
        }
    } else if !selection.quiet {
        display::print_resolve_result(&variant, &fit, hw);
        display::print_resolution_diagnostics(&diagnostics);
    }

    if !selection.quiet {
        eprintln!("Pulling {}...", terminal_line(&variant.full_ref));
    }
    local::pull_model(pull_client, ollama_host, ollama_port, &variant.full_ref, !selection.quiet)?;
    println!("{}", terminal_line(&variant.full_ref));
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ModelSelectionOptions {
    quiet: bool,
    select: Option<usize>,
    first: bool,
    fail_on_ambiguous: bool,
    interactive: bool,
}

impl ModelSelectionOptions {
    fn new(
        quiet: bool,
        select: Option<usize>,
        first: bool,
        fail_on_ambiguous: bool,
    ) -> Result<Self> {
        Self::with_interactive(
            quiet,
            select,
            first,
            fail_on_ambiguous,
            model_selection_is_interactive(),
        )
    }

    fn with_interactive(
        quiet: bool,
        select: Option<usize>,
        first: bool,
        fail_on_ambiguous: bool,
        interactive: bool,
    ) -> Result<Self> {
        if matches!(select, Some(0)) {
            return Err(ResolverError::InvalidInput(
                "--select must be a 1-based candidate number greater than 0".into(),
            ));
        }

        let selection_directives = (select.is_some() as usize)
            + (first as usize)
            + (fail_on_ambiguous as usize);
        if selection_directives > 1 {
            return Err(ResolverError::InvalidInput(
                "use only one of --select, --first, or --fail-on-ambiguous".into(),
            ));
        }

        if quiet && (select.is_some() || first) {
            return Err(ResolverError::InvalidInput(
                "--quiet cannot be combined with --select or --first; quiet resolve fails on non-exact model searches".into(),
            ));
        }

        Ok(Self {
            quiet,
            select,
            first,
            fail_on_ambiguous,
            interactive,
        })
    }
}

fn model_selection_is_interactive() -> bool {
    io::stdin().is_terminal() && io::stderr().is_terminal()
}

fn choose_model_name(
    query: &str,
    results: &[SearchResult],
    selection: ModelSelectionOptions,
) -> Result<String> {
    choose_model_name_with_picker(query, results, selection, |candidates| {
        prompt_for_model_choice(query, candidates)
    })
}

fn choose_model_name_with_picker<F>(
    query: &str,
    results: &[SearchResult],
    selection: ModelSelectionOptions,
    mut picker: F,
) -> Result<String>
where
    F: FnMut(&[SearchResult]) -> Result<Option<usize>>,
{
    if let Some(exact) = results.iter().find(|result| result.name == query) {
        return Ok(exact.name.clone());
    }

    if results.is_empty() {
        return Err(ResolverError::NoSearchResults {
            query: query.to_string(),
        });
    }

    let candidate_count = results.len().min(8);
    let candidates = &results[..candidate_count];

    if selection.quiet || selection.fail_on_ambiguous {
        return Err(ResolverError::AmbiguousModel {
            query: query.to_string(),
            candidates: candidate_list(results),
        });
    }

    if let Some(selection_number) = selection.select {
        if selection_number <= candidates.len() {
            return Ok(candidates[selection_number - 1].name.clone());
        }
        return Err(ResolverError::InvalidInput(format!(
            "--select {selection_number} is outside the candidate list; choose a number from 1 to {}",
            candidates.len()
        )));
    }

    if selection.first {
        return Ok(candidates[0].name.clone());
    }

    if !selection.interactive {
        return Err(ResolverError::Other(non_interactive_ambiguity_message(
            query,
            candidates,
        )));
    }

    match picker(candidates)? {
        Some(index) if index < candidates.len() => Ok(candidates[index].name.clone()),
        Some(index) => Err(ResolverError::InvalidInput(format!(
            "selected model index {index} is outside the candidate list"
        ))),
        None => Err(ResolverError::Other("aborted by user".into())),
    }
}

fn non_interactive_ambiguity_message(query: &str, candidates: &[SearchResult]) -> String {
    let mut lines = vec![format!(
        "No exact model match for {}. Re-run with an exact model name, use --quiet, pass --select <N>, pass --first, or pass --fail-on-ambiguous.",
        terminal_line(query)
    )];

    lines.push("Top candidates:".to_string());
    for (idx, candidate) in candidates.iter().enumerate() {
        lines.push(format!("  {}. {}", idx + 1, format_model_choice(candidate)));
    }

    lines.join("\n")
}

fn prompt_for_model_choice(query: &str, candidates: &[SearchResult]) -> Result<Option<usize>> {
    eprintln!(
        "No exact model match found for '{}'. Select one of the top matches:",
        terminal_line(query)
    );
    let items = candidates
        .iter()
        .map(format_model_choice)
        .collect::<Vec<_>>();

    Select::new()
        .with_prompt("Model")
        .items(&items)
        .default(0)
        .interact_opt()
        .map_err(|err| ResolverError::Other(format!("model selection failed: {err}")))
}

fn format_model_choice(result: &SearchResult) -> String {
    let mut parts = vec![terminal_line(&result.name)];

    let mut metadata = Vec::new();
    if !result.pulls.is_empty() {
        metadata.push(terminal_line(&result.pulls));
    }
    if !result.tag_count.is_empty() {
        metadata.push(terminal_line(&result.tag_count));
    }
    if !metadata.is_empty() {
        parts.push(format!("({})", metadata.join(", ")));
    }
    if !result.description.is_empty() {
        parts.push(format!("- {}", terminal_line(&result.description)));
    }

    parts.join(" ")
}

fn candidate_list(results: &[SearchResult]) -> String {
    results
        .iter()
        .take(8)
        .map(|result| terminal_line(&result.name))
        .collect::<Vec<_>>()
        .join(", ")
}

fn cmd_pull_exact(
    client: &Client,
    model: &str,
    quiet: bool,
    ollama_host: &str,
    ollama_port: u16,
) -> Result<()> {
    if model.trim().is_empty() {
        return Err(ResolverError::InvalidInput("model name cannot be empty".into()));
    }
    if !quiet {
        eprintln!("Pulling {}...", terminal_line(model));
    }
    local::pull_model(client, ollama_host, ollama_port, model, !quiet)?;
    println!("{}", terminal_line(model));
    Ok(())
}

fn cmd_info(client: &Client, host: &str, port: u16, hw: &HardwareProfile) -> Result<()> {
    display::print_hardware(hw);

    if local::is_reachable(client, host, port) {
        match local::list_local_models(client, host, port) {
            Ok(models) => {
                let model_list = models
                    .into_iter()
                    .map(|model| (model.name, model.size))
                    .collect::<Vec<_>>();
                display::print_local_models(&model_list);
            }
            Err(err) => eprintln!("Warning: could not list local models: {}", terminal_line(&err.to_string())),
        }
    } else {
        eprintln!("ollama is not reachable at {}", terminal_line(&local::base_url(host, port)));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_result(name: &str) -> SearchResult {
        SearchResult {
            name: name.into(),
            description: format!("description for {name}"),
            pulls: "1M Pulls".into(),
            tag_count: "10 Tags".into(),
            updated: String::new(),
        }
    }

    fn selection(
        quiet: bool,
        select: Option<usize>,
        first: bool,
        fail_on_ambiguous: bool,
        interactive: bool,
    ) -> ModelSelectionOptions {
        ModelSelectionOptions::with_interactive(
            quiet,
            select,
            first,
            fail_on_ambiguous,
            interactive,
        )
        .unwrap()
    }

    #[test]
    fn quiet_mode_requires_exact_search_match() {
        let results = vec![search_result("qwen2.5-coder")];
        let quiet = selection(true, None, false, false, false);
        assert!(choose_model_name("qwen", &results, quiet).is_err());
        assert_eq!(
            choose_model_name("qwen2.5-coder", &results, quiet).unwrap(),
            "qwen2.5-coder"
        );
    }

    #[test]
    fn interactive_mode_uses_picker_for_single_non_exact_match() {
        let results = vec![search_result("qwen2.5-coder")];
        let options = selection(false, None, false, false, true);

        let chosen = choose_model_name_with_picker("qwen", &results, options, |candidates| {
            assert_eq!(candidates.len(), 1);
            Ok(Some(0))
        })
        .unwrap();

        assert_eq!(chosen, "qwen2.5-coder");
    }

    #[test]
    fn interactive_mode_uses_picker_for_multiple_non_exact_matches() {
        let results = vec![
            search_result("qwen3"),
            search_result("qwen2.5-coder"),
            search_result("qwen2.5"),
        ];
        let options = selection(false, None, false, false, true);

        let chosen = choose_model_name_with_picker("qwen", &results, options, |candidates| {
            assert_eq!(candidates.len(), 3);
            Ok(Some(1))
        })
        .unwrap();

        assert_eq!(chosen, "qwen2.5-coder");
    }

    #[test]
    fn interactive_mode_limits_picker_to_top_eight_matches() {
        let results = (0..12)
            .map(|idx| search_result(&format!("qwen-candidate-{idx}")))
            .collect::<Vec<_>>();
        let options = selection(false, None, false, false, true);

        let chosen = choose_model_name_with_picker("qwen", &results, options, |candidates| {
            assert_eq!(candidates.len(), 8);
            Ok(Some(7))
        })
        .unwrap();

        assert_eq!(chosen, "qwen-candidate-7");
    }

    #[test]
    fn interactive_mode_reports_user_cancel() {
        let results = vec![search_result("qwen3"), search_result("qwen2.5-coder")];
        let options = selection(false, None, false, false, true);

        let err = choose_model_name_with_picker("qwen", &results, options, |_| Ok(None))
            .unwrap_err();
        assert!(err.to_string().contains("aborted by user"));
    }

    #[test]
    fn select_option_uses_one_based_candidate_number() {
        let results = vec![
            search_result("qwen3"),
            search_result("qwen2.5-coder"),
            search_result("qwen2.5"),
        ];
        let options = selection(false, Some(2), false, false, false);

        let chosen = choose_model_name_with_picker("qwen", &results, options, |_| {
            panic!("picker should not run when --select is present")
        })
        .unwrap();

        assert_eq!(chosen, "qwen2.5-coder");
    }

    #[test]
    fn select_option_rejects_out_of_range_candidate_number() {
        let results = vec![search_result("qwen3")];
        let options = selection(false, Some(2), false, false, false);

        let err = choose_model_name_with_picker("qwen", &results, options, |_| {
            panic!("picker should not run when --select is present")
        })
        .unwrap_err();

        assert!(err.to_string().contains("--select 2 is outside"));
    }

    #[test]
    fn first_option_selects_first_candidate_without_prompting() {
        let results = vec![search_result("qwen3"), search_result("qwen2.5-coder")];
        let options = selection(false, None, true, false, false);

        let chosen = choose_model_name_with_picker("qwen", &results, options, |_| {
            panic!("picker should not run when --first is present")
        })
        .unwrap();

        assert_eq!(chosen, "qwen3");
    }

    #[test]
    fn fail_on_ambiguous_returns_ambiguous_error_without_prompting() {
        let results = vec![search_result("qwen3"), search_result("qwen2.5-coder")];
        let options = selection(false, None, false, true, false);

        let err = choose_model_name_with_picker("qwen", &results, options, |_| {
            panic!("picker should not run when --fail-on-ambiguous is present")
        })
        .unwrap_err();

        assert!(err.to_string().contains("did not produce an exact match"));
    }

    #[test]
    fn non_interactive_mode_reports_remediation_without_prompting() {
        let results = vec![search_result("qwen3"), search_result("qwen2.5-coder")];
        let options = selection(false, None, false, false, false);

        let err = choose_model_name_with_picker("qwen", &results, options, |_| {
            panic!("picker should not run in non-interactive mode")
        })
        .unwrap_err();
        let message = err.to_string();

        assert!(message.contains("No exact model match for qwen"));
        assert!(message.contains("--select <N>"));
        assert!(message.contains("--first"));
        assert!(message.contains("qwen3"));
    }

    #[test]
    fn rejects_zero_select_value() {
        let err = ModelSelectionOptions::with_interactive(false, Some(0), false, false, false)
            .unwrap_err();
        assert!(err.to_string().contains("--select must be"));
    }

    #[test]
    fn rejects_conflicting_selection_directives() {
        let err = ModelSelectionOptions::with_interactive(false, Some(1), true, false, false)
            .unwrap_err();
        assert!(err.to_string().contains("use only one"));
    }

    #[test]
    fn rejects_quiet_with_select_or_first() {
        let select_err = ModelSelectionOptions::with_interactive(true, Some(1), false, false, false)
            .unwrap_err();
        assert!(select_err.to_string().contains("--quiet cannot be combined"));

        let first_err = ModelSelectionOptions::with_interactive(true, None, true, false, false)
            .unwrap_err();
        assert!(first_err.to_string().contains("--quiet cannot be combined"));
    }

    #[test]
    fn model_choice_includes_metadata_without_requiring_it() {
        let formatted = format_model_choice(&search_result("qwen3"));
        assert!(formatted.contains("qwen3"));
        assert!(formatted.contains("1M Pulls"));
        assert!(formatted.contains("10 Tags"));
    }

    fn annotated_row(
        name: &str,
        filtered: Option<FilteredReason>,
        fit: Option<crate::types::FitResult>,
    ) -> AnnotatedSearchResult {
        AnnotatedSearchResult {
            result: search_result(name),
            variant: None,
            fit,
            error: None,
            filtered,
        }
    }

    /// Minimal registry that only serves a fixed tag list (annotate_macos_only
    /// never fetches manifests).
    struct TagOnlyRegistry {
        tags: Vec<crate::types::TagInfo>,
    }

    impl Registry for TagOnlyRegistry {
        fn list_tags(&mut self, _model: &str) -> Result<Vec<crate::types::TagInfo>> {
            Ok(self.tags.clone())
        }
        fn get_manifest_size(&mut self, _model: &str, _tag: &str) -> Result<(u64, u64)> {
            Err(ResolverError::Other("get_manifest_size not used here".into()))
        }
    }

    #[test]
    fn annotate_macos_only_picks_largest_macos_variant() {
        use crate::types::{tag_info_from_str, FitResult};
        let mut reg = TagOnlyRegistry {
            tags: vec![
                tag_info_from_str("qwen3.5", "9b"),
                tag_info_from_str("qwen3.5", "9b-nvfp4"),
                tag_info_from_str("qwen3.5", "32b-nvfp4"),
            ],
        };
        let row = annotate_macos_only(&mut reg, search_result("qwen3.5"))
            .expect("model offers a macOS-only variant");
        assert!(matches!(row.fit, Some(FitResult::MacosOnly)));
        assert_eq!(row.variant.unwrap().tag, "32b-nvfp4"); // largest macOS-only variant
    }

    #[test]
    fn annotate_macos_only_none_without_macos_variant() {
        use crate::types::tag_info_from_str;
        let mut reg = TagOnlyRegistry {
            tags: vec![
                tag_info_from_str("qwen3", "8b"),
                tag_info_from_str("qwen3", "14b-q4_K_M"),
            ],
        };
        assert!(annotate_macos_only(&mut reg, search_result("qwen3")).is_none());
    }

    fn names(rows: &[AnnotatedSearchResult]) -> Vec<String> {
        rows.iter().map(|r| r.result.name.clone()).collect()
    }

    #[test]
    fn filter_hides_cloud_and_nonfitting_under_fit() {
        use crate::types::FitResult;
        let mut rows = vec![
            annotated_row("cloud", Some(FilteredReason::CloudOnly), None),
            annotated_row("too-big", None, Some(FitResult::DoesNotFit { need: 10, have: 1 })),
            annotated_row("fits", None, Some(FitResult::FitsVram)),
        ];
        let counts = filter_search_rows(&mut rows, true, false);
        assert_eq!(counts, (1, 1)); // 1 cloud, 1 non-fitting hidden
        assert_eq!(names(&rows), vec!["fits"]);
    }

    #[test]
    fn filter_all_keeps_everything() {
        use crate::types::FitResult;
        let mut rows = vec![
            annotated_row("cloud", Some(FilteredReason::CloudOnly), None),
            annotated_row("too-big", None, Some(FitResult::DoesNotFit { need: 10, have: 1 })),
            annotated_row("fits", None, Some(FitResult::FitsVram)),
        ];
        let counts = filter_search_rows(&mut rows, true, true);
        assert_eq!(counts, (0, 0));
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn cap_rows_reserves_slots_for_macos_only() {
        use crate::types::FitResult;
        let normal = vec![
            annotated_row("a", None, Some(FitResult::FitsVram)),
            annotated_row("b", None, Some(FitResult::FitsVram)),
            annotated_row("c", None, Some(FitResult::FitsVram)),
        ];
        let macos = vec![annotated_row("m", None, Some(FitResult::MacosOnly))];
        let rows = cap_rows(normal, macos, 3);
        // 1 slot reserved for macOS-only → 2 runnable + 1 macOS-only, runnable first.
        assert_eq!(names(&rows), vec!["a", "b", "m"]);
    }

    #[test]
    fn cap_rows_runnable_fill_when_no_macos() {
        use crate::types::FitResult;
        let normal = vec![
            annotated_row("a", None, Some(FitResult::FitsVram)),
            annotated_row("b", None, Some(FitResult::FitsVram)),
            annotated_row("c", None, Some(FitResult::FitsVram)),
        ];
        let rows = cap_rows(normal, vec![], 2);
        assert_eq!(names(&rows), vec!["a", "b"]);
    }

    #[test]
    fn is_macos_only_tag_detects_nvfp4() {
        use crate::types::is_macos_only_tag;
        assert!(is_macos_only_tag("9b-nvfp4"));
        assert!(is_macos_only_tag("35b-a3b-NVFP4"));
        assert!(!is_macos_only_tag("9b"));
        assert!(!is_macos_only_tag("14b-q4_K_M"));
    }
}
