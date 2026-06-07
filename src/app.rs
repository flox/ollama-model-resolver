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
use crate::types::{AnnotatedSearchResult, FilteredReason, HardwareProfile, SearchResult};

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
        Commands::Search { query, limit, fit, no_fit: _, all } => {
            let search_opts = ResolveOpts {
                fit_filter: fit,
                all,
                ..opts
            };
            if fit {
                let hw = hardware::detect_with_policy(None, cli.gpu_fit_policy)?;
                cmd_search(&metadata_client, &query, limit, &hw, &search_opts, all)
            } else {
                cmd_search_no_fit(&metadata_client, &query, limit)
            }
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

fn cmd_search(
    client: &Client,
    query: &str,
    limit: usize,
    hw: &HardwareProfile,
    opts: &ResolveOpts,
    all: bool,
) -> Result<()> {
    // Fetch slightly more than limit to compensate for filtering; most models
    // are downloadable local models, so 5 extra catches the common case where
    // a few cloud-only entries appear in the top results.
    let fetch_limit = limit.saturating_add(5);
    let mut results = registry::search_models(client, query)?;
    results.truncate(fetch_limit);

    let mut registry_client = HttpRegistry::new(client);
    let mut rows: Vec<_> = results
        .into_iter()
        .map(|result| annotate_search_result(&mut registry_client, result, hw, opts))
        .collect();

    // Filter out cloud-only, platform-restricted, and non-fitting models unless --all.
    let (cloud_count, platform_count, fit_count) = if all {
        (0, 0, 0)
    } else {
        let mut cloud = 0u64;
        let mut platform = 0u64;
        let mut fit = 0u64;
        rows.retain(|row| {
            if let Some(FilteredReason::CloudOnly) = row.filtered {
                cloud += 1;
                return false;
            }
            if let Some(FilteredReason::PlatformRestricted) = row.filtered {
                platform += 1;
                return false;
            }
            if opts.fit_filter {
                if let Some(fit_status) = &row.fit {
                    if !fit_status.fits() {
                        fit += 1;
                        return false;
                    }
                }
            }
            true
        });
        (cloud, platform, fit)
    };

    rows.truncate(limit);

    display::print_search_results(&rows, hw, cloud_count, platform_count, fit_count);
    Ok(())
}

fn cmd_search_no_fit(client: &Client, query: &str, limit: usize) -> Result<()> {
    let mut results = registry::search_models(client, query)?;
    results.truncate(limit);
    display::print_search_results_unannotated(&results);
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
}
