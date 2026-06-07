use std::collections::HashSet;

use bytesize::ByteSize;
use colored::Colorize;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, ContentArrangement, Table};

use crate::resolver::ResolutionDiagnostics;
use crate::sanitize::terminal_line;
use crate::types::{AnnotatedSearchResult, FitResult, HardwareProfile, ModelVariant};

pub fn print_hardware(hw: &HardwareProfile) {
    println!("{}", "Hardware Profile".bold());
    println!(
        "  {:<14} {}",
        "GPU:".dimmed(),
        &hw.gpu_name.as_deref().map(terminal_line).unwrap_or_else(|| "none selected".to_string())
    );
    if !hw.gpus.is_empty() {
        println!(
            "  {:<14} {} detected, {} CUDA-visible, {} used for fit",
            "NVIDIA GPUs:".dimmed(),
            hw.gpus.len(),
            hw.visible_gpu_count(),
            hw.selected_gpu_count()
        );
    }
    if let Some(value) = &hw.cuda_visible_devices {
        println!("  {:<14} {}", "CUDA_VISIBLE:".dimmed(), terminal_line(value));
    }
    if hw.has_gpu() {
        println!(
            "  {:<14} {}",
            "VRAM basis:".dimmed(),
            ByteSize(hw.vram_total).to_string().green()
        );
    }
    println!(
        "  {:<14} {} / {}",
        "RAM:".dimmed(),
        ByteSize(hw.ram_available).to_string().green(),
        ByteSize(hw.ram_total)
    );
    println!("  {:<14} {}", "Disk free:".dimmed(), ByteSize(hw.disk_free));
    println!("  {:<14} {}", "Models dir:".dimmed(), terminal_line(&hw.models_dir.display().to_string()));
    println!("  {:<14} {}", "GPU policy:".dimmed(), hw.gpu_fit_policy.to_string());
    println!("  {:<14} {}", "Fit basis:".dimmed(), terminal_line(&hw.gpu_fit_basis()));
    println!();
}

pub fn print_search_results(rows: &[AnnotatedSearchResult], hw: &HardwareProfile) {
    print_hardware(hw);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Model", "Best variant", "Fit", "Weights", "Pulls", "Tags", "Updated", "Description"]);

    for row in rows {
        let (variant, fit, weights) = match (&row.variant, &row.fit) {
            (Some(variant), Some(fit)) => (
                terminal_line(&variant.full_ref),
                fit_summary_colored(fit),
                ByteSize(variant.weights_bytes).to_string(),
            ),
            _ => {
                let err = row.error.as_deref().unwrap_or("unavailable");
                if err.contains("cloud-only") {
                    ("-".to_string(), "cloud-only".dimmed().to_string(), "-".to_string())
                } else if err.contains("platform-restricted") {
                    ("-".to_string(), "platform-restricted".dimmed().to_string(), "-".to_string())
                } else {
                    (
                        "-".to_string(),
                        terminal_line(err).dimmed().to_string(),
                        "-".to_string(),
                    )
                }
            }
        };

        table.add_row(vec![
            terminal_line(&row.result.name),
            variant,
            fit,
            weights,
            terminal_line(&row.result.pulls),
            terminal_line(&row.result.tag_count),
            terminal_line(&row.result.updated),
            truncate(&row.result.description, 72),
        ]);
    }

    println!("{table}");
}


pub fn print_search_results_unannotated(results: &[crate::types::SearchResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Model", "Pulls", "Tags", "Updated", "Description"]);

    for result in results {
        table.add_row(vec![
            terminal_line(&result.name),
            terminal_line(&result.pulls),
            terminal_line(&result.tag_count),
            terminal_line(&result.updated),
            truncate(&result.description, 96),
        ]);
    }

    println!("{}", "Search Results".bold());
    println!(
        "  {}",
        "Library-only search. Re-run with --fit for hardware-aware tag and manifest annotation.".dimmed()
    );
    println!("{table}");
}

pub fn print_resolve_result(variant: &ModelVariant, fit: &FitResult, hw: &HardwareProfile) {
    print_hardware(hw);

    println!("{}", "Resolved Model".bold());
    println!("  {:<14} {}", "Model:".dimmed(), terminal_line(&variant.full_ref).bold());
    println!("  {:<14} {}", "Weights:".dimmed(), ByteSize(variant.weights_bytes));
    println!("  {:<14} {}", "Approx. pull:".dimmed(), ByteSize(variant.total_bytes));
    println!(
        "  {:<14} {}",
        "Est. runtime:".dimmed(),
        ByteSize(variant.estimated_runtime_bytes)
    );
    println!(
        "  {:<14} {} tokens",
        "Context:".dimmed(),
        variant.context_tokens
    );
    println!(
        "  {:<14} {}%",
        "Margin:".dimmed(),
        variant.runtime_margin_pct
    );
    if let Some(params) = variant.param_billions {
        println!("  {:<14} {:.1}B {}", "Parameters:".dimmed(), params, "(tag hint)".dimmed());
    }
    if let Some(ref quant) = variant.quantization {
        println!("  {:<14} {} {}", "Quantization:".dimmed(), terminal_line(quant), "(tag hint)".dimmed());
    }

    println!();
    println!(
        "  {}",
        "Fit is an estimate based on registry weights, context tokens, GPU policy, and runtime margin.".dimmed()
    );
    println!(
        "  {}",
        "Actual memory can change with context length, architecture, quantization, and offload behavior.".dimmed()
    );
    println!(
        "  {}",
        "Approx. pull size may overstate network/disk use when layers already exist locally.".dimmed()
    );
    println!();
    print_fit_indicator(fit);
}

pub fn print_fit_indicator(fit: &FitResult) {
    match fit {
        FitResult::FitsVram => println!("  {} Fits the selected GPU VRAM basis", "✓".green().bold()),
        FitResult::FitsWithSplit { gpu_pct } => println!(
            "  {} Fits with VRAM/RAM split ({gpu_pct:.0}% GPU)",
            "~".yellow().bold()
        ),
        FitResult::FitsRamOnly => println!("  {} Fits in RAM for CPU inference", "~".yellow().bold()),
        FitResult::DoesNotFit { need, have } => println!(
            "  {} Does not fit: need {}, have {}",
            "✗".red().bold(),
            ByteSize(*need),
            ByteSize(*have)
        ),
        FitResult::InsufficientDisk { need, have } => println!(
            "  {} Estimated disk shortfall: approximately need {}, have {}",
            "✗".red().bold(),
            ByteSize(*need),
            ByteSize(*have)
        ),
    }
    println!();
}

pub fn print_resolution_diagnostics(diagnostics: &ResolutionDiagnostics) {
    if diagnostics.manifest_checked.is_empty()
        && diagnostics.approx_deferred.is_empty()
        && diagnostics.manifest_skipped_by_cap.is_empty()
    {
        return;
    }

    println!("{}", "Resolution Reasoning".bold());
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Candidate", "Page size", "Manifest", "Decision"]);

    let mut shown = HashSet::new();
    for trace in diagnostics.manifest_checked.iter().take(20) {
        shown.insert(trace.full_ref.clone());
        table.add_row(vec![
            terminal_line(&trace.full_ref),
            trace.approx_size.as_deref().map(terminal_line).unwrap_or_else(|| "-".to_string()),
            "checked".to_string(),
            terminal_line(&trace.decision),
        ]);
    }

    let remaining_slots = 20_usize.saturating_sub(shown.len());
    for trace in diagnostics.manifest_skipped_by_cap.iter().take(remaining_slots) {
        shown.insert(trace.full_ref.clone());
        table.add_row(vec![
            terminal_line(&trace.full_ref),
            trace.approx_size.as_deref().map(terminal_line).unwrap_or_else(|| "-".to_string()),
            "not checked".to_string(),
            terminal_line(&trace.decision),
        ]);
    }

    let remaining_slots = 20_usize.saturating_sub(shown.len());
    for trace in diagnostics
        .approx_deferred
        .iter()
        .filter(|trace| !shown.contains(&trace.full_ref))
        .take(remaining_slots)
    {
        table.add_row(vec![
            terminal_line(&trace.full_ref),
            trace.approx_size.as_deref().map(terminal_line).unwrap_or_else(|| "-".to_string()),
            "deferred".to_string(),
            terminal_line(&trace.decision),
        ]);
    }

    println!("{table}");
}

pub fn print_local_models(models: &[(String, u64)]) {
    if models.is_empty() {
        println!("{}", "No local models found.".dimmed());
        return;
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Model", "Size"]);

    for (name, size) in models {
        table.add_row(vec![terminal_line(name), ByteSize(*size).to_string()]);
    }

    println!("{}", "Local Models".bold());
    println!("{table}");
}

fn fit_summary_colored(fit: &FitResult) -> String {
    match fit {
        FitResult::FitsVram => fit.summary().green().to_string(),
        FitResult::FitsWithSplit { .. } | FitResult::FitsRamOnly => fit.summary().yellow().to_string(),
        FitResult::DoesNotFit { .. } | FitResult::InsufficientDisk { .. } => fit.summary().red().to_string(),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    let sanitized = terminal_line(value);
    if sanitized.chars().count() <= max_chars {
        return sanitized;
    }
    let mut out: String = sanitized.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}
