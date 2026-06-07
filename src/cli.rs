use clap::{Parser, Subcommand};

use crate::types::GpuFitPolicy;

#[derive(Debug, Parser)]
#[command(name = "ollama-model-resolver")]
#[command(about = "Resolve the best Ollama model variant for local hardware")]
pub struct Cli {
    /// Allow variants whose estimated runtime memory fits across GPU VRAM and system RAM.
    #[arg(long = "split", global = true)]
    pub allow_split: bool,

    /// Runtime memory margin percentage for KV cache and runtime overhead.
    #[arg(long, default_value_t = 20, global = true)]
    pub margin: u32,

    /// Ollama host or base URL. Non-loopback targets require --allow-remote-ollama.
    #[arg(long, default_value = "127.0.0.1", global = true)]
    pub ollama_host: String,

    /// Ollama port. Ignored when --ollama-host is a full http(s) URL.
    #[arg(long, default_value_t = 11434, global = true)]
    pub ollama_port: u16,

    /// Permit --ollama-host to target a non-loopback Ollama endpoint.
    #[arg(long, global = true)]
    pub allow_remote_ollama: bool,

    /// Maximum registry manifest lookups per model resolution. By default, all ranked candidates may be checked.
    #[arg(long, global = true)]
    pub max_manifest_lookups: Option<usize>,

    /// Seconds a pull may go without receiving stream data before failing. There is no total pull deadline.
    #[arg(long, default_value_t = 300, global = true)]
    pub pull_stall_timeout: u64,

    /// NVIDIA VRAM policy used for fit estimates. "best" is conservative; "visible-sum" models multi-GPU offload; "all-sum" ignores CUDA_VISIBLE_DEVICES.
    #[arg(long = "gpu-fit-policy", value_enum, default_value = "best", global = true)]
    pub gpu_fit_policy: GpuFitPolicy,

    /// Context length, in tokens, used to adjust the runtime-memory estimate.
    #[arg(long, default_value_t = 8192, global = true)]
    pub context_tokens: u32,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Search ollama.com. Use --fit for hardware-aware annotation.
    Search {
        /// Search query.
        query: String,

        /// Max results to show.
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// Annotate search results with hardware fit. This may fetch tag pages and registry manifests.
        #[arg(long, conflicts_with = "no_fit")]
        fit: bool,

        /// Show all search results including cloud-only and platform-restricted models.
        #[arg(long, conflicts_with = "no_fit")]
        all: bool,

        /// Compatibility flag for library-only search. Library-only is now the default.
        #[arg(long = "no-fit", visible_alias = "fast", conflicts_with = "fit")]
        no_fit: bool,

        /// Show results in a tabular format instead of the compact one-line-per-model layout.
        #[arg(long)]
        wide: bool,
    },

    /// Resolve a model ending in '?' or pull an exact model reference.
    Resolve {
        /// Model name. Append '?' to request hardware-aware resolution.
        model: String,

        /// Output only the final model:tag. Quiet mode fails when model search has no exact match.
        #[arg(long)]
        quiet: bool,

        /// Select the Nth displayed search candidate when there is no exact match. N is 1-based.
        #[arg(long)]
        select: Option<usize>,

        /// Select the first search candidate when there is no exact match.
        #[arg(long)]
        first: bool,

        /// Fail immediately when there is no exact search match.
        #[arg(long = "fail-on-ambiguous")]
        fail_on_ambiguous: bool,

        /// Skip confirmation prompts for variants that do not fit.
        #[arg(long)]
        yes: bool,
    },

    /// Show detected hardware and local Ollama models.
    Info,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_search_fit_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--fit",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { fit, no_fit, .. } => {
                assert!(fit);
                assert!(!no_fit);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn search_defaults_to_library_only() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { fit, no_fit, .. } => {
                assert!(!fit);
                assert!(!no_fit);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parses_search_no_fit_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--no-fit",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { fit, no_fit, .. } => {
                assert!(!fit);
                assert!(no_fit);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parses_search_fast_alias() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--fast",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { fit, no_fit, .. } => {
                assert!(!fit);
                assert!(no_fit);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn rejects_conflicting_search_fit_modes() {
        assert!(Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--fit",
            "--no-fit",
        ])
        .is_err());
    }

    #[test]
    fn parses_all_flag() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--fit",
            "--all",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { all, fit, no_fit, .. } => {
                assert!(all);
                assert!(fit);
                assert!(!no_fit);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn all_conflicts_with_no_fit() {
        assert!(Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--all",
            "--no-fit",
        ])
        .is_err());
    }

    #[test]
    fn all_defaults_false() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--fit",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { all, .. } => {
                assert!(!all);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parses_wide_flag() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "search",
            "qwen",
            "--wide",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { wide, .. } => {
                assert!(wide);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn parses_allow_remote_ollama_global_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "--ollama-host",
            "http://192.0.2.10:11434",
            "--allow-remote-ollama",
            "info",
        ])
        .unwrap();

        assert!(cli.allow_remote_ollama);
        assert_eq!(cli.ollama_host, "http://192.0.2.10:11434");
    }

    #[test]
    fn parses_pull_stall_timeout_global_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "--pull-stall-timeout",
            "900",
            "resolve",
            "qwen?",
        ])
        .unwrap();

        assert_eq!(cli.pull_stall_timeout, 900);
    }

    #[test]
    fn parses_gpu_fit_policy_global_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "--gpu-fit-policy",
            "visible-sum",
            "resolve",
            "qwen?",
        ])
        .unwrap();

        assert_eq!(cli.gpu_fit_policy, GpuFitPolicy::VisibleSum);
    }

    #[test]
    fn parses_context_tokens_global_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "--context-tokens",
            "32768",
            "resolve",
            "qwen?",
        ])
        .unwrap();

        assert_eq!(cli.context_tokens, 32_768);
    }

    #[test]
    fn parses_resolve_select_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "resolve",
            "qwen?",
            "--select",
            "2",
        ])
        .unwrap();

        match cli.command {
            Commands::Resolve { select, .. } => assert_eq!(select, Some(2)),
            _ => panic!("expected resolve command"),
        }
    }

    #[test]
    fn parses_resolve_first_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "resolve",
            "qwen?",
            "--first",
        ])
        .unwrap();

        match cli.command {
            Commands::Resolve { first, .. } => assert!(first),
            _ => panic!("expected resolve command"),
        }
    }

    #[test]
    fn parses_resolve_fail_on_ambiguous_option() {
        let cli = Cli::try_parse_from([
            "ollama-model-resolver",
            "resolve",
            "qwen?",
            "--fail-on-ambiguous",
        ])
        .unwrap();

        match cli.command {
            Commands::Resolve { fail_on_ambiguous, .. } => assert!(fail_on_ambiguous),
            _ => panic!("expected resolve command"),
        }
    }
}
