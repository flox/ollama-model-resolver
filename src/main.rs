use clap::Parser;
use ollama_model_resolver::{app, cli::Cli, sanitize::terminal_text};

fn main() {
    let cli = Cli::parse();

    if let Err(err) = app::run(cli) {
        eprintln!("Error: {}", terminal_text(&err.to_string()));
        std::process::exit(1);
    }
}
