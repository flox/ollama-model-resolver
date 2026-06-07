# ollama-model-resolver — Implementation Brief

## What This Is

A Rust CLI tool that resolves the best ollama model variant for the user's hardware and pulls it. It fills a gap in the ollama ecosystem: ollama has no `search` command, no hardware-aware model selection, and no way to know before downloading whether a model will fit.

## Project Location

`~/dev/builds/ollama-model-resolver/`

This project will be built as a Flox package via a Nix expression in `.flox/pkgs/`, then published with `flox publish`. See the **Nix Expression Build** section below.

## UX Contract

### The `?` convention

A trailing `?` on a model name signals "resolve this for me." Without `?`, the model name is treated as exact and passed through to ollama as-is. The `?` is **unquoted** in the shell — this is intentional and acceptable. (In bash, `?` is a glob for one character, but it will pass through literally unless a file like `qwen3X` exists in the cwd, which effectively never happens in practice.)

```bash
# Resolve mode — pick best variant for my hardware, pull it, output model:tag
ollama-model-resolver resolve qwen3?

# Exact mode — no resolution, just validate/pull this specific tag
ollama-model-resolver resolve qwen2.5-coder:14b

# Search mode — show what's available upstream, annotated with hardware fit
ollama-model-resolver search qwen

# Hardware info
ollama-model-resolver info
```

The calling wrapper (the `ollama` shell script in the agentic-playground repo) passes the model name through to the resolver unchanged — the resolver itself detects and strips the trailing `?`. When `?` is absent and the model string contains a `:` tag, the wrapper skips the resolver entirely and passes the exact model:tag to the launch script.

### Output modes

- `resolve` with `--quiet`: outputs only `model:tag` on stdout. Suitable for `$(...)` substitution in shell scripts.
- `resolve` without `--quiet`: shows hardware summary, resolution reasoning, pull progress, and final `model:tag`.
- `search`: pretty table with model names, sizes, fit indicators.
- `info`: hardware profile and locally pulled models.

## Data Sources (Verified Working)

### 1. Search models on ollama.com
```bash
curl -sf 'https://ollama.com/search?q=qwen' \
  | grep -o 'href="/library/[^"]*"' \
  | sed 's|href="/library/||;s|"||'
# Returns: qwen3.5, qwen3.6, qwen3-coder, qwen2.5-coder, ...
```
Parse HTML. Extract model names from `href="/library/<name>"` links. Descriptions and pull counts are also in the HTML.

### 2. List available tags/variants for a model
```bash
curl -sf 'https://ollama.com/library/qwen2.5-coder/tags' \
  | grep -oP 'qwen2.5-coder:[^"<]+' | sort -u
# Returns: qwen2.5-coder:0.5b, qwen2.5-coder:7b, qwen2.5-coder:14b, ...
```
Parse HTML. Tags include size labels (e.g., "4.7GB") in the page — use these for initial filtering to avoid excessive manifest API calls.

### 3. Get exact model size (JSON API)
```bash
curl -sf 'https://registry.ollama.ai/v2/library/qwen2.5-coder/manifests/latest' \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json'
```
Returns JSON:
```json
{
  "layers": [
    {"mediaType": "application/vnd.ollama.image.model", "size": 4683074048},
    {"mediaType": "application/vnd.ollama.image.system", "size": 68},
    {"mediaType": "application/vnd.ollama.image.template", "size": 1615},
    {"mediaType": "application/vnd.ollama.image.license", "size": 11343}
  ]
}
```
The `application/vnd.ollama.image.model` layer is the weights — its `size` is exact bytes. This is the authoritative source for fit calculations.

### 4. Local ollama API
```bash
# List pulled models
curl -sf http://127.0.0.1:11434/api/tags | jq '.models[].name'

# Model metadata (parameter count, quantization, context length)
curl -sf http://127.0.0.1:11434/api/show -d '{"name":"qwen2.5-coder"}'
```

### 5. Hardware detection
```bash
# GPU VRAM (MiB)
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# System RAM (kB, from /proc/meminfo)
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo

# Disk free (on ollama models dir)
# Use statvfs syscall on $OLLAMA_MODELS or ~/.ollama/models
```

## Hardware Detection Details

| Resource | Source | Fallback |
|----------|--------|----------|
| GPU VRAM | `nvidia-smi` CLI | 0 (CPU-only mode) |
| System RAM | `/proc/meminfo` MemTotal + MemAvailable | Error — required |
| Disk space | `statvfs` on `$OLLAMA_MODELS` or `~/.ollama/models` | `statvfs` on `$HOME` |

If `nvidia-smi` is absent or fails, the tool operates in CPU-only mode: models are checked against available RAM instead of VRAM. No error — just a different code path.

## Resolution Algorithm

Given a model name with `?` (e.g., `qwen2.5-coder?`):

```
1. Strip the trailing `?`
2. Search ollama.com for the model name
3. If exact match found, use it; if not, present top matches and let user pick (or fail in --quiet mode)
4. Fetch tags for the resolved model name
5. Filter tags:
   a. Prefer instruct variants over base variants
   b. Prefer Q4_K_M quantization (ollama's default) — fall back to default tag if no Q4_K_M
6. Sort remaining variants by parameter count descending (largest first)
7. For each variant (largest → smallest):
   a. Get exact weights size from registry manifest API
   b. Compute estimated runtime memory:
      estimated = weights_bytes * 1.2   (20% margin for KV cache + overhead)
   c. Check fit:
      DEFAULT MODE:  estimated <= gpu_vram_free
      --split MODE:  estimated <= gpu_vram_free + ram_available (ollama auto-splits)
      CPU-ONLY:      estimated <= ram_available
   d. Check disk:    manifest_total_bytes <= disk_free
   e. First variant that passes all checks → winner
8. If nothing fits:
   - Show warning with hardware stats and model requirements
   - Prompt yes/no: "Your system cannot run <model:tag>. Try anyway?"
   - If --yes flag: skip prompt, proceed
   - If --quiet flag: exit 1
9. Pull the resolved model:tag via ollama pull
10. Output model:tag
```

### The 20% margin

The 20% margin (`weights * 1.2`) accounts for:
- **KV cache**: memory for attention key/value states. Scales with context window length. At ollama's default context (~4K-8K tokens), this is ~5-10% of weights. At 32K context, ~20-30%.
- **CUDA context + ollama runtime**: ~300-500 MiB fixed overhead.

This is a rough heuristic. The `--margin` flag lets users adjust it (e.g., `--margin 30` for 30%). The margin applies to the fit calculation, not to the actual download size.

For reference on a 32GB VRAM GPU:
- 7B Q4_K_M (4.7 GB weights): needs ~5.6 GB → fits easily
- 14B Q4_K_M (9.0 GB weights): needs ~10.8 GB → fits easily
- 32B Q4_K_M (20 GB weights): needs ~24 GB → fits with 8 GB headroom
- 70B Q4_K_M (~40 GB weights): needs ~48 GB → does not fit (needs --split)

## CLI Interface

```
ollama-model-resolver [OPTIONS] <COMMAND>

Commands:
  search    Search ollama.com for models, annotated with hardware fit
  resolve   Resolve best variant for hardware, pull it, output model:tag
  info      Show detected hardware and locally pulled models

Global Options:
  --split              Allow models that split between VRAM and RAM
  --margin <PERCENT>   Memory safety margin percentage [default: 20]
  --ollama-host <H>    Ollama host [default: 127.0.0.1]
  --ollama-port <P>    Ollama port [default: 11434]

resolve <MODEL>
  --quiet              Output only model:tag, no decoration
  --yes                Skip confirmation prompts

search <QUERY>
  --limit <N>          Max results [default: 20]
```

## Rust Project Structure

```
ollama-model-resolver/
  Cargo.toml
  Cargo.lock
  src/
    main.rs          # Entrypoint, clap dispatch
    cli.rs           # Clap derive structs
    hardware.rs      # GPU/RAM/disk detection
    registry.rs      # ollama.com HTML scraping + registry.ollama.ai JSON API
    local.rs         # Local ollama API (list, show)
    resolver.rs      # Core resolution algorithm
    display.rs       # Pretty tables, colored fit indicators
    error.rs         # thiserror error enum
    types.rs         # HardwareProfile, ModelVariant, FitResult, etc.
  .flox/
    env.json
    env/
      manifest.toml
    pkgs/
      ollama-model-resolver/
        default.nix
```

## Crate Dependencies

```toml
[package]
name = "ollama-model-resolver"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
reqwest = { version = "0.12", features = ["blocking", "json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
scraper = "0.22"          # HTML parsing for ollama.com
comfy-table = "7"         # Pretty table output
colored = "3"             # Terminal colors
dialoguer = "0.11"        # yes/no confirmation prompts
bytesize = "2"            # Human-readable byte formatting (e.g., "4.7 GB")
thiserror = "2"           # Error type derivation
libc = "0.2"              # statvfs syscall for disk space
```

Rationale:
- `reqwest` blocking — this is a CLI tool, no need for async/tokio complexity.
- `scraper` — CSS-selector-based HTML parsing, purpose-built for this.
- No `sysinfo` crate — heavyweight. We only need three numbers: VRAM from `nvidia-smi` subprocess, RAM from `/proc/meminfo` read, disk from `statvfs` libc call.
- `libc` — thin wrapper for `statvfs` syscall. Much lighter than the `nix` crate for a single syscall.

## Core Types

```rust
pub struct HardwareProfile {
    pub gpu_name: Option<String>,
    pub vram_total: u64,      // bytes
    pub vram_free: u64,       // bytes
    pub ram_total: u64,       // bytes
    pub ram_available: u64,   // bytes
    pub disk_free: u64,       // bytes
    pub models_dir: PathBuf,  // where ollama stores models
}

pub struct ModelVariant {
    pub name: String,            // "qwen2.5-coder"
    pub tag: String,             // "14b-instruct-q4_K_M"
    pub full_ref: String,        // "qwen2.5-coder:14b-instruct-q4_K_M"
    pub weights_bytes: u64,      // from registry manifest
    pub total_bytes: u64,        // all layers summed
    pub param_billions: Option<f64>,  // parsed from tag (e.g., 14.0)
    pub quantization: Option<String>, // parsed from tag (e.g., "q4_K_M")
    pub is_instruct: bool,       // tag contains "instruct" or is default
}

pub struct SearchResult {
    pub name: String,
    pub description: String,
    pub pulls: String,           // "16.4M Pulls"
    pub tag_count: String,       // "199 Tags"
}

pub enum FitResult {
    FitsVram,
    FitsWithSplit { gpu_pct: f64 },
    FitsRamOnly,
    DoesNotFit { need: u64, have: u64 },
    InsufficientDisk { need: u64, have: u64 },
}
```

## Nix Expression Build

File: `.flox/pkgs/ollama-model-resolver/default.nix`

This builds from local source (the Rust project in the repo root). Reference pattern: `claw-code.nix` in `~/dev/builds/nix-ai-tools/.flox/pkgs/`.

```nix
{
  lib,
  rustPlatform,
  pkg-config,
  openssl,
}:
rustPlatform.buildRustPackage {
  pname = "ollama-model-resolver";
  version = "0.1.0";

  src = ../../.;

  cargoLock.lockFile = "${src}/Cargo.lock";

  nativeBuildInputs = [ pkg-config ];

  buildInputs = [ openssl ];

  doCheck = false;

  meta = {
    description = "Resolve the best ollama model variant for your hardware";
    license = lib.licenses.mit;
    mainProgram = "ollama-model-resolver";
  };
}
```

Build inputs:
- `pkg-config` (native) — needed by `openssl-sys` for `reqwest`'s TLS
- `openssl` — TLS backend for HTTPS requests to ollama.com and registry.ollama.ai

### Build & Publish Workflow

```bash
cd ~/dev/builds/ollama-model-resolver
flox init                    # create .flox/
# add rust toolchain to manifest for dev
flox build                   # builds via the nix expression
./result-ollama-model-resolver/bin/ollama-model-resolver info   # test
git add . && git commit && git push
flox publish -o <org> ollama-model-resolver
```

### Flox Dev Manifest

The `.flox/env/manifest.toml` for development (not the nix expression — this is for `flox activate` during dev):

```toml
[install]
rustc.pkg-path = "rustc"
cargo.pkg-path = "cargo"
rust-analyzer.pkg-path = "rust-analyzer"
pkg-config.pkg-path = "pkg-config"
openssl.pkg-path = "openssl"
```

## Integration Points

Once published, this tool is consumed by the `ollama` wrapper in `~/dev/fucking-around/dev/agentic-playground/bin/ollama`. The wrapper adds two new dispatch entries:

```bash
case "$TOOL" in
  search)   shift 1; exec ollama-model-resolver search "$@" ;;
  # ... existing entries ...
esac
```

And modifies the launch dispatch to detect `?` and route through the resolver:

```bash
# Before dispatching to launch-*, check for ? and resolve
MODEL_ARG=""
for arg in "$@"; do
  if [[ "$arg" == *'?' ]]; then
    # Pass to resolver as-is — it handles stripping the ?
    MODEL_ARG="$(ollama-model-resolver resolve "$arg" --quiet)" || exit 1
    # Rebuild args with resolved model
  fi
done
```

The launch scripts (`launch-codex`, `launch-gemini`, etc.) remain unchanged — they receive a fully resolved `model:tag` by the time they run.

## What This Brief Does NOT Cover

- AMD GPU support (`rocm-smi`) — NVIDIA only for v0.1.0
- macOS Metal detection — Linux only for now (ollama on Mac uses Metal automatically)
- Async/streaming pull progress — use `ollama pull` subprocess for simplicity
- Model capability matching (e.g., "pick a model that supports tool use") — future work
