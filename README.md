# ollama-model-resolver

`ollama-model-resolver` is a Linux-first Rust CLI for selecting and pulling an Ollama model variant that is likely to fit the detected machine. It searches the Ollama library, reads tag manifests from the Ollama registry, estimates runtime memory from model weights plus a configurable margin, checks disk space, and pulls the selected `model:tag` through the local Ollama API.

## Commands

```bash
ollama-model-resolver search qwen
ollama-model-resolver search qwen --fit
ollama-model-resolver search qwen --fast
ollama-model-resolver resolve 'qwen2.5-coder?'
ollama-model-resolver resolve 'qwen?' --select 2
ollama-model-resolver resolve 'qwen?' --first
ollama-model-resolver resolve qwen2.5-coder:14b
ollama-model-resolver info
ollama-model-resolver --ollama-host http://remote.example:11434 --allow-remote-ollama info
```

A trailing `?` asks the resolver to search and choose a variant. Quote model arguments that contain `?` when you can, for example `'qwen2.5-coder?'`, so the shell never treats `?` as a glob. Without `?`, the resolver treats the argument as an exact Ollama model reference and pulls it unchanged.

When search finds no exact model name in normal interactive mode, the resolver presents the top matches and prompts for a selection. In non-interactive normal mode, it fails with remediation instead of attempting an unusable prompt. Use `--select <N>` for a deterministic 1-based candidate choice, `--first` to accept the top search result, or `--fail-on-ambiguous` to make normal mode fail immediately on non-exact search matches. In `--quiet` mode, non-exact matches fail so shell substitution never receives an unapproved model name.

## Resolution policy

The resolver ranks variants by these rules:

1. Prefer non-base / instruct-style tags when present.
2. Prefer bare default tags and `Q4_K_M` tags when present.
3. Try larger parameter counts before smaller ones.
4. Use tag-page size hints to defer candidates that appear too large before manifest lookups.
5. Fetch registry manifests for plausible or unknown-size candidates first for final size math.
6. If none of those fit, fetch deferred candidates in ranked order so a pessimistic page-size hint cannot hide a fitting model.
7. Return the first fitting candidate; otherwise report the smallest evaluated non-fitting candidate for the warning / try-anyway prompt unless no manifest can be read.

Use `--max-manifest-lookups <N>` to cap manifest lookups per model resolution when performance matters more than exhaustive fallback checking. The default has no lookup cap, so correctness is favored over speed. `search` is library-only by default and performs one Ollama library request. Use `search --fit` when you want hardware-aware annotation; in that mode, the cap applies separately to each model that receives fit annotation. `search --no-fit` and `search --fast` remain accepted aliases for library-only search.

Global options include `--pull-stall-timeout <SECONDS>` for long model downloads. The default is 300 seconds without receiving any pull stream data. Pulls have no total request deadline, so a download may run for much longer as long as Ollama keeps sending stream events.

Hardware-fit controls:

```bash
ollama-model-resolver --gpu-fit-policy best resolve 'qwen?'
ollama-model-resolver --gpu-fit-policy visible-sum --split resolve 'qwen?'
ollama-model-resolver --context-tokens 32768 --margin 30 resolve 'qwen?'
```

`--gpu-fit-policy` accepts:

- `best` (default): use the single CUDA-visible NVIDIA GPU with the most free VRAM. This is conservative and matches the v0.1.0 default behavior.
- `visible-sum`: sum free VRAM across CUDA-visible NVIDIA GPUs. Use this when the target Ollama deployment can spread model layers across visible GPUs.
- `all-sum`: sum free VRAM across all detected NVIDIA GPUs, ignoring `CUDA_VISIBLE_DEVICES`. This is mainly for diagnostics and should not be used for launch decisions when CUDA visibility intentionally hides devices.

`CUDA_VISIBLE_DEVICES` is recorded in the hardware summary. The resolver marks GPUs as visible by index or GPU UUID token, then reports how many NVIDIA GPUs were detected, visible, and used for the fit estimate.

Runtime memory uses integer ceiling arithmetic with an effective margin:

```text
effective_margin_pct = margin_pct + context_margin_pct(context_tokens)
estimated_runtime_bytes = weights_bytes * (100 + effective_margin_pct) / 100
```

The default margin is 20 percent and the default context is 8192 tokens. Above 8192 tokens, the resolver adds 5 percentage points of margin for each additional 8192-token block, rounded up. This context adjustment is still a heuristic because the registry manifest does not provide layer count, hidden size, KV-cache type, or Ollama's final offload plan. Actual memory use can change with context length, model architecture, quantization, CUDA visibility, multi-GPU offload, unified memory behavior, and Ollama runtime behavior. Parameter count and quantization display values come from tag parsing and should be read as hints.

## Size and disk semantics

Registry manifests provide the model weight-layer size and summed layer size used by the resolver. If a manifest contains more than one Ollama model layer, the resolver sums all model-layer sizes with checked arithmetic before estimating runtime memory. The displayed pull size is approximate from the resolver's perspective: Ollama may already have some layers locally, which can reduce actual network transfer and new disk use.

## Ollama API use

The tool pulls through `POST /api/pull` with `stream=true`, so `--ollama-host` and `--ollama-port` apply consistently while large downloads can run beyond the metadata request timeout. By default, `resolve` and `info` only allow loopback Ollama endpoints (`127.0.0.1`, `localhost`, `::1`, `[::1]`, or full loopback `http(s)` URLs). Bare IPv6 literals such as `::1` are bracketed automatically when API URLs are built. Pass `--allow-remote-ollama` to target a non-loopback Ollama-compatible endpoint, and never populate that flag or `--ollama-host` from untrusted input. In non-quiet mode, the resolver renders streamed Ollama progress events with status, short digest, percentage, and byte counts when Ollama provides them. When stderr is a terminal, progress uses an in-place animated line. When stderr is redirected or captured, progress switches to newline-delimited events so logs and CI output do not contain carriage-return redraw sequences. Metadata calls use a bounded request timeout; pull calls use a separate client with a connection timeout and a configurable idle-read timeout (`--pull-stall-timeout`, default 300 seconds), but no total request deadline. The wrapper script avoids calling `ollama pull` from inside a binary named `ollama`, which prevents wrapper recursion.

## Hardware behavior

- NVIDIA inventory uses `nvidia-smi` with GPU index, UUID, name, total VRAM, and free VRAM.
- `CUDA_VISIBLE_DEVICES` is honored for the default `best` and `visible-sum` policies.
- The default fit basis is the single visible GPU with the most free VRAM.
- `--gpu-fit-policy visible-sum` models multi-GPU offload by summing CUDA-visible free VRAM.
- `--gpu-fit-policy all-sum` ignores `CUDA_VISIBLE_DEVICES` and exists for diagnostics.
- `resolve --select <N>` selects the 1-based Nth search candidate when there is no exact model match.
- `resolve --first` selects the top search candidate when there is no exact model match.
- `resolve --fail-on-ambiguous` fails immediately when search has no exact model match.
- `--split` allows the selected GPU VRAM basis plus system RAM for fit estimates.
- If no visible NVIDIA GPU is selected, the tool uses CPU/RAM fit checks and prints that basis in the hardware summary.
- RAM detection uses `/proc/meminfo` and targets Linux for v0.1.0.
- Disk checks use `statvfs` against `$OLLAMA_MODELS` or `~/.ollama/models`.

## Scraping and registry access

Ollama search and tag discovery currently parse `ollama.com` HTML. The parsing code lives behind the registry interface in `src/registry.rs` and includes snapshot-style parser tests for search cards, fallback links, tag links, invalid links, and manifest-size extraction. Manifest lookups are cached during a resolver run. Approximate tag-page sizes reduce registry calls by checking plausible candidates first, while manifests still provide final sizing for selected/evaluated tags. Non-quiet resolve prints a compact reasoning table showing which candidates were manifest-checked, deferred by tag-page size, or skipped by the lookup cap. Default search mode (`search`, `search --no-fit`, or `search --fast`) performs only the library search request and prints unannotated results, so it avoids the N-plus-one network cost of tag and manifest annotation. Use `search --fit` to request hardware-aware annotation.

## Wrapper integration

Install `bin/ollama` earlier in `PATH` than the real Ollama binary. It adds:

```bash
ollama search qwen
ollama resolve 'qwen2.5-coder?'
ollama resolver-info
ollama launch codex 'qwen2.5-coder?'
```

For launch subcommands, the wrapper resolves only one model argument: `--model VALUE`, `--model=VALUE`, `-m VALUE`, or otherwise the first positional argument after `launch <tool>`. Other arguments that end in `?` are passed through unchanged, so prompts and option values do not accidentally trigger model resolution. The smoke harness at `tests/wrapper_smoke.sh` checks quoted-`?` resolution, exact-tag bypass, single-model resolution, prompt-like `?` pass-through, resolver dispatch, and fallback to the real Ollama binary.

## Flox

Development shell:

```bash
flox activate
cargo test --locked
bash tests/wrapper_smoke.sh
```

Package build:

```bash
flox build ollama-model-resolver
```

The Nix expression lives at `.flox/pkgs/ollama-model-resolver/default.nix` and builds from the repository root.


Registry manifests remain the source of truth for exact weight and total layer sizes. Missing tag manifests may be skipped, but registry/network/parse failures stop resolution with the affected model tag so the resolver does not silently choose a smaller candidate from incomplete authoritative data. If every checked manifest is missing, the final error includes a compact list of the candidate tags tried and any candidates skipped by the manifest-lookup cap.

### Wrapper model argument rule

The wrapper resolves a model reference ending in `?` only in conservative locations:

- the first argument after `ollama launch <tool>`, for example `ollama launch codex 'qwen?'`;
- `--model VALUE`;
- `--model=VALUE`;
- `-m VALUE`.

If a launch option appears before the model, pass the model with `--model` or `-m`. The wrapper does not infer a later positional model after options because those later values may be prompts, option values, paths, or other tool-specific arguments.

### Terminal/log safety

The resolver treats Ollama website content, registry responses, local Ollama-compatible endpoints, and pull-stream events as untrusted for display. Before writing those values to stdout/stderr, it replaces terminal control characters with visible placeholders. Raw pull response bodies and invalid pull-stream lines are capped before they appear in diagnostics. Quiet mode still reserves stdout for the final resolved `model:tag`.

### Remote Ollama endpoint safety

`--ollama-host` is guarded because `resolve` can issue `POST /api/pull`. Non-loopback hosts require `--allow-remote-ollama`; otherwise the command fails before contacting the Ollama API. This keeps default wrapper and automation behavior local-first while preserving an explicit escape hatch for power users who intentionally operate a remote Ollama-compatible endpoint.
