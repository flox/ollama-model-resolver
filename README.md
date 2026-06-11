# ollama-model-resolver

`ollama-model-resolver` is a Rust CLI for selecting and pulling an Ollama model variant that is likely to fit the detected machine. It runs on Linux (NVIDIA VRAM) and macOS (Apple Silicon unified memory). It searches the Ollama library, reads tag manifests from the Ollama registry, estimates runtime memory from model weights plus a configurable margin, checks disk space, and pulls the selected `model:tag` through the local Ollama API.

## Commands

```bash
ollama-model-resolver search qwen
ollama-model-resolver search qwen --fit
ollama-model-resolver search qwen --fit --all
ollama-model-resolver search qwen --wide
ollama-model-resolver search qwen --fit --wide
ollama-model-resolver search qwen --quick
ollama-model-resolver search qwen --macos
ollama-model-resolver resolve 'qwen2.5-coder?'
ollama-model-resolver resolve 'qwen?' --select 2
ollama-model-resolver resolve 'qwen?' --first
ollama-model-resolver resolve 'qwen?' --yes
ollama-model-resolver resolve qwen2.5-coder:14b
ollama-model-resolver info
ollama-model-resolver --ollama-host http://remote.example:11434 --allow-remote-ollama info
```

A trailing `?` asks the resolver to search and choose a variant. Quote model arguments that contain `?` when you can, for example `'qwen2.5-coder?'`, so the shell never treats `?` as a glob. Without `?`, the resolver treats the argument as an exact Ollama model reference and pulls it unchanged.

When search finds no exact model name in normal interactive mode, the resolver presents the top matches and prompts for a selection. In non-interactive normal mode, it fails with remediation instead of attempting an unusable prompt. Use `--select <N>` for a deterministic 1-based candidate choice, `--first` to accept the top search result, or `--fail-on-ambiguous` to make normal mode fail immediately on non-exact search matches. In `--quiet` mode, non-exact matches fail so shell substitution never receives an unapproved model name.

### Search display modes

Default search (`search qwen`) resolves every matching model to an exact, pullable `model:tag` with its exact size (from the registry manifest) and a fit verdict against the detected hardware, then shows them all — fitting and not — hiding only cloud-only models. This is the same data `--fit` produces; the default simply does not filter by fit, so you can copy an exact `model:tag` to pull instead of trusting `resolve 'model?'` to pick one. It costs a manifest lookup per model plus hardware detection, so it takes a few seconds; use `--quick` for a fast approximate browse (below).

Across all search modes, results are re-ranked by name relevance before display. ollama.com orders matches by popularity, which can bury name matches under popular unrelated models — a search for `glm` otherwise surfaces `gemma4` first. The resolver promotes models whose name (or a `-`/`_`/`.`/`:`-separated name token) matches the query, and keeps ollama.com's popularity order within a relevance tier. Matching is token-based, not bare substring, so `emma` does not match `gemma4`. The trade-off: a family name buried mid-word (for example `codellama` for the query `llama`) counts as unrelated and ranks with the popularity padding. A footer note reminds you that the matches and their order originate from ollama.com, not this tool. Re-ranking only reorders what ollama.com returns (a fixed ~20 results, no pagination); it cannot surface a model ollama.com does not return.

`--fit` keeps only models that fit the detected hardware — it hides the ones the default view marks `✗ does not fit` (`DoesNotFit`, `InsufficientDisk`). `--all` shows everything, including cloud-only and platform-restricted models that are otherwise hidden. `--split` is honored: models that fit with a VRAM/RAM split pass `.fits()` and remain visible.

macOS-only models — variants the Ollama registry gates to macOS (currently `nvfp4` quantizations; the registry returns HTTP 412 "this model requires macOS" for them) — are identified by **tag name**, since their manifests are gated and cannot be sized. A model that *offers* such a variant gets an extra `macOS-only` row showing its largest such variant as the representative. A model with both runnable and macOS-only tags therefore appears twice — once for its best fitting tag, once for its macOS-only variant (e.g. `qwen3.5:9b` and `qwen3.5:35b-a3b-nvfp4`). The weight column shows `—` (unsizable) and a note explains the local Ollama daemon checks fit at pull. On macOS these rows appear in the default view, after the runnable results, with reserved slots so they are not crowded out. On Linux they cannot run, so they are hidden from the default view — reach them with `--all` (everything) or `--macos`. Only name-relevant models contribute macOS-only rows, so an unrelated popular model that happens to ship an `nvfp4` tag is not surfaced.

With `--macos`, search lists **only** models that offer a macOS-only variant, showing that variant. Detection is by tag name, not the local platform, so it works on any host OS and doubles as a way to discover macOS-only models from Linux. Name-irrelevant results are dropped first (a query like `glm` won't surface an unrelated model just because it has an `nvfp4` tag). `--macos` implies hardware annotation and conflicts with `--all` and `--no-fit`.

With `--wide`, search uses the full tabular view (bordered table) instead of the compact one-line layout. Works with or without `--fit`.

`--quick` (aliases `--fast`, `--no-fit`) is the basic browse: it lists matches with an approximate download-size range from each model's tag page — no manifest lookups, no fit checking — so it is faster than the default, but shows base model names rather than exact `model:tag`, and approximate rather than exact sizes. Models whose tag page exposes no size show `—`. Conflicts with `--fit`, `--all`, and `--macos`.

### Resolve confirmation

When `resolve` selects a variant that does not fit the detected hardware, it shows a warning and prompts for confirmation. Pass `--yes` to skip the prompt and proceed with the pull regardless. In `--quiet` mode, non-fitting variants always fail (no prompt is shown).

## Resolution policy

The resolver ranks variants by these rules:

1. Prefer non-base / instruct-style tags when present.
2. Prefer bare default tags and `Q4_K_M` tags when present.
3. Try larger parameter counts before smaller ones.
4. Use tag-page size hints to defer candidates that appear too large before manifest lookups.
5. Fetch registry manifests for plausible or unknown-size candidates first for final size math.
6. If none of those fit, fetch deferred candidates in ranked order so a pessimistic page-size hint cannot hide a fitting model.
7. Return the first fitting candidate; otherwise report the smallest evaluated non-fitting candidate for the warning / try-anyway prompt unless no manifest can be read.

Use `--max-manifest-lookups <N>` to cap manifest lookups per model resolution. For `resolve` the default is uncapped, so correctness is favored over speed. Annotated `search` (the default view and `--fit`/`--all`) instead applies a default cap of 5 lookups per model, since browsing a page of results tolerates an approximate verdict and shouldn't probe every tag of a model that doesn't fit; pass `--max-manifest-lookups <N>` to override it. The `--quick` browse (aliases `--fast`/`--no-fit`) does no manifest lookups at all.

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
- On Apple Silicon (`macOS`), the fit basis is the unified memory pool: VRAM and system RAM are one physical pool, so `--split` does not apply and is a no-op (the tool prints a note). Fit is checked against memory available right now, so the verdict tracks current system load.
- RAM detection uses `/proc/meminfo` on Linux. On macOS it uses `hw.memsize` for the total and `host_statistics64` (total minus wired and compressed pages) for available memory, mirroring the system's own "memory free" notion.
- Disk checks use `statvfs` against `$OLLAMA_MODELS` or `~/.ollama/models`.

## Scraping and registry access

Ollama search and tag discovery currently parse `ollama.com` HTML. The parsing code lives behind the registry interface in `src/registry.rs` and includes snapshot-style parser tests for search cards, fallback links, tag links, invalid links, and manifest-size extraction. Manifest lookups and per-model tag lists are cached during a resolver run. Approximate tag-page sizes reduce registry calls by checking plausible candidates first, while manifests still provide final sizing for selected/evaluated tags. Non-quiet resolve prints a compact reasoning table showing which candidates were manifest-checked, deferred by tag-page size, or skipped by the lookup cap. `search --quick` (aliases `--fast` / `--no-fit`) fetches each model's tag page for an approximate size range (N-plus-one, no manifest lookups). Plain `search` and `search --fit` resolve each match against registry manifests for exact sizes and a fit verdict (a manifest lookup per model); `--fit` additionally hides non-fitting models.

## Wrapper integration

Install `bin/ollama` earlier in `PATH` than the real Ollama binary. It adds:

```bash
ollama search qwen
ollama resolve 'qwen2.5-coder?'
ollama resolver-info
ollama launch codex 'qwen2.5-coder?'
```

For launch subcommands, the wrapper resolves only one model argument: `--model VALUE`, `--model=VALUE`, `-m VALUE`, or otherwise the first positional argument after `launch <tool>`. Other arguments that end in `?` are passed through unchanged, so prompts and option values do not accidentally trigger model resolution. The smoke harness at `tests/wrapper_smoke.sh` checks quoted-`?` resolution, exact-tag bypass, single-model resolution, prompt-like `?` pass-through, resolver dispatch, and fallback to the real Ollama binary.

## Development and packaging

A Nix flake provides the dev shell — the Rust toolchain plus the macOS link dependencies (libiconv; the default stdenv supplies clang and the Apple SDK):

```bash
nix develop                 # enter the dev shell
cargo build
cargo test
cargo run -- info
bash tests/wrapper_smoke.sh
```

Or run one-offs without entering the shell: `nix develop -c cargo test`.

Package build via Flox (Linux and macOS):

```bash
flox build ollama-model-resolver
```

The Nix package expression lives at `.flox/pkgs/ollama-model-resolver/default.nix` and builds from the repository root.


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
