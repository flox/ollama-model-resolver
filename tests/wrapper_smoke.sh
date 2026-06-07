#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/wrapper" "$tmp/bin" "$tmp/real" "$tmp/log"
cp "$repo_root/bin/ollama" "$tmp/wrapper/ollama"
chmod +x "$tmp/wrapper/ollama"

cat > "$tmp/wrapper/launch-codex" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
printf 'launch-codex:%s\n' "$*" >> "$TEST_LOG"
LAUNCH
chmod +x "$tmp/wrapper/launch-codex"

cat > "$tmp/real/ollama" <<'REAL'
#!/usr/bin/env bash
set -euo pipefail
printf 'real-ollama:%s\n' "$*" >> "$TEST_LOG"
REAL
chmod +x "$tmp/real/ollama"

cat > "$tmp/bin/ollama-model-resolver" <<'RESOLVER'
#!/usr/bin/env bash
set -euo pipefail
printf 'resolver:%s\n' "$*" >> "$TEST_LOG"
case "$1" in
  resolve)
    if [[ "${2:-}" == "qwen?" && "${3:-}" == "--quiet" ]]; then
      printf 'qwen:7b\n'
    else
      printf 'unexpected resolver args: %s\n' "$*" >&2
      exit 2
    fi
    ;;
  search|info)
    exit 0
    ;;
  *)
    printf 'unexpected resolver command: %s\n' "$1" >&2
    exit 2
    ;;
esac
RESOLVER
chmod +x "$tmp/bin/ollama-model-resolver"

export PATH="$tmp/wrapper:$tmp/bin:$tmp/real:$PATH"
export TEST_LOG="$tmp/log/events"

assert_no_resolver() {
  if grep -q '^resolver:' "$TEST_LOG"; then
    printf 'resolver was called unexpectedly\n' >&2
    cat "$TEST_LOG" >&2
    exit 1
  fi
}

assert_resolver_count() {
  local expected="$1"
  local actual
  actual="$(grep -c '^resolver:' "$TEST_LOG" || true)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'expected %s resolver calls, got %s\n' "$expected" "$actual" >&2
    cat "$TEST_LOG" >&2
    exit 1
  fi
}

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex 'qwen?'
grep -qx 'resolver:resolve qwen? --quiet' "$TEST_LOG"
grep -qx 'launch-codex:qwen:7b' "$TEST_LOG"

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex qwen2.5-coder:14b
grep -qx 'launch-codex:qwen2.5-coder:14b' "$TEST_LOG"
assert_no_resolver

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex 'qwen?' 'what is this?'
grep -qx 'resolver:resolve qwen? --quiet' "$TEST_LOG"
grep -qx 'launch-codex:qwen:7b what is this?' "$TEST_LOG"
assert_resolver_count 1

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex qwen2.5-coder:14b 'what is this?'
grep -qx 'launch-codex:qwen2.5-coder:14b what is this?' "$TEST_LOG"
assert_no_resolver

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex --model 'qwen?' 'what is this?'
grep -qx 'resolver:resolve qwen? --quiet' "$TEST_LOG"
grep -qx 'launch-codex:--model qwen:7b what is this?' "$TEST_LOG"
assert_resolver_count 1

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex --model='qwen?' 'what is this?'
grep -qx 'resolver:resolve qwen? --quiet' "$TEST_LOG"
grep -qx 'launch-codex:--model=qwen:7b what is this?' "$TEST_LOG"
assert_resolver_count 1

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex -m 'qwen?' 'what is this?'
grep -qx 'resolver:resolve qwen? --quiet' "$TEST_LOG"
grep -qx 'launch-codex:-m qwen:7b what is this?' "$TEST_LOG"
assert_resolver_count 1

# P1 regression cases: once an option precedes the model, the wrapper cannot
# safely infer a positional model. It must not resolve option values, prompts,
# or later question-mark strings.
: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex --prompt 'what is this?' 'qwen?'
grep -qx 'launch-codex:--prompt what is this? qwen?' "$TEST_LOG"
assert_no_resolver

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex --prompt 'hello there' 'qwen?'
grep -qx 'launch-codex:--prompt hello there qwen?' "$TEST_LOG"
assert_no_resolver

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex --temperature 0.2 'qwen?'
grep -qx 'launch-codex:--temperature 0.2 qwen?' "$TEST_LOG"
assert_no_resolver

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex --foo value 'qwen?'
grep -qx 'launch-codex:--foo value qwen?' "$TEST_LOG"
assert_no_resolver

: > "$TEST_LOG"
"$tmp/wrapper/ollama" launch codex 'qwen?' --prompt 'what is this?'
grep -qx 'resolver:resolve qwen? --quiet' "$TEST_LOG"
grep -qx 'launch-codex:qwen:7b --prompt what is this?' "$TEST_LOG"
assert_resolver_count 1

: > "$TEST_LOG"
"$tmp/wrapper/ollama" search qwen
grep -qx 'resolver:search qwen' "$TEST_LOG"

: > "$TEST_LOG"
"$tmp/wrapper/ollama" list
grep -qx 'real-ollama:list' "$TEST_LOG"

printf 'wrapper smoke test passed\n'
