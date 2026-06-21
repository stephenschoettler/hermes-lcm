#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="smoke"
OUTPUT_DIR=""
KEEP_GOING=0
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: scripts/validate_release.sh [--full] [--output DIR] [--keep-going]

Runs offline local release-validation gates and writes a reusable validation
checklist plus command logs under a fresh output directory.

Options:
  --full        Run full pytest, low-fd pytest, and release stress tier.
  --output DIR  Write artifacts to DIR. DIR must not already exist.
  --keep-going  Run all gates and report failures instead of stopping at first failure.
  -h, --help    Show this help.

Environment:
  PYTHON=/path/to/python  Python interpreter to use for all Python gates.
                          It must have pytest available.
EOF
}

fail_prereq() {
  cat >&2 <<EOF
Release validation prerequisite failed: $1

Using Python: $PYTHON_BIN

Fix:
  - activate a Python environment with pytest installed, or
  - run with PYTHON=/path/to/python $0 [--full]

Examples:
  python -m pip install pytest
  PYTHON=/path/to/venv/bin/python scripts/validate_release.sh --full
EOF
  exit 2
}

require_python_module() {
  local module="$1"
  local purpose="$2"
  if ! "$PYTHON_BIN" - "$module" <<'PY' >/dev/null 2>&1
import importlib.util
import sys
module = sys.argv[1]
raise SystemExit(0 if importlib.util.find_spec(module) is not None else 1)
PY
  then
    fail_prereq "Python module '$module' is required for $purpose."
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      MODE="full"
      shift
      ;;
    --output)
      if [[ $# -lt 2 ]]; then
        echo "--output requires a directory" >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail_prereq "Python interpreter not found."
fi
require_python_module "pytest" "pytest validation gates"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/tmp/hermes-lcm-release-validation-$(date -u +%Y%m%d-%H%M%S)"
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse existing output directory: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CHECKLIST="$OUTPUT_DIR/validation-checklist.md"
FAILURES=()

export PYTHONPYCACHEPREFIX="$OUTPUT_DIR/pycache"
if [[ -n "${PYTEST_ADDOPTS:-}" ]]; then
  export PYTEST_ADDOPTS="-p no:cacheprovider $PYTEST_ADDOPTS"
else
  export PYTEST_ADDOPTS="-p no:cacheprovider"
fi

cd "$REPO_ROOT"

branch="$(git branch --show-current 2>/dev/null || true)"
commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
dirty_start="$(git status --short 2>/dev/null || true)"
DIFF_CHECK_RANGE="${LCM_RELEASE_DIFF_BASE:-}"
if [[ -z "$DIFF_CHECK_RANGE" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git rev-parse --verify origin/main >/dev/null 2>&1 && ! git diff --quiet origin/main...HEAD -- .; then
    DIFF_CHECK_RANGE="origin/main...HEAD"
  elif git rev-parse --verify 'HEAD^' >/dev/null 2>&1; then
    DIFF_CHECK_RANGE="HEAD^...HEAD"
  fi
fi

cat > "$CHECKLIST" <<EOF
# hermes-lcm release validation

- generated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- mode: $MODE
- repo: ${branch:-detached}@${commit:-unknown}
- diff_check_range: ${DIFF_CHECK_RANGE:-working-tree}
- output_dir: $OUTPUT_DIR
- provider_network_side_effects: none expected
- live_profile_mutations: none expected

## Gates
EOF

run_gate() {
  local name="$1"
  shift
  local log_name
  log_name="$(printf '%s' "$name" | tr '[:upper:] /' '[:lower:]--' | tr -cd '[:alnum:]_.-')"
  local log_path="$OUTPUT_DIR/logs/${log_name}.log"

  local command_display
  printf -v command_display '%q ' "$@"
  command_display="${command_display% }"

  printf '==> %s\n' "$name"
  printf '\n### %s\n\nCommand:\n\n```bash\n%s\n```\n\n' "$name" "$command_display" >> "$CHECKLIST"

  set +e
  "$@" > "$log_path" 2>&1
  local status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    printf -- '- [x] %s -> pass (`%s`)\n' "$name" "$log_path" >> "$CHECKLIST"
  else
    printf -- '- [ ] %s -> fail exit %s (`%s`)\n' "$name" "$status" "$log_path" >> "$CHECKLIST"
    FAILURES+=("$name exit $status; log=$log_path")
    if [[ "$KEEP_GOING" -ne 1 ]]; then
      printf '\n## Result\n\nFAILED: %s (exit %s)\n' "$name" "$status" >> "$CHECKLIST"
      echo "FAILED: $name (exit $status); see $log_path" >&2
      echo "Checklist: $CHECKLIST" >&2
      exit "$status"
    fi
  fi
}

run_pytest() {
  "$PYTHON_BIN" - "$@" <<'PY'
import sys
from benchmarking.standalone import ensure_agent_context_engine_importable
ensure_agent_context_engine_importable()
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
PY
}

run_low_fd_pytest() {
  local current_limit
  current_limit="$(ulimit -n 2>/dev/null || true)"
  if [[ "$current_limit" == "unlimited" ]]; then
    ulimit -n 1024 2>/dev/null || echo "warning: could not lower file descriptor limit from unlimited to 1024" >&2
  elif [[ "$current_limit" =~ ^[0-9]+$ ]] && (( current_limit > 1024 )); then
    ulimit -n 1024 2>/dev/null || echo "warning: could not lower file descriptor limit from $current_limit to 1024" >&2
  fi
  run_pytest -q
}

if [[ -n "$DIFF_CHECK_RANGE" ]]; then
  run_gate "git diff check ($DIFF_CHECK_RANGE)" git diff --check "$DIFF_CHECK_RANGE"
  if [[ "$DIFF_CHECK_RANGE" != "HEAD" ]]; then
    run_gate "git working tree diff check" git diff --check
    run_gate "git staged diff check" git diff --cached --check
  fi
else
  run_gate "git diff check" git diff --check
fi
run_gate "python compileall" "$PYTHON_BIN" -m compileall -q .
run_gate "script py_compile" "$PYTHON_BIN" -m py_compile scripts/import_lossless_claw.py scripts/lcm_benchmark.py scripts/lcm_stress_check.py
run_gate "shell syntax" bash -n scripts/install.sh scripts/update.sh scripts/validate_release.sh
run_gate "focused pytest" run_pytest tests/test_lcm_core.py tests/test_lcm_command.py tests/test_packaging_install.py tests/test_benchmarking_cli.py tests/test_stress_release_check.py -q
run_gate "benchmark smoke" "$PYTHON_BIN" scripts/lcm_benchmark.py --synthetic-fixture release_validation_smoke:2:1:3 --policy benchmarks/policies/pressure_smoke.yaml --output "$OUTPUT_DIR/benchmark-smoke" --allow-external-output --json
run_gate "stress smoke" "$PYTHON_BIN" scripts/lcm_stress_check.py --output "$OUTPUT_DIR/stress-smoke" --tier smoke --json

if [[ "$MODE" == "full" ]]; then
  run_gate "pytest full" run_pytest -q
  run_gate "pytest low fd" run_low_fd_pytest
  run_gate "stress release" "$PYTHON_BIN" scripts/lcm_stress_check.py --output "$OUTPUT_DIR/stress-release" --tier release --json
fi

dirty_end="$(git status --short 2>/dev/null || true)"
if [[ "${dirty_end:-}" != "${dirty_start:-}" ]]; then
  FAILURES+=("validation changed git status; inspect start/end status in $CHECKLIST")
fi

cat >> "$CHECKLIST" <<'EOF'

## Doctor triage checklist

- [ ] safe/ignore warnings reviewed and intentionally left alone
- [ ] inspect warnings reviewed with source rows/session IDs before mutation
- [ ] backup-first cleanup warnings used preview commands before any apply command
- [ ] warning-only classes were not auto-cleaned: summary_quality, lifecycle_fragmentation, payload_storage suspicion, context_pressure

## Git status at validation start

```text
EOF
printf '%s\n' "${dirty_start:-clean}" >> "$CHECKLIST"
cat >> "$CHECKLIST" <<'EOF'
```

## Git status after validation

```text
EOF
printf '%s\n' "${dirty_end:-clean}" >> "$CHECKLIST"
cat >> "$CHECKLIST" <<'EOF'
```
EOF

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  {
    printf '\n## Result\n\nFAILED with %s gate failure(s):\n' "${#FAILURES[@]}"
    for failure in "${FAILURES[@]}"; do
      printf -- '- %s\n' "$failure"
    done
  } >> "$CHECKLIST"
  echo "Release validation failed; checklist: $CHECKLIST" >&2
  exit 1
fi

cat >> "$CHECKLIST" <<'EOF'

## Result

PASS: all selected release-validation gates passed.

Recommended next release-confidence step: review the generated checklist and scrubbed benchmark/stress artifacts, then run `scripts/validate_release.sh --full` before tagging if this was a smoke run.
EOF

printf 'PASS: release validation %s\nChecklist: %s\n' "$MODE" "$CHECKLIST"
