#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: test_main_params.sh [options] [-- extra-args]" \
    "" \
    "Options:" \
    "  --parallel              Run all generated commands concurrently (default sequential)." \
    "  --sequential            Force sequential execution." \
    "  --max-parallel <n>      Maximum concurrent jobs when in parallel mode (default unlimited)." \
    "  --dry-run               Print commands instead of running them." \
    "  --python-bin <path>     Python interpreter to use (default: python)." \
    "  --entry <path>          Python entry point (default: main.py)." \
    "  --log-dir <dir>         Directory for log files (default: ./logs)." \
    "  --help                  Show this message." \
    "" \
    "Only --config and --log-dir have defaults; supply every other --flag VALUE" \
    "pair (e.g. --ts 200000). Provide comma-separated VALUEs to fan out" \
    "combinations (e.g. --nsteps \"128,256\")."
}

die() {
  echo "Error: $*" >&2
  exit 1
}

trim() {
  local var="$*"
  var="${var#"${var%%[![:space:]]*}"}"
  var="${var%"${var##*[![:space:]]}"}"
  printf '%s' "$var"
}

sanitize_value() {
  local value="$1"
  value="${value// /}"
  value="${value//\//-}"
  value="${value//[^A-Za-z0-9._-]/}"
  printf '%s' "$value"
}

declare -A value_lists=()
declare -a key_order=(config ts nsteps)
declare -A pid_logs=()

ensure_key_order() {
  local key="$1"
  for existing in "${key_order[@]}"; do
    if [[ "$existing" == "$key" ]]; then
      return
    fi
  done
  key_order+=("$key")
}

set_value_list() {
  local key="$1"
  local raw="$2"
  local -a cleaned=()
  IFS=',' read -ra parts < <(printf '%s\n' "$raw")
  for part in "${parts[@]}"; do
    local trimmed
    trimmed="$(trim "$part")"
    [[ -n "$trimmed" ]] && cleaned+=("$trimmed")
  done
  if [[ ${#cleaned[@]} -eq 0 ]]; then
    die "No values supplied for --$key"
  fi
  value_lists["$key"]="$(printf '%s\n' "${cleaned[@]}")"
  ensure_key_order "$key"
}

ensure_value_list() {
  local key="$1"
  local default_value="$2"
  if [[ -z "${value_lists[$key]:-}" ]]; then
    set_value_list "$key" "$default_value"
  fi
}

require_value_list() {
  local key="$1"
  local message="$2"
  if [[ -z "${value_lists[$key]:-}" ]]; then
    die "$message"
  fi
}

enforce_parallel_limit() {
  if [[ "$MAX_PARALLEL" -le 0 ]]; then
    return
  fi
  while [[ ${#active_pids[@]} -gt "$MAX_PARALLEL" ]]; do
    local oldest="${active_pids[0]}"
    wait_for_pid "$oldest"
    active_pids=("${active_pids[@]:1}")
  done
}

wait_for_pid() {
  local pid="$1"
  [[ -z "$pid" ]] && return
  local log_file="${pid_logs[$pid]:-unknown}"
  if ! wait "$pid"; then
    echo "Error: command failed (PID $pid). Check log: $log_file" >&2
    exit 1
  fi
  unset pid_logs["$pid"]
}

DEFAULT_CONFIG="experiments/ceb_conf/ceb_index_count_penv_dft.json"
LOG_DIR="./logs"
RUN_MODE="sequential"
DRY_RUN=false
PYTHON_BIN="python"
ENTRYPOINT="main.py"
EXTRA_ARGS=()
MAX_PARALLEL=0
active_pids=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      RUN_MODE="parallel"
      shift
      ;;
    --sequential)
      RUN_MODE="sequential"
      shift
      ;;
    --max-parallel)
      [[ $# -lt 2 ]] && die "Missing value for --max-parallel"
      [[ "$2" =~ ^[0-9]+$ ]] || die "--max-parallel requires a non-negative integer"
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --python-bin)
      [[ $# -lt 2 ]] && die "Missing value for --python-bin"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --entry)
      [[ $# -lt 2 ]] && die "Missing value for --entry"
      ENTRYPOINT="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -lt 2 ]] && die "Missing value for --log-dir"
      LOG_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    --*)
      key="${1#--}"
      [[ $# -lt 2 ]] && die "Missing value for $1"
      set_value_list "$key" "$2"
      shift 2
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

ensure_value_list config "$DEFAULT_CONFIG"
require_value_list ts "--ts must be supplied (comma-separated values allowed)"

declare -a combos=("")
for key in "${key_order[@]}"; do
  value_data="${value_lists[$key]-}"
  [[ -z "$value_data" ]] && continue
  readarray -t values < <(printf '%s\n' "$value_data")
  [[ ${#values[@]} -eq 0 ]] && continue
  new_combos=()
  for combo in "${combos[@]}"; do
    for value in "${values[@]}"; do
      value="${value//$'\r'/}"
      if [[ -z "$combo" ]]; then
        new_combos+=("${key}=${value}")
      else
        new_combos+=("${combo}|${key}=${value}")
      fi
    done
  done
  combos=("${new_combos[@]}")
done

if [[ ${#combos[@]} -eq 0 ]]; then
  die "No command combinations generated"
fi

timestamp="$(date +%y%m%d%H%M%S)"

if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$LOG_DIR"
fi
for combo in "${combos[@]}"; do
  IFS='|' read -ra kv_pairs < <(printf '%s\n' "$combo")
  args=()
  labels=()
  for kv in "${kv_pairs[@]}"; do
    [[ -z "$kv" ]] && continue
    key="${kv%%=*}"
    value="${kv#*=}"
    args+=("--$key" "$value")
    labels+=("${key}$(sanitize_value "$value")")
  done
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    args+=("${EXTRA_ARGS[@]}")
  fi
  log_suffix="$(IFS=_; echo "${labels[*]}")"
  log_file="${LOG_DIR}/${timestamp}_${log_suffix}.log"

  if [[ "$DRY_RUN" == true ]]; then
    printf 'DRY RUN: nohup %q %q' "$PYTHON_BIN" "$ENTRYPOINT"
    for arg in "${args[@]}"; do
      printf ' %q' "$arg"
    done
    printf ' > %q 2>&1' "$log_file"
    if [[ "$RUN_MODE" == "parallel" ]]; then
      printf ' &'
    fi
    printf '\n'
    continue
  fi

  if [[ "$RUN_MODE" == "parallel" ]]; then
    nohup "$PYTHON_BIN" "$ENTRYPOINT" "${args[@]}" >"$log_file" 2>&1 &
    pid=$!
    active_pids+=("$pid")
    pid_logs["$pid"]="$log_file"
    echo "Launched PID $pid -> $log_file"
    enforce_parallel_limit
  else
    echo "Running -> $log_file"
    if ! nohup "$PYTHON_BIN" "$ENTRYPOINT" "${args[@]}" >"$log_file" 2>&1; then
      echo "Error: command failed. Check log: $log_file" >&2
      exit 1
    fi
  fi
done

if [[ "$RUN_MODE" == "parallel" && "$DRY_RUN" == false && ${#active_pids[@]} -gt 0 ]]; then
  for pid in "${active_pids[@]}"; do
    wait_for_pid "$pid"
  done
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "Total commands prepared: ${#combos[@]}"
else
  echo "Total commands started: ${#combos[@]}"
fi
