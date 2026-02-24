#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELS=(
  "gpt-5.1-codex-max"
  "gpt-5.1-codex-mini"
  "gpt-5.2"
  "gpt-5.2-codex"
  "gpt-5.3-codex"
  "gpt-5.3-codex-spark"
)

REASONING_ARG='model_reasoning_effort="high"'
WATCHDOG_STALE_S="${WATCHDOG_STALE_S:-480}"
WATCHDOG_POLL_S="${WATCHDOG_POLL_S:-20}"
MAX_RESTARTS="${MAX_RESTARTS:-50}"
ONLY_MODEL="${ONLY_MODEL:-}"
if [[ -n "$ONLY_MODEL" ]]; then
  MODELS=("$ONLY_MODEL")
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/artifacts/logs/codex_easy_v1_high_${STAMP}"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo "Watchdog: stale=${WATCHDOG_STALE_S}s poll=${WATCHDOG_POLL_S}s max_restarts=${MAX_RESTARTS}"

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

state_snapshot() {
  local state_file="$1"
  if [[ ! -f "$state_file" ]]; then
    return 1
  fi
  jq -r '[(.completed_episode_ids | length), .total_jobs, .next_uncommitted_episode_id, .last_updated] | @tsv' "$state_file" 2>/dev/null
}

is_run_complete() {
  local out_dir="$1"
  local state_file="$out_dir/execution_state.json"
  if [[ ! -f "$state_file" ]]; then
    return 1
  fi
  local completed total
  completed="$(jq -r '.completed_episode_ids | length' "$state_file" 2>/dev/null || true)"
  total="$(jq -r '.total_jobs' "$state_file" 2>/dev/null || true)"
  if [[ -z "$completed" || -z "$total" ]]; then
    return 1
  fi
  if [[ "$completed" == "null" || "$total" == "null" ]]; then
    return 1
  fi
  (( completed == total ))
}

run_one() {
  local model="$1"
  local game="$2"
  local slug="$3"
  local run_id="$4"
  local out_dir="$5"
  local log_file="$6"

  if is_run_complete "$out_dir"; then
    echo "[$(timestamp_utc)] SKIP model=${model} game=${game} run_id=${run_id} reason=already_complete" | tee -a "$log_file"
    return 0
  fi

  local attempt=1

  while true; do
    local last_snapshot=""
    local last_progress_epoch
    last_progress_epoch="$(date +%s)"

    local cmd=(
      uv run games-bench run
      --provider codex
      --model "$model"
      --suite easy-v1
      --game "$game"
      --run-id "$run_id"
      --parallelism 1
      --max-inflight-provider 1
      --progress
      --progress-refresh-s 5
      --codex-app-arg=-c
      --codex-app-arg="$REASONING_ARG"
    )

    local mode="fresh"
    if [[ -d "$out_dir" ]]; then
      cmd+=(--resume --strict-resume)
      mode="resume"
    fi

    echo "[$(timestamp_utc)] START model=${model} game=${game} mode=${mode} attempt=${attempt} run_id=${run_id}" | tee -a "$log_file"
    "${cmd[@]}" >>"$log_file" 2>&1 &
    local pid="$!"
    local killed_for_stall=0

    while kill -0 "$pid" 2>/dev/null; do
      sleep "$WATCHDOG_POLL_S"

      local now_epoch
      now_epoch="$(date +%s)"
      local snapshot
      if snapshot="$(state_snapshot "$out_dir/execution_state.json")"; then
        if [[ "$snapshot" != "$last_snapshot" ]]; then
          last_snapshot="$snapshot"
          last_progress_epoch="$now_epoch"
          local completed total next_id updated
          IFS=$'\t' read -r completed total next_id updated <<<"$snapshot"
          echo "[$(timestamp_utc)] PROGRESS model=${model} game=${game} attempt=${attempt} completed=${completed}/${total} next=${next_id} updated=${updated}" | tee -a "$log_file"
        fi
      fi

      if (( now_epoch - last_progress_epoch > WATCHDOG_STALE_S )); then
        echo "[$(timestamp_utc)] WATCHDOG model=${model} game=${game} attempt=${attempt} stale_for=$((now_epoch - last_progress_epoch))s action=kill pid=${pid}" | tee -a "$log_file"
        killed_for_stall=1
        kill "$pid" 2>/dev/null || true
        sleep 5
        if kill -0 "$pid" 2>/dev/null; then
          kill -9 "$pid" 2>/dev/null || true
        fi
        break
      fi
    done

    local rc=0
    if ! wait "$pid"; then
      rc="$?"
    fi

    if [[ "$rc" -eq 0 ]] && is_run_complete "$out_dir"; then
      echo "[$(timestamp_utc)] DONE model=${model} game=${game} run_id=${run_id} status=complete" | tee -a "$log_file"
      return 0
    fi

    if [[ "$rc" -eq 0 ]]; then
      local completed total
      completed="$(jq -r '.completed_episode_ids | length' "$out_dir/execution_state.json" 2>/dev/null || echo "?")"
      total="$(jq -r '.total_jobs' "$out_dir/execution_state.json" 2>/dev/null || echo "?")"
      echo "[$(timestamp_utc)] INCOMPLETE model=${model} game=${game} run_id=${run_id} completed=${completed}/${total} rc=0 action=retry" | tee -a "$log_file"
    fi

    if (( attempt >= MAX_RESTARTS )); then
      echo "[$(timestamp_utc)] FAIL model=${model} game=${game} run_id=${run_id} rc=${rc} attempts=${attempt}" | tee -a "$log_file"
      return "$rc"
    fi

    if [[ "$killed_for_stall" -eq 1 ]]; then
      echo "[$(timestamp_utc)] RETRY model=${model} game=${game} reason=watchdog_stall next_attempt=$((attempt + 1))" | tee -a "$log_file"
    else
      echo "[$(timestamp_utc)] RETRY model=${model} game=${game} reason=nonzero_exit rc=${rc} next_attempt=$((attempt + 1))" | tee -a "$log_file"
    fi
    attempt=$((attempt + 1))
    sleep 2
  done
}

for model in "${MODELS[@]}"; do
  slug="${model//\//_}"
  slug="${slug//:/_}"
  run_id="easy_v1_canonical_codex_high_20260224_codex_${slug}"

  for game in hanoi sokoban; do
    out_dir="$ROOT_DIR/artifacts/runs/${game}/codex/${slug}/${run_id}"
    log_file="$LOG_DIR/${game}__${slug}.log"
    run_one "$model" "$game" "$slug" "$run_id" "$out_dir" "$log_file"
  done
done

echo "[$(timestamp_utc)] ALL_DONE"
