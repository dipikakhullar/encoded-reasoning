#!/usr/bin/env bash
# Detached long negotiation run with progress log + periodic checkpoints.
#   experiments/negotiation/launch_long.sh <agents> <ticks> [model] [seed] [world] [parallel]
# world: a name under environments/negotiation/worlds/ (baseline_v1, free_report, ...), a
# YAML/JSON path, or an old run's config.json. Omit for Config defaults.
# Output: experiments/negotiation/runs/<slug>/<run_id>/, logs in experiments/negotiation/runs/logs/.
set -euo pipefail
cd "$(dirname "$0")/../../environments"
LOGS=../experiments/negotiation/runs/logs
mkdir -p "$LOGS"
K="${1:-50}"; N="${2:-1000}"; MODEL="${3:-gpt-6-astra}"; SEED="${4:-100}"; WORLD="${5:-}"; PAR="${6:-25}"
TAG="k${K}n${N}${WORLD:+_$(basename "${WORLD%.*}")}"
LOG="$LOGS/${TAG}_${MODEL}_s${SEED}.log"
WORLD_ARGS=(); [[ -n "$WORLD" ]] && WORLD_ARGS=(--world "$WORLD")
nohup setsid .venv/bin/python -m negotiation.run "${WORLD_ARGS[@]}" --agents "$K" --ticks "$N" --policy llm --model "$MODEL" \
  --recommender engagement --seed "$SEED" --parallel "$PAR" --tag "$TAG" --checkpoint-every 25 \
  > "$LOG" 2>&1 &
echo $! > "$LOGS/${TAG}_${MODEL}_s${SEED}.pid"
echo "started pid $(cat "$LOGS/${TAG}_${MODEL}_s${SEED}.pid")"
echo "log: $LOG"
