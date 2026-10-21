#!/usr/bin/env bash
# Detached long social run with progress log + periodic checkpoints.
# parallel defaults to the agent count: every agent of a day runs at once.
#   experiments/social/launch_long.sh <agents> <ticks> [model] [seed] [world] [parallel] [extra social.run args...]
# world: a name under environments/social/worlds/ (baseline_v1, free_report, ...), a
# YAML/JSON path, or an old run's config.json. Omit for Config defaults.
# Output: experiments/social/runs/<slug>/<run_id>/, logs in experiments/social/runs/logs/.
set -euo pipefail
cd "$(dirname "$0")/../../environments"
LOGS=../experiments/social/runs/logs
mkdir -p "$LOGS"
K="${1:-50}"; N="${2:-1000}"; MODEL="${3:-gpt-6-astra}"; SEED="${4:-100}"; WORLD="${5:-}"; PAR="${6:-$K}"
TAG="k${K}n${N}${WORLD:+_$(basename "${WORLD%.*}")}"
LOG="$LOGS/${TAG}_${MODEL}_s${SEED}.log"
WORLD_ARGS=(); [[ -n "$WORLD" ]] && WORLD_ARGS=(--world "$WORLD")
EXTRA=("${@:7}")   # e.g. --set prune_every=6
nohup setsid .venv/bin/python -m social.run "${WORLD_ARGS[@]}" --agents "$K" --ticks "$N" --policy llm --model "$MODEL" \
  --recommender engagement --seed "$SEED" --parallel "$PAR" --tag "$TAG" --checkpoint-every 25 "${EXTRA[@]}" \
  > "$LOG" 2>&1 &
echo $! > "$LOGS/${TAG}_${MODEL}_s${SEED}.pid"
echo "started pid $(cat "$LOGS/${TAG}_${MODEL}_s${SEED}.pid")"
echo "log: $LOG"
