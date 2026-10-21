#!/usr/bin/env bash
# Detached long OASIS run in its own folder (output, log and pid together).
#   experiments/oasis_social/launch_long.sh <agents> <days> [model] [extra oasis_social.run args...]
# e.g. experiments/oasis_social/launch_long.sh 50 300 gpt-6-astra --max-cost 3000
# STYLE=monitored experiments/oasis_social/launch_long.sh ...  labels the run (default "oasis")
# Output: experiments/oasis_social/runs/<name>/{oasis.db, run.log, earnings.jsonl, agent_logs/, ...}
# View:   python3 experiments/oasis_social/viz.py <run dir>   -> <run dir>/viz.html
set -euo pipefail
cd "$(dirname "$0")/../../environments"
K="${1:-50}"; N="${2:-300}"; MODEL="${3:-gpt-6-astra}"; EXTRA=("${@:4}")
NAME="${STYLE:-oasis}-${MODEL}_k${K}n${N}_$(TZ=America/Los_Angeles date +%Y-%m-%d__%H-%M-%S)"
OUT=../experiments/oasis_social/runs/$NAME
mkdir -p "$OUT"
nohup setsid oasis_social/.venv/bin/python -m oasis_social.run --agents "$K" --days "$N" --model "$MODEL" \
  --out "$OUT" "${EXTRA[@]}" > "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "started pid $(cat "$OUT/run.pid")"
echo "out: $(cd "$OUT" && pwd)"
