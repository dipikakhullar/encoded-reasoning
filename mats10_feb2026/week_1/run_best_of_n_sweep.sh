#!/usr/bin/env bash
# 1) Collect 100 rollouts per sample (construct_rollouts.py)
# 2) Run best-of-n evaluation with eligibility metrics (evaluate_best_of_n.py)
# Saves rollouts to rollouts/, results to arithmetic_problems_bon/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEMS_FILE="${SCRIPT_DIR}/arithmetic_problems.jsonl"
ROLLOUTS_PER_PROBLEM=100
MAX_SAMPLES=5
ROLLOUTS_DIR="${SCRIPT_DIR}/rollouts"
ROLLOUTS_FILE="${ROLLOUTS_DIR}/rollouts_alpha1.5_n${ROLLOUTS_PER_PROBLEM}.json"
OUTPUT_DIR="${SCRIPT_DIR}/arithmetic_problems_bon"
N_VALUES=(2 4 8 16)

echo "=== Step 1: Construct rollouts (${ROLLOUTS_PER_PROBLEM} per problem, ${MAX_SAMPLES} problems) ==="
if [[ ! -f "${ROLLOUTS_FILE}" ]]; then
    python "${SCRIPT_DIR}/construct_rollouts.py" \
        --rollouts "${ROLLOUTS_PER_PROBLEM}" \
        --max-problems "${MAX_SAMPLES}" \
        --problems-file "${PROBLEMS_FILE}" \
        --output "${ROLLOUTS_FILE}"
else
    echo "Rollouts file exists: ${ROLLOUTS_FILE} (skip construct_rollouts.py)"
fi
echo ""

echo "=== Step 2: Evaluate best-of-n with eligibility metrics ==="
echo "n values: ${N_VALUES[*]}"
python "${SCRIPT_DIR}/evaluate_best_of_n.py" \
    --rollouts "${ROLLOUTS_FILE}" \
    --n "${N_VALUES[@]}" \
    --output-dir "${OUTPUT_DIR}"
echo ""

echo "Sweep complete. Rollouts: ${ROLLOUTS_DIR}, Results: ${OUTPUT_DIR}"
