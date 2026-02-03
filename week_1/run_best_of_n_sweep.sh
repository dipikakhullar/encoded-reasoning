#!/usr/bin/env bash
# Run best-of-n evaluation for n=2,4,8,16
# Uses alpha=1.5 model, 5 samples, saves to arithmetic_problems_bon/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEMS_FILE="${SCRIPT_DIR}/arithmetic_problems.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/arithmetic_problems_bon"
MAX_SAMPLES=5
N_VALUES=(2 4 8 16)

mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Running best-of-n sweep: n in ${N_VALUES[*]}, ${MAX_SAMPLES} samples each"
echo ""

for n in "${N_VALUES[@]}"; do
    output_file="${OUTPUT_DIR}/eval_results_bon_n${n}.json"
    echo "=== best-of-${n} -> ${output_file} ==="
    python "${SCRIPT_DIR}/evaluate_best_of_n.py" \
        --n "${n}" \
        --max-problems "${MAX_SAMPLES}" \
        --problems-file "${PROBLEMS_FILE}" \
        --output-dir "${OUTPUT_DIR}"
    echo ""
done

echo "Sweep complete. Results in ${OUTPUT_DIR}"
