#!/usr/bin/env bash
# Run eval sweep over alpha values: 0.5, 1, 1.5, 2, 2.5, 3, 3.5
# Output: week_1/arithmetic_problems_<datetime>/eval_results_alpha_<alpha>.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEMS_FILE="${SCRIPT_DIR}/arithmetic_problems.jsonl"
MAX_SAMPLES=5
DATASET_NAME="arithmetic_problems"
DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/${DATASET_NAME}_${DATETIME}"
ALPHAS=(0.5 1 1.5 2 2.5 3 3.5)

mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Running eval sweep: alpha in ${ALPHAS[*]}, ${MAX_SAMPLES} samples each"
echo ""

for alpha in "${ALPHAS[@]}"; do
    output_file="${OUTPUT_DIR}/eval_results_alpha_${alpha}.json"
    echo "=== alpha=${alpha} -> ${output_file} ==="
    python "${SCRIPT_DIR}/evaluate_steering_vector.py" \
        --alpha "${alpha}" \
        --max-problems "${MAX_SAMPLES}" \
        --problems-file "${PROBLEMS_FILE}" \
        -o "${output_file}"
    echo ""
done

echo "Sweep complete. Results in ${OUTPUT_DIR}"
