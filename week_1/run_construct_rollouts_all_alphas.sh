#!/usr/bin/env bash
# Collect rollouts for each alpha value. Each run loads the model, applies that alpha,
# and saves rollouts to rollouts/rollouts_alpha<alpha>_n<rollouts>.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBLEMS_FILE="${SCRIPT_DIR}/arithmetic_problems.jsonl"
ROLLOUTS_DIR="${SCRIPT_DIR}/rollouts"
ROLLOUTS_PER_PROBLEM=20
MAX_PROBLEMS=15
ALPHAS=(0.5 1 1.5 2 2.5 3 3.5)

mkdir -p "${ROLLOUTS_DIR}"
echo "Collecting rollouts for alphas: ${ALPHAS[*]}"
echo "  Rollouts per problem: ${ROLLOUTS_PER_PROBLEM}"
echo "  Problems: ${MAX_PROBLEMS}"
echo "  Output: ${ROLLOUTS_DIR}/rollouts_alpha<alpha>_n${ROLLOUTS_PER_PROBLEM}.json"
echo ""

for alpha in "${ALPHAS[@]}"; do
    out_file="${ROLLOUTS_DIR}/rollouts_alpha${alpha}_n${ROLLOUTS_PER_PROBLEM}.json"
    if [[ -f "${out_file}" ]]; then
        echo "=== alpha=${alpha}: skip (exists ${out_file}) ==="
    else
        echo "=== alpha=${alpha} -> ${out_file} ==="
        python "${SCRIPT_DIR}/construct_rollouts.py" \
            --alpha "${alpha}" \
            --rollouts "${ROLLOUTS_PER_PROBLEM}" \
            --max-problems "${MAX_PROBLEMS}" \
            --problems-file "${PROBLEMS_FILE}" \
            --output "${out_file}"
    fi
    echo ""
done

echo "Done. Rollouts in ${ROLLOUTS_DIR}"
