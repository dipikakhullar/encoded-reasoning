#!/usr/bin/env bash
# Run GEPA with multiple reflection prompts. Each prompt file in reflection_prompts/
# gets one run; results go to gepa_iterations/<dataset>_<prompt_name>_<datetime>/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_DIR="${SCRIPT_DIR}/reflection_prompts"
VENV_ACTIVATE="${SCRIPT_DIR}/../.venv/bin/activate"

if [[ ! -d "${PROMPTS_DIR}" ]]; then
    echo "No reflection_prompts/ directory. Create .txt files with <curr_instructions> and <inputs_outputs_feedback>."
    exit 1
fi

source "${VENV_ACTIVATE}"
for f in "${PROMPTS_DIR}"/*.txt; do
    [[ -f "$f" ]] || continue
    name=$(basename "$f" .txt)
    echo "=== Reflection prompt: ${name} ==="
    python "${SCRIPT_DIR}/run_gepa_arithmetic.py" \
        --reflection-prompt-file "$f" \
        --reflection-prompt-name "$name" \
        "$@"
    echo ""
done
echo "Sweep done."
