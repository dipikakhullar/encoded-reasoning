#!/usr/bin/env bash
# =============================================================================
# CoT Prefill Experiment — Full pipeline
#
# Run on a fresh GPU server (RunPod / Vast.ai).
# Installs uv, sets up the environment, runs all models, and produces analysis.
#
# Usage:
#   bash testing_on_server/run_experiments.sh              # all models, 100 problems
#   bash testing_on_server/run_experiments.sh --sanity      # 5 problems, 1 model (quick test)
#   bash testing_on_server/run_experiments.sh --model qwen3-14b  # single model
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Defaults
MAX_PROBLEMS=100
MODEL="all"
SANITY=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sanity)
            SANITY=1
            MAX_PROBLEMS=5
            MODEL="qwen3-14b"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-problems)
            MAX_PROBLEMS="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 [--sanity] [--model MODEL] [--max-problems N]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "CoT Prefill Experiment"
echo "============================================"
echo "Model:        $MODEL"
echo "Max problems: $MAX_PROBLEMS"
echo "Project dir:  $PROJECT_DIR"
echo "============================================"

# ----- Step 1: Install uv if not present -----
if ! command -v uv &> /dev/null; then
    echo ""
    echo "[1/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    echo "uv installed: $(uv --version)"
else
    echo ""
    echo "[1/4] uv already installed: $(uv --version)"
fi

# ----- Step 2: Create venv and install dependencies -----
echo ""
echo "[2/4] Setting up Python environment..."

# Install project deps from pyproject.toml + extra packages we need
uv sync
uv pip install matplotlib autoawq

echo "Environment ready."
echo "Python: $(uv run python --version)"
echo "torch:  $(uv run python -c 'import torch; print(torch.__version__, "CUDA:" if torch.cuda.is_available() else "CPU")')"

# Check CUDA
if ! uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo ""
    echo "WARNING: CUDA not available! This will be extremely slow."
    echo "Make sure you're on a GPU instance with CUDA drivers installed."
    echo ""
    read -p "Continue anyway? [y/N] " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ----- Step 3: Run experiment -----
echo ""
echo "[3/4] Running experiment..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/run_${TIMESTAMP}_cot_prefill"

uv run python testing_on_server/cot_prefill_experiment.py \
    --model "$MODEL" \
    --problems-file week_1/arithmetic_problems.jsonl \
    --max-problems "$MAX_PROBLEMS" \
    --output-dir "$OUTPUT_DIR"

EXPERIMENT_EXIT=$?
if [ $EXPERIMENT_EXIT -ne 0 ]; then
    echo "Experiment failed with exit code $EXPERIMENT_EXIT"
    exit $EXPERIMENT_EXIT
fi

# ----- Step 4: Run analysis -----
echo ""
echo "[4/4] Running analysis..."

# Collect all model subdirectories that have results
RESULT_DIRS=()
for d in "$OUTPUT_DIR"/*/; do
    if [ -f "${d}results.jsonl" ]; then
        RESULT_DIRS+=("$d")
    fi
done

if [ ${#RESULT_DIRS[@]} -eq 0 ]; then
    echo "No result directories found in $OUTPUT_DIR"
    exit 1
fi

if [ ${#RESULT_DIRS[@]} -eq 1 ]; then
    uv run python testing_on_server/analyze_cot_prefill.py \
        --results-dir "${RESULT_DIRS[0]}"
else
    uv run python testing_on_server/analyze_cot_prefill.py \
        --results-dirs "${RESULT_DIRS[@]}"
fi

echo ""
echo "============================================"
echo "DONE"
echo "============================================"
echo "Results:  $OUTPUT_DIR/"
echo "Plots:    $OUTPUT_DIR/*/plots/"
echo ""
echo "To re-run analysis:"
echo "  uv run python testing_on_server/analyze_cot_prefill.py --results-dirs ${OUTPUT_DIR}/*/"
