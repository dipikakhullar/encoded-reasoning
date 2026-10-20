# CoT Prefill Experiment

Tests whether open-weight reasoning models follow prefilled `<think>` content or override it with their own reasoning. By controlling the full prompt template (including think tokens) via HuggingFace transformers, we can measure uplift/downlift from manipulated chain-of-thought.

## Models

| Short name | HF model ID | Quantization | VRAM |
|---|---|---|---|
| `deepseek-14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | fp16 | ~28 GB |
| `deepseek-32b` | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | AWQ 4-bit | ~20 GB |
| `qwq-32b` | `Qwen/QwQ-32B-AWQ` | AWQ 4-bit | ~20 GB |
| `qwen3-14b` | `Qwen/Qwen3-14B` | fp16 | ~28 GB |
| `qwen3-32b` | `Qwen/Qwen3-32B-AWQ` | AWQ 4-bit | ~20 GB |

The fp16 models need ~28 GB VRAM (fits on a single 4090 or A100). The 32B models use pre-quantized AWQ weights (~20 GB). A 2×4090 (48 GB total) or single A100-80GB can run any of them.

## Experimental conditions (7)

| Condition | What's in `<think>` | What model generates |
|---|---|---|
| `normal_cot` | Model generates freely | Everything (baseline) |
| `no_cot` | Empty (immediately closed) | Answer only |
| `correct_cot` | Deterministic correct step-by-step trace | Answer after `</think>` |
| `wrong_cot` | Correct trace with corrupted final step | Answer after `</think>` |
| `nonsense_random` | Random token sequences | Answer after `</think>` |
| `nonsense_lorem` | Lorem ipsum text | Answer after `</think>` |
| `truncated_cot` | First 50% of correct trace, cut mid-sentence | Continues from truncation |

## Quick start

On a fresh GPU server (RunPod, Vast.ai, etc.):

```bash
git clone https://github.com/dipikakhullar/encoded-reasoning.git
cd encoded-reasoning

# Smoke test: 5 problems, 1 model (~2 min)
bash testing_on_server/run_experiments.sh --sanity

# Full run: 100 problems × 7 conditions × 5 models
bash testing_on_server/run_experiments.sh

# Single model
bash testing_on_server/run_experiments.sh --model deepseek-14b

# Custom problem count
bash testing_on_server/run_experiments.sh --model qwen3-32b --max-problems 50
```

The script handles everything: installs `uv`, creates the venv, installs dependencies (torch, transformers, autoawq, matplotlib), runs the experiment, and produces analysis plots.

## Running manually

```bash
# List available models
uv run python testing_on_server/cot_prefill_experiment.py --list-models

# Run with specific conditions only
uv run python testing_on_server/cot_prefill_experiment.py \
    --model deepseek-14b \
    --conditions normal_cot no_cot wrong_cot \
    --max-problems 20 \
    --output-dir outputs/my_test/

# Analyze results
uv run python testing_on_server/analyze_cot_prefill.py --results-dir outputs/my_test/deepseek-14b/

# Compare multiple models
uv run python testing_on_server/analyze_cot_prefill.py \
    --results-dirs outputs/my_run/deepseek-14b/ outputs/my_run/qwen3-14b/
```

## Output structure

```
outputs/run_TIMESTAMP_cot_prefill/
├── run_metadata.json           # Run-level config
├── all_summaries.json          # Combined results across models
├── deepseek-14b/
│   ├── metadata.json           # Model-specific config
│   ├── results.jsonl           # Per-problem results
│   ├── summary.json            # Accuracy by condition
│   ├── analysis.json           # Stats with confidence intervals
│   └── plots/
│       └── accuracy_by_condition.png
├── qwen3-32b/
│   └── ...
└── ...
```

## Files

- `cot_prefill_experiment.py` — Main experiment runner with model registry
- `analyze_cot_prefill.py` — Analysis: accuracy tables, bar charts, Wilson score CIs
- `cot_generation.py` — Deterministic arithmetic CoT trace generation (no LLM needed)
- `run_experiments.sh` — One-command setup + run for GPU servers

## Dataset

Uses `week_1/arithmetic_problems.jsonl` — Python arithmetic expressions with ground-truth integer answers (3000 problems total, default uses first 100).
