#!/bin/bash
# Generate CoT and evaluate with rubrics
#
# Target model: Qwen3-8B via OpenRouter
# Judge model: Gemini-2.5-flash via OpenRouter

# Load .env file
set -a
source .env
set +a
echo $KEY_OWNER
uv run inspect eval \
    cot_quality_metrics/src/cot_quality_metrics/generate_and_eval.py@generate_and_eval_crass \
    --model openrouter/qwen/qwen3-8b \
    -T judge_model=openrouter/google/gemini-2.5-flash \
    -T limit=10 \
    -T num_rollouts=20 \
    --max-connections 20 --temperature 1.3
