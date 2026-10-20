#!/bin/bash
# Composite + legacy evaluation: 5 composite + 2 legacy = 7 rubrics
# Using Claude Sonnet 4.5 via OpenRouter
# Evaluates only alpha 2.5 data (most corrupted)

uv run inspect eval \
    cot_quality_metrics/src/cot_quality_metrics/inspect_task.py@cot_quality_comp_and_legacy \
    --model openrouter/anthropic/claude-sonnet-4.5 \
    -T data_dir=../../../rollouts_alpha2.5 \
    --max-connections 150
