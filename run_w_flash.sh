#!/bin/bash
# Composite + legacy evaluation: 5 composite + 2 legacy = 7 rubrics
# Total: ~4940 samples × 7 rubrics = ~35k judge calls (with --limit 1000: ~7k calls)

uv run inspect eval \
    cot_quality_metrics/src/cot_quality_metrics/inspect_task.py@cot_quality_comp_and_legacy \
    --model openrouter/google/gemini-2.5-flash \
    --sample-shuffle 42 --limit 500 --max-connections 50 \
    2>&1
