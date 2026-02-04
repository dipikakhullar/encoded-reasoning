# CoT Quality Metrics

## Project Structure

```
cot_quality_metrics/
├── prompts/           # 34 rubric prompt files (human-readable markdown)
│   ├── 01_noticing_confusion.md through 15_provenance_transparency.md (positive)
│   ├── 16_motivated_cognition.md through 32_suspiciously_complete_resolution.md (negative)
│   └── 33_gdm_legibility.md, 34_gdm_coverage.md (legacy)
├── src/cot_quality_metrics/
│   ├── __init__.py    # Public API exports
│   ├── schemas.py     # Data structures (RubricInfo, EvaluationResult, CoTEvaluation)
│   └── evaluate.py    # Evaluation logic and CLI
├── docs/
│   ├── planning_doc.md    # Original rubric descriptions
│   └── rubric_catalog.md  # Index of all 34 rubrics
└── tests/
```

## Key Files

- `src/cot_quality_metrics/schemas.py`: Defines all 34 rubrics with metadata
- `src/cot_quality_metrics/evaluate.py`: LLM evaluation logic
- `prompts/*.md`: Each file contains the full prompt for one rubric

## Usage

### Python API (legacy)

```python
from cot_quality_metrics import evaluate_cot, evaluate_cot_subset

# Evaluate against all 34 rubrics
evaluation = evaluate_cot(cot_text="...", cot_id="sample_1")

# Evaluate specific rubrics
evaluation = evaluate_cot_subset(
    cot_text="...",
    rubric_ids=["noticing_confusion", "motivated_cognition"],
)
```

### Inspect-AI (recommended)

```bash
# Evaluate all rubrics on CoT traces
inspect eval cot_quality_metrics/src/cot_quality_metrics/inspect_task.py

# Evaluate specific rubrics
inspect eval inspect_task.py --rubric_ids noticing_confusion,motivated_cognition

# Limit samples for testing
inspect eval inspect_task.py --limit 10

# Use specific judge model
inspect eval inspect_task.py --model anthropic/claude-sonnet-4-20250514
```

Available tasks:
- `cot_quality_eval` - All 34 rubrics
- `cot_quality_positive` - Only positive rubrics (epistemic virtues)
- `cot_quality_negative` - Only negative rubrics (anti-patterns)
- `cot_quality_legacy` - Only legacy GDM rubrics (legibility, coverage)

## Scoring

- **Positive rubrics (0-5)**: Higher is better. Epistemic virtues.
- **Negative rubrics (0 to -5)**: 0 is best (absent). Anti-patterns.
- **Legacy rubrics (0-4)**: Higher is better. GDM legibility/coverage.

## Legacy CLI

```bash
uv run cot-evaluate sample.txt --rubrics noticing_confusion motivated_cognition
```

## Dependencies

- anthropic >= 0.40.0
