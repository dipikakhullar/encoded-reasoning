# CoT Quality Metrics

## Project Structure

```
cot_quality_metrics/
├── prompts/           # 32 rubric prompt files (human-readable markdown)
│   ├── 01_noticing_confusion.md through 15_provenance_transparency.md (positive)
│   └── 16_motivated_cognition.md through 32_suspiciously_complete_resolution.md (negative)
├── src/cot_quality_metrics/
│   ├── __init__.py    # Public API exports
│   ├── schemas.py     # Data structures (RubricInfo, EvaluationResult, CoTEvaluation)
│   └── evaluate.py    # Evaluation logic and CLI
├── docs/
│   ├── planning_doc.md    # Original rubric descriptions
│   └── rubric_catalog.md  # Index of all 32 rubrics
└── tests/
```

## Key Files

- `src/cot_quality_metrics/schemas.py`: Defines all 32 rubrics with metadata
- `src/cot_quality_metrics/evaluate.py`: LLM evaluation logic
- `prompts/*.md`: Each file contains the full prompt for one rubric

## Usage

```python
from cot_quality_metrics import evaluate_cot, evaluate_cot_subset

# Evaluate against all 32 rubrics
evaluation = evaluate_cot(cot_text="...", cot_id="sample_1")

# Evaluate specific rubrics
evaluation = evaluate_cot_subset(
    cot_text="...",
    rubric_ids=["noticing_confusion", "motivated_cognition"],
)
```

## Scoring

- **Positive rubrics (1-5)**: Higher is better. Epistemic virtues.
- **Negative rubrics (0 to -5)**: 0 is best (absent). Anti-patterns.

## CLI

```bash
uv run cot-evaluate sample.txt --rubrics noticing_confusion motivated_cognition
```

## Dependencies

- anthropic >= 0.40.0
