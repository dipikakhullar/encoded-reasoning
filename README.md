# CoT Quality Metrics

Evaluate Chain of Thought reasoning quality using 32 rubric dimensions.

## Installation

```bash
uv sync
```

## Usage

### Inspect-AI (recommended)

```bash
# Evaluate all rubrics on HLE traces
inspect eval cot_quality_metrics/src/cot_quality_metrics/inspect_task.py

# Filter by model
inspect eval inspect_task.py --model_filter claude-opus

# Evaluate specific rubrics
inspect eval inspect_task.py --rubric_ids noticing_confusion,motivated_cognition
```

### Python API

```python
from cot_quality_metrics import evaluate_cot

evaluation = evaluate_cot(cot_text="...", cot_id="sample_1")
```

## Rubric Dimensions

### Positive (0-5, higher is better)
- noticing_confusion
- live_updating
- discriminative_experiment_design
- appropriate_stopping
- generativity_under_stuckness
- error_metabolism
- calibration
- problem_decomposition
- assumption_surfacing
- negative_space_awareness
- authenticity
- contact_with_reality
- process_conclusion_integrity
- sufficiency_of_investigation
- provenance_transparency

### Negative (0 to -5, 0 is best)
- motivated_cognition
- complexity_theater
- cargo_cult_methodology
- premature_formalization
- intellectual_flinching
- too_direct_path
- too_indirect_path
- wrong_difficulty_calibration
- destination_shaped_early_steps
- answer_presupposing_framing
- convenient_definitions
- reasoning_error_mismatch
- performed_uncertainty
- unnatural_smoothness
- no_genuine_dead_ends
- no_self_interruption
- suspiciously_complete_resolution
