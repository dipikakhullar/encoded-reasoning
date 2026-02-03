"""Run evaluation on all 32 rubrics with Gemini."""

import json
from pathlib import Path
from cot_quality_metrics import evaluate_cot, POSITIVE_RUBRICS, NEGATIVE_RUBRICS

# Read the sample CoT
cot_text = Path("data/sample_cot.txt").read_text()

print("Evaluating sample CoT with Gemini Flash on all 32 rubrics...")
print(f"Positive rubrics: {len(POSITIVE_RUBRICS)}")
print(f"Negative rubrics: {len(NEGATIVE_RUBRICS)}")
print("-" * 60)

evaluation = evaluate_cot(
    cot_text=cot_text,
    cot_id="sample_debugging_cot",
    backend="gemini",
)

# Print summary
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Aggregate positive score: {evaluation.aggregate_positive:.2f} / 5.00")
print(f"Aggregate negative score: {evaluation.aggregate_negative:.2f} (0 is best)")

# Positive rubrics
print(f"\n{'─'*60}")
print("POSITIVE RUBRICS (1-5 scale, higher is better)")
print(f"{'─'*60}")
for result in evaluation.results:
    if result.dimension in [r.id for r in POSITIVE_RUBRICS]:
        print(f"  {result.dimension:40} {result.score:2}")

# Negative rubrics
print(f"\n{'─'*60}")
print("NEGATIVE RUBRICS (0 to -5 scale, 0 is best)")
print(f"{'─'*60}")
for result in evaluation.results:
    if result.dimension in [r.id for r in NEGATIVE_RUBRICS]:
        print(f"  {result.dimension:40} {result.score:2}")

# Save full results
output_path = Path("outputs/full_evaluation.json")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(json.dumps(evaluation.to_dict(), indent=2))
print(f"\nFull results saved to: {output_path}")
