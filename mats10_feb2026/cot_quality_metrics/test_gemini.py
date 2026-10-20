"""Quick test of Gemini evaluation."""

from pathlib import Path
from cot_quality_metrics import evaluate_cot_subset

# Read the sample CoT
cot_text = Path("data/sample_cot.txt").read_text()

# Test with just a few rubrics to save API calls
test_rubrics = [
    "noticing_confusion",
    "error_metabolism",
    "authenticity",
]

print("Evaluating sample CoT with Gemini 2.5 Flash...")
print(f"Rubrics: {test_rubrics}")
print("-" * 50)

evaluation = evaluate_cot_subset(
    cot_text=cot_text,
    rubric_ids=test_rubrics,
    cot_id="sample_debugging_cot",
    backend="gemini",
)

print(f"\nResults for: {evaluation.cot_id}")
print(f"Aggregate positive score: {evaluation.aggregate_positive:.2f}")
print()

for result in evaluation.results:
    print(f"\n{result.dimension}:")
    print(f"  Score: {result.score}")
    print(f"  Reasoning: {result.reasoning}")
    if result.evidence:
        print(f"  Evidence:")
        for ev in result.evidence[:2]:  # Show first 2
            print(f"    - {ev[:100]}..." if len(ev) > 100 else f"    - {ev}")
