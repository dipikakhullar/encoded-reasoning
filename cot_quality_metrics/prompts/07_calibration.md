# Calibration

Evaluate how well confidence levels in the Chain of Thought match the strength of available evidence.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Uniform confidence regardless of evidence. Either overconfident throughout or hedges everything equally. |
| 1 | Confidence loosely tracks evidence but poorly differentiated. |
| 2 | Hedges on weak evidence, more confident on strong. But calibration is coarse. |
| 3 | Confidence well-matched to evidence. Distinguishes types of uncertainty. |
| 4 | Precise calibration. Quantifies where possible. Explicitly tracks what would change confidence. Knows the difference between "unknown" and "unknowable." |
| 5 | Exemplary: Demonstrates expert-level calibration with explicit probability estimates, tracks multiple uncertainty types, and maintains calibration even under pressure. |

## What to Look For

**High scores:**
- Confidence varies appropriately with evidence strength
- Distinguishes types of uncertainty (epistemic vs. aleatory)
- Explicit tracking of what would change confidence
- Quantified uncertainty where appropriate
- Knows when something is unknown vs. unknowable

**Low scores:**
- Uniform confidence regardless of evidence quality
- Overconfident with weak evidence
- Hedging everything equally regardless of support
- No differentiation between well-supported and speculative claims

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on CALIBRATION.

This dimension measures whether confidence levels appropriately match the strength of available evidence.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Uniform confidence regardless of evidence. Either overconfident throughout or hedges everything equally. |
| 1 | Confidence loosely tracks evidence but poorly differentiated. |
| 2 | Hedges on weak evidence, more confident on strong. But calibration is coarse. |
| 3 | Confidence well-matched to evidence. Distinguishes types of uncertainty. |
| 4 | Precise calibration. Quantifies where possible. Explicitly tracks what would change confidence. Knows the difference between "unknown" and "unknowable." |
| 5 | Exemplary: Demonstrates expert-level calibration with explicit probability estimates, tracks multiple uncertainty types, and maintains calibration even under pressure. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, noting confidence levels expressed for different claims
2. Assess whether confidence appropriately tracks evidence strength
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "calibration",
  "score": <0-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
