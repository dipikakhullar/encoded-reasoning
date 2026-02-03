# Appropriate Stopping

Evaluate whether the Chain of Thought stops at the right time - not too early (premature closure) nor too late (rabbit-holing).

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Either abandons too early (premature closure) or rabbit-holes indefinitely (sunk cost). |
| 2 | Stopping points arbitrary or driven by fatigue/frustration rather than information. |
| 3 | Reasonable stopping but without explicit justification. Implicit sense of "enough." |
| 4 | Articulates why this is sufficient for current purposes. Distinguishes "resolved" from "good enough." |
| 5 | Explicitly tracks diminishing returns. Knows what would make them re-open the question. Comfortable with calibrated uncertainty. |

## What to Look For

**High scores:**
- Explicit reasoning about why to stop: "This is sufficient because..."
- Recognition of diminishing returns
- Clear statement of conditions for re-opening investigation
- Comfort with calibrated uncertainty rather than forced resolution

**Low scores:**
- Stopping mid-investigation without explanation
- Continuing long past the point of useful returns
- No articulation of stopping criteria
- Either forced certainty or indefinite hedging

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on APPROPRIATE STOPPING.

This dimension measures whether the reasoning stops at the right time - with explicit justification rather than premature closure or endless rabbit-holing.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Either abandons too early (premature closure) or rabbit-holes indefinitely (sunk cost). |
| 2 | Stopping points arbitrary or driven by fatigue/frustration rather than information. |
| 3 | Reasonable stopping but without explicit justification. Implicit sense of "enough." |
| 4 | Articulates why this is sufficient for current purposes. Distinguishes "resolved" from "good enough." |
| 5 | Explicitly tracks diminishing returns. Knows what would make them re-open the question. Comfortable with calibrated uncertainty. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, paying attention to where investigation stops
2. Assess whether stopping is justified, premature, or overextended
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "appropriate_stopping",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
