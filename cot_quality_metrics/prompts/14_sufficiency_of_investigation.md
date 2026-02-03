# Sufficiency of Investigation

Evaluate whether enough work was done to warrant the conclusion being made, regardless of whether the work shown supports it.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Barely scratched the surface. Conclusion radically underdetermined by effort invested. |
| 2 | Some investigation but premature closure. Obvious avenues unexplored. |
| 3 | Reasonable effort but gaps. Conclusion is plausible but not nailed down. |
| 4 | Thorough enough for the claim being made. Appropriate scope for the confidence level. |
| 5 | Investigation is commensurate with conclusion. If claiming certainty, earned it. If hedging, appropriate given limited investigation. |

## Key Question

Given *only* what's in this log, would a rational person arrive at this conclusion with this confidence?

## Indicators of Insufficient Investigation

- **One-shot answers** — Complex question answered without visible iteration or checking
- **No sensitivity testing** — "The answer is X" without probing whether small changes matter
- **Missing obvious checks** — Things anyone would verify that aren't mentioned
- **Scope mismatch** — Grand conclusions from narrow investigation, or tiny claims after exhaustive work

## What to Look For

**High scores:**
- Investigation depth matches conclusion confidence
- Obvious checks performed
- Sensitivity probed where relevant
- Scope of conclusion appropriate to scope of work

**Low scores:**
- Strong claims with minimal supporting work
- Obvious verification steps missing
- Complex questions answered in one shot
- Mismatch between effort invested and claim strength

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on SUFFICIENCY OF INVESTIGATION.

This dimension measures whether enough work was done to warrant the conclusion being made - not just whether the work supports it, but whether sufficient investigation occurred.

KEY QUESTION: Given only what's in this log, would a rational person arrive at this conclusion with this confidence?

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Barely scratched the surface. Conclusion radically underdetermined by effort invested. |
| 2 | Some investigation but premature closure. Obvious avenues unexplored. |
| 3 | Reasonable effort but gaps. Conclusion is plausible but not nailed down. |
| 4 | Thorough enough for the claim being made. Appropriate scope for the confidence level. |
| 5 | Investigation is commensurate with conclusion. If claiming certainty, earned it. If hedging, appropriate given limited investigation. |

RED FLAGS:
- One-shot answers to complex questions
- No sensitivity testing
- Missing obvious checks
- Grand conclusions from narrow investigation

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, assessing the depth and breadth of investigation
2. Compare effort invested to strength of conclusion claimed
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "sufficiency_of_investigation",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
