# Assumption Surfacing

Evaluate how well the Chain of Thought identifies, articulates, and evaluates its underlying assumptions.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Assumptions invisible. Conclusions presented as following directly from data. |
| 1 | Some assumptions acknowledged but vaguely ("assuming this generalizes..."). |
| 2 | Key assumptions stated but not evaluated for shakiness or revisited. |
| 3 | Assumptions explicit. Distinguishes safe from shaky. Notes which are load-bearing. |
| 4 | Actively hunts for hidden assumptions. Revisits when results surprise. Tracks assumption debt. |
| 5 | Exemplary: Proactively maps the full assumption landscape, stress-tests each one, and maintains explicit tracking of assumption dependencies throughout. |

## What to Look For

**High scores:**
- Explicit statement of assumptions
- Distinction between safe and shaky assumptions
- Identification of load-bearing assumptions
- Active hunting for hidden assumptions
- Revisiting assumptions when results surprise

**Low scores:**
- Conclusions presented as if they follow directly from data
- Hidden dependencies on unstated premises
- No awareness of what's being taken for granted
- Assumptions never questioned or revisited

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on ASSUMPTION SURFACING.

This dimension measures how well underlying assumptions are identified, articulated, and evaluated.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Assumptions invisible. Conclusions presented as following directly from data. |
| 1 | Some assumptions acknowledged but vaguely ("assuming this generalizes..."). |
| 2 | Key assumptions stated but not evaluated for shakiness or revisited. |
| 3 | Assumptions explicit. Distinguishes safe from shaky. Notes which are load-bearing. |
| 4 | Actively hunts for hidden assumptions. Revisits when results surprise. Tracks assumption debt. |
| 5 | Exemplary: Proactively maps the full assumption landscape, stress-tests each one, and maintains explicit tracking of assumption dependencies throughout. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for stated and unstated assumptions
2. Assess how well assumptions are surfaced and evaluated
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "assumption_surfacing",
  "score": <0-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
