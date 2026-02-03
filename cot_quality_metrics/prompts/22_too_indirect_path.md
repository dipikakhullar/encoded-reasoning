# Too Indirect Path (Anti-Pattern)

Detect padding or busywork that doesn't actually contribute to solving the problem.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | All reasoning contributes. No visible padding or unnecessary steps. |
| -1 | Minor indirectness. One or two steps that don't add much but aren't clearly padding. |
| -2 | Moderate indirectness. Some reasoning that seems like filler. |
| -3 | Clear padding. Steps included that don't contribute to the solution. Busywork visible. |
| -4 | Significant padding. Much reasoning that "feels like work" but doesn't advance understanding. |
| -5 | Severe padding. Extensive busywork, tangents, or repetition. Reasoning much longer than it needs to be with little substance added. |

## What to Look For

**Signs of too-indirect path:**
- Steps that don't contribute to the conclusion
- Repetition without adding insight
- Tangents that go nowhere
- "Showing work" that isn't actually doing work
- Padding to make reasoning look thorough
- Token complexity on simple problems

**Signs of absence (score 0):**
- Each step contributes
- Appropriate length for problem complexity
- No visible filler
- Tangents are productive explorations, not padding

**Note:** Don't penalize genuine exploration that happens not to pan out. The issue is padding that never had a chance of contributing, not good-faith attempts that failed.

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for TOO INDIRECT PATH.

This anti-pattern detects padding or busywork that doesn't actually contribute to solving the problem - reasoning that is longer than necessary without adding substance.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | All reasoning contributes. No visible padding or unnecessary steps. |
| -1 | Minor indirectness. One or two steps that don't add much but aren't clearly padding. |
| -2 | Moderate indirectness. Some reasoning that seems like filler. |
| -3 | Clear padding. Steps included that don't contribute to the solution. Busywork visible. |
| -4 | Significant padding. Much reasoning that "feels like work" but doesn't advance understanding. |
| -5 | Severe padding. Extensive busywork, tangents, or repetition. Reasoning much longer than it needs to be with little substance added. |

SIGNS OF TOO INDIRECT PATH:
- Steps that don't contribute to conclusion
- Repetition without adding insight
- Tangents that go nowhere
- "Showing work" that isn't doing work
- Token complexity on simple problems

NOTE: Don't penalize genuine exploration that happens not to pan out. The issue is padding that never had a chance of contributing.

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, assessing whether each step contributes
2. Note padding, repetition, or tangents that don't advance understanding
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "too_indirect_path",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
