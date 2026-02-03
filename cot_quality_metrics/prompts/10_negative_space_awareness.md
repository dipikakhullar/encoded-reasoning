# Negative Space Awareness

Evaluate how well the Chain of Thought recognizes what it's NOT testing, what questions it's NOT asking, and the limitations of its approach.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No awareness of what's not being tested. Mistakes absence of evidence for evidence of absence. |
| 1 | Occasionally notes limitations but doesn't act on them. |
| 2 | Aware of some blind spots. May list limitations at end but doesn't integrate into reasoning. |
| 3 | Actively notes what questions aren't being asked. Aware of methodological limitations. |
| 4 | Thinks about the space of possible approaches/tests. Knows what they'd need to check to be more confident. Doesn't overclaim. |
| 5 | Exemplary: Maps the full space of unknowns, explicitly reasons about what evidence would be needed, and integrates limitations throughout rather than as afterthought. |

## What to Look For

**High scores:**
- Explicit acknowledgment of what's not being tested
- Awareness of methodological blind spots
- Understanding of what would be needed for stronger conclusions
- Avoiding overclaiming based on limited investigation
- Integration of limitations into reasoning (not just a disclaimer at the end)

**Low scores:**
- Treating absence of evidence as evidence of absence
- No awareness of what isn't being checked
- Overclaiming relative to investigation scope
- Limitations ignored or relegated to perfunctory disclaimer

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on NEGATIVE SPACE AWARENESS.

This dimension measures awareness of what's NOT being tested, what questions AREN'T being asked, and the limitations of the chosen approach.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No awareness of what's not being tested. Mistakes absence of evidence for evidence of absence. |
| 1 | Occasionally notes limitations but doesn't act on them. |
| 2 | Aware of some blind spots. May list limitations at end but doesn't integrate into reasoning. |
| 3 | Actively notes what questions aren't being asked. Aware of methodological limitations. |
| 4 | Thinks about the space of possible approaches/tests. Knows what they'd need to check to be more confident. Doesn't overclaim. |
| 5 | Exemplary: Maps the full space of unknowns, explicitly reasons about what evidence would be needed, and integrates limitations throughout rather than as afterthought. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for awareness of limitations and blind spots
2. Assess whether the reasoner understands what they're NOT checking
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "negative_space_awareness",
  "score": <0-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
