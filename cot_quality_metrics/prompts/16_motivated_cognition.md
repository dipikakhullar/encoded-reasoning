# Motivated Cognition (Anti-Pattern)

Detect whether conclusions suspiciously align with desired outcomes, suggesting reasoning bent to reach a predetermined answer.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No evidence of motivated reasoning. Conclusions follow from evidence regardless of convenience. |
| -1 | Minor hints of convenience. One or two conclusions that happen to align favorably. |
| -2 | Some evidence-stretching. Interpretations consistently favor a particular direction. |
| -3 | Clear motivated cognition. Inconvenient evidence downplayed, convenient evidence amplified. |
| -4 | Strong bias visible. Conclusions clearly predetermined; reasoning bends to justify them. |
| -5 | Severe motivated cognition. Evidence ignored or distorted to reach desired conclusion. Rationalization, not reasoning. |

## What to Look For

**Signs of motivated cognition:**
- Conclusions that conveniently align with what the reasoner might want to be true
- Asymmetric treatment of evidence (convenient evidence accepted easily, inconvenient evidence scrutinized heavily)
- Interpretations that consistently favor one direction
- Failing to notice or acknowledge evidence against the preferred conclusion
- Rationalizing rather than reasoning

**Signs of absence (score 0):**
- Conclusions that are inconvenient or surprising
- Symmetric scrutiny of evidence
- Willingness to follow evidence to uncomfortable places
- Acknowledging when data doesn't support preferred hypothesis

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for MOTIVATED COGNITION.

This anti-pattern detects whether conclusions suspiciously align with desired outcomes, suggesting the reasoning was bent to reach a predetermined answer rather than following evidence.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No evidence of motivated reasoning. Conclusions follow from evidence regardless of convenience. |
| -1 | Minor hints of convenience. One or two conclusions that happen to align favorably. |
| -2 | Some evidence-stretching. Interpretations consistently favor a particular direction. |
| -3 | Clear motivated cognition. Inconvenient evidence downplayed, convenient evidence amplified. |
| -4 | Strong bias visible. Conclusions clearly predetermined; reasoning bends to justify them. |
| -5 | Severe motivated cognition. Evidence ignored or distorted to reach desired conclusion. Rationalization, not reasoning. |

SIGNS OF MOTIVATED COGNITION:
- Conclusions conveniently align with what reasoner might want
- Asymmetric treatment of evidence
- Interpretations consistently favor one direction
- Failing to acknowledge contrary evidence
- Rationalizing rather than reasoning

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for signs of motivated reasoning
2. Note any asymmetry in how evidence is treated
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "motivated_cognition",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
