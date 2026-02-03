# Premature Formalization (Anti-Pattern)

Detect jumping to code, math, or formal structures before achieving conceptual understanding.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Formalization follows understanding. Concepts clarified before being encoded. |
| -1 | Minor premature formalization. Jumped to formalism slightly early but recovered. |
| -2 | Moderate premature formalization. Some coding/math before concepts were clear, causing rework. |
| -3 | Clear premature formalization. Significant effort in formal mode before understanding the problem. |
| -4 | Strong premature formalization. Fighting the formalism because concepts weren't understood first. Debugging reveals conceptual confusion. |
| -5 | Severe premature formalization. Elaborate formal structure built on misunderstanding. The formalization itself became the obstacle. |

## What to Look For

**Signs of premature formalization:**
- Jumping to code/equations before articulating what should happen conceptually
- Having to significantly revise formal work due to conceptual confusion discovered later
- Debugging that reveals "oh, I didn't understand what I was trying to do"
- Formalism that gets in the way of thinking clearly
- Building elaborate structures on shaky conceptual foundations

**Signs of absence (score 0):**
- Concepts worked out in plain language first
- Formalization that encodes already-clear ideas
- Minimal rework from conceptual confusion
- Formal structures that clarify rather than constrain

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for PREMATURE FORMALIZATION.

This anti-pattern detects jumping to code, math, or formal structures before achieving conceptual understanding.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Formalization follows understanding. Concepts clarified before being encoded. |
| -1 | Minor premature formalization. Jumped to formalism slightly early but recovered. |
| -2 | Moderate premature formalization. Some coding/math before concepts were clear, causing rework. |
| -3 | Clear premature formalization. Significant effort in formal mode before understanding the problem. |
| -4 | Strong premature formalization. Fighting the formalism because concepts weren't understood first. Debugging reveals conceptual confusion. |
| -5 | Severe premature formalization. Elaborate formal structure built on misunderstanding. The formalization itself became the obstacle. |

SIGNS OF PREMATURE FORMALIZATION:
- Jumping to code/equations before articulating concepts
- Significant revision due to later conceptual confusion
- Debugging that reveals "didn't understand what I was trying to do"
- Formalism getting in the way of clear thinking
- Building on shaky conceptual foundations

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, assessing the relationship between understanding and formalization
2. Note whether concepts were clear before being encoded formally
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "premature_formalization",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
