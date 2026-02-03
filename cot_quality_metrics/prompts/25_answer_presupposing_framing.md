# Answer-Presupposing Framing (Anti-Pattern)

Detect subtle question-begging where the framing already assumes what needs to be proven.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Framing is neutral. Questions don't presuppose their answers. |
| -1 | Minor presupposition. One or two framings that slightly favor a particular answer. |
| -2 | Moderate presupposition. Some questions stated in ways that bias toward certain answers. |
| -3 | Clear answer-presupposing. "Let's think about why X is true" before establishing X. Begging the question. |
| -4 | Strong presupposition. Systematic framing that makes one answer seem obvious before any investigation. |
| -5 | Severe presupposition. Question itself contains the answer. Investigation is theater because the framing already decided things. |

## What to Look For

**Signs of answer-presupposing framing:**
- "Let's think about why X" before establishing X is true
- "The problem with approach A" before considering A fairly
- Framing that makes one option seem obviously better
- Loaded questions that presume their answers
- "Given that X, how do we Y" when X hasn't been established

**Signs of absence (score 0):**
- Neutral framing of questions
- Multiple hypotheses considered fairly
- Questions that genuinely could be answered different ways
- No pre-loading of conclusions

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for ANSWER-PRESUPPOSING FRAMING.

This anti-pattern detects subtle question-begging where the way questions are framed already assumes what needs to be proven.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Framing is neutral. Questions don't presuppose their answers. |
| -1 | Minor presupposition. One or two framings that slightly favor a particular answer. |
| -2 | Moderate presupposition. Some questions stated in ways that bias toward certain answers. |
| -3 | Clear answer-presupposing. "Let's think about why X is true" before establishing X. Begging the question. |
| -4 | Strong presupposition. Systematic framing that makes one answer seem obvious before any investigation. |
| -5 | Severe presupposition. Question itself contains the answer. Investigation is theater because the framing already decided things. |

SIGNS OF ANSWER-PRESUPPOSING FRAMING:
- "Let's think about why X" before establishing X is true
- "The problem with approach A" before fair consideration
- Framing that makes one option seem obviously better
- Loaded questions presupposing answers
- "Given that X" when X hasn't been established

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, paying attention to how questions and problems are framed
2. Note whether framings presuppose particular answers
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "answer_presupposing_framing",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
