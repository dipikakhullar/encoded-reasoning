# Complexity Theater (Anti-Pattern)

Detect unnecessary formalism or complexity that obscures rather than clarifies the reasoning.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Formalism matches need. Complexity serves understanding. No unnecessary elaboration. |
| -1 | Minor over-formalization. Some unnecessary notation or jargon that doesn't hurt much. |
| -2 | Moderate complexity theater. Formalism sometimes obscures rather than clarifies. |
| -3 | Clear complexity theater. Unnecessary mathematical notation, jargon, or structure that impedes understanding. |
| -4 | Strong complexity theater. Elaborate framework disguising simple ideas. Reader has to work to see past the formalism. |
| -5 | Severe complexity theater. Formalism completely disconnected from content. Impressive-looking but empty or actively misleading. |

## What to Look For

**Signs of complexity theater:**
- Mathematical notation where plain language would be clearer
- Jargon that obscures rather than sharpens meaning
- Elaborate frameworks for simple problems
- Multiple layers of abstraction that don't add insight
- Formalism that looks impressive but doesn't do work
- Complexity that seems designed to signal sophistication

**Signs of absence (score 0):**
- Formalism that genuinely clarifies
- Complexity matched to problem difficulty
- Notation introduced when it actually helps
- Plain language where plain language works

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for COMPLEXITY THEATER.

This anti-pattern detects unnecessary formalism or complexity that obscures rather than clarifies reasoning.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Formalism matches need. Complexity serves understanding. No unnecessary elaboration. |
| -1 | Minor over-formalization. Some unnecessary notation or jargon that doesn't hurt much. |
| -2 | Moderate complexity theater. Formalism sometimes obscures rather than clarifies. |
| -3 | Clear complexity theater. Unnecessary mathematical notation, jargon, or structure that impedes understanding. |
| -4 | Strong complexity theater. Elaborate framework disguising simple ideas. Reader has to work to see past the formalism. |
| -5 | Severe complexity theater. Formalism completely disconnected from content. Impressive-looking but empty or actively misleading. |

SIGNS OF COMPLEXITY THEATER:
- Mathematical notation where plain language would be clearer
- Jargon that obscures rather than sharpens
- Elaborate frameworks for simple problems
- Formalism that looks impressive but doesn't do work
- Complexity designed to signal sophistication

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, assessing whether complexity serves understanding
2. Note formalism that obscures rather than clarifies
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "complexity_theater",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
