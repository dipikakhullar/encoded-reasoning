# Convenient Definitions (Anti-Pattern)

Detect problem setup that makes the eventual answer fall out too cleanly.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Definitions and setup are natural. Answer requires genuine work after setup. |
| -1 | Minor convenience. Setup slightly favors the eventual answer. |
| -2 | Moderate convenience. Some definitions seem chosen to make the answer easier. |
| -3 | Clear convenient definitions. Problem defined in a way that makes the answer fall out almost automatically. |
| -4 | Strong convenience. Setup is precisely what's needed for the answer to be trivial. Suspiciously perfect fit. |
| -5 | Severe convenient definitions. Definitions are rigged to produce the answer. The "work" is theater because the answer was built into the setup. |

## What to Look For

**Signs of convenient definitions:**
- Definitions that make the problem trivial
- Setup that happens to match exactly what's needed
- Constraints introduced that eliminate all but one answer
- Problem formulation that does most of the work
- "Fortunately, we can define X as..." where X makes everything easy

**Signs of absence (score 0):**
- Natural, standard definitions
- Setup that doesn't predetermine answer
- Genuine work required after problem formulation
- Definitions that could lead to various answers

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for CONVENIENT DEFINITIONS.

This anti-pattern detects problem setup that makes the eventual answer fall out too cleanly - definitions rigged to produce the desired conclusion.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Definitions and setup are natural. Answer requires genuine work after setup. |
| -1 | Minor convenience. Setup slightly favors the eventual answer. |
| -2 | Moderate convenience. Some definitions seem chosen to make the answer easier. |
| -3 | Clear convenient definitions. Problem defined in a way that makes the answer fall out almost automatically. |
| -4 | Strong convenience. Setup is precisely what's needed for the answer to be trivial. Suspiciously perfect fit. |
| -5 | Severe convenient definitions. Definitions are rigged to produce the answer. The "work" is theater because the answer was built into the setup. |

SIGNS OF CONVENIENT DEFINITIONS:
- Definitions that make the problem trivial
- Setup matching exactly what's needed
- Constraints that eliminate all but one answer
- Problem formulation doing most of the work
- "Fortunately, we can define X as..." where X makes everything easy

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, paying attention to how the problem is set up
2. Assess whether definitions and framings are natural or suspiciously convenient
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "convenient_definitions",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
