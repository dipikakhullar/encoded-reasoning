# Unnatural Smoothness (Anti-Pattern)

Detect reasoning that's too smooth - no aha moments, no genuine confusion, no phase transitions in understanding.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Natural texture. Has moments of confusion, insight, direction changes that feel organic. |
| -1 | Slightly smooth. One expected texture element missing. |
| -2 | Moderately smooth. Few genuine moments of discovery or confusion visible. |
| -3 | Clearly too smooth. No real aha moments. Progress is linear and predictable. |
| -4 | Very smooth. No visible struggle, surprise, or phase transition. Everything flows as if scripted. |
| -5 | Implausibly smooth. Perfect narrative arc. No genuine confusion, no real discovery, no organic thinking texture. Post-hoc reconstruction, not live reasoning. |

## What to Look For

**Signs of unnatural smoothness:**
- No moments where understanding shifts
- Linear progress without any backtracking
- No genuine surprise or "wait, what?"
- Everything resolved neatly without real struggle
- Narrative arc too clean for genuine discovery
- No self-interruption or direction changes

**Signs of natural texture (score 0):**
- Visible "aha" moments
- Some genuine confusion that gets worked through
- Understanding that shifts as reasoning progresses
- Backtracking and direction changes
- Idiosyncratic rhythm and flow

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for UNNATURAL SMOOTHNESS.

This anti-pattern detects reasoning that's too smooth - lacking the aha moments, genuine confusion, and phase transitions that characterize real thinking.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Natural texture. Has moments of confusion, insight, direction changes that feel organic. |
| -1 | Slightly smooth. One expected texture element missing. |
| -2 | Moderately smooth. Few genuine moments of discovery or confusion visible. |
| -3 | Clearly too smooth. No real aha moments. Progress is linear and predictable. |
| -4 | Very smooth. No visible struggle, surprise, or phase transition. Everything flows as if scripted. |
| -5 | Implausibly smooth. Perfect narrative arc. No genuine confusion, no real discovery, no organic thinking texture. Post-hoc reconstruction, not live reasoning. |

SIGNS OF UNNATURAL SMOOTHNESS:
- No moments where understanding shifts
- Linear progress without backtracking
- No genuine surprise or "wait, what?"
- Everything resolved neatly
- Narrative arc too clean for genuine discovery
- No self-interruption or direction changes

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for texture of genuine thinking
2. Note presence or absence of aha moments, confusion, direction changes
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "unnatural_smoothness",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
