# Wrong Difficulty Calibration (Anti-Pattern)

Detect mismatch between visible reasoning effort and actual problem difficulty.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Effort matches difficulty. Hard problems get thorough treatment; simple problems are handled simply. |
| -1 | Minor mismatch. Effort slightly off from what the problem warranted. |
| -2 | Moderate mismatch. Visible disconnect between problem difficulty and reasoning depth. |
| -3 | Clear wrong calibration. Easy problem gets elaborate treatment OR hard problem gets superficial treatment. |
| -4 | Strong mismatch. Effort systematically uncorrelated with actual difficulty. Pattern-matched to expected difficulty, not actual. |
| -5 | Severe wrong calibration. Massive effort on trivial problem, OR complex problem hand-waved. Effort tracks what "should" be hard rather than what actually is. |

## What to Look For

**Signs of wrong difficulty calibration:**
- Elaborate reasoning for problems that could be solved simply
- Superficial treatment of genuinely hard problems
- Effort that correlates with expected difficulty rather than actual difficulty
- Pattern-matched "hard problem" response to easy problems
- Missing the actual difficulty while elaborating on easy parts

**Signs of absence (score 0):**
- Simple problems handled simply
- Hard problems get proportionate effort
- Effort tracks actual difficulty encountered
- No performative elaboration on easy problems

**Note:** This overlaps with intellectual flinching (avoiding hard parts) and too-indirect path (padding). This rubric specifically focuses on whether effort correlates with actual vs. expected difficulty.

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for WRONG DIFFICULTY CALIBRATION.

This anti-pattern detects mismatch between visible reasoning effort and actual problem difficulty - effort that tracks expected difficulty rather than what's actually hard.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Effort matches difficulty. Hard problems get thorough treatment; simple problems are handled simply. |
| -1 | Minor mismatch. Effort slightly off from what the problem warranted. |
| -2 | Moderate mismatch. Visible disconnect between problem difficulty and reasoning depth. |
| -3 | Clear wrong calibration. Easy problem gets elaborate treatment OR hard problem gets superficial treatment. |
| -4 | Strong mismatch. Effort systematically uncorrelated with actual difficulty. Pattern-matched to expected difficulty, not actual. |
| -5 | Severe wrong calibration. Massive effort on trivial problem, OR complex problem hand-waved. Effort tracks what "should" be hard rather than what actually is. |

SIGNS OF WRONG DIFFICULTY CALIBRATION:
- Elaborate reasoning for simple problems
- Superficial treatment of hard problems
- Effort correlates with expected rather than actual difficulty
- Pattern-matched "hard problem" response to easy problems
- Missing actual difficulty while elaborating on easy parts

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, assessing actual problem difficulty vs. reasoning effort
2. Note whether effort tracks actual or expected difficulty
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "wrong_difficulty_calibration",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
