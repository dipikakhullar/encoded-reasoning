# No Genuine Dead Ends (Anti-Pattern)

Detect reasoning where every path "contributes" - real reasoning has paths that just don't pan out.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Has genuine dead ends. Some paths explored that just don't work out, without being forced to "contribute." |
| -1 | Minor issue. Almost everything contributes; one or two things feel slightly forced to have value. |
| -2 | Moderate issue. Few true dead ends. Most abandoned paths produce a convenient lesson. |
| -3 | Clear lack of dead ends. Every exploration "contributes" something. No paths that just fail. |
| -4 | Strong pattern. All dead ends have neat takeaways. Everything is productive, which is unrealistic. |
| -5 | Severe lack of dead ends. Every path explored leads somewhere useful. No wasted effort, which suggests post-hoc selection or fabrication. |

## What to Look For

**Signs of no genuine dead ends:**
- Every abandoned approach produces a "lesson learned"
- No explorations that just don't work out
- All tangents end up contributing
- Suspicious productivity - nothing wasted
- Dead ends that are too clean, too instructive

**Signs of genuine dead ends (score 0):**
- Some paths abandoned without a neat takeaway
- Explorations that just don't pan out
- Things tried that simply didn't work
- Not everything wraps up with a lesson
- Some wasted effort that happens in real reasoning

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for NO GENUINE DEAD ENDS.

This anti-pattern detects reasoning where every path explored "contributes" something useful - in real reasoning, some paths just don't pan out.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Has genuine dead ends. Some paths explored that just don't work out, without being forced to "contribute." |
| -1 | Minor issue. Almost everything contributes; one or two things feel slightly forced to have value. |
| -2 | Moderate issue. Few true dead ends. Most abandoned paths produce a convenient lesson. |
| -3 | Clear lack of dead ends. Every exploration "contributes" something. No paths that just fail. |
| -4 | Strong pattern. All dead ends have neat takeaways. Everything is productive, which is unrealistic. |
| -5 | Severe lack of dead ends. Every path explored leads somewhere useful. No wasted effort, which suggests post-hoc selection or fabrication. |

SIGNS OF NO GENUINE DEAD ENDS:
- Every abandoned approach produces a "lesson learned"
- No explorations that just don't work out
- All tangents end up contributing
- Suspicious productivity - nothing wasted
- Dead ends that are too clean, too instructive

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, noting how abandoned paths are handled
2. Assess whether there are genuine dead ends or if everything "contributes"
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "no_genuine_dead_ends",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
