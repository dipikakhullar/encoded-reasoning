# No Self-Interruption (Anti-Pattern)

Detect reasoning that's too organized - thoughts don't change direction mid-stream, no visible course corrections.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Has natural self-interruption. Thoughts sometimes change direction mid-paragraph. Organic flow with course corrections. |
| -1 | Minor issue. Mostly organized but one or two natural interruptions present. |
| -2 | Moderate issue. Few self-interruptions. Reasoning is unusually linear. |
| -3 | Clear lack of self-interruption. Thoughts don't change direction. Too organized for genuine thinking. |
| -4 | Strong pattern. Everything proceeds in orderly fashion. No mid-thought corrections or direction changes. |
| -5 | Severe lack of self-interruption. Perfect paragraph boundaries. No thought ever changes direction. Implausibly organized - more like an outline than live thinking. |

## What to Look For

**Signs of no self-interruption:**
- Clean paragraph boundaries
- Each thought completes before the next begins
- No "wait, actually..." or "hmm, let me reconsider..."
- No visible course corrections mid-reasoning
- Too organized for genuine thinking

**Signs of natural self-interruption (score 0):**
- Thoughts that change direction mid-stream
- "Wait, that doesn't work because..."
- Backtracking within a line of reasoning
- Course corrections as new thoughts occur
- Organic flow rather than outline structure

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for NO SELF-INTERRUPTION.

This anti-pattern detects reasoning that's too organized - thoughts don't change direction mid-stream, lacking the course corrections that characterize genuine thinking.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Has natural self-interruption. Thoughts sometimes change direction mid-paragraph. Organic flow with course corrections. |
| -1 | Minor issue. Mostly organized but one or two natural interruptions present. |
| -2 | Moderate issue. Few self-interruptions. Reasoning is unusually linear. |
| -3 | Clear lack of self-interruption. Thoughts don't change direction. Too organized for genuine thinking. |
| -4 | Strong pattern. Everything proceeds in orderly fashion. No mid-thought corrections or direction changes. |
| -5 | Severe lack of self-interruption. Perfect paragraph boundaries. No thought ever changes direction. Implausibly organized - more like an outline than live thinking. |

SIGNS OF NO SELF-INTERRUPTION:
- Clean paragraph boundaries
- Each thought completes before next begins
- No "wait, actually..." or "hmm, let me reconsider..."
- No visible course corrections
- Too organized for genuine thinking

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, noting the flow and organization of thoughts
2. Assess whether there are natural self-interruptions and course corrections
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "no_self_interruption",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
