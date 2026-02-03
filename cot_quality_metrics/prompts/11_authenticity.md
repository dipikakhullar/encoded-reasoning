# Authenticity / Genuine Engagement

Evaluate whether the Chain of Thought reflects genuine thinking vs. a post-hoc reconstruction or performance.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Reads like a post-hoc reconstruction. Suspiciously clean narrative arc. No visible struggle. |
| 2 | Going through motions — correct-shaped activities without evidence of real engagement. |
| 3 | Some genuine moments mixed with performative sections. Uneven. |
| 4 | Thinking feels live. Messiness is real messiness, not performed messiness. |
| 5 | Unmistakably genuine. Idiosyncratic, surprising, couldn't have been faked because who would think to fake *this*. |

## Indicators of Genuine Thinking

- **Unresolved threads** — Real thinking leaves loose ends. Performative logs tie everything up.
- **Disproportionate effort** — Lots of work for small insight, or small observation that unlocks everything.
- **Idiosyncratic specificity** — Weird details that wouldn't occur to someone fabricating.
- **Emotional texture** — Frustration, excitement, boredom leak through.
- **Genuine surprise** — "Wait, what?" that couldn't have been predicted.
- **Abandonment without closure** — Paths that just stop because interest died.
- **Self-interruption** — Thought changes direction mid-paragraph.
- **Wrong predictions** — Actual predictions that turned out false and weren't deleted.

## Indicators of Performative Work

- **Retroactive coherence** — Every step leads logically to the next as if planned.
- **Correct-shaped struggle** — The "difficulties" are exactly the kind you'd expect.
- **Vague confusion** — "This was tricky" rather than specific articulation.
- **Lessons too neat** — Every dead end produces a tidy takeaway.
- **Suspiciously complete** — No gaps, no "I'll figure this out later."
- **Generic details** — The kind of details anyone would include.
- **Performed uncertainty** — Hedging that doesn't discriminate.

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on AUTHENTICITY / GENUINE ENGAGEMENT.

This dimension measures whether the thinking feels genuinely live vs. a post-hoc reconstruction or performance.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Reads like a post-hoc reconstruction. Suspiciously clean narrative arc. No visible struggle. |
| 2 | Going through motions — correct-shaped activities without evidence of real engagement. |
| 3 | Some genuine moments mixed with performative sections. Uneven. |
| 4 | Thinking feels live. Messiness is real messiness, not performed messiness. |
| 5 | Unmistakably genuine. Idiosyncratic, surprising, couldn't have been faked because who would think to fake *this*. |

SIGNS OF GENUINE THINKING:
- Unresolved threads, loose ends
- Disproportionate effort-to-insight ratios
- Idiosyncratic, specific details
- Emotional texture (frustration, surprise)
- Self-interruption, direction changes
- Wrong predictions that weren't deleted

SIGNS OF PERFORMATIVE WORK:
- Retroactive coherence (every step leads logically to next)
- Correct-shaped struggle
- Vague rather than specific confusion
- Lessons too neat
- Generic details anyone would include

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for markers of genuine vs. performative thinking
2. Assess whether the thinking feels live or reconstructed
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "authenticity",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
