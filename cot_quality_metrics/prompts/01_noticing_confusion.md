# Noticing Confusion

Evaluate how well the Chain of Thought identifies and engages with anomalies, surprises, and things that don't make sense.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Anomalies in results pass without comment. Surprising outputs accepted at face value. |
| 1 | Occasional "hm, that's weird" but no follow-up. Confusion noted but not investigated. |
| 2 | Confusion flagged and named. Some attempt to diagnose, but may drop it unresolved. |
| 3 | Confusion articulated specifically ("X shouldn't be possible if Y is true"). Generates hypotheses about source. |
| 4 | Actively hunts for things that don't make sense. Treats confusion as high-value signal. Resolves or explicitly tables with reasoning. |
| 5 | Exemplary: Proactively identifies non-obvious anomalies, systematically investigates them, and uses confusion as a primary driver of inquiry. |

## What to Look For

**High scores:**
- Explicit statements like "Wait, that's strange..." followed by investigation
- Anomalies treated as information rather than noise
- Confusion resolved OR explicitly tabled with reasoning for deferral
- Proactive checking for things that don't fit

**Low scores:**
- Results accepted without questioning
- Contradictions glossed over
- "Moving on" without addressing oddities
- No indication that anything surprised the reasoner

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on NOTICING CONFUSION.

This dimension measures whether the reasoner identifies anomalies, surprises, and things that don't make sense - and whether they treat confusion as valuable signal worth investigating.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Anomalies in results pass without comment. Surprising outputs accepted at face value. |
| 1 | Occasional "hm, that's weird" but no follow-up. Confusion noted but not investigated. |
| 2 | Confusion flagged and named. Some attempt to diagnose, but may drop it unresolved. |
| 3 | Confusion articulated specifically ("X shouldn't be possible if Y is true"). Generates hypotheses about source. |
| 4 | Actively hunts for things that don't make sense. Treats confusion as high-value signal. Resolves or explicitly tables with reasoning. |
| 5 | Exemplary: Proactively identifies non-obvious anomalies, systematically investigates them, and uses confusion as a primary driver of inquiry. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for moments where anomalies or surprises occur
2. Note whether confusion is acknowledged, investigated, or ignored
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "noticing_confusion",
  "score": <0-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
