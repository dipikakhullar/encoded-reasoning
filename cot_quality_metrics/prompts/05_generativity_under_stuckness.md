# Generativity Under Stuckness

Evaluate how well the Chain of Thought generates alternatives when the initial approach isn't working.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | When stuck, repeats same approach or gives up. No visible alternative generation. |
| 2 | Tries minor variations (tweak parameters) but not genuinely different approaches. |
| 3 | Generates alternatives but they're shallow or obvious. Some reframing attempts. |
| 4 | Produces multiple substantively different hypotheses or approaches. Willing to reconceptualize. |
| 5 | Stuckness triggers creative expansion — new framings, analogies to other domains, questioning premises. Generative even when frustrated. |

## What to Look For

**High scores:**
- Multiple genuinely different approaches when stuck
- Reframing the problem itself
- Drawing analogies to other domains
- Questioning foundational assumptions
- Creativity under constraint

**Low scores:**
- Repeating the same failing approach
- Only superficial variations (different parameters, same method)
- Giving up without exploring alternatives
- No visible attempt to reconceptualize

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on GENERATIVITY UNDER STUCKNESS.

This dimension measures whether the reasoner generates genuinely different alternatives when stuck, or merely repeats variations of the same approach.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | When stuck, repeats same approach or gives up. No visible alternative generation. |
| 2 | Tries minor variations (tweak parameters) but not genuinely different approaches. |
| 3 | Generates alternatives but they're shallow or obvious. Some reframing attempts. |
| 4 | Produces multiple substantively different hypotheses or approaches. Willing to reconceptualize. |
| 5 | Stuckness triggers creative expansion — new framings, analogies to other domains, questioning premises. Generative even when frustrated. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for moments of difficulty or being stuck
2. Assess what alternatives are generated and how different they truly are
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Note: If the CoT shows no signs of being stuck, score based on whether multiple approaches were considered even without difficulty, or note insufficient evidence.

Respond with JSON in this exact format:
```json
{
  "dimension": "generativity_under_stuckness",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
