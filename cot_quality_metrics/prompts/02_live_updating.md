# Live Updating

Evaluate how well the Chain of Thought updates beliefs in response to new evidence.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Beliefs static despite evidence. Disconfirming results ignored or explained away. |
| 2 | Acknowledges contradictory evidence but doesn't change course. "Interesting, anyway..." |
| 3 | Updates on strong evidence but slow to do so. May hold onto original framing too long. |
| 4 | Visibly changes beliefs when evidence warrants. Articulates what changed and why. |
| 5 | Updates propagate — when one thing changes, revisits downstream conclusions. Actively seeks disconfirmation. |

## What to Look For

**High scores:**
- Explicit belief changes: "I thought X, but now I see Y"
- Evidence acknowledged AND integrated into subsequent reasoning
- Downstream implications traced when beliefs change
- Active search for disconfirming evidence

**Low scores:**
- Original hypothesis maintained despite contrary evidence
- "Explaining away" inconvenient results
- Evidence acknowledged but ignored in conclusions
- No visible change in direction throughout

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on LIVE UPDATING.

This dimension measures whether beliefs change appropriately in response to evidence, and whether updates propagate through to downstream conclusions.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Beliefs static despite evidence. Disconfirming results ignored or explained away. |
| 2 | Acknowledges contradictory evidence but doesn't change course. "Interesting, anyway..." |
| 3 | Updates on strong evidence but slow to do so. May hold onto original framing too long. |
| 4 | Visibly changes beliefs when evidence warrants. Articulates what changed and why. |
| 5 | Updates propagate — when one thing changes, revisits downstream conclusions. Actively seeks disconfirmation. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, tracking how beliefs/hypotheses evolve
2. Note whether evidence leads to actual changes in reasoning direction
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "live_updating",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
