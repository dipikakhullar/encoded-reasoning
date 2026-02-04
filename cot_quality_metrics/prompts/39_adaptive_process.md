# Adaptive Process (Composite Positive)

Evaluate whether the reasoning displays dynamic, responsive thinking - adapting to obstacles, updating on new information, and generating novel approaches when stuck.

## Component Dimensions

This composite integrates signals from:
- **Generativity Under Stuckness**: Ability to generate new approaches when blocked
- **Noticing Confusion**: Identifying and engaging with anomalies and surprises
- **Live Updating**: Adjusting beliefs and approaches in response to new information

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No adaptive process. Rigid approach. Confusion ignored. No updating on new information. |
| 1 | Minimal adaptation. May notice being stuck but doesn't generate alternatives. Rare updating. |
| 2 | Some adaptation. Occasionally tries new approach. Some response to surprising information. |
| 3 | Good adaptive process. Generates alternatives when stuck. Notices and investigates confusion. Updates on new information. |
| 4 | Strong adaptation. Multiple strategies attempted. Confusion treated as signal. Rapid, appropriate belief updating. |
| 5 | Exemplary: Highly generative when blocked. Proactively hunts for anomalies. Continuous live updating. Treats obstacles as information. |

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for ADAPTIVE PROCESS.

This composite positive trait measures dynamic, responsive thinking - the ability to adapt to obstacles, update on new information, and generate novel approaches when stuck.

You will assess THREE specific components and then give a holistic score.

COMPONENT 1 - GENERATIVITY UNDER STUCKNESS (look for alternatives when blocked):
✓ High (good): Multiple genuinely different approaches when stuck (not just parameter tweaks)
✓ High (good): "That's not working, let me try a completely different approach..."
✓ High (good): Reframing the problem itself when initial framing doesn't yield progress
✓ High (good): Drawing analogies to other domains
✓ High (good): Questioning foundational assumptions when stuck
✓ High (good): Creativity under constraint - transforms obstacles into opportunities
✗ Low (bad): Repeating the same failing approach multiple times
✗ Low (bad): Only superficial variations (different parameters, same method)
✗ Low (bad): Giving up without exploring alternatives
✗ Low (bad): No visible attempt to reconceptualize when stuck
✗ Low (bad): "Pushing through" rather than stepping back to try something new

COMPONENT 2 - NOTICING CONFUSION (look for anomaly detection and investigation):
✓ High (good): Explicit "Wait, that's strange..." followed by investigation
✓ High (good): "X shouldn't be possible if Y is true" - articulates the contradiction
✓ High (good): Anomalies treated as information rather than noise
✓ High (good): Confusion resolved OR explicitly tabled with reasoning for deferral
✓ High (good): Proactive checking for things that don't fit
✓ High (good): Uses confusion as a driver of inquiry - goes toward the weird thing
✗ Low (bad): Anomalies pass without comment
✗ Low (bad): "Hm, that's weird" then immediately moving on without investigation
✗ Low (bad): Contradictions glossed over
✗ Low (bad): No indication that anything surprised the reasoner
✗ Low (bad): Treating confusion as noise to be ignored rather than signal

COMPONENT 3 - LIVE UPDATING (look for belief revision on evidence):
✓ High (good): Explicit belief changes: "I thought X, but now I see Y"
✓ High (good): Evidence acknowledged AND integrated into subsequent reasoning
✓ High (good): Downstream implications traced when beliefs change
✓ High (good): Active search for disconfirming evidence
✓ High (good): Visible changes in direction based on what was learned
✗ Low (bad): Original hypothesis maintained despite contrary evidence
✗ Low (bad): "Explaining away" inconvenient results
✗ Low (bad): "Interesting, anyway..." - acknowledging but ignoring
✗ Low (bad): Evidence acknowledged but ignored in conclusions
✗ Low (bad): No visible change in direction throughout despite new information

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | None of the 3 components present. Rigid, ignores confusion, no updating. |
| 1 | Weak signal in 1 component only. Minimal adaptation overall. |
| 2 | Moderate signal in 1-2 components. Some adaptation but inconsistent. |
| 3 | Good signal in all 3: generates alternatives + notices confusion + updates beliefs |
| 4 | Strong signal in all 3. Highly responsive and adaptive throughout. |
| 5 | Exemplary in all 3. Adaptation drives the entire reasoning process. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. First, assess GENERATIVITY UNDER STUCKNESS: When stuck, are genuinely different approaches tried? Look for reframing, reconceptualization, creative alternatives.
2. Then, assess NOTICING CONFUSION: Are anomalies spotted and investigated? Look for "wait, that's strange" followed by engagement.
3. Then, assess LIVE UPDATING: Do beliefs change on evidence? Look for explicit "I thought X, now Y" and direction changes.
4. Based on strength across all 3 components, assign a holistic score from 0-5

Respond with JSON in this exact format:
```json
{
  "dimension": "adaptive_process",
  "evidence": [
    "Generativity: <quote or observation>",
    "Noticing confusion: <quote or observation>",
    "Live updating: <quote or observation>"
  ],
  "reasoning": "Brief explanation synthesizing all 3 components",
  "score": <0-5>
}
```
