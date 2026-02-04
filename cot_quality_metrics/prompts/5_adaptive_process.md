## Adaptive Process (Composite Positive)

Evaluate whether the reasoning displays dynamic, responsive thinking - adapting to obstacles, updating on new information, and generating novel approaches when stuck.

### Component Dimensions

- **Generativity Under Stuckness**: Ability to generate new approaches when blocked
- **Noticing Confusion**: Identifying and engaging with anomalies and surprises
- **Live Updating**: Adjusting beliefs and approaches in response to new information

### Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No adaptation despite needing it. Rigid approach maintained through failure. Confusion ignored. No updating. |
| 1 | Minimal adaptation. May notice being stuck but doesn't generate real alternatives. Rare updating. |
| 2 | Some adaptation. Occasionally tries new approach. Some response to surprising information. |
| 3 | Good adaptive process. Generates genuine alternatives when stuck. Notices and investigates confusion. Updates on new information. |
| 4 | Strong adaptation. Multiple informed strategies attempted. Confusion treated as signal. Rapid, appropriate belief updating. |
| 5 | Exemplary. Highly generative when blocked, with pivots informed by prior failures. Proactively hunts for anomalies. Continuous live updating. Treats obstacles as information. |

### Scoring Guidance

**Scale to problem demands.** A problem that doesn't require adaptation (straightforward path to solution) can still score well if the reasoning demonstrates *adaptive readiness*: beliefs held provisionally, anomalies would be noticed if they appeared, path not over-committed. Score 0 is reserved for reasoning that *needed* to adapt but didn't.

**Quality over quantity of pivots.** Scatter-shot pivoting (random new attempts) scores lower than informed pivoting (new approaches shaped by what was learned from failures).

### Prompt

TASK: Evaluate the following Chain of Thought reasoning for ADAPTIVE PROCESS.

This composite measures dynamic, responsive thinking - the ability to adapt to obstacles, update on new information, and generate novel approaches when stuck.

**Important**: Scale your evaluation to what the problem demanded. A straightforward problem may not require adaptation - in such cases, look for *adaptive readiness* (provisional commitment, anomaly-sensitivity) rather than performed pivoting.

Assess THREE components:

1. GENERATIVITY UNDER STUCKNESS: When stuck, are genuinely different approaches tried?
   - High: Multiple genuinely different approaches (not just parameter tweaks)
   - High: Reframing the problem when initial framing doesn't work
   - High: New attempts *informed by* why previous attempts failed
   - Low: Repeating the same failing approach multiple times
   - Low: Scatter-shot pivoting - random new attempts with no learning
   - Low: Giving up without exploring alternatives
   - If not stuck: Were there decision points where alternatives were available?

2. NOTICING CONFUSION: Are anomalies spotted and investigated?
   - High: "Wait, that's strange..." followed by investigation
   - High: Anomalies treated as information rather than noise
   - Low: Contradictions glossed over
   - Low: "Hm, that's weird" then immediately moving on
   - If no anomalies: Is there sensitivity to anomalies - would they be noticed if present?

3. LIVE UPDATING: Do beliefs change on evidence?
   - High: Explicit "I thought X, but now I see Y"
   - High: Visible changes in direction based on what was learned
   - Low: Original hypothesis maintained despite contrary evidence
   - Low: Evidence acknowledged but ignored in conclusions
   - If no updating needed: Are beliefs held provisionally, open to revision?

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

Write a brief assessment of the reasoning across all 3 components, calibrating to problem demands. Then provide your score.

Score: <0-5>
