# Discriminative Experiment Design

Evaluate how well the Chain of Thought designs tests that distinguish between competing hypotheses.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Tests only confirm what's already believed. No consideration of alternative hypotheses. |
| 2 | Experiments run, but would look similar under multiple theories. Low information value. |
| 3 | Some experiments are discriminative, others are busywork. Mixed intentionality. |
| 4 | Explicitly designs tests to distinguish hypotheses. Asks "what would falsify this?" |
| 5 | Identifies cruxes — the minimal observation that would change everything. Prioritizes high-information experiments. |

## What to Look For

**High scores:**
- Explicit consideration of what different outcomes would mean
- Questions like "If X is true, we'd expect Y; if Z is true, we'd expect W"
- Identification of crux observations that would resolve uncertainty
- Prioritizing tests by information value

**Low scores:**
- Tests that would pass regardless of which hypothesis is true
- No consideration of alternative explanations
- Running tests without clear decision criteria
- Confirmation-seeking rather than truth-seeking

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on DISCRIMINATIVE EXPERIMENT DESIGN.

This dimension measures whether tests/experiments/checks are designed to actually distinguish between competing hypotheses, or merely confirm existing beliefs.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Tests only confirm what's already believed. No consideration of alternative hypotheses. |
| 2 | Experiments run, but would look similar under multiple theories. Low information value. |
| 3 | Some experiments are discriminative, others are busywork. Mixed intentionality. |
| 4 | Explicitly designs tests to distinguish hypotheses. Asks "what would falsify this?" |
| 5 | Identifies cruxes — the minimal observation that would change everything. Prioritizes high-information experiments. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for tests, experiments, or verification steps
2. Assess whether tests would distinguish between alternatives or just confirm beliefs
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "discriminative_experiment_design",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
