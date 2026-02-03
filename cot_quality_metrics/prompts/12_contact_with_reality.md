# Contact with Reality

Evaluate whether the Chain of Thought is actually constrained by the problem and data, or theorizing unconstrained by evidence.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Theorizing unconstrained by data. Could have written this without running any experiments. |
| 2 | Results mentioned but don't seem to shape the thinking. Conclusions predetermined. |
| 3 | Some responsiveness to results but also some "I'll interpret this to fit my story." |
| 4 | Thinking clearly shaped by what's actually observed. Changes direction when reality pushes back. |
| 5 | Tight feedback loop. Every claim is grounded. Speculations are flagged as such and tested. |

## What to Look For

**High scores:**
- Specific numbers and values, not just "the accuracy was low"
- Surprise from data that shaped subsequent thinking
- Abandoned hypotheses because of evidence, not narrative convenience
- Constraints from observations propagate to next steps
- Grounded speculation: "If X is true, we'd expect Y, let's check"

**Low scores:**
- Reasoning that could have been written without seeing any data
- Results shoehorned into predetermined story
- No visible constraint from observations
- Conclusions that don't reflect what was actually found

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on CONTACT WITH REALITY.

This dimension measures whether the thinking is actually constrained by the problem and data, or whether it's theorizing that ignores or reinterprets evidence to fit a story.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 1 | Theorizing unconstrained by data. Could have written this without running any experiments. |
| 2 | Results mentioned but don't seem to shape the thinking. Conclusions predetermined. |
| 3 | Some responsiveness to results but also some "I'll interpret this to fit my story." |
| 4 | Thinking clearly shaped by what's actually observed. Changes direction when reality pushes back. |
| 5 | Tight feedback loop. Every claim is grounded. Speculations are flagged as such and tested. |

INDICATORS OF GOOD CONTACT:
- Specific numbers and values
- Surprise from data that shaped subsequent thinking
- Hypotheses abandoned because of evidence
- Constraints from observations propagate forward
- Speculation is flagged and then tested

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for how data/results constrain the reasoning
2. Assess whether thinking is responsive to reality or predetermined
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 1-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "contact_with_reality",
  "score": <1-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
