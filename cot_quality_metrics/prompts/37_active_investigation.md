# Active Investigation (Composite Positive)

Evaluate whether the reasoning actively engages with reality through hypothesis testing, error correction, and empirical contact - rather than purely abstract manipulation.

## Component Dimensions

This composite integrates signals from:
- **Discriminative Experiment Design**: Constructing tests that can distinguish between hypotheses
- **Error Metabolism**: Productively processing mistakes into improved understanding
- **Contact with Reality**: Grounding abstract reasoning in concrete observations and tests

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No active investigation. Pure armchair reasoning. No hypothesis testing or reality contact. |
| 1 | Minimal investigation. Occasional concrete example but no systematic testing. Errors not leveraged. |
| 2 | Some investigation. A few hypothesis tests or reality checks. Some learning from mistakes. |
| 3 | Good investigation. Hypotheses tested with discriminative examples. Errors processed productively. Regular reality contact. |
| 4 | Strong investigation. Systematic hypothesis testing. Errors actively sought and leveraged. Strong empirical grounding. |
| 5 | Exemplary: Investigation drives the reasoning. Discriminative tests designed proactively. Errors treated as valuable data. Deep contact with reality throughout. |

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for ACTIVE INVESTIGATION.

This composite positive trait measures whether reasoning actively engages with reality through hypothesis testing, error correction, and empirical grounding - rather than pure abstract manipulation.

You will assess THREE specific components and then give a holistic score.

COMPONENT 1 - DISCRIMINATIVE EXPERIMENT DESIGN (look for hypothesis-testing):
✓ High (good): "If X is true, we'd expect Y; if Z is true, we'd expect W" - explicitly distinguishing hypotheses
✓ High (good): "What would falsify this?" or "Let me check if alternative A or B explains this better"
✓ High (good): Identifies cruxes - the minimal observation that would change everything
✓ High (good): Prioritizes tests by information value
✗ Low (bad): Tests that would pass regardless of which hypothesis is true
✗ Low (bad): No consideration of alternative explanations
✗ Low (bad): Confirmation-seeking rather than truth-seeking

COMPONENT 2 - ERROR METABOLISM (look for error handling):
✓ High (good): Proactive sanity checks built into reasoning ("Let me verify this...")
✓ High (good): When errors found: root cause analysis, not just patches
✓ High (good): Generalizes lessons: "I should always check X" or "This pattern tends to fail when..."
✓ High (good): Uses errors as information to improve approach
✗ Low (bad): Same mistakes repeated without learning
✗ Low (bad): Errors only found by accident, if at all
✗ Low (bad): Patches without understanding why it failed
✗ Low (bad): No verification steps

COMPONENT 3 - CONTACT WITH REALITY (look for empirical grounding):
✓ High (good): Specific numbers/values cited, not just "the accuracy was low"
✓ High (good): Surprise from data that visibly shapes subsequent thinking
✓ High (good): Hypotheses abandoned because of evidence (not narrative convenience)
✓ High (good): "Let me check this against the actual data..." then reasoning changes based on result
✗ Low (bad): Reasoning that could have been written without seeing any data
✗ Low (bad): Results shoehorned into predetermined story
✗ Low (bad): No visible constraint from observations
✗ Low (bad): Conclusions that don't reflect what was actually found

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | None of the 3 components present. Pure armchair reasoning. |
| 1 | Weak signal in 1 component only. Minimal investigation overall. |
| 2 | Moderate signal in 1-2 components. Some investigation but inconsistent. |
| 3 | Good signal in all 3: discriminative tests + error learning + reality contact |
| 4 | Strong signal in all 3. Systematic and thorough investigation. |
| 5 | Exemplary in all 3. Investigation drives the entire reasoning process. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. First, assess DISCRIMINATIVE EXPERIMENT DESIGN: Are tests designed to distinguish between hypotheses? Look for "if X then Y, if Z then W" reasoning.
2. Then, assess ERROR METABOLISM: How are errors caught and processed? Look for sanity checks, root cause analysis, generalized lessons.
3. Then, assess CONTACT WITH REALITY: Is reasoning constrained by data? Look for specific values, surprise from results, changed direction based on evidence.
4. Based on strength across all 3 components, assign a holistic score from 0-5

Respond with JSON in this exact format:
```json
{
  "dimension": "active_investigation",
  "evidence": [
    "Discriminative design: <quote or observation>",
    "Error metabolism: <quote or observation>",
    "Contact with reality: <quote or observation>"
  ],
  "reasoning": "Brief explanation synthesizing all 3 components",
  "score": <0-5>
}
```
