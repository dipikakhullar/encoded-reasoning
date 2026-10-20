## Active Investigation (Composite Positive)

Evaluate whether the reasoning actively engages with reality through hypothesis testing, error correction, and empirical contact - rather than purely abstract manipulation.

### Component Dimensions

- **Discriminative Experiment Design**: Constructing tests that can distinguish between hypotheses
- **Error Metabolism**: Productively processing mistakes into improved understanding
- **Contact with Reality**: Grounding abstract reasoning in concrete observations and tests

### Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No investigation despite needing it. Pure armchair reasoning on empirical questions. No hypothesis testing or reality contact. |
| 1 | Minimal investigation. Occasional concrete example but no systematic testing. Errors not leveraged. |
| 2 | Some investigation. A few hypothesis tests or reality checks. Some learning from mistakes. |
| 3 | Good investigation. Hypotheses tested with discriminative examples. Errors processed productively. Regular reality contact. |
| 4 | Strong investigation. Systematic hypothesis testing. Errors actively sought and leveraged. Strong empirical grounding. |
| 5 | Exemplary. Investigation drives the reasoning. Discriminative tests designed proactively. Errors treated as valuable data. Deep contact with reality throughout. |

### Scoring Guidance

**Scale to problem demands.** A problem requiring no investigation (pure logic, straightforward application) can still score well if the reasoning demonstrates *investigative readiness*: falsifiable claims, checkable intermediate steps, openness to disconfirmation. Score 0 is reserved for reasoning that *needed* investigation but didn't do it.

### Prompt

TASK: Evaluate the following Chain of Thought reasoning for ACTIVE INVESTIGATION.

This composite measures whether reasoning actively engages with reality through hypothesis testing, error correction, and empirical grounding.

**Important**: Scale your evaluation to what the problem demanded. A straightforward problem may not require extensive investigation - in such cases, look for *investigative readiness* (falsifiable claims, checkable steps) rather than performed investigation.

Assess THREE components:

1. DISCRIMINATIVE EXPERIMENT DESIGN: Are tests designed to distinguish between hypotheses?
   - High: "If X is true, we'd expect Y; if Z is true, we'd expect W"
   - High: Identifies cruxes - the minimal observation that would change everything
   - Low: Tests that would pass regardless of which hypothesis is true
   - Low: No consideration of alternative explanations
   - If problem is simple: Are claims stated in falsifiable form?

2. ERROR METABOLISM: How are errors caught and processed?
   - High: Proactive sanity checks ("Let me verify this...")
   - High: When errors found: root cause analysis, generalized lessons
   - Low: Same mistakes repeated without learning
   - Low: Patches without understanding why it failed
   - If no errors occurred: Were there checkpoints where errors *could* have been caught?

3. CONTACT WITH REALITY: Is reasoning constrained by data?
   - High: Specific numbers/values cited, surprise from data shapes thinking
   - High: Hypotheses abandoned because of evidence
   - Low: Reasoning that could have been written without seeing any data
   - Low: Results shoehorned into predetermined story
   - If problem is abstract: Are concrete examples used to test intuitions?

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

Write a brief assessment of the reasoning across all 3 components, calibrating to problem demands. Then provide your score.

Score: <0-5>
