# Cargo Cult Methodology (Anti-Pattern)

Detect going through methodological motions without understanding why those steps matter.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Methods used with understanding. Each step serves a purpose the reasoner could articulate. |
| -1 | Minor cargo culting. One or two steps included without clear purpose. |
| -2 | Moderate cargo culting. Some methodological steps that seem ritualistic rather than purposeful. |
| -3 | Clear cargo cult methodology. Following a procedure without understanding why. Steps don't connect to goals. |
| -4 | Strong cargo culting. Method appears copied from elsewhere without adaptation. Ritual without understanding. |
| -5 | Severe cargo cult methodology. Complete disconnect between process and purpose. Going through elaborate motions that don't serve the actual problem. |

## What to Look For

**Signs of cargo cult methodology:**
- Steps performed because "you're supposed to" without explanation of why
- Methods borrowed from other contexts without adapting to the current problem
- Checklist-following without understanding what each item catches
- Ritual adherence to procedure over responsive problem-solving
- Can't explain why a step matters if asked
- Methods that don't actually serve the problem at hand

**Signs of absence (score 0):**
- Each step has a clear purpose
- Methods adapted to the specific problem
- Understanding of what each check is for
- Responsive methodology rather than ritual

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for CARGO CULT METHODOLOGY.

This anti-pattern detects going through methodological motions without understanding why those steps matter.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Methods used with understanding. Each step serves a purpose the reasoner could articulate. |
| -1 | Minor cargo culting. One or two steps included without clear purpose. |
| -2 | Moderate cargo culting. Some methodological steps that seem ritualistic rather than purposeful. |
| -3 | Clear cargo cult methodology. Following a procedure without understanding why. Steps don't connect to goals. |
| -4 | Strong cargo culting. Method appears copied from elsewhere without adaptation. Ritual without understanding. |
| -5 | Severe cargo cult methodology. Complete disconnect between process and purpose. Going through elaborate motions that don't serve the actual problem. |

SIGNS OF CARGO CULT METHODOLOGY:
- Steps performed because "you're supposed to"
- Methods borrowed without adapting to current problem
- Checklist-following without understanding
- Ritual adherence over responsive problem-solving
- Methods that don't actually serve the problem at hand

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, assessing whether methodology is purposeful
2. Note steps that seem ritualistic rather than reasoned
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "cargo_cult_methodology",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
