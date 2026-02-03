# Error Metabolism

Evaluate how well the Chain of Thought catches, diagnoses, and learns from errors.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Errors discovered late by accident, if at all. Same class of error recurring. |
| 1 | Errors caught but slowly. Patches applied without root cause diagnosis. |
| 2 | Some self-checking present. Errors diagnosed but fix may be shallow. |
| 3 | Active error-checking behaviors. Root cause identified. Learns from mistakes. |
| 4 | Builds in sanity checks proactively. Fast error detection. Generalizes lessons ("I should always check X"). |
| 5 | Exemplary: Anticipates error-prone areas, builds systematic verification, and uses errors as catalysts for improving the entire reasoning process. |

## What to Look For

**High scores:**
- Proactive sanity checks built into reasoning
- Quick error detection
- Root cause analysis when errors found
- Generalized lessons: "I should always check X in future"
- Prevention of error recurrence

**Low scores:**
- Same mistakes repeated
- Errors only found by accident
- Patches without understanding
- No verification steps
- No learning from mistakes

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on ERROR METABOLISM.

This dimension measures how quickly and thoroughly errors are caught, diagnosed, and learned from.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Errors discovered late by accident, if at all. Same class of error recurring. |
| 1 | Errors caught but slowly. Patches applied without root cause diagnosis. |
| 2 | Some self-checking present. Errors diagnosed but fix may be shallow. |
| 3 | Active error-checking behaviors. Root cause identified. Learns from mistakes. |
| 4 | Builds in sanity checks proactively. Fast error detection. Generalizes lessons ("I should always check X"). |
| 5 | Exemplary: Anticipates error-prone areas, builds systematic verification, and uses errors as catalysts for improving the entire reasoning process. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for errors, mistakes, or corrections
2. Assess how errors were caught, diagnosed, and what was learned
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0-5 based on the rubric

Note: If no errors occurred in the CoT, assess based on presence of proactive sanity checks and verification behaviors.

Respond with JSON in this exact format:
```json
{
  "dimension": "error_metabolism",
  "score": <0-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
