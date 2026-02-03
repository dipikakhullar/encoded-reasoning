# Problem Decomposition

Evaluate how well the Chain of Thought breaks down complex problems into manageable sub-problems.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Attacks problem as monolith. No visible breakdown into sub-questions. |
| 1 | Some implicit decomposition but not articulated. Jumps between levels without clarity. |
| 2 | Breaks problem into parts but may not prioritize correctly. Decomposition is flat. |
| 3 | Clear sub-problems identified. Knows which are load-bearing. Works hierarchically. |
| 4 | Decomposition is strategic — identifies bottlenecks, dependencies, what must be true for the rest to matter. Revises decomposition as understanding grows. |
| 5 | Exemplary: Masterful decomposition that reveals hidden structure, identifies non-obvious dependencies, and dynamically restructures as new information emerges. |

## What to Look For

**High scores:**
- Explicit identification of sub-problems
- Understanding of dependencies between parts
- Prioritization of load-bearing sub-problems
- Hierarchical rather than flat decomposition
- Decomposition revised as understanding grows

**Low scores:**
- Problem tackled as undifferentiated whole
- Jumping between levels without structure
- No articulation of what depends on what
- Flat list of considerations rather than hierarchy

## Prompt

TASK: Evaluate the following Chain of Thought reasoning on PROBLEM DECOMPOSITION.

This dimension measures how well complex problems are broken into manageable sub-problems with clear dependencies and priorities.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Attacks problem as monolith. No visible breakdown into sub-questions. |
| 1 | Some implicit decomposition but not articulated. Jumps between levels without clarity. |
| 2 | Breaks problem into parts but may not prioritize correctly. Decomposition is flat. |
| 3 | Clear sub-problems identified. Knows which are load-bearing. Works hierarchically. |
| 4 | Decomposition is strategic — identifies bottlenecks, dependencies, what must be true for the rest to matter. Revises decomposition as understanding grows. |
| 5 | Exemplary: Masterful decomposition that reveals hidden structure, identifies non-obvious dependencies, and dynamically restructures as new information emerges. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, looking for how the problem is broken down
2. Assess the quality of decomposition and understanding of dependencies
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0-5 based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "problem_decomposition",
  "score": <0-5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
