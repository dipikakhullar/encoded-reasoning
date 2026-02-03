# Intellectual Flinching (Anti-Pattern)

Detect avoiding the hard questions and staying in comfortable territory.

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Hard questions tackled directly. Discomfort doesn't prevent engagement. |
| -1 | Minor flinching. One hard question slightly avoided but overall good engagement. |
| -2 | Moderate flinching. Some hard questions addressed superficially or deflected. |
| -3 | Clear intellectual flinching. Core difficulties acknowledged but not engaged. Staying on comfortable ground. |
| -4 | Strong flinching. Systematic avoidance of the hardest parts. Elaborate work on easy aspects while hard parts are hand-waved. |
| -5 | Severe intellectual flinching. The central difficulty is never confronted. Massive effort expended to avoid the actual hard problem. |

## What to Look For

**Signs of intellectual flinching:**
- Hard questions acknowledged but then not engaged
- Disproportionate effort on easy parts vs. hard parts
- Deflection when approaching difficult territory
- Staying in comfortable zones of the problem
- Hand-waving over cruxes
- Elaborate work that avoids the core difficulty

**Signs of absence (score 0):**
- Direct engagement with hard questions
- Proportionate effort on difficult aspects
- Willingness to sit with uncomfortable uncertainty
- Tackling the crux rather than working around it

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for INTELLECTUAL FLINCHING.

This anti-pattern detects avoiding the hard questions and staying in comfortable territory.

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | Hard questions tackled directly. Discomfort doesn't prevent engagement. |
| -1 | Minor flinching. One hard question slightly avoided but overall good engagement. |
| -2 | Moderate flinching. Some hard questions addressed superficially or deflected. |
| -3 | Clear intellectual flinching. Core difficulties acknowledged but not engaged. Staying on comfortable ground. |
| -4 | Strong flinching. Systematic avoidance of the hardest parts. Elaborate work on easy aspects while hard parts are hand-waved. |
| -5 | Severe intellectual flinching. The central difficulty is never confronted. Massive effort expended to avoid the actual hard problem. |

SIGNS OF INTELLECTUAL FLINCHING:
- Hard questions acknowledged but not engaged
- Disproportionate effort on easy vs. hard parts
- Deflection when approaching difficult territory
- Hand-waving over cruxes
- Elaborate work that avoids core difficulty

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. Read the CoT carefully, noting how hard questions are handled
2. Assess whether difficult aspects are tackled or avoided
3. Identify specific evidence (quotes or observations) supporting your assessment
4. Assign a score from 0 (absent) to -5 (severely present) based on the rubric

Respond with JSON in this exact format:
```json
{
  "dimension": "intellectual_flinching",
  "score": <0 to -5>,
  "evidence": ["quote or observation 1", "quote or observation 2"],
  "reasoning": "Brief explanation of why this score"
}
```
