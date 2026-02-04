# Fake Rigor (Composite Anti-Pattern)

Evaluate whether the reasoning displays superficial markers of rigor without genuine methodological substance. This composite captures performative intellectualism - looking sophisticated without actual depth.

## Component Dimensions

This composite integrates signals from:
- **Premature Formalization**: Jumping to formal methods before understanding warrants them
- **Cargo Cult Methodology**: Following the forms of good reasoning without the substance
- **Complexity Theater**: Unnecessary complexity that obscures rather than clarifies
- **Intellectual Flinching**: Backing away from difficult implications or conclusions

## Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No fake rigor. Methodology matches problem needs. Complexity is justified. Difficult implications faced. |
| -1 | Minor hints. One instance of unnecessary formalism or methodology-for-show. |
| -2 | Some performative elements. Occasional unjustified complexity or methodological cargo-culting. |
| -3 | Clear fake rigor. Multiple instances of premature formalization, unnecessary complexity, or flinching from implications. |
| -4 | Strong pattern. Reasoning consistently prioritizes appearing rigorous over being rigorous. Systematic avoidance of hard questions. |
| -5 | Severe fake rigor. The entire approach is performative. Complexity theater dominates. Genuine engagement absent. |

## Prompt

TASK: Evaluate the following Chain of Thought reasoning for FAKE RIGOR.

This composite anti-pattern detects superficial markers of rigor without genuine methodological substance - performative intellectualism that prioritizes looking sophisticated over being correct.

You will assess FOUR specific components and then give a holistic score.

COMPONENT 1 - PREMATURE FORMALIZATION (look for sequence of understanding vs formalism):
✓ Good: Concepts worked out in plain language BEFORE formalization
✓ Good: Formalization that encodes already-clear ideas
✓ Good: Formal structures that clarify rather than constrain thinking
✗ Bad: Jumping to code/equations before articulating what should happen conceptually
✗ Bad: Having to significantly revise formal work due to conceptual confusion discovered later
✗ Bad: Debugging that reveals "oh, I didn't understand what I was trying to do"
✗ Bad: Elaborate formal structures built on shaky conceptual foundations
✗ Bad: Formalism getting in the way of clear thinking

COMPONENT 2 - CARGO CULT METHODOLOGY (look for purpose behind steps):
✓ Good: Each methodological step has a clear, articulable purpose
✓ Good: Methods adapted to the specific problem at hand
✓ Good: Understanding of what each check or step catches
✗ Bad: Steps performed because "you're supposed to" without explaining why
✗ Bad: Methods borrowed from other contexts without adapting to current problem
✗ Bad: Checklist-following without understanding what each item catches
✗ Bad: Ritual adherence to procedure over responsive problem-solving
✗ Bad: Methods that don't actually serve the problem at hand

COMPONENT 3 - COMPLEXITY THEATER (look for complexity vs clarity tradeoff):
✓ Good: Formalism matches the problem's needs
✓ Good: Notation introduced because it actually helps
✓ Good: Plain language used where plain language works
✗ Bad: Mathematical notation where plain language would be clearer
✗ Bad: Jargon that obscures rather than sharpens meaning
✗ Bad: Elaborate frameworks for simple problems
✗ Bad: Multiple layers of abstraction that don't add insight
✗ Bad: Formalism that looks impressive but doesn't do actual work
✗ Bad: Complexity that seems designed to signal sophistication

COMPONENT 4 - INTELLECTUAL FLINCHING (look for engagement with difficulty):
✓ Good: Hard questions tackled directly
✓ Good: Proportionate effort on difficult aspects
✓ Good: Willingness to sit with uncomfortable uncertainty
✓ Good: Tackling the crux rather than working around it
✗ Bad: Hard questions acknowledged but then not actually engaged
✗ Bad: Disproportionate effort on easy parts while hard parts are hand-waved
✗ Bad: Deflection when approaching difficult territory
✗ Bad: Staying in comfortable zones of the problem
✗ Bad: Elaborate work that cleverly avoids the core difficulty

RUBRIC:
| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | All 4 components look good: appropriate formalization timing, purposeful methods, justified complexity, hard questions faced |
| -1 | Minor issues in 1 component. Overall genuine rigor. |
| -2 | Clear issues in 1-2 components. Some performative elements. |
| -3 | Problems in 3+ components: premature formalization + cargo culting + complexity theater + flinching |
| -4 | Strong pattern across all components. Reasoning prioritizes appearing rigorous over being rigorous. |
| -5 | Severe: The entire approach is performative. Elaborate show with no substance. |

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

INSTRUCTIONS:
1. First, assess PREMATURE FORMALIZATION: Did formalism follow understanding, or precede it?
2. Then, assess CARGO CULT METHODOLOGY: Are methodological steps purposeful or ritualistic?
3. Then, assess COMPLEXITY THEATER: Does complexity serve understanding or obscure it?
4. Finally, assess INTELLECTUAL FLINCHING: Are hard questions tackled or avoided?
5. Based on issues across all 4 components, assign a holistic score from 0 to -5

Respond with JSON in this exact format:
```json
{
  "dimension": "fake_rigor",
  "evidence": [
    "Premature formalization: <quote or observation>",
    "Cargo cult: <quote or observation>",
    "Complexity theater: <quote or observation>",
    "Intellectual flinching: <quote or observation>"
  ],
  "reasoning": "Brief explanation synthesizing all 4 components",
  "score": <0 to -5>
}
```
