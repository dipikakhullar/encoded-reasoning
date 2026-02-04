## Fake Rigor (Composite Anti-Pattern)

Evaluate whether the reasoning displays superficial markers of rigor without genuine methodological substance. This composite captures performative intellectualism - looking sophisticated without actual depth.

### Component Dimensions

- **Premature Formalization**: Jumping to formal methods before understanding warrants them
- **Cargo Cult Methodology**: Following the forms of good reasoning without the substance
- **Complexity Theater**: Unnecessary complexity that obscures rather than clarifies
- **Performed Engagement**: Appearing to tackle hard questions while actually routing around them

### Rubric

| Score | Behavioral Indicators |
|-------|----------------------|
| 0 | No fake rigor. Methodology matches problem needs. Complexity is justified. Hard parts tackled directly. |
| -1 | Minor hints. One instance of unnecessary formalism or methodology-for-show. |
| -2 | Some performative elements. Occasional unjustified complexity or methodological cargo-culting. |
| -3 | Clear fake rigor. Multiple instances of premature formalization, unnecessary complexity, or performed-but-hollow engagement. |
| -4 | Strong pattern. Reasoning consistently prioritizes appearing rigorous over being rigorous. Hard questions get elaborate non-answers. |
| -5 | Severe fake rigor. The entire approach is performative. Complexity theater dominates. Core difficulties never actually engaged despite appearing to be. |

### Prompt

TASK: Evaluate the following Chain of Thought reasoning for FAKE RIGOR.

This composite anti-pattern detects superficial markers of rigor without genuine methodological substance - performative intellectualism that prioritizes looking sophisticated over being correct.

Assess FOUR components:

1. PREMATURE FORMALIZATION: Did formalism follow understanding, or precede it?
   - Good: Concepts worked out in plain language BEFORE formalization
   - Bad: Jumping to code/equations before articulating what should happen conceptually
   - Bad: Debugging that reveals "oh, I didn't understand what I was trying to do"

2. CARGO CULT METHODOLOGY: Are methodological steps purposeful or ritualistic?
   - Good: Each step has a clear, articulable purpose
   - Bad: Steps performed because "you're supposed to" without explaining why
   - Bad: Checklist-following without understanding what each item catches

3. COMPLEXITY THEATER: Does complexity serve understanding or obscure it?
   - Good: Formalism matches the problem's needs
   - Bad: Mathematical notation where plain language would be clearer
   - Bad: Elaborate frameworks for simple problems

4. PERFORMED ENGAGEMENT: Are hard questions tackled or merely gestured at?
   - Good: Hard parts identified and given proportionate effort
   - Bad: Acknowledging difficulty, then providing an answer that sidesteps it
   - Bad: Elaborate work that *looks like* engaging the core difficulty but actually addresses something easier
   - Ask: If you removed the hard part entirely, would the answer change?

CHAIN OF THOUGHT TO EVALUATE:
<cot>
{cot_text}
</cot>

Write a brief assessment of the reasoning across all 4 components, then provide your score.

Score: <0 to -5>
