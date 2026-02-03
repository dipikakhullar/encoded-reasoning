# Converting cot_quality_metrics to inspect-ai Framework

## Current State

### Existing Implementation
- **32 rubric prompts** in `prompts/` directory (markdown files)
  - 15 positive rubrics (scored 1-5, higher = better epistemic virtues)
  - 17 negative rubrics (scored 0 to -5, 0 = best, no anti-patterns)
- **Custom evaluation logic** in `src/cot_quality_metrics/evaluate.py`
  - Provider abstraction (Anthropic, Gemini)
  - Parallel evaluation with ThreadPoolExecutor
  - JSON output parsing
- **Schemas** in `schemas.py`: RubricInfo, EvaluationResult, CoTEvaluation

### HLE Data Format (in `hle/hle/`)
Organized by model: `anthropic-claude-opus-4.5/`, `anthropic-claude-haiku-4.5/`, etc.

Each JSON file contains:
```json
{
  "sample_id": "66fde2de0b21f205c121aca7",
  "question": "...",
  "ground_truth": "G",
  "n": 4,
  "rollouts": [
    {
      "request": { "model": "...", "messages": [...], ... },
      "response": {
        "choices": [{ "message": { "content": "COT_TEXT_HERE" } }],
        "usage": { ... }
      }
    },
    // ... more rollouts
  ]
}
```

Each rollout has a full CoT response that can be evaluated on quality metrics.

---

## inspect-ai Framework Overview

### Core Concepts
1. **Task**: Bundles dataset + solver + scorer(s)
2. **Sample**: Single evaluation instance with `input` and `target`
3. **Solver**: Strategy to generate responses (we might skip this since we have pre-generated CoTs)
4. **Scorer**: Evaluates output against target/criteria

### Key APIs for Our Use Case

#### Custom Scorer Pattern
```python
@scorer(metrics=[mean(), stderr()])
def my_scorer():
    async def score(state: TaskState, target: Target):
        return Score(value=..., answer=..., explanation=...)
    return score
```

#### Model-Graded Scorer
```python
model_graded_qa(
    template: str,           # Custom prompt template
    instructions: str,       # Grading guidance
    grade_pattern: str,      # Regex to extract grade
    model: str | list[str],  # Judge model(s)
)
```

#### Multi-Dimensional Scoring
Return a dict from `Score.value`:
```python
Score(value={"dim1": 3, "dim2": 4, "dim3": 2})
```

With metrics per dimension:
```python
@scorer(metrics={
    "dim1": [mean(), stderr()],
    "dim2": [mean(), stderr()],
})
```

---

## Conversion Strategy

### Option A: Single Multi-Dimensional Scorer
Create one scorer that evaluates all 32 dimensions at once.

**Pros:**
- Single LLM call per sample (more efficient?)
- Simpler Task structure

**Cons:**
- Very long prompt with 32 rubrics
- Harder for judge model to maintain quality across all dimensions
- Less modular

### Option B: Multiple Scorers ✅ CHOSEN
Create 32 separate scorers, one per rubric.

**Pros:**
- Focused evaluation per dimension
- Can run in parallel (inspect-ai handles this)
- Easier to iterate on individual rubrics
- Matches current implementation pattern

**Cons:**
- 32 API calls per sample
- More complex Task definition

### Option C: Batched Scorers
Group related rubrics (e.g., all authenticity-related ones) into ~6-8 scorers.

**Pros:**
- Balanced approach
- Fewer API calls than Option B

**Cons:**
- More complex prompt engineering
- May lose some scoring precision

---

## Implementation Plan

### 1. Dataset Loader
Create a function to load HLE data into inspect-ai `Sample` objects:

```python
def load_hle_dataset(hle_dir: Path, model_filter: str | None = None) -> list[Sample]:
    samples = []
    for model_dir in hle_dir.iterdir():
        if model_filter and model_filter not in model_dir.name:
            continue
        for json_file in model_dir.glob("*.json"):
            data = json.loads(json_file.read_text())
            for i, rollout in enumerate(data["rollouts"]):
                cot_text = rollout["response"]["choices"][0]["message"]["content"]
                samples.append(Sample(
                    input=cot_text,  # The CoT to evaluate
                    target=data["ground_truth"],  # Original correct answer
                    metadata={
                        "sample_id": data["sample_id"],
                        "rollout_idx": i,
                        "model": model_dir.name,
                        "question": data["question"],
                    }
                ))
    return samples
```

### 2. Rubric Scorer Factory
Create scorers from existing prompt files:

```python
def create_rubric_scorer(rubric: RubricInfo) -> Scorer:
    prompt_template = load_prompt(rubric)

    @scorer(metrics=[mean(), stderr()])
    def rubric_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            # Use model-graded approach
            grader = get_model("anthropic/claude-sonnet-4-20250514")
            prompt = prompt_template.format(cot_text=state.input_text)
            result = await grader.generate(prompt)

            # Parse JSON response
            parsed = parse_evaluation_response(result.completion)

            return Score(
                value=parsed.score,
                answer=str(parsed.score),
                explanation=parsed.reasoning,
                metadata={"evidence": parsed.evidence}
            )
        return score

    rubric_scorer.__name__ = rubric.id
    return rubric_scorer()
```

### 3. Task Definition
```python
@task
def cot_quality_eval(
    hle_dir: str = "hle/hle",
    model_filter: str | None = None,
    rubric_ids: list[str] | None = None,
):
    dataset = load_hle_dataset(Path(hle_dir), model_filter)

    rubrics = ALL_RUBRICS if rubric_ids is None else [
        get_rubric_by_id(rid) for rid in rubric_ids
    ]
    scorers = [create_rubric_scorer(r) for r in rubrics]

    return Task(
        dataset=dataset,
        solver=[],  # No solver - we're evaluating pre-generated CoTs
        scorer=scorers,
    )
```

### 4. CLI Usage
```bash
# Evaluate all rubrics on all models
inspect eval cot_quality_eval.py

# Evaluate specific rubrics
inspect eval cot_quality_eval.py --rubric_ids noticing_confusion,motivated_cognition

# Evaluate specific model's CoTs
inspect eval cot_quality_eval.py --model_filter claude-opus-4.5

# Use specific judge model
inspect eval cot_quality_eval.py --model anthropic/claude-sonnet-4-20250514
```

---

## Open Questions

1. **Empty Solver?** Can we have a Task with no solver (just evaluate pre-existing text)?
   - **ANSWER**: Yes! Create a custom solver that assigns pre-generated text to `state.output`:
   ```python
   from inspect_ai.model import ModelOutput, ChatMessageAssistant

   @solver
   def passthrough():
       """Solver that uses input as the 'response' for scoring."""
       async def solve(state: TaskState, generate: Generate):
           # The CoT text is in state.input_text (from Sample.input)
           state.output = ModelOutput(
               model="pre-generated",
               choices=[Choice(
                   message=ChatMessageAssistant(content=state.input_text),
                   stop_reason="stop"
               )]
           )
           state.completed = True
           return state
       return solve
   ```
   - Alternatively, set `state.completed = True` to halt execution early

2. **Judge Model Configuration**: How to set the judge model for all rubric scorers?
   - Use `model` parameter in Task?
   - Use eval CLI `--model` flag?
   - Need to check if this applies to scorers or just solvers

3. **Score Normalization**: ✅ DECIDED
   - **Positive rubrics**: 0 to 5 (shifted from 1-5)
   - **Negative rubrics**: 0 to -5 (unchanged)
   - Both start at 0, symmetric range of 5 points

4. **Parallelization**: Does inspect-ai automatically parallelize multiple scorers?
   - Check docs on eval concurrency

5. **Logging**: How to preserve detailed evidence/reasoning in inspect logs?
   - Use `Score.metadata` field
   - Check log viewer capabilities

---

## Dependencies to Add

```toml
[project]
dependencies = [
    "inspect-ai>=0.3",  # Check latest version
]
```

---

## Next Steps

1. [x] Update `schemas.py` to use 0-5 for positive rubrics (was 1-5)
2. [x] Update all 15 positive rubric prompts to use 0-5 scale
3. [x] Create `cot_quality_metrics/src/cot_quality_metrics/inspect_task.py`
4. [x] Implement dataset loader for HLE format
5. [x] Create rubric scorer factory
6. [ ] Test with single rubric on single sample
7. [ ] Verify parallelization and logging
8. [ ] Run full evaluation
9. [ ] Compare results with existing implementation

---

## TODO: "One Prompt to Rule Them All" Experiment

Create a single consolidated rubric prompt that scores all 32 dimensions in one model call.

**Approach**: Treat this as scorer #33 - just another scorer among the many. Run it alongside the individual scorers to collect comparison data. May eventually replace the individual scorers if correlation is high enough.

**Purpose**: Compare accuracy/correlation of:
- **All-in-one prompt**: Single lossy approximation, one API call
- **Individual prompts**: 32 separate focused evaluations, 32 API calls

**Hypothesis**: The individual prompts should be more accurate (focused attention), but the all-in-one might be "good enough" for some use cases and much cheaper.

**Implementation**:
1. [ ] Create `prompts/00_all_dimensions.md` with condensed rubric table
2. [ ] Create corresponding `all_dimensions` scorer
3. [ ] Run both approaches on same HLE samples
4. [ ] Compute correlation between all-in-one scores and individual scores per dimension
5. [ ] Analyze which dimensions degrade most in the consolidated prompt

**Expected Output Format** (all-in-one):
```json
{
  "scores": {
    "noticing_confusion": 3,
    "live_updating": 4,
    "motivated_cognition": -2,
    // ... all 32 dimensions
  },
  "summary_reasoning": "Brief overall assessment..."
}
```

**Questions to Answer**:
- What's the correlation per dimension?
- Which dimensions are most robust to consolidation?
- Is there a "sweet spot" (e.g., 4-8 dimensions per prompt)?
- Cost/accuracy tradeoff curve
