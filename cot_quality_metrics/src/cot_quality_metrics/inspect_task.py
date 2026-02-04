"""Inspect-AI integration for CoT quality evaluation.

This module provides an inspect-ai Task for evaluating Chain of Thought quality
using the 32 rubric dimensions. It loads pre-generated CoT traces from HLE
(Humanity's Last Exam) format and scores them using model-graded evaluation.

Usage:
    # From command line:
    inspect eval cot_quality_metrics/src/cot_quality_metrics/inspect_task.py

    # With options:
    inspect eval inspect_task.py --model anthropic/claude-sonnet-4-20250514
"""

import json
import re
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver

try:
    from .evaluate import load_prompt, parse_evaluation_response
    from .schemas import ALL_RUBRICS, COMPOSITE_RUBRICS, LEGACY_RUBRICS, NEGATIVE_RUBRICS, POSITIVE_RUBRICS, RubricInfo, RubricType
except ImportError:
    # Support running directly via inspect eval
    from cot_quality_metrics.evaluate import load_prompt, parse_evaluation_response
    from cot_quality_metrics.schemas import ALL_RUBRICS, COMPOSITE_RUBRICS, LEGACY_RUBRICS, NEGATIVE_RUBRICS, POSITIVE_RUBRICS, RubricInfo, RubricType

# Project root: encoded-reasoning/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def load_cot_dataset(data_dir: Path) -> list[Sample]:
    """Load CoT traces into inspect-ai Sample format.

    Args:
        data_dir: Directory containing data organized by model subdirectories.

    Returns:
        List of Sample objects for inspect-ai evaluation.
        Use inspect-ai's --limit flag to control how many samples are evaluated.
    """
    samples = []

    for model_dir in sorted(data_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith(".") or model_dir.name.startswith("__"):
            continue

        model_name = model_dir.name

        for json_file in sorted(model_dir.glob("*.json")):
            filename = json_file.stem  # filename without extension (experimental condition)

            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError:
                continue

            sample_id = data.get("sample_id", filename)
            question = data.get("question", "")
            ground_truth = data.get("ground_truth", "")

            for i, rollout in enumerate(data.get("rollouts", [])):
                # Extract CoT text from response
                try:
                    cot_text = rollout["response"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    continue

                samples.append(
                    Sample(
                        input=cot_text,  # The CoT to evaluate
                        target=ground_truth,  # Original correct answer (for reference)
                        id=f"{model_name}/{filename}/rollout_{i}",
                        metadata={
                            "model": model_name,
                            "filename": filename,
                            "sample_id": sample_id,
                            "rollout_idx": i,
                            "question": question,
                            "ground_truth": ground_truth,
                        },
                    )
                )

    return samples


@solver
def passthrough() -> Solver:
    """Solver that passes input directly to output without model generation.

    Used for evaluating pre-generated CoT traces where we don't need
    to generate new responses.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # The CoT text is already in state.input_text
        # We set it as the assistant response for scoring
        state.messages.append(ChatMessageAssistant(content=state.input_text))
        state.output.completion = state.input_text
        state.completed = True
        return state

    return solve


def create_stub_scorer(name: str = "stub") -> Scorer:
    """Create a stub scorer for testing that makes no LLM calls."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value=1.0,
            answer="stub",
            explanation="Stub scorer - no LLM call made",
            metadata={"dimension": name},
        )

    return scorer(metrics=[mean(), stderr()], name=name)(lambda: score)()


def create_rubric_scorer(
    rubric: RubricInfo,
    prompts_dir: Path | None = None,
) -> Scorer:
    """Create an inspect-ai scorer for a specific rubric.

    Args:
        rubric: The rubric to create a scorer for.
        prompts_dir: Directory containing prompt templates.

    Returns:
        An inspect-ai Scorer function.
    """
    # Load the prompt template
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"

    prompt_template = load_prompt(rubric, prompts_dir)

    async def score(state: TaskState, target: Target) -> Score:
        # Get the CoT text to evaluate
        cot_text = state.output.completion

        # Build the evaluation prompt
        prompt = prompt_template.replace("{cot_text}", cot_text)

        # Get the grader model (uses the model specified in eval config)
        grader = get_model()

        # Retry loop for parse failures
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            # Generate the evaluation
            result = await grader.generate(prompt)

            # Parse the response
            try:
                parsed = parse_evaluation_response(result.completion)
                score_value = parsed.score

                # Validate score range
                if rubric.rubric_type == RubricType.POSITIVE:
                    if not (0 <= score_value <= 5):
                        raise ValueError(f"Score {score_value} out of range for positive rubric")
                elif rubric.rubric_type == RubricType.LEGACY:
                    if not (0 <= score_value <= 4):
                        raise ValueError(f"Score {score_value} out of range for legacy rubric")
                else:
                    if not (-5 <= score_value <= 0):
                        raise ValueError(f"Score {score_value} out of range for negative rubric")

                return Score(
                    value=score_value,
                    answer=str(score_value),
                    explanation=parsed.reasoning,
                    metadata={
                        "dimension": rubric.id,
                        "evidence": parsed.evidence,
                        "rubric_type": rubric.rubric_type.value,
                        "attempts": attempt + 1,
                    },
                )
            except Exception as e:
                last_error = e
                # Continue to retry

        # All retries exhausted - return error score
        error_score = rubric.min_score
        return Score(
            value=error_score,
            answer="error",
            explanation=f"Failed to parse after {max_retries} attempts: {last_error}",
            metadata={"dimension": rubric.id, "error": str(last_error), "attempts": max_retries},
        )

    # Create scorer with the rubric id as the name
    return scorer(metrics=[mean(), stderr()], name=rubric.id)(lambda: score)()


def create_all_scorers(
    rubric_ids: list[str] | None = None,
    prompts_dir: Path | None = None,
    include_composites: bool = False,
) -> list[Scorer]:
    """Create scorers for all specified rubrics.

    Args:
        rubric_ids: List of rubric IDs to create scorers for.
                   If None, creates scorers for all 34 rubrics (or 39 with composites).
        prompts_dir: Directory containing prompt templates.
        include_composites: Whether to include composite rubrics when rubric_ids is None.

    Returns:
        List of Scorer functions.
    """
    all_available = ALL_RUBRICS + COMPOSITE_RUBRICS if include_composites else ALL_RUBRICS

    if rubric_ids is None:
        rubrics = all_available
    else:
        rubrics = [r for r in all_available if r.id in rubric_ids]
        # Also check COMPOSITE_RUBRICS if not found in ALL_RUBRICS
        if len(rubrics) < len(rubric_ids):
            rubrics.extend([r for r in COMPOSITE_RUBRICS if r.id in rubric_ids and r not in rubrics])

    return [create_rubric_scorer(r, prompts_dir) for r in rubrics]


@task
def cot_quality_eval(
    data_dir: str | None = None,
    rubric_ids: str | None = None,
) -> Task:
    """Evaluate CoT quality on HLE traces using inspect-ai.

    Args:
        data_dir: Path to data directory containing model subdirectories.
        rubric_ids: Comma-separated list of rubric IDs to evaluate.
                   If None, evaluates all 34 rubrics.

    Returns:
        An inspect-ai Task for CoT quality evaluation.

    Example:
        inspect eval inspect_task.py --limit 10
        inspect eval inspect_task.py -T rubric_ids=noticing_confusion,motivated_cognition
    """
    # Parse rubric IDs if provided
    rubric_id_list = rubric_ids.split(",") if rubric_ids else None

    # Load dataset
    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    # Create scorers
    scorers = create_all_scorers(rubric_ids=rubric_id_list)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_positive(data_dir: str | None = None) -> Task:
    """Evaluate only positive rubrics (epistemic virtues, 0-5 scale)."""
    rubric_ids = [r.id for r in POSITIVE_RUBRICS]

    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_negative(data_dir: str | None = None) -> Task:
    """Evaluate only negative rubrics (anti-patterns, 0 to -5 scale)."""
    rubric_ids = [r.id for r in NEGATIVE_RUBRICS]

    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_legacy(data_dir: str | None = None) -> Task:
    """Evaluate only legacy GDM rubrics (legibility and coverage, 0-4 scale)."""
    rubric_ids = [r.id for r in LEGACY_RUBRICS]

    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_composite(data_dir: str | None = None) -> Task:
    """Evaluate only composite rubrics (5 combined metrics).

    Composite rubrics:
    - fake_rigor (negative): premature_formalization + cargo_cult + complexity_theater + intellectual_flinching
    - too_clean (negative): no_self_interruption + unnatural_smoothness + suspiciously_complete_resolution
    - active_investigation (positive): discriminative_experiment_design + error_metabolism + contact_with_reality
    - epistemic_honesty (positive): calibration + provenance_transparency + process_conclusion_integrity
    - adaptive_process (positive): generativity_under_stuckness + noticing_confusion + live_updating
    """
    rubric_ids = [r.id for r in COMPOSITE_RUBRICS]

    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_comp_and_legacy(data_dir: str | None = None) -> Task:
    """Evaluate composite rubrics + legacy GDM rubrics (7 total metrics).

    Composite rubrics (5):
    - fake_rigor (negative)
    - too_clean (negative)
    - active_investigation (positive)
    - epistemic_honesty (positive)
    - adaptive_process (positive)

    Legacy rubrics (2):
    - gdm_legibility (0-4)
    - gdm_coverage (0-4)
    """
    rubric_ids = [r.id for r in COMPOSITE_RUBRICS] + [r.id for r in LEGACY_RUBRICS]

    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_all(data_dir: str | None = None) -> Task:
    """Evaluate ALL rubrics: 32 novel + 2 legacy + 5 composite (39 total).

    This runs everything in one pass for direct comparison between
    component rubrics and composite rubrics on the same samples.
    """
    # All 34 base rubrics + 5 composites = 39 total
    rubric_ids = [r.id for r in ALL_RUBRICS] + [r.id for r in COMPOSITE_RUBRICS]

    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_stub(data_dir: str | None = None) -> Task:
    """Stub task for testing dataset loading - no LLM calls."""
    dataset = load_cot_dataset(
        data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
    )

    print(f"[DEBUG] Loaded {len(dataset)} samples")
    for i, s in enumerate(dataset[:5]):
        print(f"  [{i}] {s.id}")
    if len(dataset) > 5:
        print(f"  ... and {len(dataset) - 5} more")

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=[create_stub_scorer()],
    )
