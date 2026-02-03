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
    from .schemas import ALL_RUBRICS, NEGATIVE_RUBRICS, POSITIVE_RUBRICS, RubricInfo, RubricType
except ImportError:
    # Support running directly via inspect eval
    from cot_quality_metrics.evaluate import load_prompt, parse_evaluation_response
    from cot_quality_metrics.schemas import ALL_RUBRICS, NEGATIVE_RUBRICS, POSITIVE_RUBRICS, RubricInfo, RubricType


def load_hle_dataset(
    data_dir: Path,
    limit: int | None = None,
) -> list[Sample]:
    """Load HLE CoT traces into inspect-ai Sample format.

    Args:
        data_dir: Directory containing HLE data organized by model subdirectories.
        limit: Maximum number of samples to load.

    Returns:
        List of Sample objects for inspect-ai evaluation.
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

                if limit and len(samples) >= limit:
                    return samples

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

    @scorer(metrics=[mean(), stderr()])
    def rubric_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            # Get the CoT text to evaluate
            cot_text = state.output.completion

            # Build the evaluation prompt
            prompt = prompt_template.replace("{cot_text}", cot_text)

            # Get the grader model (uses the model specified in eval config)
            grader = get_model()

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
                    },
                )
            except Exception as e:
                return Score(
                    value=0 if rubric.rubric_type == RubricType.POSITIVE else -5,
                    answer="error",
                    explanation=f"Failed to parse evaluation: {e}",
                    metadata={"dimension": rubric.id, "error": str(e)},
                )

        return score

    # Set the scorer name to the rubric id for identification in logs
    rubric_scorer.__name__ = rubric.id
    return rubric_scorer()


def create_all_scorers(
    rubric_ids: list[str] | None = None,
    prompts_dir: Path | None = None,
) -> list[Scorer]:
    """Create scorers for all specified rubrics.

    Args:
        rubric_ids: List of rubric IDs to create scorers for.
                   If None, creates scorers for all 32 rubrics.
        prompts_dir: Directory containing prompt templates.

    Returns:
        List of Scorer functions.
    """
    if rubric_ids is None:
        rubrics = ALL_RUBRICS
    else:
        rubrics = [r for r in ALL_RUBRICS if r.id in rubric_ids]

    return [create_rubric_scorer(r, prompts_dir) for r in rubrics]


@task
def cot_quality_eval(
    data_dir: str = "data",
    rubric_ids: str | None = None,
    limit: int | None = None,
) -> Task:
    """Evaluate CoT quality on HLE traces using inspect-ai.

    Args:
        data_dir: Path to data directory containing model subdirectories.
        rubric_ids: Comma-separated list of rubric IDs to evaluate.
                   If None, evaluates all 32 rubrics.
        limit: Maximum number of samples to evaluate.

    Returns:
        An inspect-ai Task for CoT quality evaluation.

    Example:
        inspect eval inspect_task.py --data_dir ./data
        inspect eval inspect_task.py --rubric_ids noticing_confusion,motivated_cognition
    """
    # Parse rubric IDs if provided
    rubric_id_list = rubric_ids.split(",") if rubric_ids else None

    # Load dataset
    dataset = load_hle_dataset(
        data_dir=Path(data_dir),
        limit=limit,
    )

    # Create scorers
    scorers = create_all_scorers(rubric_ids=rubric_id_list)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_positive(
    data_dir: str = "data",
    limit: int | None = None,
) -> Task:
    """Evaluate only positive rubrics (epistemic virtues, 0-5 scale)."""
    rubric_ids = [r.id for r in POSITIVE_RUBRICS]

    dataset = load_hle_dataset(
        data_dir=Path(data_dir),
        limit=limit,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )


@task
def cot_quality_negative(
    data_dir: str = "data",
    limit: int | None = None,
) -> Task:
    """Evaluate only negative rubrics (anti-patterns, 0 to -5 scale)."""
    rubric_ids = [r.id for r in NEGATIVE_RUBRICS]

    dataset = load_hle_dataset(
        data_dir=Path(data_dir),
        limit=limit,
    )

    scorers = create_all_scorers(rubric_ids=rubric_ids)

    return Task(
        dataset=dataset,
        solver=[passthrough()],
        scorer=scorers,
    )
