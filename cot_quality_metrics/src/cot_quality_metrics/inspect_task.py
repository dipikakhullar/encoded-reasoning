"""Inspect-AI integration for CoT quality evaluation.

This module provides an inspect-ai Task for evaluating Chain of Thought quality
using the rubric dimensions. It supports multiple data formats with automatic
detection:

Supported formats:
    1. HLE format: Model subdirectories with nested response structure
       - data/{model_name}/*.json with rollouts[].response.choices[0].message.content

    2. Rollouts format: Task vector steering experiment files
       - data/*.json with alpha, problems[], rollouts[][] structure

Usage:
    # From command line (uses default data/ directory):
    inspect eval cot_quality_metrics/src/cot_quality_metrics/inspect_task.py

    # With custom data directory:
    inspect eval inspect_task.py -T data_dir=/path/to/data

    # With specific model:
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


class FileFormat:
    """Detected file format types."""
    HLE = "hle"  # Original HLE format with nested response structure
    ROLLOUTS = "rollouts"  # Task vector rollouts format
    UNKNOWN = "unknown"


def detect_file_format(data: dict) -> str:
    """Detect the format of a JSON data file.

    Args:
        data: Parsed JSON data from a file.

    Returns:
        FileFormat constant indicating the detected format.
    """
    # Rollouts format has: alpha, model, problems, rollouts (list of lists)
    if all(k in data for k in ["alpha", "problems", "rollouts"]):
        if isinstance(data.get("rollouts"), list) and data["rollouts"]:
            # Check if rollouts is list of lists (rollouts format)
            if isinstance(data["rollouts"][0], list):
                return FileFormat.ROLLOUTS

    # HLE format has: rollouts (list of dicts with response.choices structure)
    if "rollouts" in data and isinstance(data.get("rollouts"), list):
        if data["rollouts"]:
            rollout = data["rollouts"][0]
            if isinstance(rollout, dict) and "response" in rollout:
                if isinstance(rollout["response"], dict) and "choices" in rollout["response"]:
                    return FileFormat.HLE

    return FileFormat.UNKNOWN


def load_rollouts_file(json_file: Path, data: dict) -> list[Sample]:
    """Load samples from a task vector rollouts format file.

    Args:
        json_file: Path to the JSON file (for ID generation).
        data: Parsed JSON data.

    Returns:
        List of Sample objects.
    """
    samples = []
    alpha = data.get("alpha", "unknown")
    model = data.get("model", "unknown")
    # Clean model name for ID (remove slashes)
    model_short = model.split("/")[-1] if "/" in model else model

    for prob_idx, problem in enumerate(data.get("problems", [])):
        question = problem.get("problem", "")
        expected = problem.get("answer")

        for roll_idx, rollout in enumerate(data.get("rollouts", [[]])[prob_idx]):
            cot_text = rollout.get("response", "")
            predicted = rollout.get("pred")

            samples.append(
                Sample(
                    input=cot_text,
                    target=str(expected) if expected is not None else "",
                    id=f"{model_short}/alpha{alpha}/prob{prob_idx}/roll{roll_idx}",
                    metadata={
                        "model": model,
                        "alpha": alpha,
                        "problem_idx": prob_idx,
                        "rollout_idx": roll_idx,
                        "question": question,
                        "expected": expected,
                        "predicted": predicted,
                        "correct": predicted == expected if expected is not None else None,
                        "source_file": json_file.name,
                        "format": FileFormat.ROLLOUTS,
                    },
                )
            )

    return samples


def load_hle_file(json_file: Path, data: dict, model_name: str) -> list[Sample]:
    """Load samples from an HLE format file.

    Args:
        json_file: Path to the JSON file.
        data: Parsed JSON data.
        model_name: Model name (usually from parent directory).

    Returns:
        List of Sample objects.
    """
    samples = []
    filename = json_file.stem
    sample_id = data.get("sample_id", filename)
    question = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    for i, rollout in enumerate(data.get("rollouts", [])):
        try:
            cot_text = rollout["response"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            continue

        samples.append(
            Sample(
                input=cot_text,
                target=ground_truth,
                id=f"{model_name}/{filename}/rollout_{i}",
                metadata={
                    "model": model_name,
                    "filename": filename,
                    "sample_id": sample_id,
                    "rollout_idx": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "format": FileFormat.HLE,
                },
            )
        )

    return samples


def load_dataset_autodetect(data_dir: Path) -> list[Sample]:
    """Load CoT traces with automatic format detection.

    Supports both HLE format (model subdirectories with nested response structure)
    and rollouts format (flat files with alpha/problems/rollouts structure).

    Args:
        data_dir: Directory containing data files. Can be:
            - A directory with model subdirectories (HLE format)
            - A directory with rollouts JSON files directly
            - A mixed directory with both formats

    Returns:
        List of Sample objects for inspect-ai evaluation.
    """
    samples = []

    # First, check for JSON files directly in data_dir (rollouts format)
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            continue

        fmt = detect_file_format(data)
        if fmt == FileFormat.ROLLOUTS:
            samples.extend(load_rollouts_file(json_file, data))
        elif fmt == FileFormat.HLE:
            # HLE file in root - use filename as model name
            samples.extend(load_hle_file(json_file, data, json_file.stem))

    # Then, check subdirectories (HLE format with model subdirs)
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith(".") or subdir.name.startswith("__"):
            continue

        model_name = subdir.name

        for json_file in sorted(subdir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError:
                continue

            fmt = detect_file_format(data)
            if fmt == FileFormat.ROLLOUTS:
                # Rollouts file in a subdir - still use its own metadata
                samples.extend(load_rollouts_file(json_file, data))
            elif fmt == FileFormat.HLE:
                samples.extend(load_hle_file(json_file, data, model_name))

    return samples


def load_cot_dataset(data_dir: Path) -> list[Sample]:
    """Load CoT traces into inspect-ai Sample format with automatic format detection.

    Supports both HLE format (model subdirectories with nested response structure)
    and rollouts format (flat files with alpha/problems/rollouts structure).

    Args:
        data_dir: Directory containing data. Can be:
            - A directory with model subdirectories (HLE format)
            - A directory with rollouts JSON files directly
            - A mixed directory with both formats

    Returns:
        List of Sample objects for inspect-ai evaluation.
        Use inspect-ai's --limit flag to control how many samples are evaluated.
    """
    return load_dataset_autodetect(data_dir)


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
