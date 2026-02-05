"""Inspect-AI task for generating CoT from a model and evaluating it.

This task extracts problems from a rollout file, sends them to a target model
for generation, then evaluates the CoT quality using the rubric scorers.

Usage:
    # Using HuggingFace router (DeepSeek-R1-Distill-Qwen-1.5B):
    inspect eval generate_and_eval.py@generate_and_eval \
        --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale \
        -M base_url=https://router.huggingface.co/v1 \
        -M api_key=$HF_TOKEN \
        -T problems_file=rollouts/rollouts_alpha0.5_n20.json

    # With a different judge model:
    inspect eval generate_and_eval.py@generate_and_eval \
        --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale \
        -M base_url=https://router.huggingface.co/v1 \
        -T problems_file=rollouts/rollouts_alpha0.5_n20.json \
        -T judge_model=anthropic/claude-sonnet-4-20250514
"""

import csv
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, generate

try:
    from .evaluate import load_prompt, parse_evaluation_response
    from .schemas import COMPOSITE_RUBRICS, LEGACY_RUBRICS, RubricInfo, RubricType
except ImportError:
    from cot_quality_metrics.evaluate import load_prompt, parse_evaluation_response
    from cot_quality_metrics.schemas import COMPOSITE_RUBRICS, LEGACY_RUBRICS, RubricInfo, RubricType


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_crass_dataset(
    csv_file: Path,
    limit: int | None = None,
    num_rollouts: int = 1,
) -> list[Sample]:
    """Load problems from the CRASS counterfactual reasoning dataset.

    Args:
        csv_file: Path to CRASS_FTM_main_data_set.csv
        limit: Optional limit on number of problems to load.
        num_rollouts: Number of rollouts per problem (each gets separate generation).

    Returns:
        List of Sample objects for generation.
    """
    samples = []

    with open(csv_file) as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    for row in rows:
        premise = row["Premise"]
        question = row["QCC"]
        correct_answer = row["CorrectAnswer"]

        # Collect answer choices (filter empty ones)
        choices = [row["CorrectAnswer"], row["Answer1"], row["Answer2"]]
        if row.get("PossibleAnswer3"):
            choices.append(row["PossibleAnswer3"])

        prompt = f"""Consider this scenario and answer the counterfactual question.

Scenario: {premise}

Question: {question}

Think through this step by step, then give your final answer."""

        # Create num_rollouts samples for this problem
        for rollout_idx in range(num_rollouts):
            samples.append(
                Sample(
                    input=prompt,
                    target=correct_answer,
                    id=f"crass_{row['PCTID']}_r{rollout_idx}",
                    metadata={
                        "pctid": row["PCTID"],
                        "batch_id": row["BatchID"],
                        "premise": premise,
                        "question": question,
                        "correct_answer": correct_answer,
                        "choices": choices,
                        "source": "CRASS",
                        "rollout_idx": rollout_idx,
                        "problem_idx": int(row["PCTID"]) - 1,
                    },
                )
            )

    return samples


def load_problems_dataset(problems_file: Path) -> list[Sample]:
    """Load problems from a rollouts file into a dataset for generation.

    Args:
        problems_file: Path to a rollouts JSON file containing problems.

    Returns:
        List of Sample objects where input is the problem and target is the answer.
    """
    with open(problems_file) as f:
        data = json.load(f)

    samples = []
    source_file = problems_file.name

    for prob_idx, problem in enumerate(data.get("problems", [])):
        question = problem.get("problem", "")
        expected = problem.get("answer")

        # Create a prompt that asks the model to think through the problem
        prompt = f"""Solve this problem step by step, showing your reasoning:

{question}

Think through this carefully, then give your final answer."""

        samples.append(
            Sample(
                input=prompt,
                target=str(expected) if expected is not None else "",
                id=f"prob{prob_idx}",
                metadata={
                    "problem_idx": prob_idx,
                    "question": question,
                    "expected": expected,
                    "source_file": source_file,
                },
            )
        )

    return samples


def create_rubric_scorer_with_judge(
    rubric: RubricInfo,
    judge_model_name: str | None = None,
    prompts_dir: Path | None = None,
) -> Scorer:
    """Create a scorer that uses a specific judge model.

    Args:
        rubric: The rubric to score with.
        judge_model_name: Model to use for judging (e.g., "anthropic/claude-sonnet-4-20250514").
                         If None, uses the same model as generation.
        prompts_dir: Directory containing prompt templates.

    Returns:
        An inspect-ai Scorer.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"

    prompt_template = load_prompt(rubric, prompts_dir)

    async def score(state: TaskState, target: Target) -> Score:
        cot_text = state.output.completion

        # Skip judging if model failed to generate
        if not cot_text or not cot_text.strip():
            return Score(
                value=rubric.min_score,
                answer="no_generation",
                explanation="Model failed to generate a response",
                metadata={"dimension": rubric.id, "error": "empty_completion"},
            )

        prompt = prompt_template.replace("{cot_text}", cot_text)

        # Get the judge model
        if judge_model_name:
            grader = get_model(judge_model_name)
        else:
            grader = get_model()

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            result = await grader.generate(prompt)

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
                        "judge_model": judge_model_name or "default",
                    },
                )
            except Exception as e:
                last_error = e

        error_score = rubric.min_score
        return Score(
            value=error_score,
            answer="error",
            explanation=f"Failed to parse after {max_retries} attempts: {last_error}",
            metadata={"dimension": rubric.id, "error": str(last_error), "attempts": max_retries},
        )

    return scorer(metrics=[mean(), stderr()], name=rubric.id)(lambda: score)()


@task
def generate_and_eval(
    problems_file: str | None = None,
    judge_model: str | None = None,
) -> Task:
    """Generate CoT from target model and evaluate with rubrics.

    Args:
        problems_file: Path to rollouts JSON file containing problems.
                      Defaults to rollouts/rollouts_alpha0.5_n20.json
        judge_model: Model to use for judging (e.g., "anthropic/claude-sonnet-4-20250514").
                    If None, uses the same model as generation.

    Returns:
        An inspect-ai Task.

    Example:
        inspect eval generate_and_eval.py@generate_and_eval \
            --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale \
            -M base_url=https://router.huggingface.co/v1 \
            -M api_key=$HF_TOKEN \
            -T problems_file=rollouts/rollouts_alpha0.5_n20.json \
            -T judge_model=anthropic/claude-sonnet-4-20250514
    """
    if problems_file is None:
        problems_path = PROJECT_ROOT / "rollouts" / "rollouts_alpha0.5_n20.json"
    else:
        problems_path = Path(problems_file)
        if not problems_path.is_absolute():
            problems_path = PROJECT_ROOT / problems_path

    dataset = load_problems_dataset(problems_path)

    # Create scorers with the judge model
    rubrics = COMPOSITE_RUBRICS + LEGACY_RUBRICS
    scorers = [create_rubric_scorer_with_judge(r, judge_model) for r in rubrics]

    return Task(
        dataset=dataset,
        solver=[generate()],  # Actually generate from the model
        scorer=scorers,
    )


@task
def generate_and_eval_crass(
    judge_model: str | None = None,
    limit: int = 10,
    num_rollouts: int = 1,
) -> Task:
    """Generate CoT on CRASS counterfactual reasoning and evaluate.

    Args:
        judge_model: Model to use for judging.
        limit: Number of problems to use (default 10).
        num_rollouts: Number of rollouts per problem (default 1).

    Returns:
        An inspect-ai Task.

    Example:
        inspect eval generate_and_eval.py@generate_and_eval_crass \
            --model openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B:nscale \
            -M base_url=https://router.huggingface.co/v1 \
            -M api_key=$HF_TOKEN \
            -T judge_model=openrouter/anthropic/claude-sonnet-4.5 \
            -T limit=10 \
            -T num_rollouts=5
    """
    crass_path = PROJECT_ROOT / "data" / "CRASS_FTM_main_data_set.csv"
    dataset = load_crass_dataset(crass_path, limit=limit, num_rollouts=num_rollouts)

    rubrics = COMPOSITE_RUBRICS + LEGACY_RUBRICS
    scorers = [create_rubric_scorer_with_judge(r, judge_model) for r in rubrics]

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=scorers,
    )


@task
def generate_only(
    problems_file: str | None = None,
) -> Task:
    """Generate CoT from target model without evaluation (for testing).

    Args:
        problems_file: Path to rollouts JSON file containing problems.

    Returns:
        An inspect-ai Task with a stub scorer.
    """
    from inspect_ai.scorer import accuracy

    if problems_file is None:
        problems_path = PROJECT_ROOT / "rollouts" / "rollouts_alpha0.5_n20.json"
    else:
        problems_path = Path(problems_file)
        if not problems_path.is_absolute():
            problems_path = PROJECT_ROOT / problems_path

    dataset = load_problems_dataset(problems_path)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=[accuracy()],  # Simple scorer just to complete the task
    )
