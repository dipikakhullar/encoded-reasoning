"""CoT Quality Metrics - Evaluate Chain of Thought reasoning quality."""

from .evaluate import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    evaluate_cot,
    evaluate_cot_subset,
    evaluate_rubric,
    get_provider,
    load_prompt,
)
from .schemas import (
    ALL_RUBRICS,
    LEGACY_RUBRICS,
    NEGATIVE_RUBRICS,
    POSITIVE_RUBRICS,
    CoTEvaluation,
    EvaluationResult,
    RubricInfo,
    RubricType,
    get_rubric_by_id,
)

# Inspect-AI integration (lazy import to avoid dependency issues)
def __getattr__(name: str):
    """Lazy load inspect-ai components."""
    if name in (
        "cot_quality_eval",
        "cot_quality_positive",
        "cot_quality_negative",
        "load_cot_dataset",
        "create_rubric_scorer",
        "create_all_scorers",
        "passthrough",
    ):
        from . import inspect_task

        return getattr(inspect_task, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Evaluation functions
    "evaluate_cot",
    "evaluate_cot_subset",
    "evaluate_rubric",
    "load_prompt",
    "get_provider",
    # Provider classes
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    # Schema classes
    "CoTEvaluation",
    "EvaluationResult",
    "RubricInfo",
    "RubricType",
    # Rubric collections
    "ALL_RUBRICS",
    "POSITIVE_RUBRICS",
    "NEGATIVE_RUBRICS",
    "LEGACY_RUBRICS",
    "get_rubric_by_id",
    # Inspect-AI integration
    "cot_quality_eval",
    "cot_quality_positive",
    "cot_quality_negative",
    "load_cot_dataset",
    "create_rubric_scorer",
    "create_all_scorers",
    "passthrough",
]
