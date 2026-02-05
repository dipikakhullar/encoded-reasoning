"""Output schemas for CoT quality evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class RubricType(Enum):
    """Type of rubric - positive (epistemic virtues) or negative (anti-patterns)."""

    POSITIVE = "positive"  # 0-5 scale (higher is better)
    NEGATIVE = "negative"  # 0 to -5 scale (0 is best)
    LEGACY = "legacy"  # 0-4 scale (GDM rubrics, higher is better)


@dataclass
class RubricInfo:
    """Metadata about a rubric."""

    id: str
    name: str
    rubric_type: RubricType
    description: str
    prompt_file: str

    @property
    def min_score(self) -> int:
        return 0 if self.rubric_type in (RubricType.POSITIVE, RubricType.LEGACY) else -5

    @property
    def max_score(self) -> int:
        if self.rubric_type == RubricType.POSITIVE:
            return 5
        elif self.rubric_type == RubricType.LEGACY:
            return 4
        else:
            return 0


@dataclass
class EvaluationResult:
    """Result from a single rubric evaluation."""

    dimension: str
    score: float  # Can be decimal (e.g., 3.5) or integer
    evidence: list[str]
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
        }


@dataclass
class CoTEvaluation:
    """Complete evaluation of a Chain of Thought across all rubrics."""

    cot_id: str
    cot_text: str
    results: list[EvaluationResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def positive_scores(self) -> dict[str, int]:
        """Get scores for positive rubrics (1-5 scale)."""
        return {
            r.dimension: r.score
            for r in self.results
            if r.score > 0 or r.dimension
            in [rubric.id for rubric in POSITIVE_RUBRICS]
        }

    @property
    def negative_scores(self) -> dict[str, int]:
        """Get scores for negative rubrics (0 to -5 scale)."""
        return {
            r.dimension: r.score
            for r in self.results
            if r.score <= 0 and r.dimension in [rubric.id for rubric in NEGATIVE_RUBRICS]
        }

    @property
    def aggregate_positive(self) -> float:
        """Average of positive rubric scores."""
        scores = self.positive_scores
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    @property
    def aggregate_negative(self) -> float:
        """Average of negative rubric scores (will be <= 0)."""
        scores = self.negative_scores
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    def to_dict(self) -> dict:
        return {
            "cot_id": self.cot_id,
            "cot_text": self.cot_text,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
            "aggregate_positive": self.aggregate_positive,
            "aggregate_negative": self.aggregate_negative,
        }


# Legacy rubrics (GDM)
LEGACY_RUBRICS = [
    RubricInfo(
        id="gdm_legibility",
        name="GDM Legibility",
        rubric_type=RubricType.LEGACY,
        description="Can the user follow along with the model's reasoning without external aid?",
        prompt_file="gdm_legibility.md",
    ),
    RubricInfo(
        id="gdm_coverage",
        name="GDM Coverage",
        rubric_type=RubricType.LEGACY,
        description="Does the CoT contain all the reasoning needed to arrive at the final output?",
        prompt_file="gdm_coverage.md",
    ),
]

COMPOSITE_RUBRICS = [
    RubricInfo(
        id="fake_rigor",
        name="Fake Rigor (Composite)",
        rubric_type=RubricType.NEGATIVE,
        description="Superficial markers of rigor without genuine methodological substance.",
        prompt_file="1_fake_rigor.md",
    ),
    RubricInfo(
        id="reportive_fidelity",
        name="Reportive Fidelity (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Accurately represents actual cognitive process; positions followers to catch errors.",
        prompt_file="2_reportive_fidelity.md",
    ),
    RubricInfo(
        id="active_investigation",
        name="Active Investigation (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Actively engages with reality through hypothesis testing and error correction.",
        prompt_file="3_active_investigation.md",
    ),
    RubricInfo(
        id="epistemic_honesty",
        name="Epistemic Honesty (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Honest self-assessment with calibrated confidence and transparent sourcing.",
        prompt_file="4_epistemic_honesty.md",
    ),
    RubricInfo(
        id="adaptive_process",
        name="Adaptive Process (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Dynamic, responsive thinking that adapts to obstacles and updates on new information.",
        prompt_file="5_adaptive_process.md",
    ),
]

# For backwards compatibility, create empty lists
POSITIVE_RUBRICS: list[RubricInfo] = []
NEGATIVE_RUBRICS: list[RubricInfo] = []

ALL_RUBRICS = LEGACY_RUBRICS + COMPOSITE_RUBRICS
ALL_RUBRICS_WITH_COMPOSITES = ALL_RUBRICS  # Same now


def get_rubric_by_id(rubric_id: str) -> RubricInfo | None:
    """Look up a rubric by its ID."""
    for rubric in ALL_RUBRICS:
        if rubric.id == rubric_id:
            return rubric
    return None
