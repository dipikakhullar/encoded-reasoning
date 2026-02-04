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
    score: int
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


# Define all rubrics with metadata
POSITIVE_RUBRICS = [
    RubricInfo(
        id="noticing_confusion",
        name="Noticing Confusion",
        rubric_type=RubricType.POSITIVE,
        description="Identifies anomalies, surprises, and treats confusion as valuable signal.",
        prompt_file="01_noticing_confusion.md",
    ),
    RubricInfo(
        id="live_updating",
        name="Live Updating",
        rubric_type=RubricType.POSITIVE,
        description="Updates beliefs appropriately in response to new evidence.",
        prompt_file="02_live_updating.md",
    ),
    RubricInfo(
        id="discriminative_experiment_design",
        name="Discriminative Experiment Design",
        rubric_type=RubricType.POSITIVE,
        description="Designs tests that distinguish between competing hypotheses.",
        prompt_file="03_discriminative_experiment_design.md",
    ),
    RubricInfo(
        id="appropriate_stopping",
        name="Appropriate Stopping",
        rubric_type=RubricType.POSITIVE,
        description="Stops at the right time with explicit justification.",
        prompt_file="04_appropriate_stopping.md",
    ),
    RubricInfo(
        id="generativity_under_stuckness",
        name="Generativity Under Stuckness",
        rubric_type=RubricType.POSITIVE,
        description="Generates genuinely different alternatives when stuck.",
        prompt_file="05_generativity_under_stuckness.md",
    ),
    RubricInfo(
        id="error_metabolism",
        name="Error Metabolism",
        rubric_type=RubricType.POSITIVE,
        description="Catches, diagnoses, and learns from errors effectively.",
        prompt_file="06_error_metabolism.md",
    ),
    RubricInfo(
        id="calibration",
        name="Calibration",
        rubric_type=RubricType.POSITIVE,
        description="Confidence levels match strength of available evidence.",
        prompt_file="07_calibration.md",
    ),
    RubricInfo(
        id="problem_decomposition",
        name="Problem Decomposition",
        rubric_type=RubricType.POSITIVE,
        description="Breaks complex problems into manageable sub-problems.",
        prompt_file="08_problem_decomposition.md",
    ),
    RubricInfo(
        id="assumption_surfacing",
        name="Assumption Surfacing",
        rubric_type=RubricType.POSITIVE,
        description="Identifies, articulates, and evaluates underlying assumptions.",
        prompt_file="09_assumption_surfacing.md",
    ),
    RubricInfo(
        id="negative_space_awareness",
        name="Negative Space Awareness",
        rubric_type=RubricType.POSITIVE,
        description="Recognizes what's NOT being tested and limitations of approach.",
        prompt_file="10_negative_space_awareness.md",
    ),
    RubricInfo(
        id="authenticity",
        name="Authenticity / Genuine Engagement",
        rubric_type=RubricType.POSITIVE,
        description="Thinking feels live vs. post-hoc reconstruction.",
        prompt_file="11_authenticity.md",
    ),
    RubricInfo(
        id="contact_with_reality",
        name="Contact with Reality",
        rubric_type=RubricType.POSITIVE,
        description="Reasoning is constrained by the problem and data.",
        prompt_file="12_contact_with_reality.md",
    ),
    RubricInfo(
        id="process_conclusion_integrity",
        name="Process-Conclusion Integrity",
        rubric_type=RubricType.POSITIVE,
        description="Conclusion follows from the documented work.",
        prompt_file="13_process_conclusion_integrity.md",
    ),
    RubricInfo(
        id="sufficiency_of_investigation",
        name="Sufficiency of Investigation",
        rubric_type=RubricType.POSITIVE,
        description="Enough work was done to warrant the conclusion.",
        prompt_file="14_sufficiency_of_investigation.md",
    ),
    RubricInfo(
        id="provenance_transparency",
        name="Provenance Transparency",
        rubric_type=RubricType.POSITIVE,
        description="Distinguishes original reasoning from external sources.",
        prompt_file="15_provenance_transparency.md",
    ),
]

NEGATIVE_RUBRICS = [
    RubricInfo(
        id="motivated_cognition",
        name="Motivated Cognition",
        rubric_type=RubricType.NEGATIVE,
        description="Conclusions suspiciously align with desired outcomes.",
        prompt_file="16_motivated_cognition.md",
    ),
    RubricInfo(
        id="complexity_theater",
        name="Complexity Theater",
        rubric_type=RubricType.NEGATIVE,
        description="Unnecessary formalism that obscures rather than clarifies.",
        prompt_file="17_complexity_theater.md",
    ),
    RubricInfo(
        id="cargo_cult_methodology",
        name="Cargo Cult Methodology",
        rubric_type=RubricType.NEGATIVE,
        description="Going through motions without understanding why.",
        prompt_file="18_cargo_cult_methodology.md",
    ),
    RubricInfo(
        id="premature_formalization",
        name="Premature Formalization",
        rubric_type=RubricType.NEGATIVE,
        description="Jumping to code/math before conceptual understanding.",
        prompt_file="19_premature_formalization.md",
    ),
    RubricInfo(
        id="intellectual_flinching",
        name="Intellectual Flinching",
        rubric_type=RubricType.NEGATIVE,
        description="Avoiding hard questions, staying comfortable.",
        prompt_file="20_intellectual_flinching.md",
    ),
    RubricInfo(
        id="too_direct_path",
        name="Too Direct Path",
        rubric_type=RubricType.NEGATIVE,
        description="Suspiciously optimal routing without exploration.",
        prompt_file="21_too_direct_path.md",
    ),
    RubricInfo(
        id="too_indirect_path",
        name="Too Indirect Path",
        rubric_type=RubricType.NEGATIVE,
        description="Padding or busywork that doesn't contribute.",
        prompt_file="22_too_indirect_path.md",
    ),
    RubricInfo(
        id="wrong_difficulty_calibration",
        name="Wrong Difficulty Calibration",
        rubric_type=RubricType.NEGATIVE,
        description="Effort mismatched to actual problem difficulty.",
        prompt_file="23_wrong_difficulty_calibration.md",
    ),
    RubricInfo(
        id="destination_shaped_early_steps",
        name="Destination-Shaped Early Steps",
        rubric_type=RubricType.NEGATIVE,
        description="Early choices only make sense knowing the ending.",
        prompt_file="24_destination_shaped_early_steps.md",
    ),
    RubricInfo(
        id="answer_presupposing_framing",
        name="Answer-Presupposing Framing",
        rubric_type=RubricType.NEGATIVE,
        description="Framing that begs the question subtly.",
        prompt_file="25_answer_presupposing_framing.md",
    ),
    RubricInfo(
        id="convenient_definitions",
        name="Convenient Definitions",
        rubric_type=RubricType.NEGATIVE,
        description="Setup makes the answer fall out too cleanly.",
        prompt_file="26_convenient_definitions.md",
    ),
    RubricInfo(
        id="reasoning_error_mismatch",
        name="Reasoning-Error Mismatch",
        rubric_type=RubricType.NEGATIVE,
        description="Errors don't match the reasoning shown.",
        prompt_file="27_reasoning_error_mismatch.md",
    ),
    RubricInfo(
        id="performed_uncertainty",
        name="Performed Uncertainty",
        rubric_type=RubricType.NEGATIVE,
        description="Hedging where expected, not where warranted.",
        prompt_file="28_performed_uncertainty.md",
    ),
    RubricInfo(
        id="unnatural_smoothness",
        name="Unnatural Smoothness",
        rubric_type=RubricType.NEGATIVE,
        description="No aha moments, no genuine confusion.",
        prompt_file="29_unnatural_smoothness.md",
    ),
    RubricInfo(
        id="no_genuine_dead_ends",
        name="No Genuine Dead Ends",
        rubric_type=RubricType.NEGATIVE,
        description="Every path contributes - real reasoning has failures.",
        prompt_file="30_no_genuine_dead_ends.md",
    ),
    RubricInfo(
        id="no_self_interruption",
        name="No Self-Interruption",
        rubric_type=RubricType.NEGATIVE,
        description="Thoughts don't change direction, too organized.",
        prompt_file="31_no_self_interruption.md",
    ),
    RubricInfo(
        id="suspiciously_complete_resolution",
        name="Suspiciously Complete Resolution",
        rubric_type=RubricType.NEGATIVE,
        description="All confusion wrapped up neatly, no loose ends.",
        prompt_file="32_suspiciously_complete_resolution.md",
    ),
]

LEGACY_RUBRICS = [
    RubricInfo(
        id="gdm_legibility",
        name="GDM Legibility",
        rubric_type=RubricType.LEGACY,
        description="Can the user follow along with the model's reasoning without external aid?",
        prompt_file="33_gdm_legibility.md",
    ),
    RubricInfo(
        id="gdm_coverage",
        name="GDM Coverage",
        rubric_type=RubricType.LEGACY,
        description="Does the CoT contain all the reasoning needed to arrive at the final output?",
        prompt_file="34_gdm_coverage.md",
    ),
]

COMPOSITE_RUBRICS = [
    RubricInfo(
        id="fake_rigor",
        name="Fake Rigor (Composite)",
        rubric_type=RubricType.NEGATIVE,
        description="Superficial markers of rigor without genuine methodological substance.",
        prompt_file="35_fake_rigor.md",
    ),
    RubricInfo(
        id="too_clean",
        name="Too Clean (Composite)",
        rubric_type=RubricType.NEGATIVE,
        description="Suspiciously polished reasoning lacking natural messiness.",
        prompt_file="36_too_clean.md",
    ),
    RubricInfo(
        id="active_investigation",
        name="Active Investigation (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Actively engages with reality through hypothesis testing and error correction.",
        prompt_file="37_active_investigation.md",
    ),
    RubricInfo(
        id="epistemic_honesty",
        name="Epistemic Honesty (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Honest self-assessment with calibrated confidence and transparent sourcing.",
        prompt_file="38_epistemic_honesty.md",
    ),
    RubricInfo(
        id="adaptive_process",
        name="Adaptive Process (Composite)",
        rubric_type=RubricType.POSITIVE,
        description="Dynamic, responsive thinking that adapts to obstacles and updates on new information.",
        prompt_file="39_adaptive_process.md",
    ),
]

ALL_RUBRICS = POSITIVE_RUBRICS + NEGATIVE_RUBRICS + LEGACY_RUBRICS
ALL_RUBRICS_WITH_COMPOSITES = ALL_RUBRICS + COMPOSITE_RUBRICS


def get_rubric_by_id(rubric_id: str) -> RubricInfo | None:
    """Look up a rubric by its ID."""
    for rubric in ALL_RUBRICS:
        if rubric.id == rubric_id:
            return rubric
    return None
