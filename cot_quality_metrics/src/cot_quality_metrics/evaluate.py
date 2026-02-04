"""Evaluation runner for CoT quality metrics."""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from .schemas import (
    ALL_RUBRICS,
    CoTEvaluation,
    EvaluationResult,
    RubricInfo,
    RubricType,
    get_rubric_by_id,
)

# Default prompts directory relative to this file
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# Load environment variables from common locations
load_dotenv(Path.home() / "projects" / ".env")
load_dotenv()  # Also check local .env


def load_prompt(rubric: RubricInfo, prompts_dir: Path | None = None) -> str:
    """Load the evaluation prompt for a rubric.

    Args:
        rubric: The rubric to load the prompt for.
        prompts_dir: Directory containing prompt files. Defaults to prompts/.

    Returns:
        The prompt template with {cot_text} placeholder.
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    prompt_path = prompts_dir / rubric.prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    content = prompt_path.read_text()

    # Extract just the ## Prompt section
    prompt_match = re.search(r"## Prompt\n\n(.+)", content, re.DOTALL)
    if prompt_match:
        return prompt_match.group(1).strip()

    # Fallback: return everything after the rubric table
    return content


def parse_evaluation_response(response_text: str) -> EvaluationResult:
    """Parse the evaluation response from the LLM.

    Supports two formats:
    1. Plain text with "Score: X" at the end (preferred)
    2. JSON format (legacy fallback)

    Args:
        response_text: The raw response text from the LLM.

    Returns:
        Parsed EvaluationResult.

    Raises:
        ValueError: If response cannot be parsed.
    """
    # First, try plain text format - find the LAST score mention in the response
    # Handle various formats: "Score: -3", "**Score:** -3", "Score: 3.5/5", "Rating: 4", etc.
    # Look for patterns like "Score: X" or "Score: X/5" anywhere in text, take the last one
    # Supports decimal scores like 3.5, -1.5, etc.
    score_matches = list(re.finditer(
        r"\*{0,2}(?:Overall\s+)?(?:Final\s+)?(?:Score|Rating)\s*[:=]?\s*\*{0,2}\s*(-?\d+\.?\d*)(?:/\d+)?\*{0,2}",
        response_text,
        re.IGNORECASE
    ))
    if score_matches:
        last_match = score_matches[-1]
        score = float(last_match.group(1))
        reasoning = response_text[:last_match.start()].strip()
        return EvaluationResult(
            dimension="unknown",  # Will be set by caller context
            score=score,
            evidence=[],
            reasoning=reasoning,
        )

    # Fallback: look for a standalone number at the very end (with optional bold/formatting)
    end_number_match = re.search(r"\*{0,2}(-?\d+\.?\d*)\*{0,2}\s*$", response_text.strip())
    if end_number_match:
        score = float(end_number_match.group(1))
        reasoning = response_text[:end_number_match.start()].strip()
        return EvaluationResult(
            dimension="unknown",
            score=score,
            evidence=[],
            reasoning=reasoning,
        )

    # Fallback: try JSON format
    # Look for ```json ... ``` blocks first
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r"\{[^{}]*\"dimension\"[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Could not find score in response: {response_text[:200]}...")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    return EvaluationResult(
        dimension=data.get("dimension", "unknown"),
        score=data.get("score", 0),
        evidence=data.get("evidence", []),
        reasoning=data.get("reasoning", ""),
    )


# Provider abstraction
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(
        self,
        model: str = "models/gemini-flash-latest",
        thinking_level: str | None = None,
    ):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.thinking_level = thinking_level
        self._types = types

    def generate(self, prompt: str) -> str:
        contents = [
            self._types.Content(
                role="user",
                parts=[self._types.Part.from_text(text=prompt)],
            ),
        ]

        # Only add thinking config if explicitly requested
        if self.thinking_level:
            config = self._types.GenerateContentConfig(
                thinking_config=self._types.ThinkingConfig(
                    thinking_level=self.thinking_level.upper(),
                ),
            )
        else:
            config = None

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Extract text from response, handling thinking parts if present
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        return "\n".join(text_parts)


def get_provider(
    backend: Literal["anthropic", "gemini"] = "anthropic",
    model: str | None = None,
    **kwargs,
) -> LLMProvider:
    """Get an LLM provider instance.

    Args:
        backend: Which provider to use ("anthropic" or "gemini").
        model: Model name to use. If None, uses provider default.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured LLMProvider instance.
    """
    if backend == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514")
    elif backend == "gemini":
        return GeminiProvider(
            model=model or "models/gemini-flash-latest",
            thinking_level=kwargs.get("thinking_level"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def evaluate_rubric(
    cot_text: str,
    rubric: RubricInfo,
    provider: LLMProvider,
    prompts_dir: Path | None = None,
) -> EvaluationResult:
    """Evaluate a CoT against a single rubric.

    Args:
        cot_text: The Chain of Thought text to evaluate.
        rubric: The rubric to evaluate against.
        provider: LLM provider instance.
        prompts_dir: Directory containing prompt files.

    Returns:
        EvaluationResult with score and evidence.
    """
    prompt_template = load_prompt(rubric, prompts_dir)
    prompt = prompt_template.replace("{cot_text}", cot_text)

    response_text = provider.generate(prompt)
    result = parse_evaluation_response(response_text)

    # Validate score is in expected range
    if rubric.rubric_type == RubricType.POSITIVE:
        if not (0 <= result.score <= 5):
            raise ValueError(
                f"Score {result.score} out of range for positive rubric (expected 0-5)"
            )
    else:
        if not (-5 <= result.score <= 0):
            raise ValueError(
                f"Score {result.score} out of range for negative rubric (expected -5 to 0)"
            )

    return result


def evaluate_cot(
    cot_text: str,
    cot_id: str = "unnamed",
    rubrics: list[RubricInfo] | None = None,
    provider: LLMProvider | None = None,
    backend: Literal["anthropic", "gemini"] = "anthropic",
    model: str | None = None,
    prompts_dir: Path | None = None,
    max_workers: int = 16,
    **provider_kwargs,
) -> CoTEvaluation:
    """Evaluate a CoT against all specified rubrics.

    Args:
        cot_text: The Chain of Thought text to evaluate.
        cot_id: Identifier for this CoT sample.
        rubrics: List of rubrics to evaluate against. Defaults to all rubrics.
        provider: LLM provider instance. If None, creates one from backend/model.
        backend: Which provider to use if provider is None.
        model: Model to use for evaluation.
        prompts_dir: Directory containing prompt files.
        max_workers: Maximum parallel API calls (default: 16).
        **provider_kwargs: Additional provider-specific arguments.

    Returns:
        CoTEvaluation with results from all rubrics.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if rubrics is None:
        rubrics = ALL_RUBRICS

    if provider is None:
        provider = get_provider(backend=backend, model=model, **provider_kwargs)

    evaluation = CoTEvaluation(cot_id=cot_id, cot_text=cot_text)

    def eval_single(rubric: RubricInfo) -> EvaluationResult:
        try:
            return evaluate_rubric(
                cot_text=cot_text,
                rubric=rubric,
                provider=provider,
                prompts_dir=prompts_dir,
            )
        except Exception as e:
            return EvaluationResult(
                dimension=rubric.id,
                score=0,
                evidence=[],
                reasoning=f"Evaluation failed: {e}",
            )

    # Run evaluations in parallel
    results_map: dict[str, EvaluationResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_rubric = {
            executor.submit(eval_single, rubric): rubric for rubric in rubrics
        }
        for future in as_completed(future_to_rubric):
            rubric = future_to_rubric[future]
            results_map[rubric.id] = future.result()

    # Preserve original rubric order
    for rubric in rubrics:
        evaluation.results.append(results_map[rubric.id])

    return evaluation


def evaluate_cot_subset(
    cot_text: str,
    rubric_ids: list[str],
    cot_id: str = "unnamed",
    provider: LLMProvider | None = None,
    backend: Literal["anthropic", "gemini"] = "anthropic",
    model: str | None = None,
    prompts_dir: Path | None = None,
    **provider_kwargs,
) -> CoTEvaluation:
    """Evaluate a CoT against a subset of rubrics by ID.

    Args:
        cot_text: The Chain of Thought text to evaluate.
        rubric_ids: List of rubric IDs to evaluate.
        cot_id: Identifier for this CoT sample.
        provider: LLM provider instance.
        backend: Which provider to use if provider is None.
        model: Model to use for evaluation.
        prompts_dir: Directory containing prompt files.
        **provider_kwargs: Additional provider-specific arguments.

    Returns:
        CoTEvaluation with results from specified rubrics.
    """
    rubrics = []
    for rid in rubric_ids:
        rubric = get_rubric_by_id(rid)
        if rubric is None:
            raise ValueError(f"Unknown rubric ID: {rid}")
        rubrics.append(rubric)

    return evaluate_cot(
        cot_text=cot_text,
        cot_id=cot_id,
        rubrics=rubrics,
        provider=provider,
        backend=backend,
        model=model,
        prompts_dir=prompts_dir,
        **provider_kwargs,
    )


def main():
    """CLI entrypoint for CoT evaluation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate Chain of Thought quality")
    parser.add_argument("cot_file", type=Path, help="File containing CoT text to evaluate")
    parser.add_argument(
        "--rubrics",
        nargs="+",
        help="Specific rubric IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        help="Model to use for evaluation (default: provider-specific)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (default: stdout)",
    )

    args = parser.parse_args()

    if not args.cot_file.exists():
        print(f"Error: CoT file not found: {args.cot_file}", file=sys.stderr)
        sys.exit(1)

    cot_text = args.cot_file.read_text()

    if args.rubrics:
        evaluation = evaluate_cot_subset(
            cot_text=cot_text,
            rubric_ids=args.rubrics,
            cot_id=args.cot_file.stem,
            backend=args.backend,
            model=args.model,
        )
    else:
        evaluation = evaluate_cot(
            cot_text=cot_text,
            cot_id=args.cot_file.stem,
            backend=args.backend,
            model=args.model,
        )

    output = json.dumps(evaluation.to_dict(), indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
