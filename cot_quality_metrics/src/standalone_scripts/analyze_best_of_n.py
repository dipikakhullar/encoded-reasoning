#!/usr/bin/env python3
"""
Analyze how best-of-n sampling affects rubric scores.

Given eval results for all rollouts, simulates best-of-n sampling at various n values
and plots how scores change with n.

Usage:
    uv run python analyze_best_of_n.py logs/EVAL_FILE.eval
    uv run python analyze_best_of_n.py --n-values 1 2 4 8 16 20
"""

import argparse
import gzip
import json
import random
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


# =============================================================================
# REPETITION METRICS
# =============================================================================

def tokenize_text(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def trigram_repetition_rate(text: str) -> float:
    """Fraction of unique trigrams that appear more than once.

    Higher = more repetitive = worse quality.
    """
    tokens = tokenize_text(text)
    ngrams = get_ngrams(tokens, 3)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    return repeated / len(counts) if counts else 0.0


def compression_ratio(text: str) -> float:
    """Ratio of gzip-compressed to original size.

    Lower = more compressible = more repetitive = worse quality.
    """
    if not text:
        return 1.0
    encoded = text.encode('utf-8')
    compressed = gzip.compress(encoded)
    return len(compressed) / len(encoded)


def compute_repetition_metrics(text: str) -> dict[str, float]:
    """Compute repetition metrics for a text."""
    return {
        "trigram_repetition": trigram_repetition_rate(text),
        "compression_ratio": compression_ratio(text),
    }


# Repetition metrics (added to samples, not LLM-judged)
REPETITION_METRICS = {"trigram_repetition", "compression_ratio"}

# =============================================================================


# Selection rules (must match sample_rollouts.py)
def select_shortest(rollouts: list[dict]) -> dict:
    """Select rollout with shortest response."""
    return min(rollouts, key=lambda r: len(r.get("input", "")))


def select_longest(rollouts: list[dict]) -> dict:
    """Select rollout with longest response."""
    return max(rollouts, key=lambda r: len(r.get("input", "")))


SELECTION_RULES = {
    "shortest": select_shortest,
    "longest": select_longest,
    "random": lambda rollouts: random.choice(rollouts),
}

# Rubric categories for normalization to 0-1 scale
NEGATIVE_RUBRICS = {"fake_rigor"}  # 0 to -5 scale, shift by +5 then /5
GDM_RUBRICS = {"gdm_legibility", "gdm_coverage"}  # 0-4 scale, /4
POSITIVE_RUBRICS_0_5 = {  # 0-5 scale, /5
    "active_investigation", "epistemic_honesty", "adaptive_process",
    "reportive_fidelity", "noticing_confusion", "live_updating",
    "discriminative_experiment_design", "appropriate_stopping",
    "generativity_under_stuckness", "error_metabolism", "calibration",
    "problem_decomposition", "assumption_surfacing", "negative_space_awareness",
    "authenticity", "contact_with_reality", "process_conclusion_integrity",
    "sufficiency_of_investigation", "provenance_transparency",
}

# Default model for tokenization (matches rollouts)
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = DEFAULT_MODEL) -> AutoTokenizer:
    """Load and cache the tokenizer."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_tokens(text: str, model_name: str = DEFAULT_MODEL) -> int:
    """Count tokens in text using the model's tokenizer."""
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))


def normalize_score(rubric: str, value: float) -> float:
    """Normalize score to 0-1 scale where higher is always better.

    - Negative rubrics (0 to -5): shift by +5, then /5
    - GDM rubrics (0-4): /4
    - Positive rubrics (0-5): /5
    - trigram_repetition (0-1, higher=worse): invert to 1-value
    - compression_ratio (0-1, higher=better): keep as-is
    """
    if rubric in NEGATIVE_RUBRICS:
        return (value + 5) / 5
    elif rubric in GDM_RUBRICS:
        return value / 4
    elif rubric == "trigram_repetition":
        return 1 - value  # Invert so higher = better (less repetitive)
    elif rubric == "compression_ratio":
        return value  # Already 0-1, higher = better (less compressible)
    elif rubric in POSITIVE_RUBRICS_0_5:
        return value / 5
    else:
        # Unknown rubric, assume 0-5 scale
        return value / 5


def extract_samples_from_eval(eval_path: Path, add_repetition_metrics: bool = True) -> list[dict]:
    """Extract all samples from eval file.

    Args:
        eval_path: Path to the .eval file
        add_repetition_metrics: If True, compute and add trigram_repetition and
            compression_ratio metrics to each sample's scores.
    """
    samples = []
    with zipfile.ZipFile(eval_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as f:
                    data = json.load(f)

                    # Add repetition metrics computed from the CoT text
                    if add_repetition_metrics:
                        cot_text = data.get("input", "")
                        if isinstance(cot_text, str) and cot_text:
                            rep_metrics = compute_repetition_metrics(cot_text)
                            if "scores" not in data:
                                data["scores"] = {}
                            for metric_name, value in rep_metrics.items():
                                data["scores"][metric_name] = {"value": value}

                    samples.append(data)
    return samples


def extract_header(eval_path: Path) -> dict:
    """Extract header from eval file."""
    with zipfile.ZipFile(eval_path, 'r') as zf:
        if "header.json" in zf.namelist():
            with zf.open("header.json") as f:
                return json.load(f)
    return {}


def group_samples_by_problem(samples: list[dict]) -> dict[tuple, list[dict]]:
    """Group samples by (alpha, problem_idx).

    Returns:
        {(alpha, problem_idx): [list of rollout samples]}

    If alpha is not present in metadata, uses "none" as a placeholder
    to support single-condition evals.
    """
    groups = defaultdict(list)
    for s in samples:
        meta = s.get("metadata", {})
        alpha = meta.get("alpha", "none")  # Default to "none" if no alpha
        prob_idx = meta.get("problem_idx")
        if prob_idx is not None:
            groups[(alpha, prob_idx)].append(s)
    return dict(groups)


def simulate_best_of_n(
    rollouts: list[dict],
    n: int,
    rule: str,
    rng: random.Random,
) -> dict:
    """Simulate best-of-n selection: sample n, pick winner by rule."""
    if n > len(rollouts):
        n = len(rollouts)  # Cap at available rollouts

    sampled = rng.sample(rollouts, n)
    selector = SELECTION_RULES[rule]
    return selector(sampled)


def compute_scores_for_n(
    grouped_samples: dict[tuple, list[dict]],
    n: int,
    rule: str,
    seed: int,
    rubrics: list[str],
) -> dict[str, list[float]]:
    """Compute rubric scores for best-of-n selection across all problems.

    Returns:
        {rubric: [scores across all (alpha, problem) combinations]}
    """
    rng = random.Random(seed)
    scores = {r: [] for r in rubrics}

    for (alpha, prob_idx), rollouts in sorted(grouped_samples.items()):
        # Select winner
        winner = simulate_best_of_n(rollouts, n, rule, rng)

        # Extract scores (normalize negative rubrics)
        for rubric in rubrics:
            score = winner.get("scores", {}).get(rubric, {}).get("value")
            if score is not None:
                scores[rubric].append(normalize_score(rubric, score))

    return scores


def compute_aggregate_scores_for_n(
    grouped_samples: dict[tuple, list[dict]],
    n: int,
    rule: str,
    seed: int,
    rubrics: list[str],
) -> dict[str, float]:
    """Compute mean scores for best-of-n across all problems.

    Returns:
        {rubric: mean_score, "non_gdm_avg": mean_avg, ...}
    """
    scores = compute_scores_for_n(grouped_samples, n, rule, seed, rubrics)

    result = {}
    for rubric, values in scores.items():
        if values:
            result[rubric] = statistics.mean(values)
            result[f"{rubric}_stderr"] = (
                statistics.stdev(values) / (len(values) ** 0.5)
                if len(values) > 1 else 0
            )

    # Compute non-GDM average
    gdm_rubrics = {"gdm_legibility", "gdm_coverage"}
    non_gdm = [r for r in rubrics if r not in gdm_rubrics]

    # Average per sample, then average across samples
    rng = random.Random(seed)
    non_gdm_avgs = []
    for (alpha, prob_idx), rollouts in sorted(grouped_samples.items()):
        winner = simulate_best_of_n(rollouts, n, rule, rng)
        values = [
            normalize_score(r, winner.get("scores", {}).get(r, {}).get("value", 0))
            for r in non_gdm
        ]
        non_gdm_avgs.append(statistics.mean(values) if values else 0)

    if non_gdm_avgs:
        result["non_gdm_avg"] = statistics.mean(non_gdm_avgs)
        result["non_gdm_avg_stderr"] = (
            statistics.stdev(non_gdm_avgs) / (len(non_gdm_avgs) ** 0.5)
            if len(non_gdm_avgs) > 1 else 0
        )

    return result


def save_best_of_n_plot(
    n_values: list[int],
    results: dict[int, dict[str, float]],
    rubrics: list[str],
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Plot rubric scores vs n (best-of-n)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(rubrics)))

    for i, rubric in enumerate(rubrics):
        means = [results[n].get(rubric, 0) for n in n_values]
        ax.plot(n_values, means, label=rubric,
                marker='o', linewidth=2, markersize=6, color=colors[i])

    ax.set_xlabel("n (best-of-n)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    title = "Rubric Scores vs Best-of-n"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_aggregate_best_of_n_plot(
    n_values: list[int],
    results: dict[int, dict[str, float]],
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Plot aggregate scores (non-GDM avg + GDM metrics) vs n."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Non-GDM average
    means = [results[n].get("non_gdm_avg", 0) for n in n_values]
    ax.plot(n_values, means, label="Non-GDM Avg",
            marker='o', linewidth=2, markersize=8, color='steelblue')

    # GDM legibility
    means = [results[n].get("gdm_legibility", 0) for n in n_values]
    ax.plot(n_values, means, label="GDM Legibility",
            marker='s', linewidth=2, markersize=8, color='forestgreen')

    # GDM coverage
    means = [results[n].get("gdm_coverage", 0) for n in n_values]
    ax.plot(n_values, means, label="GDM Coverage",
            marker='^', linewidth=2, markersize=8, color='darkorange')

    ax.set_xlabel("n (best-of-n)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    title = "Aggregate Scores vs Best-of-n"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_grid_by_alpha(
    n_values: list[int],
    samples: list[dict],
    rubrics: list[str],
    rule: str,
    seed: int,
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save 2x2 grid of plots, one per alpha value."""
    # Group by alpha first
    alpha_groups = defaultdict(list)
    for s in samples:
        alpha = s.get("metadata", {}).get("alpha", "none")  # Default to "none"
        alpha_groups[alpha].append(s)

    alpha_values = sorted(alpha_groups.keys(), key=lambda x: (x == "none", x))
    if not alpha_values:
        return

    n_alphas = len(alpha_values)
    n_cols = 2
    n_rows = (n_alphas + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    gdm_rubrics = {"gdm_legibility", "gdm_coverage"}
    non_gdm = [r for r in rubrics if r not in gdm_rubrics]

    for idx, alpha in enumerate(alpha_values):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Group this alpha's samples by problem
        alpha_samples = alpha_groups[alpha]
        prob_groups = defaultdict(list)
        for s in alpha_samples:
            prob_idx = s.get("metadata", {}).get("problem_idx")
            if prob_idx is not None:
                prob_groups[prob_idx].append(s)

        # Compute scores for each n
        non_gdm_means = []
        non_gdm_errs = []
        gdm_leg_means = []
        gdm_leg_errs = []
        gdm_cov_means = []
        gdm_cov_errs = []

        for n in n_values:
            rng = random.Random(seed)
            non_gdm_avgs = []
            leg_vals = []
            cov_vals = []

            for prob_idx, rollouts in sorted(prob_groups.items()):
                winner = simulate_best_of_n(rollouts, n, rule, rng)

                values = [
                    normalize_score(r, winner.get("scores", {}).get(r, {}).get("value", 0))
                    for r in non_gdm
                ]
                non_gdm_avgs.append(statistics.mean(values) if values else 0)

                leg_vals.append(winner.get("scores", {}).get("gdm_legibility", {}).get("value", 0))
                cov_vals.append(winner.get("scores", {}).get("gdm_coverage", {}).get("value", 0))

            non_gdm_means.append(statistics.mean(non_gdm_avgs) if non_gdm_avgs else 0)
            non_gdm_errs.append(
                statistics.stdev(non_gdm_avgs) / (len(non_gdm_avgs) ** 0.5)
                if len(non_gdm_avgs) > 1 else 0
            )
            gdm_leg_means.append(statistics.mean(leg_vals) if leg_vals else 0)
            gdm_leg_errs.append(
                statistics.stdev(leg_vals) / (len(leg_vals) ** 0.5)
                if len(leg_vals) > 1 else 0
            )
            gdm_cov_means.append(statistics.mean(cov_vals) if cov_vals else 0)
            gdm_cov_errs.append(
                statistics.stdev(cov_vals) / (len(cov_vals) ** 0.5)
                if len(cov_vals) > 1 else 0
            )

        # Plot with error bars (stderr across problems)
        ax.errorbar(n_values, non_gdm_means, yerr=non_gdm_errs,
                    label=f"Non-GDM Avg ({len(non_gdm)})", marker='o',
                    capsize=2, linewidth=1.5, markersize=5, color='steelblue')
        ax.errorbar(n_values, gdm_leg_means, yerr=gdm_leg_errs,
                    label="GDM Legibility", marker='s',
                    capsize=2, linewidth=1.5, markersize=5, color='forestgreen')
        ax.errorbar(n_values, gdm_cov_means, yerr=gdm_cov_errs,
                    label="GDM Coverage", marker='^',
                    capsize=2, linewidth=1.5, markersize=5, color='darkorange')

        ax.set_xlabel("n (best-of-n)", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(f"alpha={alpha}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_ylim(0, 1.2)

    # Hide unused
    for idx in range(n_alphas, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)

    # Title
    title = "Best-of-n Scores by Alpha"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_token_counts_for_n(
    grouped_samples: dict[tuple, list[dict]],
    n: int,
    rule: str,
    seed: int,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, float]:
    """Compute token count stats for best-of-n across all problems.

    Returns:
        {"mean": mean_tokens, "stderr": stderr, "neg_mean": -mean_tokens, ...}
    """
    rng = random.Random(seed)
    token_counts = []

    for (alpha, prob_idx), rollouts in sorted(grouped_samples.items()):
        winner = simulate_best_of_n(rollouts, n, rule, rng)
        reasoning_text = winner.get("input", "")
        tokens = count_tokens(reasoning_text, model_name)
        token_counts.append(tokens)

    if not token_counts:
        return {"mean": 0, "stderr": 0, "neg_mean": 0, "neg_stderr": 0}

    mean = statistics.mean(token_counts)
    stderr = statistics.stdev(token_counts) / (len(token_counts) ** 0.5) if len(token_counts) > 1 else 0

    return {
        "mean": mean,
        "stderr": stderr,
        "neg_mean": -mean,
        "neg_stderr": stderr,
    }


def save_token_count_plot(
    n_values: list[int],
    token_results: dict[int, dict[str, float]],
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Plot token count vs n (best-of-n) as negative reward."""
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [token_results[n]["neg_mean"] for n in n_values]

    ax.plot(n_values, means, label="−Token Count",
            marker='o', linewidth=2, markersize=8, color='purple')

    ax.set_xlabel("n (best-of-n)", fontsize=12)
    ax.set_ylabel("−Token Count (reward)", fontsize=12)

    title = "Token Count vs Best-of-n"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_individual_metrics_grid_by_alpha(
    n_values: list[int],
    samples: list[dict],
    rubrics: list[str],
    rule: str,
    seed: int,
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save 2x2 grid showing all individual metrics, one subplot per alpha value."""
    # Group by alpha first
    alpha_groups = defaultdict(list)
    for s in samples:
        alpha = s.get("metadata", {}).get("alpha", "none")
        alpha_groups[alpha].append(s)

    alpha_values = sorted(alpha_groups.keys(), key=lambda x: (x == "none", x))
    if not alpha_values:
        return

    n_alphas = len(alpha_values)
    n_cols = 2
    n_rows = (n_alphas + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color palette for rubrics
    colors = plt.cm.tab10(np.linspace(0, 1, len(rubrics)))

    for idx, alpha in enumerate(alpha_values):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Group this alpha's samples by problem
        alpha_samples = alpha_groups[alpha]
        prob_groups = defaultdict(list)
        for s in alpha_samples:
            prob_idx = s.get("metadata", {}).get("problem_idx")
            if prob_idx is not None:
                prob_groups[prob_idx].append(s)

        # Compute scores for each rubric at each n
        for rubric_idx, rubric in enumerate(rubrics):
            means = []
            errs = []

            for n in n_values:
                rng = random.Random(seed)
                values = []

                for prob_idx, rollouts in sorted(prob_groups.items()):
                    winner = simulate_best_of_n(rollouts, n, rule, rng)
                    score = winner.get("scores", {}).get(rubric, {}).get("value")
                    if score is not None:
                        values.append(normalize_score(rubric, score))

                means.append(statistics.mean(values) if values else 0)
                errs.append(
                    statistics.stdev(values) / (len(values) ** 0.5)
                    if len(values) > 1 else 0
                )

            ax.errorbar(n_values, means, yerr=errs, label=rubric, marker='o',
                        capsize=2, linewidth=1.5, markersize=4, color=colors[rubric_idx])

        ax.set_xlabel("n (best-of-n)", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(f"alpha={alpha}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_ylim(0, 1.2)

    # Hide unused
    for idx in range(n_alphas, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Legend outside
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=8)

    # Title
    title = "Individual Metrics by Alpha"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_token_count_grid_by_alpha(
    n_values: list[int],
    samples: list[dict],
    rule: str,
    seed: int,
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save 2x2 grid of token count plots (as -1 * tokens), one per alpha value.

    We plot -1 * tokens because selecting shortest = negative reward for tokens.
    """
    # Group by alpha first
    alpha_groups = defaultdict(list)
    for s in samples:
        alpha = s.get("metadata", {}).get("alpha", "none")
        alpha_groups[alpha].append(s)

    alpha_values = sorted(alpha_groups.keys(), key=lambda x: (x == "none", x))
    if not alpha_values:
        return

    n_alphas = len(alpha_values)
    n_cols = 2
    n_rows = (n_alphas + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Get model name for tokenizer
    model_name = samples[0].get("metadata", {}).get("model", DEFAULT_MODEL)

    # First pass: compute all token means and errors to find global min/max
    all_token_data = {}  # {alpha: {"means": [...], "errs": [...]}}
    for alpha in alpha_values:
        alpha_samples = alpha_groups[alpha]
        prob_groups = defaultdict(list)
        for s in alpha_samples:
            prob_idx = s.get("metadata", {}).get("problem_idx")
            if prob_idx is not None:
                prob_groups[prob_idx].append(s)

        token_means = []
        token_errs = []
        for n in n_values:
            rng = random.Random(seed)
            token_counts = []
            for prob_idx, rollouts in sorted(prob_groups.items()):
                winner = simulate_best_of_n(rollouts, n, rule, rng)
                reasoning_text = winner.get("input", "")
                tokens = count_tokens(reasoning_text, model_name)
                token_counts.append(-tokens)
            token_means.append(statistics.mean(token_counts) if token_counts else 0)
            token_errs.append(
                statistics.stdev(token_counts) / (len(token_counts) ** 0.5)
                if len(token_counts) > 1 else 0
            )
        all_token_data[alpha] = {"means": token_means, "errs": token_errs}

    # Find global min/max across all alphas (including error bars)
    all_mins = []
    all_maxs = []
    for alpha, data in all_token_data.items():
        for m, e in zip(data["means"], data["errs"]):
            all_mins.append(m - e)
            all_maxs.append(m + e)
    global_min = min(all_mins)
    global_max = max(all_maxs)
    # Add small padding
    padding = (global_max - global_min) * 0.05
    y_min = global_min - padding
    y_max = global_max + padding

    # Second pass: plot with consistent y-axis
    for idx, alpha in enumerate(alpha_values):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        data = all_token_data[alpha]
        ax.errorbar(n_values, data["means"], yerr=data["errs"],
                    label="−Token Count", marker='o',
                    capsize=2, linewidth=1.5, markersize=5, color='purple')

        ax.set_xlabel("n (best-of-n)", fontsize=10)
        ax.set_ylabel("−Token Count (reward)", fontsize=10)
        ax.set_title(f"alpha={alpha}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_ylim(y_min, y_max)

    # Hide unused
    for idx in range(n_alphas, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)

    # Title
    title = "Best-of-n Token Count by Alpha"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_vs_alpha(
    samples: list[dict],
    rubrics: list[str],
    n: int,
    rule: str,
    seed: int,
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save plot showing all metrics vs alpha (x-axis) for fixed n.

    This shows how metrics degrade as corruption (alpha) increases.
    """
    # Group by alpha
    alpha_groups = defaultdict(list)
    for s in samples:
        alpha = s.get("metadata", {}).get("alpha")
        if alpha is not None:  # Only include samples with actual alpha values
            alpha_groups[alpha].append(s)

    alpha_values = sorted(alpha_groups.keys())
    if len(alpha_values) < 2:  # Need at least 2 alpha values for a meaningful plot
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(rubrics)))

    for rubric_idx, rubric in enumerate(rubrics):
        means = []
        errs = []

        for alpha in alpha_values:
            # Group this alpha's samples by problem
            alpha_samples = alpha_groups[alpha]
            prob_groups = defaultdict(list)
            for s in alpha_samples:
                prob_idx = s.get("metadata", {}).get("problem_idx")
                if prob_idx is not None:
                    prob_groups[prob_idx].append(s)

            # Compute scores for this alpha
            rng = random.Random(seed)
            values = []

            for prob_idx, rollouts in sorted(prob_groups.items()):
                winner = simulate_best_of_n(rollouts, n, rule, rng)
                score = winner.get("scores", {}).get(rubric, {}).get("value")
                if score is not None:
                    values.append(normalize_score(rubric, score))

            means.append(statistics.mean(values) if values else 0)
            errs.append(
                statistics.stdev(values) / (len(values) ** 0.5)
                if len(values) > 1 else 0
            )

        ax.errorbar(alpha_values, means, yerr=errs, label=rubric, marker='o',
                    capsize=3, linewidth=2, markersize=6, color=colors[rubric_idx])

    ax.set_xlabel("Alpha (corruption level)", fontsize=12)
    ax.set_ylabel("Score (normalized 0-1, higher=better)", fontsize=12)
    ax.set_xticks(alpha_values)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    title = f"All Metrics vs Alpha (best-of-{n})"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if subtitle_parts:
        title += f"\n({', '.join(subtitle_parts)})"

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_most_recent_eval(logs_dir: Path) -> Path | None:
    """Find most recent .eval file."""
    eval_files = list(logs_dir.glob("*.eval"))
    if not eval_files:
        return None
    return max(eval_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best-of-n sampling effect on rubric scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "eval_file",
        nargs="?",
        type=Path,
        help="Path to .eval file (default: most recent in logs/)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        help="n values to analyze (default: 1 2 4 6 8 10 12 14 16 18 20)",
    )
    parser.add_argument(
        "--rule",
        type=str,
        choices=list(SELECTION_RULES.keys()),
        default="shortest",
        help="Selection rule for best-of-n (default: shortest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: logs/analyses/)",
    )
    args = parser.parse_args()

    # Project root
    project_root = Path(__file__).parent.parent.parent.parent

    # Find eval file
    if args.eval_file:
        eval_path = args.eval_file
    else:
        logs_dir = project_root / "logs"
        eval_path = find_most_recent_eval(logs_dir)
        if eval_path is None:
            print(f"No .eval files found in {logs_dir}")
            return 1

    if not eval_path.exists():
        print(f"File not found: {eval_path}")
        return 1

    print(f"Analyzing: {eval_path}")
    print(f"n values: {args.n_values}")
    print(f"Selection rule: {args.rule}")

    # Extract data
    samples = extract_samples_from_eval(eval_path)
    header = extract_header(eval_path)

    if not samples:
        print("No samples found")
        return 1

    print(f"Total samples: {len(samples)}")

    # Extract model names
    target_model = samples[0].get("metadata", {}).get("model", "").split("/")[-1]
    target_model = target_model.replace("-Instruct", "")
    judge_model = header.get("eval", {}).get("model", "").split("/")[-1]

    print(f"Target model: {target_model}")
    print(f"Judge model: {judge_model}")

    # Get rubrics
    rubrics = list(samples[0].get("scores", {}).keys())
    print(f"Rubrics: {', '.join(rubrics)}")

    # Group samples by alpha
    alpha_groups = defaultdict(list)
    for s in samples:
        alpha = s.get("metadata", {}).get("alpha")
        if alpha is not None:
            alpha_groups[alpha].append(s)
    print(f"Alpha values: {sorted(alpha_groups.keys())}")
    print(f"Problems per alpha: {len(set(s.get('metadata', {}).get('problem_idx') for s in samples))}")

    # Output directory
    output_dir = args.output_dir or (project_root / "logs" / "analyses")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"best_of_n_{eval_path.stem}"

    # Save plots (all by-alpha, no cross-alpha averaging)
    # 1. Grid by alpha (aggregate scores)
    grid_path = output_dir / f"{base_name}_grid_by_alpha.png"
    save_grid_by_alpha(args.n_values, samples, rubrics, args.rule, args.seed,
                       grid_path, target_model=target_model, judge_model=judge_model)
    print(f"\nGrid by alpha saved to: {grid_path}")

    # 2. Grid by alpha (individual metrics)
    metrics_grid_path = output_dir / f"{base_name}_metrics_grid_by_alpha.png"
    save_individual_metrics_grid_by_alpha(args.n_values, samples, rubrics, args.rule,
                                          args.seed, metrics_grid_path,
                                          target_model=target_model, judge_model=judge_model)
    print(f"Individual metrics grid saved to: {metrics_grid_path}")

    # 3. Token count grid by alpha
    print("\nCounting tokens (loading tokenizer)...")
    token_grid_path = output_dir / f"{base_name}_tokens_grid_by_alpha.png"
    save_token_count_grid_by_alpha(args.n_values, samples, args.rule, args.seed,
                                    token_grid_path, target_model=target_model,
                                    judge_model=judge_model)
    print(f"Token count grid saved to: {token_grid_path}")

    # 4. Metrics vs alpha (standalone plot at n=1 and max n)
    for n in [1, max(args.n_values)]:
        metrics_vs_alpha_path = output_dir / f"{base_name}_metrics_vs_alpha_n{n}.png"
        save_metrics_vs_alpha(samples, rubrics, n, args.rule, args.seed,
                              metrics_vs_alpha_path, target_model=target_model,
                              judge_model=judge_model)
        print(f"Metrics vs alpha (n={n}) saved to: {metrics_vs_alpha_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
