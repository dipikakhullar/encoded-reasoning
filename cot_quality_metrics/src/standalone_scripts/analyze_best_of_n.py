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
import json
import random
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

# Negative rubrics are shifted by +5 to make them comparable to positive ones
NEGATIVE_RUBRICS = {"fake_rigor"}


def normalize_score(rubric: str, value: float) -> float:
    """Normalize score: shift negative rubrics by +5 to positive scale."""
    if rubric in NEGATIVE_RUBRICS:
        return value + 5
    return value


def extract_samples_from_eval(eval_path: Path) -> list[dict]:
    """Extract all samples from eval file."""
    samples = []
    with zipfile.ZipFile(eval_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as f:
                    samples.append(json.load(f))
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
    """
    groups = defaultdict(list)
    for s in samples:
        meta = s.get("metadata", {})
        alpha = meta.get("alpha")
        prob_idx = meta.get("problem_idx")
        if alpha is not None and prob_idx is not None:
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
        errs = [results[n].get(f"{rubric}_stderr", 0) for n in n_values]

        ax.errorbar(n_values, means, yerr=errs, label=rubric,
                   marker='o', capsize=3, linewidth=2, markersize=6, color=colors[i])

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
    errs = [results[n].get("non_gdm_avg_stderr", 0) for n in n_values]
    ax.errorbar(n_values, means, yerr=errs, label="Non-GDM Avg",
               marker='o', capsize=3, linewidth=2, markersize=8, color='steelblue')

    # GDM legibility
    means = [results[n].get("gdm_legibility", 0) for n in n_values]
    errs = [results[n].get("gdm_legibility_stderr", 0) for n in n_values]
    ax.errorbar(n_values, means, yerr=errs, label="GDM Legibility",
               marker='s', capsize=3, linewidth=2, markersize=8, color='forestgreen')

    # GDM coverage
    means = [results[n].get("gdm_coverage", 0) for n in n_values]
    errs = [results[n].get("gdm_coverage_stderr", 0) for n in n_values]
    ax.errorbar(n_values, means, yerr=errs, label="GDM Coverage",
               marker='^', capsize=3, linewidth=2, markersize=8, color='darkorange')

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
        alpha = s.get("metadata", {}).get("alpha")
        if alpha is not None:
            alpha_groups[alpha].append(s)

    alpha_values = sorted(alpha_groups.keys())
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

        # Plot
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

    # Group samples
    grouped = group_samples_by_problem(samples)
    print(f"Problem groups: {len(grouped)} (alpha × problem combinations)")

    # Compute scores for each n
    results = {}
    for n in args.n_values:
        results[n] = compute_aggregate_scores_for_n(
            grouped, n, args.rule, args.seed, rubrics
        )
        print(f"  n={n}: non_gdm_avg={results[n].get('non_gdm_avg', 0):.2f}")

    # Output directory
    output_dir = args.output_dir or (project_root / "logs" / "analyses")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"best_of_n_{eval_path.stem}"

    # Save plots
    # 1. All rubrics vs n
    plot_path = output_dir / f"{base_name}_rubrics.png"
    save_best_of_n_plot(args.n_values, results, rubrics, plot_path,
                        target_model=target_model, judge_model=judge_model)
    print(f"\nRubrics plot saved to: {plot_path}")

    # 2. Aggregate vs n
    agg_path = output_dir / f"{base_name}_aggregate.png"
    save_aggregate_best_of_n_plot(args.n_values, results, agg_path,
                                   target_model=target_model, judge_model=judge_model)
    print(f"Aggregate plot saved to: {agg_path}")

    # 3. Grid by alpha
    grid_path = output_dir / f"{base_name}_grid_by_alpha.png"
    save_grid_by_alpha(args.n_values, samples, rubrics, args.rule, args.seed,
                       grid_path, target_model=target_model, judge_model=judge_model)
    print(f"Grid by alpha saved to: {grid_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Best-of-n Analysis Summary")
    print("=" * 60)
    print(f"{'n':<6} {'Non-GDM Avg':<15} {'GDM Legibility':<15} {'GDM Coverage':<15}")
    print("-" * 60)
    for n in args.n_values:
        r = results[n]
        print(f"{n:<6} {r.get('non_gdm_avg', 0):>7.2f}±{r.get('non_gdm_avg_stderr', 0):<6.2f}"
              f"{r.get('gdm_legibility', 0):>7.2f}±{r.get('gdm_legibility_stderr', 0):<6.2f}"
              f"{r.get('gdm_coverage', 0):>7.2f}±{r.get('gdm_coverage_stderr', 0):<6.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
