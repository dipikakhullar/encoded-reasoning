"""Analyze results from CoT prefill experiment.

Usage:
    uv run python testing_on_server/analyze_cot_prefill.py --results-dir outputs/run_TIMESTAMP/
    uv run python testing_on_server/analyze_cot_prefill.py --results-dirs outputs/run_A/ outputs/run_B/ ...
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (point_estimate, lower, upper).
    """
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)


def load_results(results_path: Path) -> list[dict]:
    """Load results from JSONL."""
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute per-condition statistics from results."""
    by_condition = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_condition[r["condition"]].append(r)

    stats = {}
    for condition, rows in by_condition.items():
        total = len(rows)
        correct = sum(1 for r in rows if r.get("is_correct"))
        parsed = sum(1 for r in rows if r.get("extracted_answer") is not None)
        acc, ci_lo, ci_hi = wilson_ci(correct, total)

        entry = {
            "total": total,
            "correct": correct,
            "parsed": parsed,
            "accuracy": acc,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "parse_rate": parsed / total if total > 0 else 0,
        }

        # For wrong_cot: how often did model follow the wrong answer?
        if condition == "wrong_cot":
            followed = sum(1 for r in rows if r.get("followed_wrong_cot"))
            entry["followed_wrong_cot"] = followed
            entry["followed_wrong_rate"] = followed / total if total > 0 else 0

        stats[condition] = entry

    return stats


CONDITION_ORDER = [
    "normal_cot",
    "correct_cot",
    "no_cot",
    "truncated_cot",
    "wrong_cot",
    "nonsense_lorem",
    "nonsense_random",
]

CONDITION_COLORS = {
    "normal_cot": "#2ecc71",
    "correct_cot": "#27ae60",
    "no_cot": "#3498db",
    "truncated_cot": "#f39c12",
    "wrong_cot": "#e74c3c",
    "nonsense_lorem": "#9b59b6",
    "nonsense_random": "#8e44ad",
}

CONDITION_LABELS = {
    "normal_cot": "Normal CoT",
    "correct_cot": "Correct CoT\n(prefilled)",
    "no_cot": "No CoT\n(empty think)",
    "truncated_cot": "Truncated CoT\n(50%)",
    "wrong_cot": "Wrong CoT\n(prefilled)",
    "nonsense_lorem": "Nonsense\n(lorem)",
    "nonsense_random": "Nonsense\n(random)",
}


def plot_single_model(stats: dict, model_name: str, output_path: Path):
    """Bar chart of accuracy by condition for a single model."""
    conditions = [c for c in CONDITION_ORDER if c in stats]
    accuracies = [stats[c]["accuracy"] for c in conditions]
    ci_lo = [stats[c]["accuracy"] - stats[c]["ci_lower"] for c in conditions]
    ci_hi = [stats[c]["ci_upper"] - stats[c]["accuracy"] for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#95a5a6") for c in conditions]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(conditions))
    bars = ax.bar(x, accuracies, yerr=[ci_lo, ci_hi], capsize=4, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"CoT Prefill Experiment: {model_name}", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add wrong_cot follow rate annotation if present
    if "wrong_cot" in stats and "followed_wrong_rate" in stats["wrong_cot"]:
        rate = stats["wrong_cot"]["followed_wrong_rate"]
        idx = conditions.index("wrong_cot")
        ax.text(idx, -0.08, f"Followed wrong: {rate:.0%}", ha="center", fontsize=8,
                color="#e74c3c", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_multi_model(all_stats: dict[str, dict], output_path: Path):
    """Grouped bar chart comparing accuracy across models."""
    models = list(all_stats.keys())
    # Find conditions present in all models
    common_conditions = None
    for stats in all_stats.values():
        conds = set(stats.keys())
        common_conditions = conds if common_conditions is None else common_conditions & conds
    conditions = [c for c in CONDITION_ORDER if c in common_conditions]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(conditions))
    width = 0.8 / len(models)
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model_name, stats) in enumerate(all_stats.items()):
        accs = [stats[c]["accuracy"] for c in conditions]
        ci_lo = [stats[c]["accuracy"] - stats[c]["ci_lower"] for c in conditions]
        ci_hi = [stats[c]["ci_upper"] - stats[c]["accuracy"] for c in conditions]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, yerr=[ci_lo, ci_hi], capsize=2,
                       label=model_name.split("/")[-1], color=model_colors[i], alpha=0.85)

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("CoT Prefill Experiment: Model Comparison", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_accuracy_table(all_stats: dict[str, dict]):
    """Print accuracy matrix: models × conditions."""
    models = list(all_stats.keys())
    all_conds = set()
    for stats in all_stats.values():
        all_conds.update(stats.keys())
    conditions = [c for c in CONDITION_ORDER if c in all_conds]

    # Header
    header = f"{'Model':>40s} | " + " | ".join(f"{c:>15s}" for c in conditions)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name in models:
        stats = all_stats[model_name]
        short_name = model_name.split("/")[-1]
        row = f"{short_name:>40s} | "
        cells = []
        for c in conditions:
            if c in stats:
                acc = stats[c]["accuracy"]
                cells.append(f"{acc:>14.1%}")
            else:
                cells.append(f"{'N/A':>15s}")
        row += " | ".join(cells)
        print(row)

    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Analyze CoT Prefill Experiment Results")
    parser.add_argument("--results-dir", type=str, help="Single results directory")
    parser.add_argument("--results-dirs", nargs="+", type=str, help="Multiple results directories (multi-model)")

    args = parser.parse_args()

    if args.results_dir:
        dirs = [args.results_dir]
    elif args.results_dirs:
        dirs = args.results_dirs
    else:
        parser.error("Provide --results-dir or --results-dirs")

    all_stats = {}

    for d in dirs:
        d = Path(d)
        results_path = d / "results.jsonl"
        if not results_path.exists():
            print(f"Warning: {results_path} not found, skipping")
            continue

        # Get model name from metadata
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_name = meta.get("model", d.name)
        else:
            model_name = d.name

        results = load_results(results_path)
        stats = compute_stats(results)
        all_stats[model_name] = stats

        # Per-model plot
        plots_dir = d / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_single_model(stats, model_name, plots_dir / "accuracy_by_condition.png")

        # Save stats
        with open(d / "analysis.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

    # Print table
    print_accuracy_table(all_stats)

    # Multi-model comparison plot (if multiple)
    if len(all_stats) > 1:
        # Save to first directory
        first_dir = Path(dirs[0])
        plot_multi_model(all_stats, first_dir / "plots" / "model_comparison.png")

        # Also save combined analysis
        with open(first_dir / "combined_analysis.json", "w") as f:
            json.dump(all_stats, f, indent=2, default=str)


if __name__ == "__main__":
    main()
