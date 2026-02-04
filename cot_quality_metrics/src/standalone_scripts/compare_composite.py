#!/usr/bin/env python3
"""Compare composite rubrics against full 34-rubric evaluation.

Usage:
    uv run python cot_quality_metrics/src/standalone_scripts/compare_composite.py
    uv run python cot_quality_metrics/src/standalone_scripts/compare_composite.py full.eval composite.eval
"""

import json
import statistics
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Composite rubric definitions - which original rubrics they combine
COMPOSITE_COMPONENTS = {
    "fake_rigor": [
        "premature_formalization",
        "cargo_cult_methodology",
        "complexity_theater",
        "intellectual_flinching",
    ],
    "too_clean": [
        "no_self_interruption",
        "unnatural_smoothness",
        "suspiciously_complete_resolution",
    ],
    "active_investigation": [
        "discriminative_experiment_design",
        "error_metabolism",
        "contact_with_reality",
    ],
    "epistemic_honesty": [
        "calibration",
        "provenance_transparency",
        "process_conclusion_integrity",
    ],
    "adaptive_process": [
        "generativity_under_stuckness",
        "noticing_confusion",
        "live_updating",
    ],
}

COMPOSITE_RUBRICS = list(COMPOSITE_COMPONENTS.keys())


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def extract_samples(eval_path: Path) -> tuple[dict, dict]:
    """Extract samples and header from eval file."""
    samples = {}
    header = {}
    with zipfile.ZipFile(eval_path, "r") as zf:
        if "header.json" in zf.namelist():
            with zf.open("header.json") as f:
                header = json.load(f)
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as f:
                    data = json.load(f)
                    samples[data["id"]] = data
    return header, samples


def find_eval_files(logs_dir: Path) -> tuple[Path | None, Path | None]:
    """Find the most recent full and composite eval files."""
    eval_files = list(logs_dir.glob("*.eval"))

    full_evals = [f for f in eval_files if "cot-quality-eval" in f.name]
    composite_evals = [f for f in eval_files if "cot-quality-composite" in f.name]

    full_eval = max(full_evals, key=lambda p: p.stat().st_mtime) if full_evals else None
    composite_eval = max(composite_evals, key=lambda p: p.stat().st_mtime) if composite_evals else None

    return full_eval, composite_eval


def compute_synthetic_composite(sample: dict, composite_name: str) -> float:
    """Compute what the composite score would be by averaging component rubrics."""
    components = COMPOSITE_COMPONENTS[composite_name]
    scores = sample.get("scores", {})

    component_scores = []
    for comp in components:
        if comp in scores:
            component_scores.append(scores[comp].get("value", 0))

    if not component_scores:
        return 0.0
    return statistics.mean(component_scores)


def generate_comparison_report(
    full_path: Path,
    composite_path: Path,
    full_header: dict,
    full_samples: dict,
    composite_header: dict,
    composite_samples: dict,
) -> str:
    """Generate comparison report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("COMPOSITE VS FULL RUBRIC COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Full eval: {full_path.name}")
    lines.append(f"  Model: {full_header.get('eval', {}).get('model', 'unknown')}")
    lines.append(f"  Samples: {len(full_samples)}")
    lines.append("")
    lines.append(f"Composite eval: {composite_path.name}")
    lines.append(f"  Model: {composite_header.get('eval', {}).get('model', 'unknown')}")
    lines.append(f"  Samples: {len(composite_samples)}")
    lines.append("")
    lines.append(f"Analysis time: {datetime.now().isoformat()}")
    lines.append("")

    # Find common samples
    common_ids = set(full_samples.keys()) & set(composite_samples.keys())
    lines.append(f"Common samples: {len(common_ids)}")
    lines.append("")

    if not common_ids:
        lines.append("ERROR: No common samples found!")
        return "\n".join(lines)

    # For each composite, compare:
    # 1. Composite score vs average of component scores from full eval
    # 2. Correlation between composite and each component

    lines.append("-" * 80)
    lines.append("COMPOSITE VS SYNTHETIC (averaged components from full eval)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("For each composite, we compare:")
    lines.append("  - Actual composite score (from composite eval)")
    lines.append("  - Synthetic score (average of component rubrics from full eval)")
    lines.append("")

    all_composite_scores = []
    all_synthetic_scores = []

    for composite_name in COMPOSITE_RUBRICS:
        composite_scores = []
        synthetic_scores = []

        for sid in common_ids:
            # Get actual composite score
            comp_score = composite_samples[sid]["scores"].get(composite_name, {}).get("value", 0)
            composite_scores.append(comp_score)

            # Compute synthetic from full eval
            synth_score = compute_synthetic_composite(full_samples[sid], composite_name)
            synthetic_scores.append(synth_score)

        all_composite_scores.extend(composite_scores)
        all_synthetic_scores.extend(synthetic_scores)

        corr = pearson_correlation(composite_scores, synthetic_scores)
        comp_mean = statistics.mean(composite_scores)
        synth_mean = statistics.mean(synthetic_scores)
        diff = comp_mean - synth_mean

        lines.append(f"{composite_name}:")
        lines.append(f"  Composite mean: {comp_mean:>7.2f}")
        lines.append(f"  Synthetic mean: {synth_mean:>7.2f}")
        lines.append(f"  Difference:     {diff:>+7.2f}")
        lines.append(f"  Correlation:    {corr:>7.3f}" if corr else "  Correlation:      N/A")
        lines.append(f"  Components: {', '.join(COMPOSITE_COMPONENTS[composite_name])}")
        lines.append("")

    overall_corr = pearson_correlation(all_composite_scores, all_synthetic_scores)
    lines.append(f"Overall correlation (all composites): {overall_corr:.3f}" if overall_corr else "N/A")
    lines.append("")

    # Detailed component correlations
    lines.append("-" * 80)
    lines.append("COMPOSITE VS INDIVIDUAL COMPONENT CORRELATIONS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("How well does each composite correlate with its component rubrics?")
    lines.append("")

    for composite_name in COMPOSITE_RUBRICS:
        lines.append(f"{composite_name}:")

        composite_scores = [
            composite_samples[sid]["scores"].get(composite_name, {}).get("value", 0)
            for sid in common_ids
        ]

        for component in COMPOSITE_COMPONENTS[composite_name]:
            component_scores = [
                full_samples[sid]["scores"].get(component, {}).get("value", 0)
                for sid in common_ids
            ]
            corr = pearson_correlation(composite_scores, component_scores)
            corr_str = f"{corr:>6.3f}" if corr else "   N/A"
            lines.append(f"  vs {component:<35} r = {corr_str}")
        lines.append("")

    # Total score comparison
    lines.append("-" * 80)
    lines.append("TOTAL SCORE COMPARISON")
    lines.append("-" * 80)
    lines.append("")

    # Compute total scores for each sample
    full_totals = []
    composite_totals = []

    for sid in common_ids:
        # Full eval: sum of all rubric scores
        full_scores = full_samples[sid].get("scores", {})
        full_total = sum(s.get("value", 0) for s in full_scores.values())
        full_totals.append(full_total)

        # Composite: sum of 5 composite scores
        comp_scores = composite_samples[sid].get("scores", {})
        comp_total = sum(s.get("value", 0) for s in comp_scores.values())
        composite_totals.append(comp_total)

    total_corr = pearson_correlation(full_totals, composite_totals)

    lines.append(f"Full eval total (34 rubrics) mean:    {statistics.mean(full_totals):>7.2f}")
    lines.append(f"Composite total (5 rubrics) mean:     {statistics.mean(composite_totals):>7.2f}")
    lines.append(f"Correlation between totals:           {total_corr:>7.3f}" if total_corr else "N/A")
    lines.append("")
    lines.append("This correlation indicates how well the 5 composites rank samples")
    lines.append("compared to the full 34-rubric evaluation.")
    lines.append("")

    # Per-sample comparison
    lines.append("-" * 80)
    lines.append("PER-SAMPLE COMPARISON (first 20 samples)")
    lines.append("-" * 80)
    lines.append("")

    sample_data = []
    for sid in common_ids:
        full_total = sum(
            full_samples[sid]["scores"].get(r, {}).get("value", 0)
            for r in full_samples[sid].get("scores", {})
        )
        comp_total = sum(
            composite_samples[sid]["scores"].get(r, {}).get("value", 0)
            for r in COMPOSITE_RUBRICS
        )
        sample_data.append((sid, full_total, comp_total))

    # Sort by full total
    sample_data.sort(key=lambda x: x[1], reverse=True)

    lines.append(f"{'Sample ID':<50} {'Full':>8} {'Comp':>8} {'Rank':>8}")
    lines.append("-" * 80)

    for i, (sid, full_total, comp_total) in enumerate(sample_data[:20]):
        # Find composite rank
        comp_rank = sorted(sample_data, key=lambda x: x[2], reverse=True).index((sid, full_total, comp_total)) + 1
        lines.append(f"{sid[-48:]:<50} {full_total:>8.1f} {comp_total:>8.1f} {comp_rank:>8}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_comparison_plots(
    full_samples: dict,
    composite_samples: dict,
    common_ids: set,
    output_dir: Path,
    base_name: str,
) -> None:
    """Generate and save comparison plots."""

    # Plot 1: Composite vs Synthetic scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, composite_name in enumerate(COMPOSITE_RUBRICS):
        ax = axes[i]

        composite_scores = []
        synthetic_scores = []

        for sid in common_ids:
            comp_score = composite_samples[sid]["scores"].get(composite_name, {}).get("value", 0)
            synth_score = compute_synthetic_composite(full_samples[sid], composite_name)
            composite_scores.append(comp_score)
            synthetic_scores.append(synth_score)

        ax.scatter(synthetic_scores, composite_scores, alpha=0.5)

        # Add diagonal line
        min_val = min(min(synthetic_scores), min(composite_scores))
        max_val = max(max(synthetic_scores), max(composite_scores))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        corr = pearson_correlation(composite_scores, synthetic_scores)
        ax.set_xlabel("Synthetic (avg of components)")
        ax.set_ylabel("Actual Composite")
        ax.set_title(f"{composite_name}\nr = {corr:.3f}" if corr else composite_name)

    # Hide extra subplot
    axes[5].axis('off')

    plt.suptitle("Composite Scores vs Synthetic (Averaged Components)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Total score correlation
    fig, ax = plt.subplots(figsize=(8, 8))

    full_totals = []
    composite_totals = []

    for sid in common_ids:
        full_scores = full_samples[sid].get("scores", {})
        full_total = sum(s.get("value", 0) for s in full_scores.values())
        full_totals.append(full_total)

        comp_scores = composite_samples[sid].get("scores", {})
        comp_total = sum(s.get("value", 0) for s in comp_scores.values())
        composite_totals.append(comp_total)

    ax.scatter(full_totals, composite_totals, alpha=0.5)

    corr = pearson_correlation(full_totals, composite_totals)
    ax.set_xlabel("Full Eval Total (34 rubrics)")
    ax.set_ylabel("Composite Total (5 rubrics)")
    ax.set_title(f"Total Score Correlation\nr = {corr:.3f}" if corr else "Total Score Correlation")

    # Add trend line
    z = np.polyfit(full_totals, composite_totals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(full_totals), max(full_totals), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_total_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    logs_dir = project_root / "logs"

    # Determine which eval files to compare
    if len(sys.argv) >= 3:
        full_path = Path(sys.argv[1])
        composite_path = Path(sys.argv[2])
    else:
        full_path, composite_path = find_eval_files(logs_dir)
        if full_path is None:
            print("No full eval file found (cot-quality-eval)")
            sys.exit(1)
        if composite_path is None:
            print("No composite eval file found (cot-quality-composite)")
            sys.exit(1)

    print(f"Full eval: {full_path}")
    print(f"Composite eval: {composite_path}")

    # Extract data
    full_header, full_samples = extract_samples(full_path)
    composite_header, composite_samples = extract_samples(composite_path)

    if not full_samples:
        print("No samples in full eval")
        sys.exit(1)
    if not composite_samples:
        print("No samples in composite eval")
        sys.exit(1)

    # Generate report
    report = generate_comparison_report(
        full_path, composite_path,
        full_header, full_samples,
        composite_header, composite_samples,
    )

    # Save outputs
    analyses_dir = logs_dir / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    base_name = "composite_comparison"

    # Save text report
    output_path = analyses_dir / f"{base_name}.txt"
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")

    # Save plots
    common_ids = set(full_samples.keys()) & set(composite_samples.keys())
    if common_ids:
        save_comparison_plots(
            full_samples, composite_samples, common_ids,
            analyses_dir, base_name
        )
        print(f"Plots saved to: {analyses_dir}/{base_name}_*.png")

    print("")
    print(report)


if __name__ == "__main__":
    main()
