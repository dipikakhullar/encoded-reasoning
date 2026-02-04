#!/usr/bin/env python3
"""Analyze inspect-ai eval results comparing new rubrics to GDM legacy rubrics.

Usage:
    uv run python cot_quality_metrics/src/standalone_scripts/analyze_eval.py
    uv run python cot_quality_metrics/src/standalone_scripts/analyze_eval.py path/to/file.eval
"""

import json
import statistics
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Rubric definitions
POSITIVE_RUBRICS = [
    "noticing_confusion", "live_updating", "discriminative_experiment_design",
    "appropriate_stopping", "generativity_under_stuckness", "error_metabolism",
    "calibration", "problem_decomposition", "assumption_surfacing",
    "negative_space_awareness", "authenticity", "contact_with_reality",
    "process_conclusion_integrity", "sufficiency_of_investigation", "provenance_transparency",
]

NEGATIVE_RUBRICS = [
    "motivated_cognition", "complexity_theater", "cargo_cult_methodology",
    "premature_formalization", "intellectual_flinching", "too_direct_path",
    "too_indirect_path", "wrong_difficulty_calibration", "destination_shaped_early_steps",
    "answer_presupposing_framing", "convenient_definitions", "reasoning_error_mismatch",
    "performed_uncertainty", "unnatural_smoothness", "no_genuine_dead_ends",
    "no_self_interruption", "suspiciously_complete_resolution",
]

GDM_RUBRICS = ["gdm_legibility", "gdm_coverage"]

# Score ranges
POSITIVE_MAX = 5 * len(POSITIVE_RUBRICS)  # 75
NEGATIVE_MIN = -5 * len(NEGATIVE_RUBRICS)  # -85
GDM_MAX = 4 * len(GDM_RUBRICS)  # 8

# Combined new rubrics range: -85 to 75, span of 160
NEW_RUBRICS_MIN = NEGATIVE_MIN  # -85
NEW_RUBRICS_MAX = POSITIVE_MAX  # 75
NEW_RUBRICS_SPAN = NEW_RUBRICS_MAX - NEW_RUBRICS_MIN  # 160

ALL_RUBRICS = POSITIVE_RUBRICS + NEGATIVE_RUBRICS + GDM_RUBRICS

# Short names for correlation matrix display
RUBRIC_SHORT_NAMES = {
    "noticing_confusion": "notic_conf",
    "live_updating": "live_upd",
    "discriminative_experiment_design": "discrim_exp",
    "appropriate_stopping": "approp_stop",
    "generativity_under_stuckness": "gen_stuck",
    "error_metabolism": "err_metab",
    "calibration": "calibrat",
    "problem_decomposition": "prob_decomp",
    "assumption_surfacing": "assump_surf",
    "negative_space_awareness": "neg_space",
    "authenticity": "authentic",
    "contact_with_reality": "contact_real",
    "process_conclusion_integrity": "proc_concl",
    "sufficiency_of_investigation": "suff_invest",
    "provenance_transparency": "prov_trans",
    "motivated_cognition": "motiv_cog",
    "complexity_theater": "compl_theat",
    "cargo_cult_methodology": "cargo_cult",
    "premature_formalization": "premat_form",
    "intellectual_flinching": "intel_flinch",
    "too_direct_path": "too_direct",
    "too_indirect_path": "too_indir",
    "wrong_difficulty_calibration": "wrong_diff",
    "destination_shaped_early_steps": "dest_shaped",
    "answer_presupposing_framing": "ans_presup",
    "convenient_definitions": "conv_def",
    "reasoning_error_mismatch": "reason_err",
    "performed_uncertainty": "perf_uncert",
    "unnatural_smoothness": "unnat_smooth",
    "no_genuine_dead_ends": "no_dead_end",
    "no_self_interruption": "no_self_int",
    "suspiciously_complete_resolution": "susp_compl",
    "gdm_legibility": "gdm_legib",
    "gdm_coverage": "gdm_cover",
}


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return None

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return None

    return numerator / (denom_x * denom_y)


def compute_correlation_matrix(samples: list[dict]) -> dict[str, dict[str, float | None]]:
    """Compute correlation matrix for all rubrics across samples."""
    # Extract per-rubric scores across all samples
    rubric_scores = {r: [] for r in ALL_RUBRICS}

    for sample in samples:
        scores = sample.get("scores", {})
        for rubric in ALL_RUBRICS:
            value = scores.get(rubric, {}).get("value", 0)
            rubric_scores[rubric].append(value)

    # Compute pairwise correlations
    matrix = {}
    for r1 in ALL_RUBRICS:
        matrix[r1] = {}
        for r2 in ALL_RUBRICS:
            matrix[r1][r2] = pearson_correlation(rubric_scores[r1], rubric_scores[r2])

    return matrix


def format_correlation_matrix(matrix: dict[str, dict[str, float | None]]) -> list[str]:
    """Format correlation matrix as text lines."""
    lines = []

    # Use short names for display
    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:10])

    # Column width
    col_width = 11
    name_width = 12

    # Header row
    header = " " * name_width + "".join(f"{short(r):>{col_width}}" for r in ALL_RUBRICS)
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for r1 in ALL_RUBRICS:
        row = f"{short(r1):<{name_width}}"
        for r2 in ALL_RUBRICS:
            corr = matrix[r1][r2]
            if corr is None:
                row += f"{'N/A':>{col_width}}"
            else:
                row += f"{corr:>{col_width}.2f}"
        lines.append(row)

    return lines


def save_correlation_heatmap(matrix: dict[str, dict[str, float | None]], output_path: Path) -> None:
    """Save correlation matrix as a matplotlib heatmap."""
    # Convert to numpy array
    n = len(ALL_RUBRICS)
    data = np.zeros((n, n))
    mask = np.zeros((n, n), dtype=bool)

    for i, r1 in enumerate(ALL_RUBRICS):
        for j, r2 in enumerate(ALL_RUBRICS):
            corr = matrix[r1][r2]
            if corr is None:
                mask[i, j] = True
                data[i, j] = 0
            else:
                data[i, j] = corr

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))

    # Create heatmap
    cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    # Mask N/A cells with gray
    masked_data = np.ma.array(data, mask=mask)
    ax.imshow(np.where(mask, 1, np.nan), cmap=plt.cm.gray, vmin=0, vmax=2, aspect='equal')

    # Labels
    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:10])
    labels = [short(r) for r in ALL_RUBRICS]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom", fontsize=10)

    # Add grid lines to separate rubric types
    # Positive: 0-14, Negative: 15-31, GDM: 32-33
    for pos in [len(POSITIVE_RUBRICS), len(POSITIVE_RUBRICS) + len(NEGATIVE_RUBRICS)]:
        ax.axhline(y=pos - 0.5, color='black', linewidth=2)
        ax.axvline(x=pos - 0.5, color='black', linewidth=2)

    # Add section labels
    ax.text(-2, len(POSITIVE_RUBRICS) / 2, 'Positive', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax.text(-2, len(POSITIVE_RUBRICS) + len(NEGATIVE_RUBRICS) / 2, 'Negative', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax.text(-2, len(POSITIVE_RUBRICS) + len(NEGATIVE_RUBRICS) + 1, 'GDM', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

    ax.set_title("Rubric Correlation Matrix", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_most_recent_eval(logs_dir: Path) -> Path | None:
    """Find the most recent .eval file in the logs directory."""
    eval_files = list(logs_dir.glob("*.eval"))
    if not eval_files:
        return None
    return max(eval_files, key=lambda p: p.stat().st_mtime)


def extract_samples_from_eval(eval_path: Path) -> list[dict]:
    """Extract sample data from an inspect-ai .eval zip file."""
    samples = []
    with zipfile.ZipFile(eval_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as f:
                    data = json.load(f)
                    samples.append(data)
    return samples


def extract_header(eval_path: Path) -> dict:
    """Extract header info from eval file."""
    with zipfile.ZipFile(eval_path, 'r') as zf:
        if "header.json" in zf.namelist():
            with zf.open("header.json") as f:
                return json.load(f)
    return {}


def compute_sample_scores(sample: dict) -> dict:
    """Compute normalized scores for a single sample."""
    scores = sample.get("scores", {})

    # Extract raw scores
    positive_scores = {r: scores.get(r, {}).get("value", 0) for r in POSITIVE_RUBRICS}
    negative_scores = {r: scores.get(r, {}).get("value", 0) for r in NEGATIVE_RUBRICS}
    gdm_scores = {r: scores.get(r, {}).get("value", 0) for r in GDM_RUBRICS}

    # Compute sums
    positive_sum = sum(positive_scores.values())
    negative_sum = sum(negative_scores.values())  # Will be <= 0
    gdm_sum = sum(gdm_scores.values())

    # Combined new rubrics score (positive + negative, since negative is already negative)
    new_rubrics_sum = positive_sum + negative_sum

    # Normalize to 0-1
    # New rubrics: range is -85 to 75, normalize so -85 -> 0, 75 -> 1
    new_rubrics_normalized = (new_rubrics_sum - NEW_RUBRICS_MIN) / NEW_RUBRICS_SPAN

    # GDM: range is 0 to 8, normalize so 0 -> 0, 8 -> 1
    gdm_normalized = gdm_sum / GDM_MAX if GDM_MAX > 0 else 0

    return {
        "id": sample.get("id", "unknown"),
        "positive_sum": positive_sum,
        "positive_scores": positive_scores,
        "negative_sum": negative_sum,
        "negative_scores": negative_scores,
        "new_rubrics_sum": new_rubrics_sum,
        "new_rubrics_normalized": new_rubrics_normalized,
        "gdm_sum": gdm_sum,
        "gdm_scores": gdm_scores,
        "gdm_normalized": gdm_normalized,
    }


def generate_analysis_report(eval_path: Path, samples: list[dict], header: dict) -> str:
    """Generate a human-readable analysis report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("COT QUALITY METRICS ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Eval file: {eval_path.name}")
    lines.append(f"Analysis time: {datetime.now().isoformat()}")
    if header:
        lines.append(f"Task: {header.get('eval', {}).get('task', 'unknown')}")
        lines.append(f"Model: {header.get('eval', {}).get('model', 'unknown')}")
    lines.append(f"Samples analyzed: {len(samples)}")
    lines.append("")

    # Scoring explanation
    lines.append("-" * 80)
    lines.append("SCORING METHODOLOGY")
    lines.append("-" * 80)
    lines.append("")
    lines.append("New Rubrics (32 total):")
    lines.append(f"  - Positive rubrics (15): 0-5 scale each, sum range 0-{POSITIVE_MAX}")
    lines.append(f"  - Negative rubrics (17): 0 to -5 scale each, sum range {NEGATIVE_MIN}-0")
    lines.append(f"  - Combined range: {NEW_RUBRICS_MIN} to {NEW_RUBRICS_MAX}")
    lines.append(f"  - Normalized: (sum - {NEW_RUBRICS_MIN}) / {NEW_RUBRICS_SPAN}")
    lines.append("")
    lines.append("GDM Legacy Rubrics (2 total):")
    lines.append(f"  - Legibility + Coverage: 0-4 scale each, sum range 0-{GDM_MAX}")
    lines.append(f"  - Normalized: sum / {GDM_MAX}")
    lines.append("")

    # Compute per-sample scores
    sample_results = [compute_sample_scores(s) for s in samples]

    # Summary statistics
    lines.append("-" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)
    lines.append("")

    new_norm = [r["new_rubrics_normalized"] for r in sample_results]
    gdm_norm = [r["gdm_normalized"] for r in sample_results]

    lines.append(f"{'Metric':<30} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    lines.append("-" * 70)

    for name, values in [("New Rubrics (normalized)", new_norm), ("GDM (normalized)", gdm_norm)]:
        mean_val = statistics.mean(values)
        min_val = min(values)
        max_val = max(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        lines.append(f"{name:<30} {mean_val:>10.3f} {min_val:>10.3f} {max_val:>10.3f} {std_val:>10.3f}")

    lines.append("")

    # Correlation (New vs GDM aggregate)
    if len(sample_results) > 1:
        corr = pearson_correlation(new_norm, gdm_norm)
        if corr is not None:
            lines.append(f"Pearson correlation (New vs GDM): {corr:.3f}")
        else:
            lines.append("Pearson correlation: N/A (no variance)")
        lines.append("")

    # Full correlation matrix
    lines.append("-" * 80)
    lines.append("RUBRIC CORRELATION MATRIX")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Pearson correlations between all rubrics across samples.")
    lines.append("Positive rubrics: rows 1-15, Negative rubrics: rows 16-32, GDM: rows 33-34")
    lines.append("")

    if len(samples) > 1:
        corr_matrix = compute_correlation_matrix(samples)
        matrix_lines = format_correlation_matrix(corr_matrix)
        lines.extend(matrix_lines)
    else:
        lines.append("(Need at least 2 samples to compute correlations)")

    lines.append("")

    # Per-sample details
    lines.append("-" * 80)
    lines.append("PER-SAMPLE RESULTS")
    lines.append("-" * 80)
    lines.append("")

    # Find max sample ID length for alignment
    max_id_len = max(len(r["id"]) for r in sample_results)
    lines.append(f"{'Sample ID':<{max_id_len}}  {'New':>8} {'GDM':>8} {'Diff':>8}")
    lines.append("-" * (max_id_len + 30))

    for r in sample_results:
        diff = r["new_rubrics_normalized"] - r["gdm_normalized"]
        lines.append(f"{r['id']:<{max_id_len}}  {r['new_rubrics_normalized']:>8.3f} {r['gdm_normalized']:>8.3f} {diff:>+8.3f}")

    lines.append("")

    # Detailed breakdown for first few samples
    lines.append("-" * 80)
    lines.append("DETAILED BREAKDOWN (first 3 samples)")
    lines.append("-" * 80)

    for r in sample_results[:3]:
        lines.append("")
        lines.append(f"Sample: {r['id']}")
        lines.append("")

        lines.append("  Positive Rubrics:")
        for rubric, score in sorted(r["positive_scores"].items()):
            lines.append(f"    {rubric:<40} {score:>3}/5")
        lines.append(f"    {'SUBTOTAL':<40} {r['positive_sum']:>3}/{POSITIVE_MAX}")

        lines.append("")
        lines.append("  Negative Rubrics (0 is best):")
        for rubric, score in sorted(r["negative_scores"].items()):
            lines.append(f"    {rubric:<40} {score:>3}")
        lines.append(f"    {'SUBTOTAL':<40} {r['negative_sum']:>3}")

        lines.append("")
        lines.append("  GDM Legacy Rubrics:")
        for rubric, score in sorted(r["gdm_scores"].items()):
            lines.append(f"    {rubric:<40} {score:>3}/4")
        lines.append(f"    {'SUBTOTAL':<40} {r['gdm_sum']:>3}/{GDM_MAX}")

        lines.append("")
        lines.append(f"  New Rubrics Combined: {r['new_rubrics_sum']} → normalized: {r['new_rubrics_normalized']:.3f}")
        lines.append(f"  GDM Combined: {r['gdm_sum']} → normalized: {r['gdm_normalized']:.3f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    # Project root: encoded-reasoning/
    # Script is at: cot_quality_metrics/src/standalone_scripts/analyze_eval.py
    project_root = Path(__file__).parent.parent.parent.parent

    # Determine which eval file to analyze
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1])
    else:
        # Find most recent eval in logs/
        logs_dir = project_root / "logs"
        eval_path = find_most_recent_eval(logs_dir)
        if eval_path is None:
            print(f"No .eval files found in {logs_dir}")
            sys.exit(1)

    if not eval_path.exists():
        print(f"File not found: {eval_path}")
        sys.exit(1)

    print(f"Analyzing: {eval_path}")

    # Extract data
    samples = extract_samples_from_eval(eval_path)
    header = extract_header(eval_path)

    if not samples:
        print("No samples found in eval file")
        sys.exit(1)

    # Generate report
    report = generate_analysis_report(eval_path, samples, header)

    # Save to analyses directory
    analyses_dir = project_root / "logs" / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"analysis_{timestamp}_{eval_path.stem}"

    # Save text report
    output_path = analyses_dir / f"{base_name}.txt"
    output_path.write_text(report)
    print(f"Analysis saved to: {output_path}")

    # Save correlation heatmap
    if len(samples) > 1:
        corr_matrix = compute_correlation_matrix(samples)
        heatmap_path = analyses_dir / f"{base_name}_correlation.png"
        save_correlation_heatmap(corr_matrix, heatmap_path)
        print(f"Heatmap saved to: {heatmap_path}")

    print("")
    print(report)


if __name__ == "__main__":
    main()
