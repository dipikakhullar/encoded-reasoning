#!/usr/bin/env python3
"""Analyze inspect-ai eval results comparing new rubrics to GDM legacy rubrics.

Usage:
    uv run python cot_quality_metrics/src/standalone_scripts/analyze_eval.py
    uv run python cot_quality_metrics/src/standalone_scripts/analyze_eval.py path/to/file.eval
"""

import gzip
import json
import re
import statistics
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


# =============================================================================
# REPETITION METRICS
# =============================================================================
# These metrics detect corrupted/repetitive CoT traces that LLM judges may miss.


def tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def trigram_repetition_rate(text: str) -> float:
    """Fraction of unique trigrams that appear more than once.

    Higher values indicate more repetitive text.
    Clean CoT: ~0.05-0.10
    Corrupted CoT: ~0.30-0.50
    """
    tokens = tokenize(text)
    ngrams = get_ngrams(tokens, 3)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    return repeated / len(counts) if counts else 0.0


def compression_ratio(text: str) -> float:
    """Ratio of gzip-compressed to original size.

    Lower values indicate more compressible (repetitive) text.
    Clean CoT: ~0.40-0.50
    Corrupted CoT: ~0.15-0.25
    """
    if not text:
        return 1.0
    encoded = text.encode('utf-8')
    compressed = gzip.compress(encoded)
    return len(compressed) / len(encoded)


def compute_repetition_metrics(text: str) -> dict[str, float]:
    """Compute all repetition metrics for a text.

    Returns dict with metric names and values, ready to be added to scores.
    """
    return {
        "trigram_repetition": trigram_repetition_rate(text),
        "compression_ratio": compression_ratio(text),
    }


# =============================================================================

# Rubric definitions (used for categorization when present)
POSITIVE_RUBRICS = [
    "noticing_confusion", "live_updating", "discriminative_experiment_design",
    "appropriate_stopping", "generativity_under_stuckness", "error_metabolism",
    "calibration", "problem_decomposition", "assumption_surfacing",
    "negative_space_awareness", "authenticity", "contact_with_reality",
    "process_conclusion_integrity", "sufficiency_of_investigation", "provenance_transparency",
    # Composite positives
    "active_investigation", "epistemic_honesty", "adaptive_process",
]

NEGATIVE_RUBRICS = [
    "motivated_cognition", "complexity_theater", "cargo_cult_methodology",
    "premature_formalization", "intellectual_flinching", "too_direct_path",
    "too_indirect_path", "wrong_difficulty_calibration", "destination_shaped_early_steps",
    "answer_presupposing_framing", "convenient_definitions", "reasoning_error_mismatch",
    "performed_uncertainty", "unnatural_smoothness", "no_genuine_dead_ends",
    "no_self_interruption", "suspiciously_complete_resolution",
    # Composite negatives
    "fake_rigor", "too_clean",
]

GDM_RUBRICS = ["gdm_legibility", "gdm_coverage"]

COMPOSITE_RUBRICS = [
    "fake_rigor", "too_clean", "active_investigation", "epistemic_honesty", "adaptive_process",
]

# Repetition-based metrics (computed from text, not LLM-judged)
REPETITION_METRICS = ["trigram_repetition", "compression_ratio"]

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
    # Composite rubrics
    "fake_rigor": "fake_rigor",
    "too_clean": "too_clean",
    "active_investigation": "active_inv",
    "epistemic_honesty": "epist_hon",
    "adaptive_process": "adapt_proc",
    # Repetition metrics
    "trigram_repetition": "trigram_rep",
    "compression_ratio": "compress",
}


def detect_rubrics_from_samples(samples: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Detect which rubrics are present in the eval samples.

    Returns:
        Tuple of (positive_rubrics, negative_rubrics, legacy_rubrics, repetition_metrics) found in samples.
    """
    if not samples:
        return [], [], [], []

    # Get all rubric names from first sample
    all_found = list(samples[0].get("scores", {}).keys())

    # Categorize
    positive = [r for r in all_found if r in POSITIVE_RUBRICS]
    negative = [r for r in all_found if r in NEGATIVE_RUBRICS]
    legacy = [r for r in all_found if r in GDM_RUBRICS]
    repetition = [r for r in all_found if r in REPETITION_METRICS]

    # Any uncategorized go to positive (safe default for 0-5 scale)
    categorized = set(positive + negative + legacy + repetition)
    uncategorized = [r for r in all_found if r not in categorized]
    positive.extend(uncategorized)

    return positive, negative, legacy, repetition


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


def compute_correlation_matrix(samples: list[dict], rubrics: list[str]) -> dict[str, dict[str, float | None]]:
    """Compute correlation matrix for specified rubrics across samples."""
    # Extract per-rubric scores across all samples
    rubric_scores = {r: [] for r in rubrics}

    for sample in samples:
        scores = sample.get("scores", {})
        for rubric in rubrics:
            value = scores.get(rubric, {}).get("value", 0)
            rubric_scores[rubric].append(value)

    # Compute pairwise correlations
    matrix = {}
    for r1 in rubrics:
        matrix[r1] = {}
        for r2 in rubrics:
            matrix[r1][r2] = pearson_correlation(rubric_scores[r1], rubric_scores[r2])

    return matrix


def format_correlation_matrix(matrix: dict[str, dict[str, float | None]], rubrics: list[str]) -> list[str]:
    """Format correlation matrix as text lines."""
    lines = []

    # Use short names for display
    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:10])

    # Column width
    col_width = 11
    name_width = 12

    # Header row
    header = " " * name_width + "".join(f"{short(r):>{col_width}}" for r in rubrics)
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for r1 in rubrics:
        row = f"{short(r1):<{name_width}}"
        for r2 in rubrics:
            corr = matrix[r1][r2]
            if corr is None:
                row += f"{'N/A':>{col_width}}"
            else:
                row += f"{corr:>{col_width}.2f}"
        lines.append(row)

    return lines


def save_correlation_heatmap(
    matrix: dict[str, dict[str, float | None]],
    rubrics: list[str],
    output_path: Path,
    positive_rubrics: list[str] | None = None,
    negative_rubrics: list[str] | None = None,
) -> None:
    """Save correlation matrix as a matplotlib heatmap."""
    # Convert to numpy array
    n = len(rubrics)
    data = np.zeros((n, n))
    mask = np.zeros((n, n), dtype=bool)

    for i, r1 in enumerate(rubrics):
        for j, r2 in enumerate(rubrics):
            corr = matrix[r1][r2]
            if corr is None:
                mask[i, j] = True
                data[i, j] = 0
            else:
                data[i, j] = corr

    # Adjust figure size based on number of rubrics
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    # Create heatmap
    cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    # Mask N/A cells with gray
    masked_data = np.ma.array(data, mask=mask)
    ax.imshow(np.where(mask, 1, np.nan), cmap=plt.cm.gray, vmin=0, vmax=2, aspect='equal')

    # Labels
    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:10])
    labels = [short(r) for r in rubrics]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom", fontsize=10)

    # Add grid lines to separate rubric types (if categorization provided)
    if positive_rubrics and negative_rubrics:
        n_pos = len([r for r in rubrics if r in positive_rubrics])
        n_neg = len([r for r in rubrics if r in negative_rubrics])
        if n_pos > 0 and n_neg > 0:
            ax.axhline(y=n_pos - 0.5, color='black', linewidth=2)
            ax.axvline(x=n_pos - 0.5, color='black', linewidth=2)

    # Add section labels (only for larger rubric sets with categorization)
    if positive_rubrics and negative_rubrics and n > 10:
        n_pos = len([r for r in rubrics if r in positive_rubrics])
        n_neg = len([r for r in rubrics if r in negative_rubrics])
        if n_pos > 0:
            ax.text(-2, n_pos / 2, 'Positive', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
        if n_neg > 0:
            ax.text(-2, n_pos + n_neg / 2, 'Negative', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

    ax.set_title("Rubric Correlation Matrix", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_rubric_scores_matrix(samples: list[dict], rubrics: list[str] | None = None) -> np.ndarray:
    """Extract rubric scores as a (n_samples, n_rubrics) matrix."""
    if rubrics is None:
        rubrics = ALL_RUBRICS
    n_samples = len(samples)
    n_rubrics = len(rubrics)
    data = np.zeros((n_samples, n_rubrics))

    for i, sample in enumerate(samples):
        scores = sample.get("scores", {})
        for j, rubric in enumerate(rubrics):
            data[i, j] = scores.get(rubric, {}).get("value", 0)

    return data


def run_pca(data: np.ndarray) -> dict:
    """Run PCA on the rubric scores matrix. Returns dict with components, loadings, variance explained."""
    # Center the data (subtract mean)
    centered = data - data.mean(axis=0)

    # Handle zero-variance columns
    std = centered.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    standardized = centered / std

    # SVD for PCA
    U, S, Vt = np.linalg.svd(standardized, full_matrices=False)

    # Explained variance
    n_samples = data.shape[0]
    explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else explained_variance

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Loadings (correlations between original variables and components)
    # loadings = Vt.T * S / sqrt(n-1)
    loadings = Vt.T

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "loadings": loadings,  # Shape: (n_rubrics, n_components)
        "n_components_90": int(np.searchsorted(cumulative_variance, 0.90) + 1),
        "n_components_95": int(np.searchsorted(cumulative_variance, 0.95) + 1),
    }


def format_pca_results(pca_results: dict, rubrics: list[str]) -> list[str]:
    """Format PCA results as text lines."""
    lines = []

    var_ratio = pca_results["explained_variance_ratio"]
    cum_var = pca_results["cumulative_variance"]
    loadings = pca_results["loadings"]

    # Show variance explained by first 10 components
    lines.append("Principal Components - Variance Explained:")
    lines.append(f"  {'PC':<6} {'Variance %':>12} {'Cumulative %':>14}")
    lines.append("  " + "-" * 34)

    n_show = min(10, len(var_ratio))
    for i in range(n_show):
        lines.append(f"  PC{i+1:<4} {var_ratio[i]*100:>11.1f}% {cum_var[i]*100:>13.1f}%")

    lines.append("")
    lines.append(f"  Components for 90% variance: {pca_results['n_components_90']}")
    lines.append(f"  Components for 95% variance: {pca_results['n_components_95']}")
    lines.append("")

    # Show top loadings for first 3 PCs
    lines.append("Top Rubric Loadings per Component:")
    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:12])

    for pc_idx in range(min(3, loadings.shape[1])):
        pc_loadings = loadings[:, pc_idx]
        # Sort by absolute loading
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]

        lines.append(f"\n  PC{pc_idx + 1} (explains {var_ratio[pc_idx]*100:.1f}% variance):")
        for rank, idx in enumerate(sorted_indices[:5]):
            rubric = rubrics[idx]
            loading = pc_loadings[idx]
            lines.append(f"    {rank+1}. {short(rubric):<14} {loading:>+.3f}")

    return lines


def run_hierarchical_clustering(corr_matrix: dict[str, dict[str, float | None]], rubrics: list[str]) -> dict:
    """Run hierarchical clustering on rubrics based on correlation matrix."""
    n = len(rubrics)

    # Convert correlation to distance: d = 1 - |r|
    # Using absolute correlation so both positive and negative correlations cluster
    dist_matrix = np.zeros((n, n))
    for i, r1 in enumerate(rubrics):
        for j, r2 in enumerate(rubrics):
            if i == j:
                dist_matrix[i, j] = 0.0  # Diagonal must be exactly zero
            else:
                corr = corr_matrix[r1][r2]
                if corr is None:
                    dist_matrix[i, j] = 1.0  # Max distance for undefined
                else:
                    # Clamp to handle floating-point precision issues
                    dist_matrix[i, j] = max(0.0, 1 - abs(corr))

    # Convert to condensed form for linkage
    condensed_dist = squareform(dist_matrix)

    # Hierarchical clustering with average linkage
    linkage_matrix = linkage(condensed_dist, method='average')

    # Get clusters at different thresholds
    # threshold 0.3 means correlation >= 0.7
    clusters_strict = fcluster(linkage_matrix, t=0.3, criterion='distance')
    # threshold 0.5 means correlation >= 0.5
    clusters_moderate = fcluster(linkage_matrix, t=0.5, criterion='distance')

    return {
        "linkage_matrix": linkage_matrix,
        "dist_matrix": dist_matrix,
        "clusters_strict": clusters_strict,  # r >= 0.7
        "clusters_moderate": clusters_moderate,  # r >= 0.5
    }


def format_cluster_results(cluster_results: dict, rubrics: list[str]) -> list[str]:
    """Format clustering results as text lines."""
    lines = []
    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:12])

    for name, clusters, threshold in [
        ("Strict (|r| >= 0.7)", cluster_results["clusters_strict"], 0.7),
        ("Moderate (|r| >= 0.5)", cluster_results["clusters_moderate"], 0.5),
    ]:
        lines.append(f"\n{name}:")
        # Group rubrics by cluster
        cluster_groups: dict[int, list[str]] = {}
        for rubric, cluster_id in zip(rubrics, clusters):
            cluster_groups.setdefault(cluster_id, []).append(rubric)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(cluster_groups.items(), key=lambda x: -len(x[1]))

        n_clusters = len(sorted_clusters)
        n_singletons = sum(1 for _, members in sorted_clusters if len(members) == 1)

        lines.append(f"  {n_clusters} clusters total ({n_singletons} singletons)")

        # Show non-singleton clusters
        for cluster_id, members in sorted_clusters:
            if len(members) > 1:
                member_names = ", ".join(short(r) for r in members)
                lines.append(f"  Cluster {cluster_id} ({len(members)} rubrics): {member_names}")

    return lines


def save_dendrogram(cluster_results: dict, rubrics: list[str], output_path: Path) -> None:
    """Save hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(16, 10))

    short = lambda r: RUBRIC_SHORT_NAMES.get(r, r[:10])
    labels = [f"{i}_{short(r)}" for i, r in enumerate(rubrics)]

    # Color threshold at 0.5 (correlation 0.5)
    dendrogram(
        cluster_results["linkage_matrix"],
        labels=labels,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=8,
        color_threshold=0.5,
    )

    # Adjust x-axis labels: right-align so label ends point at tick marks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax.set_ylabel("Distance (1 - |correlation|)", fontsize=10)
    ax.set_title("Rubric Clustering Dendrogram\n(Distance = 1 - |correlation|, lower = more similar)", fontsize=12)

    # Add horizontal lines at key thresholds
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='|r| = 0.7')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='|r| = 0.5')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_pca_variance_plot(pca_results: dict, output_path: Path) -> None:
    """Save PCA explained variance plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    var_ratio = pca_results["explained_variance_ratio"]
    cum_var = pca_results["cumulative_variance"]
    n_components = min(20, len(var_ratio))

    x = np.arange(1, n_components + 1)

    # Individual variance
    ax1.bar(x, var_ratio[:n_components] * 100, color='steelblue', alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Individual Variance per Component")
    ax1.set_xticks(x)

    # Cumulative variance
    ax2.plot(x, cum_var[:n_components] * 100, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(x)
    ax2.legend()
    ax2.set_ylim(0, 105)

    plt.suptitle("PCA Analysis: How Many Independent Dimensions?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_most_recent_eval(logs_dir: Path) -> Path | None:
    """Find the most recent .eval file in the logs directory."""
    eval_files = list(logs_dir.glob("*.eval"))
    if not eval_files:
        return None
    return max(eval_files, key=lambda p: p.stat().st_mtime)


def extract_samples_from_eval(eval_path: Path, add_repetition_metrics: bool = True) -> list[dict]:
    """Extract sample data from an inspect-ai .eval zip file.

    Args:
        eval_path: Path to the .eval file
        add_repetition_metrics: If True, compute and add trigram_repetition and
            compression_ratio metrics to each sample's scores based on the input text.
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
    """Extract header info from eval file."""
    with zipfile.ZipFile(eval_path, 'r') as zf:
        if "header.json" in zf.namelist():
            with zf.open("header.json") as f:
                return json.load(f)
    return {}


def generate_analysis_report(eval_path: Path, samples: list[dict], header: dict) -> str:
    """Generate a human-readable analysis report."""
    lines = []

    # Detect which rubrics are present
    positive_found, negative_found, legacy_found, repetition_found = detect_rubrics_from_samples(samples)
    all_rubrics_found = positive_found + negative_found + legacy_found + repetition_found
    non_legacy_rubrics = positive_found + negative_found

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

    # Rubrics found
    lines.append("-" * 80)
    lines.append("RUBRICS DETECTED")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Positive rubrics ({len(positive_found)}): {', '.join(positive_found) if positive_found else 'None'}")
    lines.append(f"Negative rubrics ({len(negative_found)}): {', '.join(negative_found) if negative_found else 'None'}")
    lines.append(f"Legacy rubrics ({len(legacy_found)}): {', '.join(legacy_found) if legacy_found else 'None'}")
    lines.append(f"Repetition metrics ({len(repetition_found)}): {', '.join(repetition_found) if repetition_found else 'None'}")
    lines.append(f"Total: {len(all_rubrics_found)} metrics")
    lines.append("")

    # Summary statistics per rubric
    lines.append("-" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)
    lines.append("")

    lines.append(f"{'Rubric':<35} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    lines.append("-" * 70)

    for rubric in all_rubrics_found:
        values = [s["scores"].get(rubric, {}).get("value", 0) for s in samples]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)
        lines.append(f"{rubric:<35} {mean_val:>8.2f} {std_val:>8.2f} {min_val:>6.0f} {max_val:>6.0f}")

    lines.append("")

    # Total score summary
    if len(samples) > 0:
        total_scores = []
        for s in samples:
            total = sum(s["scores"].get(r, {}).get("value", 0) for r in all_rubrics_found)
            total_scores.append(total)

        lines.append(f"Total Score (sum of all rubrics):")
        lines.append(f"  Mean: {statistics.mean(total_scores):.2f}")
        lines.append(f"  Std:  {statistics.stdev(total_scores) if len(total_scores) > 1 else 0:.2f}")
        lines.append(f"  Range: {min(total_scores):.0f} to {max(total_scores):.0f}")
        lines.append("")

    # Full correlation matrix
    lines.append("-" * 80)
    lines.append("RUBRIC CORRELATION MATRIX")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Pearson correlations between all rubrics across samples.")
    lines.append("")

    if len(samples) > 1 and len(all_rubrics_found) > 1:
        corr_matrix = compute_correlation_matrix(samples, all_rubrics_found)
        matrix_lines = format_correlation_matrix(corr_matrix, all_rubrics_found)
        lines.extend(matrix_lines)
    else:
        lines.append("(Need at least 2 samples and 2 rubrics to compute correlations)")

    lines.append("")

    # PCA Analysis (only if we have enough non-legacy rubrics)
    if len(non_legacy_rubrics) >= 3:
        lines.append("-" * 80)
        lines.append("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        lines.append("-" * 80)
        lines.append("")
        lines.append("PCA reveals the underlying dimensionality of the rubrics.")
        lines.append("If N components explain 90%+ variance, you effectively have N independent constructs.")
        lines.append("")

        if len(samples) >= 3:
            data_matrix = extract_rubric_scores_matrix(samples, non_legacy_rubrics)
            pca_results = run_pca(data_matrix)
            pca_lines = format_pca_results(pca_results, non_legacy_rubrics)
            lines.extend(pca_lines)
        else:
            lines.append("(Need at least 3 samples for meaningful PCA)")

        lines.append("")

        # Hierarchical Clustering
        lines.append("-" * 80)
        lines.append("HIERARCHICAL CLUSTERING")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Clusters rubrics by correlation similarity.")
        lines.append("Rubrics in the same cluster may be measuring the same underlying construct.")
        lines.append("")

        if len(samples) > 1:
            corr_matrix = compute_correlation_matrix(samples, non_legacy_rubrics)
            cluster_results = run_hierarchical_clustering(corr_matrix, non_legacy_rubrics)
            cluster_lines = format_cluster_results(cluster_results, non_legacy_rubrics)
            lines.extend(cluster_lines)
        else:
            lines.append("(Need at least 2 samples for clustering)")

        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines), positive_found, negative_found, legacy_found, repetition_found


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

    # Generate report (also returns detected rubrics for visualizations)
    report, positive_found, negative_found, legacy_found, repetition_found = generate_analysis_report(eval_path, samples, header)
    all_rubrics_found = positive_found + negative_found + legacy_found + repetition_found
    non_legacy_rubrics = positive_found + negative_found

    # Save to analyses directory
    analyses_dir = project_root / "logs" / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"analysis_{eval_path.stem}"

    # Save text report
    output_path = analyses_dir / f"{base_name}.txt"
    output_path.write_text(report)
    print(f"Analysis saved to: {output_path}")

    # Save visualizations
    if len(samples) > 1 and len(all_rubrics_found) > 1:
        corr_matrix = compute_correlation_matrix(samples, all_rubrics_found)

        # Correlation heatmap
        heatmap_path = analyses_dir / f"{base_name}_correlation.png"
        save_correlation_heatmap(
            corr_matrix, all_rubrics_found, heatmap_path,
            positive_rubrics=positive_found,
            negative_rubrics=negative_found,
        )
        print(f"Heatmap saved to: {heatmap_path}")

        # Hierarchical clustering dendrogram (non-legacy metrics only)
        if len(non_legacy_rubrics) >= 3:
            cluster_corr = compute_correlation_matrix(samples, non_legacy_rubrics)
            cluster_results = run_hierarchical_clustering(cluster_corr, non_legacy_rubrics)
            dendrogram_path = analyses_dir / f"{base_name}_dendrogram.png"
            save_dendrogram(cluster_results, non_legacy_rubrics, dendrogram_path)
            print(f"Dendrogram saved to: {dendrogram_path}")

    if len(samples) >= 3 and len(non_legacy_rubrics) >= 3:
        # PCA variance plot (non-legacy metrics only)
        data_matrix = extract_rubric_scores_matrix(samples, non_legacy_rubrics)
        pca_results = run_pca(data_matrix)
        pca_path = analyses_dir / f"{base_name}_pca.png"
        save_pca_variance_plot(pca_results, pca_path)
        print(f"PCA plot saved to: {pca_path}")

    print("")
    print(report)


if __name__ == "__main__":
    main()
