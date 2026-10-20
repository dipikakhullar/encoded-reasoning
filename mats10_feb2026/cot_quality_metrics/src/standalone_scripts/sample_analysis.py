#!/usr/bin/env python3
"""Analyze how rubric scores vary across experimental conditions.

Creates line plots showing rubric scores (y-axis) vs experimental variable (x-axis).

Usage:
    uv run python cot_quality_metrics/src/standalone_scripts/sample_analysis.py
    uv run python cot_quality_metrics/src/standalone_scripts/sample_analysis.py path/to/file.eval
    uv run python cot_quality_metrics/src/standalone_scripts/sample_analysis.py --var problem_idx  # by problem
    uv run python cot_quality_metrics/src/standalone_scripts/sample_analysis.py --var alpha  # by alpha
"""

import argparse
import json
import re
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Rubric display names (short versions for plots)
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
    "reportive_fidelity": "report_fid",
}

# Categorization for coloring
POSITIVE_RUBRICS = {
    "noticing_confusion", "live_updating", "discriminative_experiment_design",
    "appropriate_stopping", "generativity_under_stuckness", "error_metabolism",
    "calibration", "problem_decomposition", "assumption_surfacing",
    "negative_space_awareness", "authenticity", "contact_with_reality",
    "process_conclusion_integrity", "sufficiency_of_investigation", "provenance_transparency",
    "active_investigation", "epistemic_honesty", "adaptive_process",
    "gdm_legibility", "gdm_coverage", "reportive_fidelity",
}

NEGATIVE_RUBRICS = {
    "motivated_cognition", "complexity_theater", "cargo_cult_methodology",
    "premature_formalization", "intellectual_flinching", "too_direct_path",
    "too_indirect_path", "wrong_difficulty_calibration", "destination_shaped_early_steps",
    "answer_presupposing_framing", "convenient_definitions", "reasoning_error_mismatch",
    "performed_uncertainty", "unnatural_smoothness", "no_genuine_dead_ends",
    "no_self_interruption", "suspiciously_complete_resolution",
    "fake_rigor", "too_clean",
}


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


def extract_model_name(samples: list[dict], header: dict) -> str:
    """Extract target model name from samples or header."""
    # Try from sample metadata first
    if samples:
        model = samples[0].get("metadata", {}).get("model", "")
        if model:
            # Clean up model name (e.g., "Qwen/Qwen3-VL-8B-Instruct" -> "Qwen3-VL-8B")
            model = model.split("/")[-1]  # Remove org prefix
            model = model.replace("-Instruct", "").replace("_Instruct", "")
            return model

    # Try from header
    if header:
        model = header.get("eval", {}).get("model", "")
        if model:
            return model.split("/")[-1]

    return "Unknown"


def extract_judge_model(header: dict) -> str:
    """Extract judge model name from header."""
    if header:
        model = header.get("eval", {}).get("model", "")
        if model:
            # Clean up (e.g., "openrouter/anthropic/claude-sonnet-4.5" -> "claude-sonnet-4.5")
            return model.split("/")[-1]
    return "Unknown"


def parse_var_from_sample(sample: dict, var_name: str) -> float | None:
    """Extract experimental variable value from sample.

    Tries multiple sources:
    1. metadata.{var_name} (direct field)
    2. metadata.source_file (parse from filename)
    3. sample id (parse from id like 'Model/alpha0.5/prob0/roll0')
    """
    metadata = sample.get("metadata", {})

    # Try direct metadata field (handles problem_idx, alpha, rollout_idx, etc.)
    if var_name in metadata:
        try:
            return float(metadata[var_name])
        except (ValueError, TypeError):
            pass

    # Try parsing from source_file
    source_file = metadata.get("source_file", "")
    match = re.search(rf'{var_name}([\d.]+)', source_file)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try parsing from sample id
    sample_id = sample.get("id", "")
    # Handle patterns like "prob0" -> problem_idx=0
    if var_name == "problem_idx":
        match = re.search(r'prob(\d+)', sample_id)
        if match:
            return float(match.group(1))
    else:
        match = re.search(rf'{var_name}([\d.]+)', sample_id)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    return None


def group_samples_by_var(samples: list[dict], var_name: str) -> dict[float, list[dict]]:
    """Group samples by experimental variable value."""
    groups = defaultdict(list)
    for sample in samples:
        var_val = parse_var_from_sample(sample, var_name)
        if var_val is not None:
            groups[var_val].append(sample)
    return dict(groups)


def compute_rubric_stats_by_var(
    groups: dict[float, list[dict]],
    rubrics: list[str],
) -> dict[str, dict[float, dict]]:
    """Compute mean and stderr for each rubric at each variable value.

    Returns:
        {rubric: {var_val: {"mean": float, "stderr": float, "n": int}}}
    """
    stats = {r: {} for r in rubrics}

    for var_val, samples in groups.items():
        for rubric in rubrics:
            values = []
            for s in samples:
                score = s.get("scores", {}).get(rubric, {}).get("value")
                if score is not None:
                    values.append(score)

            if values:
                mean = statistics.mean(values)
                stderr = statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0
                stats[rubric][var_val] = {
                    "mean": mean,
                    "stderr": stderr,
                    "n": len(values),
                }

    return stats


def save_line_plot(
    stats: dict[str, dict[float, dict]],
    rubrics: list[str],
    var_name: str,
    output_path: Path,
    title: str = "",
    n_samples: int = 0,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save line plot of rubric scores vs experimental variable."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get sorted x values
    all_x = set()
    for rubric_stats in stats.values():
        all_x.update(rubric_stats.keys())
    x_values = sorted(all_x)

    if not x_values:
        print("No data to plot")
        return

    # Color palette - use different colors for positive vs negative rubrics
    n_rubrics = len(rubrics)
    colors_positive = plt.cm.Blues(np.linspace(0.4, 0.9, n_rubrics))
    colors_negative = plt.cm.Reds(np.linspace(0.4, 0.9, n_rubrics))

    for i, rubric in enumerate(rubrics):
        rubric_stats = stats.get(rubric, {})
        if not rubric_stats:
            continue

        xs = []
        ys = []
        errs = []
        for x in x_values:
            if x in rubric_stats:
                xs.append(x)
                ys.append(rubric_stats[x]["mean"])
                errs.append(rubric_stats[x]["stderr"])

        if not xs:
            continue

        # Choose color based on rubric type
        if rubric in NEGATIVE_RUBRICS:
            color = colors_negative[i % len(colors_negative)]
            linestyle = '--'
        else:
            color = colors_positive[i % len(colors_positive)]
            linestyle = '-'

        label = RUBRIC_SHORT_NAMES.get(rubric, rubric)
        ax.errorbar(xs, ys, yerr=errs, label=label, marker='o',
                   linestyle=linestyle, capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel(var_name, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    # Build title with model info
    title_text = title or f"Rubric Scores vs {var_name}"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if n_samples:
        subtitle_parts.append(f"n={n_samples}")

    if subtitle_parts:
        title_text += f"\n({', '.join(subtitle_parts)})"

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to actual values (use int if all are whole numbers)
    if all(x == int(x) for x in x_values):
        ax.set_xticks([int(x) for x in x_values])
    else:
        ax.set_xticks(x_values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_aggregate_plot(
    groups: dict[float, list[dict]],
    rubrics: list[str],
    var_name: str,
    output_path: Path,
    n_samples: int = 0,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save plot with 3 lines: sum of non-GDM metrics, gdm_legibility, gdm_coverage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = sorted(groups.keys())
    gdm_rubrics = ["gdm_legibility", "gdm_coverage"]
    non_gdm_rubrics = [r for r in rubrics if r not in gdm_rubrics]

    # Compute aggregates per x value
    non_gdm_means = []
    non_gdm_errs = []
    gdm_leg_means = []
    gdm_leg_errs = []
    gdm_cov_means = []
    gdm_cov_errs = []

    for x in x_values:
        samples = groups[x]

        # Sum of non-GDM for each sample
        non_gdm_sums = []
        for s in samples:
            total = sum(
                s.get("scores", {}).get(r, {}).get("value", 0)
                for r in non_gdm_rubrics
            )
            non_gdm_sums.append(total)

        if non_gdm_sums:
            non_gdm_means.append(statistics.mean(non_gdm_sums))
            non_gdm_errs.append(
                statistics.stdev(non_gdm_sums) / (len(non_gdm_sums) ** 0.5)
                if len(non_gdm_sums) > 1 else 0
            )
        else:
            non_gdm_means.append(0)
            non_gdm_errs.append(0)

        # GDM legibility
        leg_vals = [
            s.get("scores", {}).get("gdm_legibility", {}).get("value", 0)
            for s in samples
        ]
        if leg_vals:
            gdm_leg_means.append(statistics.mean(leg_vals))
            gdm_leg_errs.append(
                statistics.stdev(leg_vals) / (len(leg_vals) ** 0.5)
                if len(leg_vals) > 1 else 0
            )
        else:
            gdm_leg_means.append(0)
            gdm_leg_errs.append(0)

        # GDM coverage
        cov_vals = [
            s.get("scores", {}).get("gdm_coverage", {}).get("value", 0)
            for s in samples
        ]
        if cov_vals:
            gdm_cov_means.append(statistics.mean(cov_vals))
            gdm_cov_errs.append(
                statistics.stdev(cov_vals) / (len(cov_vals) ** 0.5)
                if len(cov_vals) > 1 else 0
            )
        else:
            gdm_cov_means.append(0)
            gdm_cov_errs.append(0)

    # Convert x_values to int if appropriate
    if all(x == int(x) for x in x_values):
        x_plot = [int(x) for x in x_values]
    else:
        x_plot = x_values

    # Plot
    ax.errorbar(x_plot, non_gdm_means, yerr=non_gdm_errs,
                label=f"Non-GDM Sum ({len(non_gdm_rubrics)} rubrics)",
                marker='o', capsize=3, linewidth=2, markersize=8, color='steelblue')
    ax.errorbar(x_plot, gdm_leg_means, yerr=gdm_leg_errs,
                label="GDM Legibility",
                marker='s', capsize=3, linewidth=2, markersize=8, color='forestgreen')
    ax.errorbar(x_plot, gdm_cov_means, yerr=gdm_cov_errs,
                label="GDM Coverage",
                marker='^', capsize=3, linewidth=2, markersize=8, color='darkorange')

    ax.set_xlabel(var_name, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    # Build title with model info
    title_text = f"Aggregate Scores vs {var_name}"
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    if n_samples:
        subtitle_parts.append(f"n={n_samples}")

    if subtitle_parts:
        title_text += f"\n({', '.join(subtitle_parts)})"

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_plot)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_grid_plot_by_condition(
    samples: list[dict],
    rubrics: list[str],
    condition_var: str,
    x_var: str,
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save a grid of plots, one per condition value (e.g., alpha).

    Each subplot shows x_var on x-axis (e.g., problem_idx) and rubric scores on y-axis.
    """
    # Group samples by condition (e.g., alpha)
    condition_groups = group_samples_by_var(samples, condition_var)
    condition_values = sorted(condition_groups.keys())

    if not condition_values:
        print(f"No data for condition variable '{condition_var}'")
        return

    # Determine grid size
    n_conditions = len(condition_values)
    n_cols = 2
    n_rows = (n_conditions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color palette
    n_rubrics = len(rubrics)
    colors_positive = plt.cm.Blues(np.linspace(0.4, 0.9, n_rubrics))
    colors_negative = plt.cm.Reds(np.linspace(0.4, 0.9, n_rubrics))

    for idx, condition_val in enumerate(condition_values):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Get samples for this condition and group by x_var
        condition_samples = condition_groups[condition_val]
        x_groups = group_samples_by_var(condition_samples, x_var)
        x_values = sorted(x_groups.keys())

        # Compute stats for each rubric at each x value
        stats = compute_rubric_stats_by_var(x_groups, rubrics)

        # Plot each rubric
        for i, rubric in enumerate(rubrics):
            rubric_stats = stats.get(rubric, {})
            if not rubric_stats:
                continue

            xs = []
            ys = []
            errs = []
            for x in x_values:
                if x in rubric_stats:
                    xs.append(x)
                    ys.append(rubric_stats[x]["mean"])
                    errs.append(rubric_stats[x]["stderr"])

            if not xs:
                continue

            # Choose color based on rubric type
            if rubric in NEGATIVE_RUBRICS:
                color = colors_negative[i % len(colors_negative)]
                linestyle = '--'
            else:
                color = colors_positive[i % len(colors_positive)]
                linestyle = '-'

            label = RUBRIC_SHORT_NAMES.get(rubric, rubric)
            ax.errorbar(xs, ys, yerr=errs, label=label, marker='o',
                       linestyle=linestyle, capsize=2, linewidth=1.5, markersize=4)

        # Format subplot
        ax.set_xlabel(x_var, fontsize=10)
        ax.set_ylabel("Score", fontsize=10)

        # Format condition value in title
        if condition_val == int(condition_val):
            cond_str = f"{condition_var}={int(condition_val)}"
        else:
            cond_str = f"{condition_var}={condition_val}"
        ax.set_title(cond_str, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Set x-ticks
        if all(x == int(x) for x in x_values):
            ax.set_xticks([int(x) for x in x_values])
        else:
            ax.set_xticks(x_values)

    # Hide unused subplots
    for idx in range(n_conditions, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Add shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=8)

    # Build suptitle with model info
    title_parts = [f"Rubric Scores by {condition_var}"]
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    subtitle_parts.append(f"n={len(samples)}")

    suptitle = title_parts[0]
    if subtitle_parts:
        suptitle += f"\n({', '.join(subtitle_parts)})"

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_aggregate_grid_plot(
    samples: list[dict],
    rubrics: list[str],
    condition_var: str,
    x_var: str,
    output_path: Path,
    target_model: str = "",
    judge_model: str = "",
) -> None:
    """Save a grid of aggregate plots, one per condition value.

    Each subplot shows non-GDM sum, gdm_legibility, gdm_coverage.
    """
    # Group samples by condition
    condition_groups = group_samples_by_var(samples, condition_var)
    condition_values = sorted(condition_groups.keys())

    if not condition_values:
        print(f"No data for condition variable '{condition_var}'")
        return

    gdm_rubrics = ["gdm_legibility", "gdm_coverage"]
    non_gdm_rubrics = [r for r in rubrics if r not in gdm_rubrics]

    # Determine grid size
    n_conditions = len(condition_values)
    n_cols = 2
    n_rows = (n_conditions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, condition_val in enumerate(condition_values):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Get samples for this condition and group by x_var
        condition_samples = condition_groups[condition_val]
        x_groups = group_samples_by_var(condition_samples, x_var)
        x_values = sorted(x_groups.keys())

        # Compute aggregates
        non_gdm_means = []
        non_gdm_errs = []
        gdm_leg_means = []
        gdm_leg_errs = []
        gdm_cov_means = []
        gdm_cov_errs = []

        for x in x_values:
            group_samples = x_groups[x]

            # Sum of non-GDM
            non_gdm_sums = [
                sum(s.get("scores", {}).get(r, {}).get("value", 0) for r in non_gdm_rubrics)
                for s in group_samples
            ]
            if non_gdm_sums:
                non_gdm_means.append(statistics.mean(non_gdm_sums))
                non_gdm_errs.append(
                    statistics.stdev(non_gdm_sums) / (len(non_gdm_sums) ** 0.5)
                    if len(non_gdm_sums) > 1 else 0
                )
            else:
                non_gdm_means.append(0)
                non_gdm_errs.append(0)

            # GDM legibility
            leg_vals = [s.get("scores", {}).get("gdm_legibility", {}).get("value", 0) for s in group_samples]
            gdm_leg_means.append(statistics.mean(leg_vals) if leg_vals else 0)
            gdm_leg_errs.append(
                statistics.stdev(leg_vals) / (len(leg_vals) ** 0.5) if len(leg_vals) > 1 else 0
            )

            # GDM coverage
            cov_vals = [s.get("scores", {}).get("gdm_coverage", {}).get("value", 0) for s in group_samples]
            gdm_cov_means.append(statistics.mean(cov_vals) if cov_vals else 0)
            gdm_cov_errs.append(
                statistics.stdev(cov_vals) / (len(cov_vals) ** 0.5) if len(cov_vals) > 1 else 0
            )

        # Convert x to int if appropriate
        x_plot = [int(x) if x == int(x) else x for x in x_values]

        # Plot
        ax.errorbar(x_plot, non_gdm_means, yerr=non_gdm_errs,
                    label=f"Non-GDM Sum ({len(non_gdm_rubrics)})",
                    marker='o', capsize=2, linewidth=1.5, markersize=5, color='steelblue')
        ax.errorbar(x_plot, gdm_leg_means, yerr=gdm_leg_errs,
                    label="GDM Legibility",
                    marker='s', capsize=2, linewidth=1.5, markersize=5, color='forestgreen')
        ax.errorbar(x_plot, gdm_cov_means, yerr=gdm_cov_errs,
                    label="GDM Coverage",
                    marker='^', capsize=2, linewidth=1.5, markersize=5, color='darkorange')

        # Format subplot
        ax.set_xlabel(x_var, fontsize=10)
        ax.set_ylabel("Score", fontsize=10)

        if condition_val == int(condition_val):
            cond_str = f"{condition_var}={int(condition_val)}"
        else:
            cond_str = f"{condition_var}={condition_val}"
        ax.set_title(cond_str, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_plot)

    # Hide unused subplots
    for idx in range(n_conditions, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Add shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)

    # Build suptitle
    title_parts = [f"Aggregate Scores by {condition_var}"]
    subtitle_parts = []
    if target_model:
        subtitle_parts.append(f"Target: {target_model}")
    if judge_model:
        subtitle_parts.append(f"Judge: {judge_model}")
    subtitle_parts.append(f"n={len(samples)}")

    suptitle = title_parts[0]
    if subtitle_parts:
        suptitle += f"\n({', '.join(subtitle_parts)})"

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_table(
    stats: dict[str, dict[float, dict]],
    rubrics: list[str],
    var_name: str,
) -> str:
    """Generate a text summary table of scores by variable value."""
    lines = []

    # Get sorted x values
    all_x = set()
    for rubric_stats in stats.values():
        all_x.update(rubric_stats.keys())
    x_values = sorted(all_x)

    if not x_values:
        return "No data available"

    # Determine if x values are integers
    is_int = all(x == int(x) for x in x_values)

    # Header
    col_width = 12
    if is_int:
        header = f"{'Rubric':<20}" + "".join(f"{var_name}={int(x):<{col_width-len(var_name)-1}}" for x in x_values)
    else:
        header = f"{'Rubric':<20}" + "".join(f"{var_name}={x:<{col_width-len(var_name)-1}.1f}" for x in x_values)
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for rubric in rubrics:
        short = RUBRIC_SHORT_NAMES.get(rubric, rubric[:18])
        row = f"{short:<20}"
        rubric_stats = stats.get(rubric, {})
        for x in x_values:
            if x in rubric_stats:
                mean = rubric_stats[x]["mean"]
                stderr = rubric_stats[x]["stderr"]
                row += f"{mean:>5.2f}±{stderr:<5.2f}"
            else:
                row += " " * col_width
        lines.append(row)

    return "\n".join(lines)


def find_most_recent_eval(logs_dir: Path) -> Path | None:
    """Find the most recent .eval file in the logs directory."""
    eval_files = list(logs_dir.glob("*.eval"))
    if not eval_files:
        return None
    return max(eval_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rubric scores across experimental conditions",
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
        "--var",
        type=str,
        default="problem_idx",
        help="Name of experimental variable to analyze (default: problem_idx)",
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
    print(f"Experimental variable: {args.var}")

    # Extract data
    samples = extract_samples_from_eval(eval_path)
    header = extract_header(eval_path)

    if not samples:
        print("No samples found in eval file")
        return 1

    # Extract model names
    target_model = extract_model_name(samples, header)
    judge_model = extract_judge_model(header)
    print(f"Target model: {target_model}")
    print(f"Judge model: {judge_model}")

    # Detect rubrics from first sample
    rubrics = list(samples[0].get("scores", {}).keys())
    print(f"Found {len(rubrics)} rubrics: {', '.join(rubrics)}")

    # Group samples by variable
    groups = group_samples_by_var(samples, args.var)
    if not groups:
        print(f"Could not extract variable '{args.var}' from samples")
        return 1

    print(f"Found {len(groups)} unique {args.var} values: {sorted(groups.keys())}")
    for var_val, group_samples in sorted(groups.items()):
        print(f"  {args.var}={var_val}: {len(group_samples)} samples")

    # Compute statistics
    stats = compute_rubric_stats_by_var(groups, rubrics)

    # Output directory
    output_dir = args.output_dir or (project_root / "logs" / "analyses")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"sample_analysis_{eval_path.stem}"

    # Generate and save summary table
    summary = generate_summary_table(stats, rubrics, args.var)
    summary_path = output_dir / f"{base_name}.txt"

    full_report = f"""Sample Analysis Report
=====================
Eval file: {eval_path.name}
Analysis time: {datetime.now().isoformat()}
Target model: {target_model}
Judge model: {judge_model}
Experimental variable: {args.var}
Total samples: {len(samples)}
Rubrics: {len(rubrics)}

{summary}
"""
    summary_path.write_text(full_report)
    print(f"\nSummary saved to: {summary_path}")

    # Save combined plot (all rubrics)
    combined_path = output_dir / f"{base_name}_combined.png"
    save_line_plot(stats, rubrics, args.var, combined_path,
                   title=f"All Rubric Scores vs {args.var}",
                   n_samples=len(samples),
                   target_model=target_model,
                   judge_model=judge_model)
    print(f"Combined plot saved to: {combined_path}")

    # Save aggregate plot (non-GDM sum + GDM metrics)
    aggregate_path = output_dir / f"{base_name}_aggregate.png"
    save_aggregate_plot(groups, rubrics, args.var, aggregate_path,
                        n_samples=len(samples),
                        target_model=target_model,
                        judge_model=judge_model)
    print(f"Aggregate plot saved to: {aggregate_path}")

    # Save grid plots by alpha (if alpha data exists)
    alpha_groups = group_samples_by_var(samples, "alpha")
    if alpha_groups and len(alpha_groups) > 1:
        # Grid of all rubrics by alpha
        grid_path = output_dir / f"{base_name}_grid_by_alpha.png"
        save_grid_plot_by_condition(
            samples, rubrics, "alpha", args.var, grid_path,
            target_model=target_model, judge_model=judge_model
        )
        print(f"Grid plot (by alpha) saved to: {grid_path}")

        # Grid of aggregates by alpha
        grid_agg_path = output_dir / f"{base_name}_grid_aggregate_by_alpha.png"
        save_aggregate_grid_plot(
            samples, rubrics, "alpha", args.var, grid_agg_path,
            target_model=target_model, judge_model=judge_model
        )
        print(f"Aggregate grid plot (by alpha) saved to: {grid_agg_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
