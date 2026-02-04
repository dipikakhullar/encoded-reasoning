#!/usr/bin/env python3
"""Analyze evaluation results comparing composite rubrics to component rubrics.

Supports two modes:
1. Single file: All 39 rubrics in one eval file
2. Two files: Component rubrics in one file, composite rubrics in another

Compares:
- Actual composite scores vs synthetic composites (component averages)
- Composite totals vs full 32-rubric totals
- GDM legacy rubrics vs composites

Usage:
    # Auto-detect mode (single file with all rubrics, or find matching pair)
    uv run python cot_quality_metrics/src/standalone_scripts/compare_rubric_selections.py

    # Explicit two-file mode
    uv run python cot_quality_metrics/src/standalone_scripts/compare_rubric_selections.py --component-eval FILE1 --composite-eval FILE2
"""

import argparse
import json
import statistics
import sys
import zipfile
from pathlib import Path

from scipy.stats import spearmanr

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

LEGACY_RUBRICS = ["gdm_legibility", "gdm_coverage"]

COMPOSITES = ["fake_rigor", "reportive_fidelity", "active_investigation", "epistemic_honesty", "adaptive_process"]

ALL_NOVEL = POSITIVE_RUBRICS + NEGATIVE_RUBRICS

# Composite component definitions (for synthetic composite comparison)
# Note: reportive_fidelity is holistic and doesn't map directly to component rubrics
COMPOSITE_COMPONENTS = {
    "fake_rigor": ["premature_formalization", "cargo_cult_methodology", "complexity_theater", "intellectual_flinching"],
    "reportive_fidelity": [],  # Holistic - no direct component mapping
    "active_investigation": ["discriminative_experiment_design", "error_metabolism", "contact_with_reality"],
    "epistemic_honesty": ["calibration", "provenance_transparency", "process_conclusion_integrity"],
    "adaptive_process": ["generativity_under_stuckness", "noticing_confusion", "live_updating"],
}


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def extract_samples(eval_path: Path) -> dict:
    samples = {}
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as f:
                    data = json.load(f)
                    samples[data["id"]] = data
    return samples


def get_score(sample: dict, rubric: str) -> float | None:
    sc = sample.get("scores", {}).get(rubric, {})
    if "Failed to parse" in sc.get("explanation", ""):
        return None
    return sc.get("value")


def compute_total(sample: dict, rubrics: list[str]) -> float | None:
    total = 0
    for r in rubrics:
        v = get_score(sample, r)
        if v is None:
            return None
        total += v
    return total


def compute_synthetic_composite(sample: dict, composite: str) -> float | None:
    components = COMPOSITE_COMPONENTS[composite]
    if not components:  # Holistic composite with no component mapping
        return None
    scores = [get_score(sample, c) for c in components]
    if None in scores:
        return None
    return statistics.mean(scores)


def merge_samples(component_samples: dict, composite_samples: dict) -> dict:
    """Merge samples from two eval files, combining scores by sample ID."""
    merged = {}
    common_ids = set(component_samples.keys()) & set(composite_samples.keys())

    for sid in common_ids:
        comp_sample = component_samples[sid]
        composite_sample = composite_samples[sid]

        # Start with component sample and add composite scores
        merged_sample = comp_sample.copy()
        merged_sample["scores"] = comp_sample.get("scores", {}).copy()

        # Add composite scores
        for rubric, score_data in composite_sample.get("scores", {}).items():
            if rubric in COMPOSITES:
                merged_sample["scores"][rubric] = score_data

        merged[sid] = merged_sample

    return merged


def get_sample_shuffle_seed(eval_path: Path) -> int | bool | None:
    """Extract sample_shuffle value from eval file config."""
    with zipfile.ZipFile(eval_path, "r") as zf:
        try:
            with zf.open("_journal/start.json") as f:
                data = json.load(f)
                return data.get("eval", {}).get("config", {}).get("sample_shuffle")
        except (KeyError, json.JSONDecodeError):
            return None


def find_matching_evals(logs_dir: Path) -> tuple[Path | None, Path | None]:
    """Find a matching pair of component and composite eval files with same seed."""
    # Find all evals with seed 42
    component_evals = []
    composite_evals = []

    for eval_path in logs_dir.glob("*cot-quality*.eval"):
        seed = get_sample_shuffle_seed(eval_path)
        if seed != 42:
            continue

        # Check what rubrics are present
        samples = extract_samples(eval_path)
        if not samples:
            continue

        sample = next(iter(samples.values()))
        rubrics_present = set(sample.get("scores", {}).keys())

        has_composites = bool(rubrics_present & set(COMPOSITES))
        has_components = bool(rubrics_present & set(ALL_NOVEL))

        if has_composites and not has_components:
            composite_evals.append((eval_path, len(samples)))
        elif has_components and not has_composites:
            component_evals.append((eval_path, len(samples)))

    # Return largest of each type
    component_evals.sort(key=lambda x: x[1], reverse=True)
    composite_evals.sort(key=lambda x: x[1], reverse=True)

    component_path = component_evals[0][0] if component_evals else None
    composite_path = composite_evals[0][0] if composite_evals else None

    return component_path, composite_path


def main():
    parser = argparse.ArgumentParser(description="Compare composite vs component rubric evaluations")
    parser.add_argument("--component-eval", type=Path, help="Eval file with component rubrics (32 novel + 2 legacy)")
    parser.add_argument("--composite-eval", type=Path, help="Eval file with composite rubrics (5 composites)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    logs_dir = project_root / "logs"

    # Determine mode based on arguments
    if args.component_eval and args.composite_eval:
        # Explicit two-file mode
        component_path = args.component_eval
        composite_path = args.composite_eval
        two_file_mode = True
    else:
        # Auto-detect mode: first try to find a single file with all rubrics
        all_evals = sorted(logs_dir.glob("*cot-quality*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not all_evals:
            print("No eval files found")
            sys.exit(1)

        # Check if most recent file has both components and composites
        eval_path = all_evals[0]
        test_samples = extract_samples(eval_path)
        if test_samples:
            sample = next(iter(test_samples.values()))
            rubrics_present = set(sample.get("scores", {}).keys())
            has_both = (rubrics_present & set(ALL_NOVEL)) and (rubrics_present & set(COMPOSITES))

            if has_both:
                # Single file mode
                print(f"Analyzing single file: {eval_path.name}")
                samples = test_samples
                two_file_mode = False
            else:
                # Try to find matching pair
                print("Most recent file doesn't have all rubrics, searching for matching pair...")
                component_path, composite_path = find_matching_evals(logs_dir)
                two_file_mode = True
        else:
            component_path, composite_path = find_matching_evals(logs_dir)
            two_file_mode = True

    if two_file_mode:
        if not component_path or not composite_path:
            print("Could not find matching component and composite eval files with seed 42")
            sys.exit(1)

        print(f"Component eval: {component_path.name}")
        print(f"Composite eval: {composite_path.name}")

        component_samples = extract_samples(component_path)
        composite_samples = extract_samples(composite_path)

        print(f"Component samples: {len(component_samples)}")
        print(f"Composite samples: {len(composite_samples)}")

        samples = merge_samples(component_samples, composite_samples)
        print(f"Merged samples (common IDs): {len(samples)}")

    # Check which rubrics are present
    sample = next(iter(samples.values()))
    rubrics_present = set(sample.get("scores", {}).keys())

    has_novel = bool(rubrics_present & set(ALL_NOVEL))
    has_legacy = bool(rubrics_present & set(LEGACY_RUBRICS))
    has_composites = bool(rubrics_present & set(COMPOSITES))

    print(f"Novel rubrics: {'Yes' if has_novel else 'No'}")
    print(f"Legacy rubrics: {'Yes' if has_legacy else 'No'}")
    print(f"Composite rubrics: {'Yes' if has_composites else 'No'}")
    print()

    # Get valid samples (all scores parsed successfully)
    all_rubrics_to_check = []
    if has_novel:
        all_rubrics_to_check.extend(ALL_NOVEL)
    if has_legacy:
        all_rubrics_to_check.extend(LEGACY_RUBRICS)
    if has_composites:
        all_rubrics_to_check.extend(COMPOSITES)

    valid_samples = {
        sid: s for sid, s in samples.items()
        if all(get_score(s, r) is not None for r in all_rubrics_to_check if r in rubrics_present)
    }
    print(f"Valid samples (all scores parsed): {len(valid_samples)}")
    print()

    if has_novel and has_composites:
        print("=" * 70)
        print("COMPOSITE vs COMPONENT COMPARISON")
        print("=" * 70)
        print()

        # Per-composite analysis
        print(f"{'Composite':<25} {'vs Synthetic':>12} {'vs Components':>14}")
        print("-" * 55)

        for comp in COMPOSITES:
            components = COMPOSITE_COMPONENTS[comp]

            actual = []
            synthetic = []
            component_totals = []

            for sid, s in valid_samples.items():
                a = get_score(s, comp)
                syn = compute_synthetic_composite(s, comp)
                comp_total = sum(get_score(s, c) or 0 for c in components)

                if a is not None and syn is not None:
                    actual.append(a)
                    synthetic.append(syn)
                    component_totals.append(comp_total)

            if len(actual) > 2:
                r_syn = pearson_correlation(actual, synthetic)
                r_comp = pearson_correlation(actual, component_totals)
                print(f"{comp:<25} {r_syn:>12.3f} {r_comp:>14.3f}")

        print()

        # Overall comparison
        print("=" * 70)
        print("OVERALL CORRELATIONS")
        print("=" * 70)
        print()

        novel_totals = []
        composite_totals = []
        synthetic_totals = []

        for sid, s in valid_samples.items():
            novel = compute_total(s, ALL_NOVEL)
            comp = compute_total(s, COMPOSITES)

            syn_total = 0
            for c in COMPOSITES:
                syn = compute_synthetic_composite(s, c)
                if syn is not None:
                    syn_total += syn

            if novel is not None and comp is not None:
                novel_totals.append(novel)
                composite_totals.append(comp)
                synthetic_totals.append(syn_total)

        if len(novel_totals) > 2:
            r_actual = pearson_correlation(novel_totals, composite_totals)
            rho_actual, _ = spearmanr(novel_totals, composite_totals)

            r_synth = pearson_correlation(novel_totals, synthetic_totals)
            rho_synth, _ = spearmanr(novel_totals, synthetic_totals)

            print(f"{'Comparison':<35} {'Pearson r':>10} {'Spearman ρ':>11}")
            print("-" * 60)
            print(f"{'Actual Composites vs 32 Components':<35} {r_actual:>10.3f} {rho_actual:>11.3f}")
            print(f"{'Synthetic Composites vs 32 Components':<35} {r_synth:>10.3f} {rho_synth:>11.3f}")
            print(f"{'N samples':<35} {len(novel_totals):>10}")

    if has_legacy and has_composites:
        print()
        print("=" * 70)
        print("LEGACY vs COMPOSITE CORRELATIONS")
        print("=" * 70)
        print()

        # Correlate legacy rubrics with composites
        print(f"{'':>20}", end="")
        for comp in COMPOSITES:
            print(f"{comp[:10]:>12}", end="")
        print()

        for leg in LEGACY_RUBRICS:
            print(f"{leg:<20}", end="")
            for comp in COMPOSITES:
                leg_scores = [get_score(s, leg) for s in valid_samples.values()]
                comp_scores = [get_score(s, comp) for s in valid_samples.values()]

                pairs = [(l, c) for l, c in zip(leg_scores, comp_scores) if l is not None and c is not None]
                if len(pairs) > 2:
                    r = pearson_correlation([p[0] for p in pairs], [p[1] for p in pairs])
                    print(f"{r:>12.2f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()

    # Score distributions summary
    print()
    print("=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)
    print()
    print(f"{'Rubric':<30} {'Mean':>7} {'Std':>7} {'Min':>5} {'Max':>5} {'N':>6}")
    print("-" * 65)

    for rubric_group, rubrics in [("COMPOSITES", COMPOSITES), ("LEGACY", LEGACY_RUBRICS)]:
        for r in rubrics:
            if r not in rubrics_present:
                continue
            scores = [get_score(s, r) for s in valid_samples.values()]
            scores = [s for s in scores if s is not None]
            if scores:
                std = statistics.stdev(scores) if len(scores) > 1 else 0
                print(f"{r:<30} {statistics.mean(scores):>7.2f} {std:>7.2f} {min(scores):>5.0f} {max(scores):>5.0f} {len(scores):>6}")


if __name__ == "__main__":
    main()
