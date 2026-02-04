#!/usr/bin/env python3
"""Reparse eval files to fix score extraction from explanations.

Use this after fixing the parser to re-extract scores from existing
judge explanations without re-running expensive API calls.

Usage:
    uv run python cot_quality_metrics/src/standalone_scripts/reparse_eval.py logs/some_eval.eval
    uv run python cot_quality_metrics/src/standalone_scripts/reparse_eval.py logs/some_eval.eval --output logs/reparsed.eval
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def extract_score_method_1(explanation: str) -> float | None:
    """Method 1: Find last 'Score: X' pattern."""
    pattern = r"\*{0,2}(?:Overall\s+)?(?:Final\s+)?(?:Score|Rating)\s*[:=]?\s*\*{0,2}\s*(-?\d+\.?\d*)(?:/\d+)?\*{0,2}"
    matches = list(re.finditer(pattern, explanation, re.IGNORECASE))
    if matches:
        return float(matches[-1].group(1))
    return None


def extract_score_method_2(explanation: str) -> float | None:
    """Method 2: Find standalone number at end of text."""
    match = re.search(r"\*{0,2}(-?\d+\.?\d*)\*{0,2}\s*$", explanation.strip())
    if match:
        return float(match.group(1))
    return None


def extract_score_method_3(explanation: str) -> float | None:
    """Method 3: Find 'X/5' or 'X/4' pattern (component scores)."""
    # Look for final score pattern like "4.5/5" at end
    match = re.search(r"(-?\d+\.?\d*)\s*/\s*[45]\s*\**\s*$", explanation.strip())
    if match:
        return float(match.group(1))
    return None


def extract_score_from_explanation(explanation: str, rubric: str = "") -> tuple[float | None, bool]:
    """Re-extract score from explanation text using multiple methods.

    Args:
        explanation: The judge's explanation text
        rubric: Rubric name (used to determine valid score range)

    Returns:
        Tuple of (extracted score, confidence flag)
        - score: Extracted score if valid, None otherwise
        - confident: True if methods agree, False if disagreement detected
    """
    # Determine valid range based on rubric type
    if rubric.startswith("gdm_"):
        valid_range = (0, 4)
    elif rubric == "fake_rigor":
        valid_range = (-5, 0)
    elif rubric in ("reportive_fidelity", "active_investigation", "epistemic_honesty", "adaptive_process"):
        valid_range = (0, 5)
    else:
        valid_range = (-5, 5)  # Default: accept anything in full range

    def clamp(score):
        """Clamp score to valid range."""
        if score is None:
            return None
        return max(valid_range[0], min(valid_range[1], score))

    # Try all methods
    m1 = extract_score_method_1(explanation)
    m2 = extract_score_method_2(explanation)
    m3 = extract_score_method_3(explanation)

    # Clamp all scores to valid range
    scores = [clamp(s) for s in [m1, m2, m3] if s is not None]

    if not scores:
        return None, True  # No score found

    # Check if methods agree (after clamping)
    unique_scores = set(scores)
    confident = len(unique_scores) == 1

    # Prefer method 1 (explicit Score: pattern), then method 3, then method 2
    if m1 is not None:
        return clamp(m1), confident
    if m3 is not None:
        return clamp(m3), confident
    if m2 is not None:
        return clamp(m2), confident

    return scores[0], confident


def reparse_eval(input_path: Path, output_path: Path | None = None) -> dict:
    """Reparse an eval file and save corrected version.

    Args:
        input_path: Path to input .eval file
        output_path: Path to output .eval file (default: overwrite input)

    Returns:
        Statistics about the reparsing
    """
    if output_path is None:
        output_path = input_path

    stats = {
        "total_scores": 0,
        "changed": 0,
        "unchanged": 0,
        "failed": 0,
        "changes": []
    }

    # Create a temporary file for the new zip
    with tempfile.NamedTemporaryFile(delete=False, suffix='.eval') as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(input_path, 'r') as zf_in:
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                for name in zf_in.namelist():
                    if name.startswith("samples/") and name.endswith(".json"):
                        # Process sample file
                        with zf_in.open(name) as f:
                            data = json.load(f)

                        sample_id = data.get("id", name)

                        for rubric, score_data in data.get("scores", {}).items():
                            stats["total_scores"] += 1

                            old_value = score_data.get("value")
                            explanation = score_data.get("explanation", "")

                            # Skip if already marked as parse failure
                            if "Failed to parse" in explanation:
                                stats["failed"] += 1
                                continue

                            new_value, confident = extract_score_from_explanation(explanation, rubric)

                            if new_value is not None and new_value != old_value:
                                score_data["value"] = new_value
                                score_data["answer"] = str(new_value)
                                stats["changed"] += 1
                                stats["changes"].append({
                                    "sample": sample_id,
                                    "rubric": rubric,
                                    "old": old_value,
                                    "new": new_value,
                                    "confident": confident
                                })
                            else:
                                stats["unchanged"] += 1

                        # Write updated sample
                        zf_out.writestr(name, json.dumps(data, indent=2))
                    else:
                        # Copy non-sample files as-is
                        zf_out.writestr(name, zf_in.read(name))

        # Move temp file to output path
        shutil.move(tmp_path, output_path)

    except Exception as e:
        # Clean up temp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return stats


def main():
    parser = argparse.ArgumentParser(description="Reparse eval file to fix score extraction")
    parser.add_argument("input", type=Path, help="Input .eval file")
    parser.add_argument("--output", "-o", type=Path, help="Output .eval file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        # Just analyze without saving
        with zipfile.ZipFile(args.input, 'r') as zf:
            changes = []
            for name in zf.namelist():
                if name.startswith("samples/") and name.endswith(".json"):
                    with zf.open(name) as f:
                        data = json.load(f)

                    sample_id = data.get("id", name)
                    for rubric, score_data in data.get("scores", {}).items():
                        old_value = score_data.get("value")
                        explanation = score_data.get("explanation", "")

                        if "Failed to parse" in explanation:
                            continue

                        new_value, confident = extract_score_from_explanation(explanation, rubric)
                        if new_value is not None and new_value != old_value:
                            changes.append({
                                "sample": sample_id,
                                "rubric": rubric,
                                "old": old_value,
                                "new": new_value,
                                "confident": confident
                            })

        confident_changes = [c for c in changes if c['confident']]
        uncertain_changes = [c for c in changes if not c['confident']]

        print(f"Dry run: {len(changes)} scores would change")
        print(f"  - {len(confident_changes)} confident (methods agree)")
        print(f"  - {len(uncertain_changes)} uncertain (methods disagree) ⚠️")
        print()

        if uncertain_changes:
            print("UNCERTAIN CHANGES (methods disagree - review manually):")
            for c in uncertain_changes[:10]:
                print(f"  ⚠️ {c['rubric']:<22} {c['old']:>6} -> {c['new']:<6}  ({c['sample'][:40]}...)")
            if len(uncertain_changes) > 10:
                print(f"  ... and {len(uncertain_changes) - 10} more uncertain")
            print()

        if confident_changes:
            print("CONFIDENT CHANGES (methods agree):")
            for c in confident_changes[:15]:
                print(f"  ✓ {c['rubric']:<22} {c['old']:>6} -> {c['new']:<6}  ({c['sample'][:40]}...)")
            if len(confident_changes) > 15:
                print(f"  ... and {len(confident_changes) - 15} more confident")
    else:
        output = args.output or args.input
        print(f"Reparsing: {args.input}")
        print(f"Output: {output}")

        stats = reparse_eval(args.input, output)

        print(f"\nResults:")
        print(f"  Total scores: {stats['total_scores']}")
        print(f"  Changed: {stats['changed']}")
        print(f"  Unchanged: {stats['unchanged']}")
        print(f"  Already failed: {stats['failed']}")

        if stats['changes']:
            print(f"\nSample changes:")
            for c in stats['changes'][:10]:
                print(f"  {c['rubric']:<22} {c['old']:>6} -> {c['new']:<6}")
            if len(stats['changes']) > 10:
                print(f"  ... and {len(stats['changes']) - 10} more")


if __name__ == "__main__":
    main()
