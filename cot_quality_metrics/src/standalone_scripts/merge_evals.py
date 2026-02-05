#!/usr/bin/env python3
"""
Merge multiple .eval files into a single combined eval file.

Usage:
    uv run python merge_evals.py eval1.eval eval2.eval -o combined.eval
    uv run python merge_evals.py logs/*.eval -o logs/merged.eval
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path


def extract_samples(eval_path: Path) -> list[dict]:
    """Extract all samples from an eval file."""
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


def merge_eval_files(eval_paths: list[Path], output_path: Path) -> None:
    """Merge multiple eval files into one."""
    all_samples = []
    headers = []

    for eval_path in eval_paths:
        print(f"Reading {eval_path.name}...")
        samples = extract_samples(eval_path)
        header = extract_header(eval_path)

        # Track alpha values found
        alphas = set()
        for s in samples:
            alpha = s.get("metadata", {}).get("alpha")
            if alpha is not None:
                alphas.add(alpha)

        print(f"  Found {len(samples)} samples, alphas: {sorted(alphas)}")
        all_samples.extend(samples)
        headers.append(header)

    print(f"\nTotal samples: {len(all_samples)}")

    # Get all unique alphas
    all_alphas = set()
    for s in all_samples:
        alpha = s.get("metadata", {}).get("alpha")
        if alpha is not None:
            all_alphas.add(alpha)
    print(f"All alphas: {sorted(all_alphas)}")

    # Create merged eval file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write merged header (use first header as base, update some fields)
        merged_header = headers[0].copy() if headers else {}
        merged_header["merged_from"] = [str(p) for p in eval_paths]
        zf.writestr("header.json", json.dumps(merged_header, indent=2))

        # Write all samples
        for i, sample in enumerate(all_samples):
            sample_json = json.dumps(sample, indent=2)
            zf.writestr(f"samples/{i:06d}.json", sample_json)

    print(f"\nMerged eval saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple .eval files into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "eval_files",
        nargs="+",
        type=Path,
        help="Eval files to merge",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path for merged eval file",
    )
    args = parser.parse_args()

    # Validate inputs
    for p in args.eval_files:
        if not p.exists():
            print(f"Error: {p} not found")
            return 1

    merge_eval_files(args.eval_files, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
