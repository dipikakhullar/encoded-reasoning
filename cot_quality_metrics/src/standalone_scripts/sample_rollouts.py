#!/usr/bin/env python3
"""
Sample rollouts from rollout files and select winners based on a rule.

For each sample (problem), randomly picks n rollouts without replacement,
then applies a selection rule (e.g., shortest) to pick the winning rollout.
Saves the result with 1 rollout per sample.

Example usage:
    # Sample at multiple n values, creates nested folders
    uv run python sample_rollouts.py --input-dir ../../rollouts --n 1 2 4 6 8 10 12 14 16 18 20 --output-dir ../../rollouts_samples

    # Single n value (backwards compatible)
    uv run python sample_rollouts.py --input-dir ../../rollouts --n 4 --output-dir ../../rollouts_sample_n4

    # Different selection rule
    uv run python sample_rollouts.py --input-dir ../../rollouts --n 4 8 --rule longest --output-dir ../../rollouts_samples_longest
"""

import argparse
import json
import random
from pathlib import Path
from typing import Callable


# Selection rules: given a list of rollouts, return the winning one
SELECTION_RULES: dict[str, Callable[[list[dict]], dict]] = {
    "shortest": lambda rollouts: min(rollouts, key=lambda r: len(r.get("response", ""))),
    "longest": lambda rollouts: max(rollouts, key=lambda r: len(r.get("response", ""))),
    "first": lambda rollouts: rollouts[0],  # just take first (useful for debugging)
    "random": lambda rollouts: random.choice(rollouts),  # random from the sampled set
}


def sample_and_select(
    rollouts_for_problem: list[dict],
    n: int,
    rule: str,
    rng: random.Random,
) -> dict:
    """
    Sample n rollouts without replacement, then apply selection rule to pick winner.

    Args:
        rollouts_for_problem: List of all rollouts for a single problem
        n: Number of rollouts to sample
        rule: Selection rule name (e.g., "shortest")
        rng: Random number generator for reproducibility

    Returns:
        The winning rollout dict
    """
    if n > len(rollouts_for_problem):
        raise ValueError(
            f"Cannot sample {n} rollouts from {len(rollouts_for_problem)} available"
        )

    # Sample n rollouts without replacement
    sampled = rng.sample(rollouts_for_problem, n)

    # Apply selection rule
    selector = SELECTION_RULES[rule]
    return selector(sampled)


def process_rollout_file(
    input_path: Path,
    output_path: Path,
    n: int,
    rule: str,
    seed: int,
) -> dict:
    """
    Process a single rollout file: sample and select for each problem.

    Returns metadata about the processing.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Create deterministic RNG for this file
    rng = random.Random(seed)

    problems = data.get("problems", [])
    all_rollouts = data.get("rollouts", [])

    if len(problems) != len(all_rollouts):
        raise ValueError(
            f"Mismatch: {len(problems)} problems but {len(all_rollouts)} rollout lists"
        )

    # Process each problem
    selected_rollouts = []
    for i, (problem, rollouts_for_problem) in enumerate(zip(problems, all_rollouts)):
        winner = sample_and_select(rollouts_for_problem, n, rule, rng)
        selected_rollouts.append([winner])  # Keep as list of 1 for consistent structure

    # Build output data (preserve metadata, update rollouts)
    output_data = {
        "alpha": data.get("alpha"),
        "model": data.get("model"),
        "task_vector_path": data.get("task_vector_path"),
        "num_problems": data.get("num_problems"),
        "rollouts_per_problem": 1,  # Now just 1 per problem
        "temperature": data.get("temperature"),
        "max_new_tokens": data.get("max_new_tokens"),
        "sampling_metadata": {
            "original_rollouts_per_problem": data.get("rollouts_per_problem"),
            "n_sampled": n,
            "selection_rule": rule,
            "seed": seed,
            "source_file": input_path.name,
        },
        "problems": problems,
        "rollouts": selected_rollouts,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "num_problems": len(problems),
    }


def process_n_value(
    input_files: list[Path],
    output_dir: Path,
    n: int,
    rule: str,
    seed: int,
) -> bool:
    """Process all input files for a single n value."""
    print(f"\n  n={n}:")
    for input_path in input_files:
        output_path = output_dir / input_path.name
        try:
            result = process_rollout_file(
                input_path=input_path,
                output_path=output_path,
                n=n,
                rule=rule,
                seed=seed,
            )
            print(f"    {input_path.name} -> {result['num_problems']} problems")
        except Exception as e:
            print(f"    ERROR processing {input_path.name}: {e}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sample rollouts and select winners based on a rule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing rollout JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory for sampled rollout files (will create n{value}/ subfolders)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        required=True,
        help="Number(s) of rollouts to sample per problem. Multiple values create nested folders.",
    )
    parser.add_argument(
        "--rule",
        type=str,
        choices=list(SELECTION_RULES.keys()),
        default="shortest",
        help="Selection rule to apply to sampled rollouts (default: shortest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for input files (default: *.json)",
    )
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob(args.pattern))
    if not input_files:
        print(f"No files matching '{args.pattern}' in {args.input_dir}")
        return 1

    n_values = sorted(args.n)
    use_nested = len(n_values) > 1

    print(f"Processing {len(input_files)} files from {args.input_dir}")
    print(f"Sampling at n values: {n_values}")
    print(f"Selection rule: {args.rule}")
    print(f"Output directory: {args.output_dir}")

    for n in n_values:
        if use_nested:
            # Create nested folder: output_dir/n{value}/
            n_output_dir = args.output_dir / f"n{n}"
        else:
            # Single n value: use output_dir directly
            n_output_dir = args.output_dir

        success = process_n_value(
            input_files=input_files,
            output_dir=n_output_dir,
            n=n,
            rule=args.rule,
            seed=args.seed,
        )
        if not success:
            return 1

    print(f"\nDone. Output saved to {args.output_dir}")
    if use_nested:
        print(f"Created {len(n_values)} subfolders: {', '.join(f'n{n}' for n in n_values)}")
    return 0


if __name__ == "__main__":
    exit(main())
