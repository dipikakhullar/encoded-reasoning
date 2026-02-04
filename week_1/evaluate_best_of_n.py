"""
Evaluate best-of-N using pre-collected rollouts (from construct_rollouts.py).
Loads rollouts, samples n per problem, and applies multiple eligibility/selection
metrics to pick the chosen response. Saves results and transcripts under arithmetic_problems_bon/.
"""

import argparse
import json
import os
import random
from collections import Counter

# ---------------------------------------------------------------------------
# Eligibility / selection metrics (pick one response from n rollouts)
# ---------------------------------------------------------------------------


def metric_first_correct(rollouts: list[dict], gt_answer: int | float]) -> int:
    """Pick the first response that is correct; else pick index 0."""
    for i, r in enumerate(rollouts):
        pred = r.get("pred")
        if pred is not None and pred == gt_answer:
            return i
    return 0


def metric_random_correct(rollouts: list[dict], gt_answer: int | float) -> int:
    """Pick uniformly among correct responses; if none correct, pick random among all."""
    correct_indices = [i for i, r in enumerate(rollouts) if r.get("pred") is not None and r["pred"] == gt_answer]
    if correct_indices:
        return random.choice(correct_indices)
    return random.randrange(len(rollouts))


def metric_majority_vote(rollouts: list[dict], gt_answer: int | float) -> int:
    """Pick the response whose extracted answer is the most common among the n (ties: first)."""
    preds = [r.get("pred") for r in rollouts]
    value_counts: Counter = Counter(p for p in preds if p is not None)
    if not value_counts:
        return 0
    best_value = value_counts.most_common(1)[0][0]
    for i, p in enumerate(preds):
        if p == best_value:
            return i
    return 0


def metric_last(rollouts: list[dict], gt_answer: int | float) -> int:
    """Always pick the last (n-th) response."""
    return len(rollouts) - 1


def metric_random(rollouts: list[dict], gt_answer: int | float) -> int:
    """Pick a random response among the n."""
    return random.randrange(len(rollouts))


# Registry of metric name -> (function, description)
ELIGIBILITY_METRICS = {
    "first_correct": (metric_first_correct, "First response that is correct; else first"),
    "random_correct": (metric_random_correct, "Uniform among correct; else random among all"),
    "majority_vote": (metric_majority_vote, "Most common extracted answer (ties: first)"),
    "last": (metric_last, "Always the last (n-th) response"),
    "random": (metric_random, "Uniform random among n"),
}


def load_rollouts(path: str) -> dict:
    """Load rollouts JSON produced by construct_rollouts.py."""
    with open(path) as f:
        return json.load(f)


def evaluate_best_of_n_from_rollouts(
    rollouts_data: dict,
    n_values: list[int],
    metrics: list[str],
    num_trials: int,
    output_dir: str | None,
    seed: int | None,
) -> dict:
    """
    For each n in n_values, sample n rollouts per problem (without replacement)
    and evaluate each eligibility metric. Optionally run num_trials trials and
    average (for stochastic metrics).
    """
    if seed is not None:
        random.seed(seed)

    problems = rollouts_data["problems"]
    all_rollouts = rollouts_data["rollouts"]
    rollouts_per_problem = rollouts_data["rollouts_per_problem"]

    if any(len(r) < max(n_values) for r in all_rollouts):
        raise ValueError(
            f"Rollouts per problem ({rollouts_per_problem}) must be >= max(n)={max(n_values)}"
        )

    results_by_n = {}

    for n in n_values:
        metric_accuracies = {m: [] for m in metrics}
        # For each problem, we'll sample n indices and apply each metric
        problem_results = []

        for prob_idx, (prob_meta, rollouts) in enumerate(zip(problems, all_rollouts)):
            gt = prob_meta["answer"]
            problem_text = prob_meta["problem"]

            trial_correct = {m: [] for m in metrics}

            # One representative sample for stored transcripts
            indices_rep = random.sample(range(len(rollouts)), n)
            sampled_rep = [rollouts[i] for i in indices_rep]
            prompt = f"{problem_text}\n\nPut your final answer in \\boxed{{}}."
            transcripts = [
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": r["response"]}]
                for r in sampled_rep
            ]
            chosen_per_metric = {}
            for m in metrics:
                fn, _ = ELIGIBILITY_METRICS[m]
                idx = fn(sampled_rep, gt)
                chosen_per_metric[m] = {"response": sampled_rep[idx]["response"], "pred": sampled_rep[idx].get("pred")}

            for trial in range(num_trials):
                indices = random.sample(range(len(rollouts)), n)
                sampled = [rollouts[i] for i in indices]

                for metric_name in metrics:
                    fn, _ = ELIGIBILITY_METRICS[metric_name]
                    chosen_idx = fn(sampled, gt)
                    chosen = sampled[chosen_idx]
                    pred = chosen.get("pred")
                    correct = pred is not None and pred == gt
                    trial_correct[metric_name].append(1 if correct else 0)

            for m in metrics:
                metric_accuracies[m].append(sum(trial_correct[m]) / num_trials)

            problem_results.append({
                "problem": problem_text,
                "gt": gt,
                "n": n,
                "rollout_indices_sample": indices_rep,
                "transcripts": transcripts,
                "chosen_per_metric": chosen_per_metric,
                "accuracy_per_metric": {m: metric_accuracies[m][-1] for m in metrics},
            })

        acc_per_metric = {
            m: sum(metric_accuracies[m]) / len(problems) if problems else 0.0
            for m in metrics
        }
        results_by_n[n] = {
            "accuracy_per_metric": acc_per_metric,
            "num_problems": len(problems),
            "num_trials": num_trials,
            "results": problem_results,
        }

        if output_dir:
            out_path = os.path.join(output_dir, f"eval_results_bon_n{n}.json")
            payload = {
                "rollouts_path": rollouts_data.get("_path", ""),
                "alpha": rollouts_data.get("alpha"),
                "n": n,
                "metrics": metrics,
                "num_trials": num_trials,
                "seed": seed,
                **results_by_n[n],
            }
            os.makedirs(output_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  n={n}: saved to {out_path}")

    return results_by_n


def main():
    parser = argparse.ArgumentParser(description="Evaluate best-of-N from rollouts with eligibility metrics")
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts JSON (from construct_rollouts.py)")
    parser.add_argument("--n", type=int, nargs="+", default=[2, 4, 8, 16], help="Best-of-N values")
    parser.add_argument("--metrics", type=str, nargs="+", default=list(ELIGIBILITY_METRICS), help="Eligibility metrics to use")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per problem (for stochastic metrics)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory (default: arithmetic_problems_bon)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "arithmetic_problems_bon")

    if not os.path.exists(args.rollouts):
        print(f"Error: Rollouts file not found: {args.rollouts}")
        print("Run construct_rollouts.py first.")
        return 1

    rollouts_data = load_rollouts(args.rollouts)
    rollouts_data["_path"] = args.rollouts

    print(f"Loaded {rollouts_data['num_problems']} problems × {rollouts_data['rollouts_per_problem']} rollouts")
    print(f"Best-of-N values: {args.n}")
    print(f"Metrics: {args.metrics}")
    print(f"Trials per problem: {args.trials}")
    print()

    results_by_n = evaluate_best_of_n_from_rollouts(
        rollouts_data,
        n_values=args.n,
        metrics=args.metrics,
        num_trials=args.trials,
        output_dir=output_dir,
        seed=args.seed,
    )

    print("\n--- Best-of-N accuracy by metric ---")
    for n in args.n:
        acc = results_by_n[n]["accuracy_per_metric"]
        print(f"  n={n}: " + "  ".join(f"{m}={acc[m]:.2%}" for m in args.metrics))
    print(f"\nResults saved under {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
