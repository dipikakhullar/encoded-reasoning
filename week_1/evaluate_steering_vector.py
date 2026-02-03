"""
Evaluate steering vector: O = M + α * τ, where τ = R - M (reasoning task vector).
- M = non-reasoning instruct (Qwen3-VL-8B-Instruct)
- R = reasoning/distilled (Qwen3-VL-8B-Thinking)
- α=0 → M (instruct), α=1 → R (thinking)
"""

import argparse
import json
import os
import re
import urllib.request
from collections import defaultdict

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# M (Instruct) and τ (R - M)
M_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TASK_VECTOR_PATH = os.path.join(os.path.dirname(__file__), "qwen3_vl_8b_thinking_task_vector.pt")

ARITHMETIC_PROBLEMS_URL = "https://raw.githubusercontent.com/rgreenblatt/no_cot_math_public/master/arithmetic_problems.jsonl"


def load_arithmetic_problems_from_file(path: str, max_problems: int | None = None) -> list[dict]:
    """Load arithmetic problems from a local JSONL file."""
    print(f"Loading arithmetic problems from {path}...")
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
            if max_problems is not None and len(problems) >= max_problems:
                break
    print(f"Loaded {len(problems)} problems.")
    return problems


def fetch_arithmetic_problems(url: str = ARITHMETIC_PROBLEMS_URL, max_problems: int | None = None) -> list[dict]:
    """Fetch arithmetic problems from the no_cot_math JSONL URL."""
    print(f"Fetching arithmetic problems from {url}...")
    with urllib.request.urlopen(url) as resp:
        content = resp.read().decode()
    problems = []
    for line in content.strip().split("\n"):
        if not line:
            continue
        problems.append(json.loads(line))
        if max_problems is not None and len(problems) >= max_problems:
            break
    print(f"Loaded {len(problems)} problems.")
    return problems


def load_model_and_task_vector(task_vector_path: str):
    """Load M (Instruct) and τ (R - M)."""
    print(f"Loading model M (Instruct): {M_MODEL}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        M_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(M_MODEL)

    print(f"Loading task vector τ (R - M) from {task_vector_path}...")
    task_vector = torch.load(task_vector_path, map_location="cpu", weights_only=True)

    return model, processor, task_vector


def apply_task_vector(model, task_vector: dict, alpha: float):
    """
    Apply τ to M: O[k] = M[k] + α * τ[k].
    Modifies model in-place.
    """
    with torch.no_grad():
        for name, delta in task_vector.items():
            if name not in dict(model.named_parameters()):
                continue
            param = model.get_parameter(name)
            param.add_(delta.to(param.device, dtype=param.dtype) * alpha)
    return model


def compute_layer_norm_stats(task_vector: dict) -> dict:
    """Compute per-layer L2 norm statistics for τ."""
    layer_norms = defaultdict(list)
    for name, tensor in task_vector.items():
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            norm = tensor.float().norm(2).item()
            layer_norms[layer_idx].append(norm)

    stats = {}
    for layer_idx in sorted(layer_norms.keys()):
        norms = layer_norms[layer_idx]
        stats[layer_idx] = {
            "mean_norm": sum(norms) / len(norms),
            "max_norm": max(norms),
            "num_params": len(norms),
        }
    return stats


def run_sanity_checks(task_vector: dict):
    """Print parameter-norm statistics per layer for τ."""
    print("\n--- Sanity checks: task vector τ norm statistics per layer ---")
    stats = compute_layer_norm_stats(task_vector)
    for layer_idx, s in stats.items():
        print(f"  Layer {layer_idx:3d}: mean_norm={s['mean_norm']:.4f}, max_norm={s['max_norm']:.4f}, params={s['num_params']}")
    total_norm = sum(t.float().norm(2).item() ** 2 for t in task_vector.values()) ** 0.5
    print(f"  Total task vector L2 norm: {total_norm:.4f}")
    print()


def extract_answer_from_output(text: str) -> int | float | None:
    """Extract the final answer from \\boxed{...} in model output."""
    match = re.search(r"\\boxed\{([^{}]*)\}", text)
    if not match:
        return None
    content = match.group(1).strip().replace("\u2212", "-")
    if not content:
        return None
    try:
        return float(content)
    except ValueError:
        return None


def evaluate_model(model, processor, problems: list[dict], max_new_tokens: int, max_problems: int | None, output_path: str | None, output_meta: dict | None) -> dict:
    """Evaluate VL model on arithmetic (text-only)."""
    problems = problems[:max_problems] if max_problems else problems
    correct = 0
    results = []
    meta = output_meta or {}

    for i, item in enumerate(problems):
        problem = item["problem"]
        gt_answer = item["answer"]
        prompt = f"{problem}\n\nPut your final answer in \\boxed{{}}."
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        pred = extract_answer_from_output(response)
        is_correct = pred is not None and pred == gt_answer
        if is_correct:
            correct += 1

        result = {
            "problem": problem,
            "response": response,
            "transcript": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "gt": gt_answer,
            "pred": pred,
            "correct": is_correct,
        }
        results.append(result)

        if output_path:
            accuracy_so_far = correct / len(results) if results else 0.0
            output_data = {**meta, "accuracy": accuracy_so_far, "correct": correct, "total": len(results), "results": results}
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(problems)} problems...")

    accuracy = correct / len(problems) if problems else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(problems), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering vector: O = M + α * τ")
    parser.add_argument("--alpha", type=float, default=0.5, help="τ coefficient (0=M/instruct, 1=R/thinking)")
    parser.add_argument("--max-problems", type=int, default=100, help="Max number of arithmetic problems")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max new tokens per generation")
    parser.add_argument("--sanity-only", action="store_true", help="Only run sanity checks, skip evaluation")
    parser.add_argument("--baseline", action="store_true", help="Evaluate M only (alpha=0)")
    parser.add_argument("--task-vector-path", type=str, default=TASK_VECTOR_PATH, help="Path to τ .pt file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--problems-file", type=str, default=None, help="Path to local arithmetic_problems.jsonl (default: fetch from URL)")
    args = parser.parse_args()

    if not os.path.exists(args.task_vector_path):
        print(f"Error: Task vector not found at {args.task_vector_path}")
        print("Run construct_task_vector.py first.")
        return 1

    model, processor, task_vector = load_model_and_task_vector(args.task_vector_path)
    run_sanity_checks(task_vector)

    if args.sanity_only:
        print("Sanity-only mode: skipping evaluation.")
        return 0

    if args.baseline:
        args.alpha = 0.0
        print("\nBaseline mode: evaluating M only (alpha=0).")
    else:
        print(f"\nApplying O = M + {args.alpha} * τ...")
        apply_task_vector(model, task_vector, args.alpha)

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), f"eval_results_alpha_{args.alpha}.json")

    if args.problems_file:
        problems = load_arithmetic_problems_from_file(args.problems_file, max_problems=args.max_problems)
    else:
        problems = fetch_arithmetic_problems(max_problems=args.max_problems)
    print(f"\nEvaluating on {len(problems)} arithmetic problems (alpha={args.alpha})...")
    output_meta = {"alpha": args.alpha, "model": M_MODEL, "task_vector_path": args.task_vector_path} if args.output else None

    eval_result = evaluate_model(model, processor, problems, args.max_tokens, args.max_problems, args.output, output_meta)

    print(f"\n--- Results (alpha={args.alpha}) ---")
    print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

    if args.output:
        print(f"\nSaved to {args.output} (updated after each sample)")

    return 0


if __name__ == "__main__":
    exit(main())
