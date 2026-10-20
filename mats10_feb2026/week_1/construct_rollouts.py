"""
Collect rollouts for best-of-N evaluation.
For each sample (problem), generate a fixed number of rollouts (e.g. 100) with
O = M + α * τ. Alpha is configurable (--alpha). Saves rollouts to a JSON file for
evaluate_best_of_n.py.
"""

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

M_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TASK_VECTOR_PATH = os.path.join(os.path.dirname(__file__), "qwen3_vl_8b_thinking_task_vector.pt")
DEFAULT_ALPHA = 1.5
DEFAULT_ROLLOUTS_PER_PROBLEM = 100


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
    """Apply τ to M: O[k] = M[k] + α * τ[k]. Modifies model in-place."""
    with torch.no_grad():
        for name, delta in task_vector.items():
            if name not in dict(model.named_parameters()):
                continue
            param = model.get_parameter(name)
            param.add_(delta.to(param.device, dtype=param.dtype) * alpha)
    return model


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


def collect_rollouts(
    model,
    processor,
    problems: list[dict],
    rollouts_per_problem: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int = 1,
) -> tuple[list[dict], list[list[dict]]]:
    """
    For each problem, generate rollouts_per_problem responses.
    When batch_size > 1, replicates the same prompt and runs batched generate() on GPU
    for better throughput. Returns (problems_meta, rollouts).
    """
    problems_meta = []
    all_rollouts = []
    batch_size = min(batch_size, rollouts_per_problem)
    if batch_size < 1:
        batch_size = 1

    total_rollouts = len(problems) * rollouts_per_problem
    mininterval = 1.0 if sys.stderr.isatty() else 10.0  # less spam when logging to file
    pbar = tqdm(
        total=total_rollouts,
        unit="rollout",
        desc="Rollouts",
        mininterval=mininterval,
        file=sys.stderr,
    )

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
        input_len = inputs["input_ids"].shape[1]
        device = next(model.parameters()).device

        problems_meta.append({"problem": problem, "answer": gt_answer})
        problem_rollouts = []

        j = 0
        while j < rollouts_per_problem:
            current_batch = min(batch_size, rollouts_per_problem - j)
            # Replicate inputs for this batch (same prompt, current_batch copies)
            batched = {}
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    v_dev = v.to(device)
                    # (1, ...) -> (current_batch, ...) with real copies so generation can diverge
                    batched[k] = v_dev.repeat(current_batch, *([1] * (v_dev.dim() - 1)))
                else:
                    batched[k] = v

            with torch.no_grad():
                outputs = model.generate(
                    **batched,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            decoded = processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for response in decoded:
                pred = extract_answer_from_output(response)
                problem_rollouts.append({"response": response, "pred": pred})

            j += current_batch
            pbar.update(current_batch)
            pbar.set_postfix(problem=f"{i + 1}/{len(problems)}", rollouts=f"{j}/{rollouts_per_problem}")

        all_rollouts.append(problem_rollouts)

    pbar.close()

    return problems_meta, all_rollouts


def main():
    parser = argparse.ArgumentParser(description="Collect rollouts for best-of-N evaluation")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="τ coefficient (0=M, 1=R)")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS_PER_PROBLEM, help="Rollouts per problem (e.g. 100)")
    parser.add_argument("--max-problems", type=int, default=5, help="Number of problems")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--task-vector-path", type=str, default=TASK_VECTOR_PATH, help="Path to τ .pt file")
    parser.add_argument("--problems-file", type=str, default=None, help="Path to arithmetic_problems.jsonl")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset label for output dir (default: stem of problems file, e.g. arithmetic_problems)")
    parser.add_argument("--steering-vector-name", type=str, default=None, help="Steering vector label for output dir (default: task vector filename without .pt)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path (overrides dataset/steering-vector layout)")
    parser.add_argument("--batch-size", type=int, default=1, help="Rollouts per GPU batch (same prompt replicated); higher = faster if memory allows")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    problems_file = args.problems_file or os.path.join(script_dir, "arithmetic_problems.jsonl")
    if args.output is not None:
        output_path = args.output
    else:
        dataset_name = args.dataset_name or os.path.splitext(os.path.basename(problems_file))[0] or "problems"
        steering_name = args.steering_vector_name or os.path.splitext(os.path.basename(args.task_vector_path))[0] or "vector"
        output_path = os.path.join(
            script_dir,
            f"rollouts_{dataset_name}",
            steering_name,
            f"rollouts_alpha{args.alpha}_n{args.rollouts}.json",
        )

    if not os.path.exists(args.task_vector_path):
        print(f"Error: Task vector not found at {args.task_vector_path}")
        return 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model, processor, task_vector = load_model_and_task_vector(args.task_vector_path)
    print(f"\nApplying O = M + {args.alpha} * τ (alpha={args.alpha})...")
    apply_task_vector(model, task_vector, args.alpha)

    problems = load_arithmetic_problems_from_file(problems_file, max_problems=args.max_problems)
    print(f"\nCollecting {args.rollouts} rollouts per problem ({len(problems)} problems)...")

    problems_meta, rollouts = collect_rollouts(
        model, processor, problems,
        rollouts_per_problem=args.rollouts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    data = {
        "alpha": args.alpha,
        "model": M_MODEL,
        "task_vector_path": args.task_vector_path,
        "num_problems": len(problems_meta),
        "rollouts_per_problem": args.rollouts,
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "problems": problems_meta,
        "rollouts": rollouts,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(rollouts)} problems × {args.rollouts} rollouts to {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
