"""
Best-of-N evaluation: generate N responses per problem, count correct if any is correct.
Uses alpha=1.5 model (O = M + 1.5 * τ) from eval_results_alpha_1.5.json setup.
Saves all transcripts under arithmetic_problems_bon/.
"""

import argparse
import json
import os
import re

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

M_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TASK_VECTOR_PATH = os.path.join(os.path.dirname(__file__), "qwen3_vl_8b_thinking_task_vector.pt")
ALPHA = 1.5


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


def evaluate_best_of_n(
    model,
    processor,
    problems: list[dict],
    n: int,
    max_new_tokens: int,
    temperature: float,
    output_path: str | None,
    output_meta: dict | None,
) -> dict:
    """Best-of-N: generate N responses per problem, correct if any is correct. Save all transcripts."""
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
        input_len = inputs["input_ids"].shape[1]

        responses = []
        transcripts = []
        any_correct = False

        for j in range(n):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            response = processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pred = extract_answer_from_output(response)
            responses.append({"response": response, "pred": pred})
            transcripts.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ])
            if pred is not None and pred == gt_answer:
                any_correct = True

        if any_correct:
            correct += 1

        result = {
            "problem": problem,
            "gt": gt_answer,
            "correct": any_correct,
            "n": n,
            "responses": responses,
            "transcripts": transcripts,
        }
        results.append(result)

        if output_path:
            accuracy_so_far = correct / len(results) if results else 0.0
            output_data = {**meta, "n": n, "accuracy": accuracy_so_far, "correct": correct, "total": len(results), "results": results}
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

        print(f"  Problem {i + 1}/{len(problems)}: {'correct' if any_correct else 'incorrect'} (n={n})")

    accuracy = correct / len(problems) if problems else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(problems), "results": results}


def main():
    parser = argparse.ArgumentParser(description="Best-of-N evaluation with alpha=1.5 model")
    parser.add_argument("--n", type=int, required=True, help="Number of samples per problem (best-of-n)")
    parser.add_argument("--max-problems", type=int, default=5, help="Number of problems to evaluate")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--task-vector-path", type=str, default=TASK_VECTOR_PATH, help="Path to τ .pt file")
    parser.add_argument("--problems-file", type=str, default=None, help="Path to arithmetic_problems.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: arithmetic_problems_bon)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    problems_file = args.problems_file or os.path.join(script_dir, "arithmetic_problems.jsonl")
    output_dir = args.output_dir or os.path.join(script_dir, "arithmetic_problems_bon")

    if not os.path.exists(args.task_vector_path):
        print(f"Error: Task vector not found at {args.task_vector_path}")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"eval_results_bon_n{args.n}.json")

    model, processor, task_vector = load_model_and_task_vector(args.task_vector_path)
    print(f"\nApplying O = M + {ALPHA} * τ (alpha=1.5)...")
    apply_task_vector(model, task_vector, ALPHA)

    problems = load_arithmetic_problems_from_file(problems_file, max_problems=args.max_problems)
    print(f"\nBest-of-{args.n} evaluation on {len(problems)} problems (temperature={args.temperature})...")

    output_meta = {
        "alpha": ALPHA,
        "n": args.n,
        "temperature": args.temperature,
        "model": M_MODEL,
        "task_vector_path": args.task_vector_path,
    }

    eval_result = evaluate_best_of_n(
        model, processor, problems,
        n=args.n,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        output_path=output_path,
        output_meta=output_meta,
    )

    print(f"\n--- Best-of-{args.n} Results ---")
    print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")
    print(f"\nSaved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
