"""CoT Prefill Experiment: Measure uplift/downlift from manipulated reasoning traces.

Tests whether open-weight reasoning models follow prefilled <think> content or
override it with their own reasoning.

Usage:
    uv run python testing_on_server/cot_prefill_experiment.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --problems-file week_1/arithmetic_problems.jsonl \
        --max-problems 100 \
        --output-dir outputs/run_20260225_cot_prefill/
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as `uv run python src/cot_prefill_experiment.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_generation import (
    LOREM_IPSUM,
    generate_correct_cot,
    generate_random_tokens,
    generate_wrong_cot,
)

ALL_CONDITIONS = [
    "no_cot",
    "normal_cot",
    "correct_cot",
    "wrong_cot",
    "nonsense_random",
    "nonsense_lorem",
    "truncated_cot",
]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def load_problems(path: str, max_problems: int) -> list[dict]:
    """Load arithmetic problems from JSONL."""
    problems = []
    with open(path) as f:
        for line in f:
            if len(problems) >= max_problems:
                break
            problems.append(json.loads(line))
    return problems


def extract_expression(problem_text: str) -> str:
    """Extract the arithmetic expression from the problem text."""
    marker = "Evaluate this Python expression. "
    if marker in problem_text:
        return problem_text.split(marker, 1)[1].strip()
    return problem_text.strip()


def load_model_and_tokenizer(model_id: str, quantize: str):
    """Load model with optional quantization."""
    print(f"Loading model: {model_id} (quantize={quantize})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "awq":
        # AWQ models load directly — the quantization is baked into the weights
        # Just need to make sure we don't set conflicting dtype
        kwargs.pop("torch_dtype", None)
    elif quantize != "none":
        raise ValueError(f"Unknown quantize mode: {quantize}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    print(f"Model loaded. Device map: {getattr(model, 'hf_device_map', 'single device')}")
    return model, tokenizer


def build_prompt_text(tokenizer, problem_text: str) -> str:
    """Build the chat-formatted prompt string up to the assistant turn."""
    user_msg = f"{problem_text}\n\nPut your final answer in \\boxed{{}}."
    messages = [{"role": "user", "content": user_msg}]

    # Try Qwen3-style with enable_thinking
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    return prompt


def build_input(
    tokenizer,
    problem_text: str,
    condition: str,
    correct_cot: str | None = None,
    wrong_cot: str | None = None,
    random_tokens: str | None = None,
) -> tuple[torch.Tensor, str]:
    """Build tokenized input with appropriate prefill for each condition.

    Returns:
        (input_ids tensor [1, seq_len], prefill_text string)
    """
    prompt = build_prompt_text(tokenizer, problem_text)

    if condition == "normal_cot":
        # Model generates everything including <think>
        # Check if template already includes <think> — if so, return as-is
        if THINK_OPEN in prompt:
            ids = tokenizer.encode(prompt, return_tensors="pt")
            return ids, ""
        # Otherwise, add <think>\n to start the thinking
        prefill = f"{THINK_OPEN}\n"
        ids = tokenizer.encode(prompt + prefill, return_tensors="pt")
        return ids, prefill

    elif condition == "no_cot":
        prefill = f"{THINK_OPEN}\n{THINK_CLOSE}\n"

    elif condition == "correct_cot":
        prefill = f"{THINK_OPEN}\n{correct_cot}\n{THINK_CLOSE}\n"

    elif condition == "wrong_cot":
        prefill = f"{THINK_OPEN}\n{wrong_cot}\n{THINK_CLOSE}\n"

    elif condition == "nonsense_random":
        prefill = f"{THINK_OPEN}\n{random_tokens}\n{THINK_CLOSE}\n"

    elif condition == "nonsense_lorem":
        prefill = f"{THINK_OPEN}\n{LOREM_IPSUM}\n{THINK_CLOSE}\n"

    elif condition == "truncated_cot":
        half = correct_cot[: len(correct_cot) // 2]
        prefill = f"{THINK_OPEN}\n{half}"
        # NO closing tag — model continues from truncation point

    else:
        raise ValueError(f"Unknown condition: {condition}")

    # If the template already added <think>, we need to handle that
    # Check if the prompt already ends with <think>\n
    if prompt.rstrip().endswith(THINK_OPEN):
        # Template already has <think>, strip our duplicate
        prefill_no_open = prefill[len(THINK_OPEN):]
        ids = tokenizer.encode(prompt + prefill_no_open, return_tensors="pt")
        return ids, prefill_no_open

    ids = tokenizer.encode(prompt + prefill, return_tensors="pt")
    return ids, prefill


def extract_answer(text: str) -> int | None:
    """Extract integer answer from \\boxed{...} in generated text."""
    # Find the last \boxed{...}
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    ans_str = matches[-1].strip()
    # Handle negative numbers, commas
    ans_str = ans_str.replace(",", "").strip()
    try:
        return int(float(ans_str))
    except (ValueError, OverflowError):
        return None


def run_generation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """Run generation and return (generated_text, n_tokens).

    Returns only the newly generated text (not the input).
    """
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        if temperature == 0.0:
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    new_tokens = output[0, input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    n_tokens = len(new_tokens)

    return generated_text, n_tokens


def run_experiment(args):
    """Main experiment loop."""
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # Save metadata
    metadata = {
        "model": args.model,
        "problems_file": args.problems_file,
        "max_problems": args.max_problems,
        "conditions": args.conditions,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "quantize": args.quantize,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Load problems
    problems = load_problems(args.problems_file, args.max_problems)
    print(f"Loaded {len(problems)} problems")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.quantize)

    # Pre-generate CoT traces and random tokens (deterministic)
    rng = random.Random(42)
    cot_cache = {}
    for p in problems:
        pid = p["problem_number"]
        expr = extract_expression(p["problem"])
        answer = p["answer"]
        try:
            correct = generate_correct_cot(expr, answer)
            wrong, wrong_ans = generate_wrong_cot(expr, answer, rng=rng)
            cot_cache[pid] = {
                "correct_cot": correct,
                "wrong_cot": wrong,
                "wrong_answer": wrong_ans,
                "random_tokens": generate_random_tokens(200, rng=rng),
            }
        except Exception as e:
            print(f"Warning: Could not generate CoT for problem {pid}: {e}")
            cot_cache[pid] = None

    # Open results file
    results_path = output_dir / "results.jsonl"
    results_file = open(results_path, "w")

    # Counters for summary
    condition_correct = {c: 0 for c in args.conditions}
    condition_total = {c: 0 for c in args.conditions}
    condition_parsed = {c: 0 for c in args.conditions}

    total_runs = len(problems) * len(args.conditions)
    pbar = tqdm(total=total_runs, desc="Running experiment")

    for p in problems:
        pid = p["problem_number"]
        expr = extract_expression(p["problem"])
        answer = p["answer"]

        if cot_cache[pid] is None:
            pbar.update(len(args.conditions))
            continue

        cache = cot_cache[pid]

        for condition in args.conditions:
            try:
                input_ids, prefill = build_input(
                    tokenizer,
                    p["problem"],
                    condition,
                    correct_cot=cache["correct_cot"],
                    wrong_cot=cache["wrong_cot"],
                    random_tokens=cache["random_tokens"],
                )

                generated_text, n_tokens = run_generation(
                    model, tokenizer, input_ids, args.max_tokens, args.temperature
                )

                # For analysis, combine prefill + generated for full picture
                full_output = prefill + generated_text
                extracted = extract_answer(full_output)
                is_correct = extracted == answer

                result = {
                    "problem_id": pid,
                    "expression": expr,
                    "correct_answer": answer,
                    "condition": condition,
                    "prefill_text": prefill[:500],  # Truncate for storage
                    "generated_text": generated_text[:2000],  # Truncate for storage
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "generation_tokens": n_tokens,
                }

                if condition == "wrong_cot":
                    result["wrong_cot_answer"] = cache["wrong_answer"]
                    result["followed_wrong_cot"] = extracted == cache["wrong_answer"]

                results_file.write(json.dumps(result) + "\n")
                results_file.flush()

                condition_total[condition] += 1
                if extracted is not None:
                    condition_parsed[condition] += 1
                if is_correct:
                    condition_correct[condition] += 1

            except Exception as e:
                print(f"\nError on problem {pid}, condition {condition}: {e}")
                result = {
                    "problem_id": pid,
                    "expression": expr,
                    "correct_answer": answer,
                    "condition": condition,
                    "error": str(e),
                }
                results_file.write(json.dumps(result) + "\n")
                results_file.flush()
                condition_total[condition] += 1

            pbar.update(1)

    pbar.close()
    results_file.close()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    summary = {"model": args.model, "conditions": {}}
    for c in args.conditions:
        total = condition_total[c]
        correct = condition_correct[c]
        parsed = condition_parsed[c]
        acc = correct / total if total > 0 else 0
        parse_rate = parsed / total if total > 0 else 0
        print(f"  {c:20s}: {correct:3d}/{total:3d} = {acc:.1%} accuracy  (parsed: {parse_rate:.0%})")
        summary["conditions"][c] = {
            "total": total,
            "correct": correct,
            "parsed": parsed,
            "accuracy": acc,
            "parse_rate": parse_rate,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="CoT Prefill Experiment")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--problems-file", required=True, help="Path to arithmetic_problems.jsonl")
    parser.add_argument("--max-problems", type=int, default=100, help="Max problems to use")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Conditions to run",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit", "awq"],
        default="none",
        help="Quantization strategy",
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
