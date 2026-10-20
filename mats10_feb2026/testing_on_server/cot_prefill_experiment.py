"""CoT Prefill Experiment: Measure uplift/downlift from manipulated reasoning traces.

Tests whether open-weight reasoning models follow prefilled <think> content or
override it with their own reasoning.

Usage:
    # Run a single model by short name:
    uv run python testing_on_server/cot_prefill_experiment.py \
        --model deepseek-14b \
        --problems-file week_1/arithmetic_problems.jsonl

    # Run all models:
    uv run python testing_on_server/cot_prefill_experiment.py \
        --model all \
        --problems-file week_1/arithmetic_problems.jsonl

    # List available model short names:
    uv run python testing_on_server/cot_prefill_experiment.py --list-models
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

# Allow running from project root
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

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry: short_name -> {hf_id, quantize, description}
# "quantize" is auto-applied unless overridden on the CLI.
MODEL_REGISTRY = {
    "deepseek-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantize": "none",
        "description": "DeepSeek-R1-Distill-Qwen-14B (fp16, ~28 GB)",
    },
    "deepseek-32b": {
        "hf_id": "casperhansen/deepseek-r1-distill-qwen-32b-awq",
        "quantize": "awq",
        "description": "DeepSeek-R1-Distill-Qwen-32B (AWQ 4-bit, ~20 GB)",
    },
    "qwq-32b": {
        "hf_id": "Qwen/QwQ-32B-AWQ",
        "quantize": "awq",
        "description": "QwQ-32B (AWQ 4-bit, ~20 GB)",
    },
    "qwen3-14b": {
        "hf_id": "Qwen/Qwen3-14B",
        "quantize": "none",
        "description": "Qwen3-14B (fp16, ~28 GB)",
    },
    "qwen3-32b": {
        "hf_id": "Qwen/Qwen3-32B-AWQ",
        "quantize": "awq",
        "description": "Qwen3-32B (AWQ 4-bit, ~20 GB)",
    },
}

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


def resolve_model(model_arg: str, quantize_override: str | None) -> list[dict]:
    """Resolve a --model argument to a list of {hf_id, quantize, short_name} dicts.

    Accepts:
      - A registry short name (e.g. "deepseek-14b")
      - "all" to run every model in the registry
      - A raw HuggingFace model ID (passed through as-is)
    """
    if model_arg == "all":
        entries = []
        for short_name, info in MODEL_REGISTRY.items():
            entries.append({
                "hf_id": info["hf_id"],
                "quantize": quantize_override or info["quantize"],
                "short_name": short_name,
            })
        return entries

    if model_arg in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model_arg]
        return [{
            "hf_id": info["hf_id"],
            "quantize": quantize_override or info["quantize"],
            "short_name": model_arg,
        }]

    # Treat as raw HF model ID
    # Try to auto-detect AWQ from the model name
    quantize = quantize_override or ("awq" if "awq" in model_arg.lower() else "none")
    short_name = model_arg.split("/")[-1].lower()
    return [{
        "hf_id": model_arg,
        "quantize": quantize,
        "short_name": short_name,
    }]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, quantize: str):
    """Load model with optional quantization."""
    print(f"Loading model: {model_id} (quantize={quantize})")
    t0 = time.time()

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
        # AWQ models have quantization baked in — don't set conflicting dtype
        kwargs.pop("torch_dtype", None)
    elif quantize != "none":
        raise ValueError(f"Unknown quantize mode: {quantize}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s. Device map: {getattr(model, 'hf_device_map', 'single device')}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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
    template_has_think = prompt.rstrip().endswith(THINK_OPEN)

    if condition == "normal_cot":
        # Model generates everything including <think>
        if template_has_think:
            ids = tokenizer.encode(prompt, return_tensors="pt")
            return ids, ""
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
        prefill = f"{THINK_OPEN}\n{half}"  # NO closing tag
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # If the template already appended <think>, strip our duplicate open tag
    if template_has_think:
        prefill_no_open = prefill[len(THINK_OPEN):]
        ids = tokenizer.encode(prompt + prefill_no_open, return_tensors="pt")
        return ids, prefill_no_open

    ids = tokenizer.encode(prompt + prefill, return_tensors="pt")
    return ids, prefill


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> int | None:
    """Extract integer answer from \\boxed{...} in generated text."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    ans_str = matches[-1].strip().replace(",", "").strip()
    try:
        return int(float(ans_str))
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """Run generation and return (generated_text, n_new_tokens)."""
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(input_ids, **gen_kwargs)

    new_tokens = output[0, input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return generated_text, len(new_tokens)


# ---------------------------------------------------------------------------
# Experiment runner (single model)
# ---------------------------------------------------------------------------

def run_single_model(
    model_id: str,
    short_name: str,
    quantize: str,
    problems: list[dict],
    conditions: list[str],
    max_tokens: int,
    temperature: float,
    output_dir: Path,
    cot_cache: dict,
):
    """Run the experiment for one model. Returns the summary dict."""
    model_dir = output_dir / short_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "plots").mkdir(exist_ok=True)

    # Save metadata
    metadata = {
        "model": model_id,
        "short_name": short_name,
        "quantize": quantize,
        "n_problems": len(problems),
        "conditions": conditions,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_id, quantize)

    # Open results file
    results_path = model_dir / "results.jsonl"
    results_file = open(results_path, "w")

    # Counters
    condition_correct = {c: 0 for c in conditions}
    condition_total = {c: 0 for c in conditions}
    condition_parsed = {c: 0 for c in conditions}

    total_runs = len(problems) * len(conditions)
    pbar = tqdm(total=total_runs, desc=f"{short_name}")

    for p in problems:
        pid = p["problem_number"]
        expr = extract_expression(p["problem"])
        answer = p["answer"]

        if cot_cache.get(pid) is None:
            pbar.update(len(conditions))
            continue

        cache = cot_cache[pid]

        for condition in conditions:
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
                    model, tokenizer, input_ids, max_tokens, temperature
                )

                full_output = prefill + generated_text
                extracted = extract_answer(full_output)
                is_correct = extracted == answer

                result = {
                    "problem_id": pid,
                    "expression": expr,
                    "correct_answer": answer,
                    "condition": condition,
                    "prefill_text": prefill[:500],
                    "generated_text": generated_text[:2000],
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

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {short_name} ({model_id})")
    print(f"{'=' * 60}")
    summary = {"model": model_id, "short_name": short_name, "conditions": {}}
    for c in conditions:
        total = condition_total[c]
        correct = condition_correct[c]
        parsed = condition_parsed[c]
        acc = correct / total if total > 0 else 0
        parse_rate = parsed / total if total > 0 else 0
        print(f"  {c:20s}: {correct:3d}/{total:3d} = {acc:.1%}  (parsed: {parse_rate:.0%})")
        summary["conditions"][c] = {
            "total": total,
            "correct": correct,
            "parsed": parsed,
            "accuracy": acc,
            "parse_rate": parse_rate,
        }

    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {model_dir}/")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    """Top-level: resolve models, prepare data, run each model."""
    models = resolve_model(args.model, args.quantize)
    print(f"Will run {len(models)} model(s): {[m['short_name'] for m in models]}")

    # Timestamp for the whole run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/run_{timestamp}_cot_prefill")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load problems
    problems = load_problems(args.problems_file, args.max_problems)
    print(f"Loaded {len(problems)} problems from {args.problems_file}")

    # Pre-generate CoT traces (deterministic, shared across models)
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

    # Save run-level metadata
    run_meta = {
        "models": [m["short_name"] for m in models],
        "n_problems": len(problems),
        "conditions": args.conditions,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": timestamp,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Run each model
    all_summaries = {}
    for i, m in enumerate(models):
        print(f"\n{'#' * 60}")
        print(f"# Model {i+1}/{len(models)}: {m['short_name']}  ({m['hf_id']})")
        print(f"# Quantize: {m['quantize']}")
        print(f"{'#' * 60}\n")

        summary = run_single_model(
            model_id=m["hf_id"],
            short_name=m["short_name"],
            quantize=m["quantize"],
            problems=problems,
            conditions=args.conditions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_dir=output_dir,
            cot_cache=cot_cache,
        )
        all_summaries[m["short_name"]] = summary

    # Save combined summary
    with open(output_dir / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ALL DONE. Results in: {output_dir}/")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="CoT Prefill Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model short names: " + ", ".join(MODEL_REGISTRY.keys()) + ", all",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help=(
            "Model to run. Use a registry short name (e.g. 'deepseek-14b'), "
            "'all' for every model, or a raw HuggingFace model ID."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument("--problems-file", default="week_1/arithmetic_problems.jsonl")
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Conditions to run (default: all 7)",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None, help="Output dir (default: auto-timestamped)")
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit", "awq"],
        default=None,
        help="Override quantization (default: use registry setting per model)",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:20s}  {info['description']}")
            print(f"  {'':20s}  HF: {info['hf_id']}  (quantize: {info['quantize']})")
        sys.exit(0)

    if args.model is None:
        parser.error("--model is required (or use --list-models)")

    run_experiment(args)


if __name__ == "__main__":
    main()
