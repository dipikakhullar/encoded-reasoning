"""
Collect rollouts with activation noise at a fixed decoder layer (25th, 50th, or 75th percentile).
Same structure as construct_rollouts but instead of steering vector M + α*τ we use base model M
and add α * Gaussian noise to the activations at the chosen layer. Alpha controls noise strength.
Outputs: rollouts_<dataset_name>/noise_<size>_layer<percentile>/rollouts_alpha<alpha>_n<n>.json
Requires noise configs from noise_to_activations.py (noise_configs/qwen3_vl_<size>_noise_layers.json).
"""

import argparse
import json
import os
import re

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

SIZE_TO_MODEL = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}

DEFAULT_ALPHA = 0.5
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


def load_noise_config(script_dir: str, size: str) -> dict:
    """Load noise layer config for this model size (from noise_to_activations.py output)."""
    path = os.path.join(script_dir, "noise_configs", f"qwen3_vl_{size}_noise_layers.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Noise config not found: {path}. Run noise_to_activations.py first."
        )
    with open(path) as f:
        return json.load(f)


def get_decoder_layers_module(model):
    """Return the decoder layers list (model.model.layers for Qwen3-VL)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not find decoder layers on model (expected model.model.layers)")


def register_noise_hook(model, layer_idx: int, alpha: float, seed: int | None = None):
    """
    Add a forward hook to the decoder layer at layer_idx that adds alpha * N(0,1) noise
    to the layer output (first element of the tuple). Returns the handle so caller can remove it.
    """
    layers = get_decoder_layers_module(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx={layer_idx} out of range [0, {len(layers)})")
    layer = layers[layer_idx]

    def hook(module, args, output):
        if not isinstance(output, tuple):
            return output + alpha * torch.randn_like(
                output, device=output.device, dtype=output.dtype
            )
        hidden = output[0]
        if seed is not None:
            gen = torch.Generator(device=hidden.device).manual_seed(seed)
            noise = alpha * torch.randn_like(hidden, generator=gen, dtype=hidden.dtype)
        else:
            noise = alpha * torch.randn_like(hidden, device=hidden.device, dtype=hidden.dtype)
        return (hidden + noise,) + output[1:]

    handle = layer.register_forward_hook(hook)
    return handle


def load_model_and_processor(model_id: str):
    """Load base Instruct model and processor (no steering vector)."""
    print(f"Loading model: {model_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


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
    """Generate rollouts (same logic as construct_rollouts)."""
    problems_meta = []
    all_rollouts = []
    batch_size = min(max(1, batch_size), rollouts_per_problem)
    if batch_size < 1:
        batch_size = 1
    device = next(model.parameters()).device

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

        problems_meta.append({"problem": problem, "answer": gt_answer})
        problem_rollouts = []

        j = 0
        while j < rollouts_per_problem:
            current_batch = min(batch_size, rollouts_per_problem - j)
            batched = {}
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    v_dev = v.to(device)
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
            if j % 20 == 0 or j == rollouts_per_problem:
                print(f"  Problem {i + 1}/{len(problems)}: {j}/{rollouts_per_problem} rollouts")

        all_rollouts.append(problem_rollouts)
        print(f"  Problem {i + 1}/{len(problems)}: done ({rollouts_per_problem} rollouts)")

    return problems_meta, all_rollouts


def main():
    parser = argparse.ArgumentParser(
        description="Collect rollouts with activation noise at a decoder layer (alpha = noise scale)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["2b", "4b", "8b"],
        default="8b",
        help="Model size (determines base model and noise config)",
    )
    parser.add_argument(
        "--layer-percentile",
        type=int,
        choices=[25, 50, 75],
        default=50,
        help="Which decoder layer to noise (25th, 50th, or 75th percentile)",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Noise scale (std of added Gaussian)")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS_PER_PROBLEM, help="Rollouts per problem")
    parser.add_argument("--max-problems", type=int, default=5, help="Number of problems")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--problems-file", type=str, default=None, help="Path to arithmetic_problems.jsonl")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset label for output dir")
    parser.add_argument("--output", "-o", type=str, default=None, help="Override output JSON path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation (same prompt replicated)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for noise (default: different each time)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    problems_file = args.problems_file or os.path.join(script_dir, "arithmetic_problems.jsonl")

    noise_config = load_noise_config(script_dir, args.model_size)
    layer_key = f"layer_{args.layer_percentile}"
    layer_idx = noise_config.get(layer_key)
    if layer_idx is None:
        raise KeyError(f"No {layer_key} in noise config. Keys: {list(noise_config.keys())}")

    if args.output is not None:
        output_path = args.output
    else:
        dataset_name = args.dataset_name or os.path.splitext(os.path.basename(problems_file))[0] or "problems"
        steering_name = f"noise_{args.model_size}_layer{args.layer_percentile}"
        output_path = os.path.join(
            script_dir,
            f"rollouts_{dataset_name}",
            steering_name,
            f"rollouts_alpha{args.alpha}_n{args.rollouts}.json",
        )

    if not os.path.exists(problems_file):
        print(f"Error: Problems file not found: {problems_file}")
        return 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model_id = SIZE_TO_MODEL[args.model_size]
    model, processor = load_model_and_processor(model_id)
    handle = register_noise_hook(model, layer_idx, args.alpha, seed=args.seed)
    try:
        print(f"\nNoise: alpha={args.alpha} at layer {layer_idx} ({args.layer_percentile}th percentile)")
        problems = load_arithmetic_problems_from_file(problems_file, max_problems=args.max_problems)
        print(f"Collecting {args.rollouts} rollouts per problem ({len(problems)} problems)...")
        problems_meta, rollouts = collect_rollouts(
            model, processor, problems,
            rollouts_per_problem=args.rollouts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=getattr(args, "batch_size", 1),
        )
    finally:
        handle.remove()

    data = {
        "alpha": args.alpha,
        "model": model_id,
        "noise_config_path": os.path.join(script_dir, "noise_configs", f"qwen3_vl_{args.model_size}_noise_layers.json"),
        "layer_percentile": args.layer_percentile,
        "layer_index": layer_idx,
        "num_problems": len(problems_meta),
        "rollouts_per_problem": args.rollouts,
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "problems": problems_meta,
        "rollouts": rollouts,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
