"""
Construct reasoning task vector: τ_reason = θ_R - θ_M
- M = non-reasoning instruct (Qwen3-VL-8B-Instruct)
- R = reasoning/distilled (Qwen3-VL-8B-Thinking)
- τ[k] = R[k] - M[k] for each parameter k
"""

import argparse
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch

# VL model pair: R (Thinking) - M (Instruct)
R_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
M_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TASK_VECTOR_PATH = "qwen3_vl_8b_thinking_task_vector.pt"


def load_models():
    """Load R (Thinking) and M (Instruct) models."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"Loading processor for {R_MODEL}...")
    processor = AutoProcessor.from_pretrained(R_MODEL)

    print(f"Loading R (Thinking): {R_MODEL}...")
    r_model = Qwen3VLForConditionalGeneration.from_pretrained(
        R_MODEL, torch_dtype=torch.bfloat16
    )

    print(f"Loading M (Instruct): {M_MODEL}...")
    m_model = Qwen3VLForConditionalGeneration.from_pretrained(
        M_MODEL, torch_dtype=torch.bfloat16
    )

    return r_model, m_model, processor


def verify_model_compatibility(model1, model2):
    """Verifies that two models have identical architectures and parameter shapes."""
    params1 = {n: p.shape for n, p in model1.named_parameters()}
    params2 = {n: p.shape for n, p in model2.named_parameters()}

    if set(params1.keys()) != set(params2.keys()):
        missing_in_2 = set(params1.keys()) - set(params2.keys())
        missing_in_1 = set(params2.keys()) - set(params1.keys())
        print(f"Error: Parameter keys do not match.")
        print(f"  Missing in M: {missing_in_2}")
        print(f"  Missing in R: {missing_in_1}")
        raise ValueError("Model parameter keys do not match.")

    for name in params1:
        if params1[name] != params2[name]:
            print(f"Error: Parameter '{name}' shapes differ: {params1[name]} vs {params2[name]}")
            raise ValueError(f"Parameter '{name}' shapes do not match.")

    print("Model architectures and parameter shapes are compatible.")


def compute_task_vector(r_model, m_model, exclude_embeddings=True):
    """
    Computes τ = R - M.
    Optionally excludes embedding and LM head weights.
    """
    task_vector = {}
    for name, param_r in r_model.named_parameters():
        param_m = m_model.state_dict()[name]

        if exclude_embeddings and ("embed_tokens" in name or "lm_head" in name):
            print(f"Excluding {name} from task vector.")
            continue

        if param_r.shape != param_m.shape:
            raise ValueError(
                f"Shape mismatch for parameter {name}: "
                f"R {param_r.shape}, M {param_m.shape}"
            )

        task_vector[name] = param_r - param_m
    return task_vector


def save_task_vector(task_vector, path):
    """Saves the computed task vector to a file."""
    torch.save(task_vector, path)
    print(f"Task vector saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Construct τ = R - M (Qwen3-VL-8B-Thinking - Instruct)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to save task vector .pt file",
    )
    args = parser.parse_args()

    try:
        print("\n=== Building τ = R - M (Qwen3-VL-8B-Thinking - Qwen3-VL-8B-Instruct) ===")
        r_model, m_model, _ = load_models()
        verify_model_compatibility(r_model, m_model)
        task_vector = compute_task_vector(r_model, m_model)
        path = os.path.join(args.save_dir, TASK_VECTOR_PATH)
        save_task_vector(task_vector, path)
        print("\nDone.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
