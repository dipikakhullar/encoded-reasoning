"""
Compute and save layer indices for activation noise injection for Qwen3-VL 2B, 4B, 8B.
Noise is applied at a chosen percentile of the decoder depth: 25th, 50th, or 75th.
Output: one JSON per model under noise_configs/ with layer indices for each percentile.
"""

import argparse
import json
import os

# Model IDs for each size (Instruct; we only need config, not weights)
SIZE_TO_MODEL = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}

PERCENTILES = (25, 50, 75)


def get_num_decoder_layers(model_id: str) -> int:
    """Load config only and return number of text decoder layers."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "text_config") and config.text_config is not None:
        return getattr(config.text_config, "num_hidden_layers", None) or getattr(
            config.text_config, "num_layers", None
        )
    return getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None)


def layer_index_for_percentile(num_layers: int, percentile: int) -> int:
    """Return 0-based layer index for the given percentile (25, 50, or 75)."""
    if num_layers is None or num_layers < 1:
        raise ValueError(f"Invalid num_layers={num_layers}")
    # 25th -> early, 50th -> middle, 75th -> late
    idx = int((percentile / 100.0) * num_layers)
    return min(idx, num_layers - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Write noise layer configs for 2B, 4B, 8B at 25th/50th/75th percentile"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["2b", "4b", "8b"],
        default=["2b", "4b", "8b"],
        help="Model sizes to generate configs for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for noise_configs (default: script_dir/noise_configs)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(script_dir, "noise_configs")
    os.makedirs(out_dir, exist_ok=True)

    for size in args.sizes:
        model_id = SIZE_TO_MODEL[size]
        print(f"Loading config for {model_id}...")
        num_layers = get_num_decoder_layers(model_id)
        if num_layers is None:
            print(f"  Warning: could not get num_layers for {model_id}, skipping")
            continue
        indices = {
            f"layer_{p}": layer_index_for_percentile(num_layers, p)
            for p in PERCENTILES
        }
        config = {
            "model_id": model_id,
            "size": size,
            "num_decoder_layers": num_layers,
            "percentiles": list(PERCENTILES),
            **indices,
        }
        out_name = f"qwen3_vl_{size}_noise_layers.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  num_layers={num_layers} -> {out_path}")
        print(f"    layer_25={indices['layer_25']}, layer_50={indices['layer_50']}, layer_75={indices['layer_75']}")

    print(f"\nConfigs written to {out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
