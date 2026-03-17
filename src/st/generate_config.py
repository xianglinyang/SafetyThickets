"""
Generate expert perturbation configs and store model metadata for reproducibility.

Config file layout
──────────────────
{
  "_meta": {
    "base_model":              "meta-llama/Llama-3.1-8B-Instruct",
    "target_module_keywords":  ["layers.24", "self_attn.o_proj", ...],
    "target_param_names":      [...],   # ordered list — MUST use this order in perturb()
    "target_param_shapes":     {"name": [dim0, dim1]},
    "global_seed":             42,
    "population_size":         32,
    "sigma_list":              [0.0001, 0.0005, 0.001],
    "created_at":              "2026-03-17T..."
  },
  "0": {"expert_id": 0, "seed": ..., "sigma": ...},
  "1": {"expert_id": 1, "seed": ..., "sigma": ...},
  ...
}

The `target_param_names` list fixes the iteration order used when generating
noise in `perturb_and_infer.py`, ensuring that:
  - Two runs with the same seed produce identical perturbations.
  - SVD analysis in `svd_analysis.py` can reconstruct noise vectors exactly.
"""
from __future__ import annotations

import json
import logging
import os
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_target_param_info(
    base_model: str,
    target_module_keywords: List[str],
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Load the model on the 'meta' device (zero memory cost, no weights downloaded
    beyond the config) and return the ordered list of target parameter names and
    their shapes.

    Using device_map="meta" means we get the exact architecture — and therefore
    the exact named_parameters() order — without allocating any GPU/CPU memory.
    """
    from transformers import AutoModelForCausalLM

    logger = logging.getLogger(__name__)
    logger.info(f"Inspecting model architecture: {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="meta")

    target_param_names: List[str] = []
    target_param_shapes: Dict[str, List[int]] = {}

    for name, param in model.named_parameters():
        if any(k in name for k in target_module_keywords):
            target_param_names.append(name)
            target_param_shapes[name] = list(param.shape)

    del model
    return target_param_names, target_param_shapes


# ─────────────────────────────────────────────────────────────────────────────
# Config generation
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_and_save_configs(
    base_model: str,
    target_module_keywords: List[str],
    population_size: int,
    sigma_list: List[float],
    global_seed: int,
    output_file: str,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting config generation with global_seed={global_seed}, population_size={population_size}")
    set_seed(global_seed)

    # ── 1. Capture ordered param names from the model ────────────────────────
    target_param_names, target_param_shapes = get_target_param_info(
        base_model, target_module_keywords
    )

    n_params = len(target_param_names)
    total_numel = sum(
        int(np.prod(s)) for s in target_param_shapes.values()
    )
    logger.info(f"Found {n_params} target parameters ({total_numel:,} elements):")
    for name in target_param_names:
        logger.info(f"  {name:60s}  shape={target_param_shapes[name]}")

    # ── 2. Sample seeds and sigmas ───────────────────────────────────────────
    logger.info(f"Sampling {population_size} expert configurations...")
    rng = np.random.default_rng(global_seed)
    seeds  = rng.choice(2**31 - 1, size=population_size, replace=False).tolist()
    sigmas = rng.choice(sigma_list,  size=population_size).tolist()
    logger.info(f"Sigma distribution: {dict(zip(*np.unique(sigmas, return_counts=True)))}")

    # ── 3. Build config dict ─────────────────────────────────────────────────
    configs: Dict = {
        "_meta": {
            "base_model":             base_model,
            "target_module_keywords": target_module_keywords,
            # Ordered list: perturb() MUST iterate in this exact order so the
            # torch.Generator RNG sequence is reproducible across runs.
            "target_param_names":     target_param_names,
            "target_param_shapes":    target_param_shapes,
            "global_seed":            global_seed,
            "population_size":        population_size,
            "sigma_list":             [float(s) for s in sigma_list],
            "created_at":             datetime.now().isoformat(),
        }
    }
    for i, (seed, sigma) in enumerate(zip(seeds, sigmas)):
        configs[str(i)] = {
            "expert_id": i,
            "seed":      int(seed),
            "sigma":     float(sigma),
        }

    # ── 4. Save ──────────────────────────────────────────────────────────────
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(configs, f, indent=4)

    logger.info(f"Saved {population_size} expert configs → {output_file}")
    logger.info("Configuration generation completed successfully")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate perturbation configs with model metadata."
    )
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--target_module_keywords", type=str, nargs="+",
                        default=[
                            "layers.24", "layers.25", "layers.26", "layers.27",
                            "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj",
                        ],
                        help="Substrings matched against parameter names to "
                             "select which layers to perturb.")
    parser.add_argument("--population_size", type=int, default=32)
    parser.add_argument("--sigma_list", type=float, nargs="+",
                        default=[1e-4, 5e-4, 1e-3])
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_file", type=str,
                        default="outputs/Llama-3.1-8B-Instruct.json")
    args = parser.parse_args()

    # Setup logging
    from src.utils.logging_utils import setup_logging
    setup_logging(
        task_name="generate_config",
        log_level=logging.INFO,
        log_dir="logs",
        run_id=f"seed{args.global_seed}_pop{args.population_size}"
    )

    generate_and_save_configs(
        base_model=args.base_model,
        target_module_keywords=args.target_module_keywords,
        population_size=args.population_size,
        sigma_list=args.sigma_list,
        global_seed=args.global_seed,
        output_file=args.output_file,
    )
