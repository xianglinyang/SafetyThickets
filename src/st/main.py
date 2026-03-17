"""
perturb_and_infer.py

Pipeline per expert:
  1. perturb()  — apply noise to target layers with an isolated torch.Generator
                  (seeded from config) and save perturbed weights to disk
  2. infer()    — single vLLM session; only runs for datasets whose responses
                  are not already cached on disk
  3. reward     — per-dataset safety + utility scores
  4. write-back — scores written atomically back into the shared config file

Output directory layout
───────────────────────
All outputs are stored under a deterministic path derived from the model name
and run configuration, so every experiment is self-contained and easy to find:

  {output_root}/
    {model_shortname}/              e.g.  Llama-3.1-8B-Instruct
      seed{global_seed}_pop{N}/    e.g.  seed42_pop32
        expert_0000/
          safety/
            PolyGuardMix.json      [{prompt, generated_text}, ...]
            harmbench.json
          utility/
            gsm8k.json
            mmlu.json
          reward.json

Caching
───────
On second (or later) runs, if a dataset's JSON already exists at the expected
path it is loaded from disk instead of re-running vLLM.  If ALL datasets for
a given expert are already cached, both perturb() and vLLM are skipped entirely.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from src.data_utils.harmful_datasets import data_reader as harmful_data_reader
from src.data_utils.reasoning_datasets import data_reader as utility_data_reader
from src.st.reward import (
    compute_safety_rewards,
    compute_utility_rewards,
    write_scores_to_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_shortname(base_model: str) -> str:
    return base_model.rstrip("/").split("/")[-1]


def _run_tag(meta: Dict) -> str:
    """e.g. "seed42_pop32" — uniquely identifies a generation config."""
    seed = meta.get("global_seed", "unk")
    pop  = meta.get("population_size", "unk")
    return f"seed{seed}_pop{pop}"


def get_expert_output_dir(output_root: str, base_model: str, meta: Dict, expert_id: int) -> str:
    """
    Deterministic path for a single expert's inference outputs and reward record.

      {output_root}/{model_shortname}/{run_tag}/expert_{id:04d}/
    """
    return os.path.join(
        output_root,
        _model_shortname(base_model),
        _run_tag(meta),
        f"expert_{expert_id:04d}",
    )


def get_expert_temp_dir(temp_root: str, base_model: str, meta: Dict, expert_id: int) -> str:
    """
    Deterministic path for the perturbed model weights (vLLM loads from here).

      {temp_root}/{model_shortname}/{run_tag}/expert_{id:04d}/
    """
    return os.path.join(
        temp_root,
        _model_shortname(base_model),
        _run_tag(meta),
        f"expert_{expert_id:04d}",
    )


def _result_path(expert_dir: str, tag: str, dataset_name: str) -> str:
    """Path for one dataset's raw responses: {expert_dir}/{tag}/{dataset_name}.json"""
    return os.path.join(expert_dir, tag, f"{dataset_name}.json")


def _save_responses(
    expert_dir: str,
    tag: str,
    dataset_name: str,
    prompts: List[str],
    responses: List[str],
) -> str:
    path = _result_path(expert_dir, tag, dataset_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [{"prompt": p, "generated_text": r} for p, r in zip(prompts, responses)],
            f, indent=4,
        )
    print(f"  → saved {len(responses)} responses  {path}")
    return path


def _load_cached_responses(
    expert_dir: str,
    tag: str,
    dataset_name: str,
    expected_len: int,
) -> Optional[List[str]]:
    """
    Return cached responses if the file exists and has the expected number of
    entries.  Returns None on any mismatch so inference will be re-run.
    """
    path = _result_path(expert_dir, tag, dataset_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if len(data) != expected_len:
            print(f"  [cache stale] {path}  ({len(data)} != {expected_len}), re-running")
            return None
        print(f"  [cache hit]   {path}")
        return [item["generated_text"] for item in data]
    except Exception as exc:
        print(f"  [cache error] {path}: {exc}, re-running")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_file: str) -> Dict:
    with open(config_file, "r") as f:
        return json.load(f)


def get_expert(configs: Dict, expert_id: int) -> Dict:
    if isinstance(configs, list):
        return next(c for c in configs if c["expert_id"] == expert_id)
    key = str(expert_id)
    if key not in configs:
        raise KeyError(f"expert_id {expert_id} not found  (keys: {list(configs.keys())})")
    return configs[key]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Perturb
# ─────────────────────────────────────────────────────────────────────────────

def perturb(expert_id: int, config_file: str, base_model: str, temp_dir: str) -> str:
    """
    Apply Gaussian noise to the model's target layers and save to temp_dir.

    Uses an isolated torch.Generator (not the global RNG) seeded with the
    expert's seed, iterating in the order fixed by _meta.target_param_names.
    This guarantees identical perturbations regardless of environment.
    """
    configs = load_config(config_file)
    meta    = configs.get("_meta", {})
    config  = get_expert(configs, expert_id)

    seed, sigma = config["seed"], config["sigma"]
    target_param_names: Optional[List[str]] = meta.get("target_param_names")

    print(f"Perturbing expert {expert_id}  seed={seed}  sigma={sigma:.2e}")
    print(f"  → {len(target_param_names or [])} target params  →  {temp_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu"
    )

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    named_params = dict(model.named_parameters())

    with torch.no_grad():
        if target_param_names:
            for name in target_param_names:
                if name not in named_params:
                    continue
                param = named_params[name]
                eps = torch.randn(param.shape, generator=g, dtype=torch.float32)
                param.add_(eps.to(dtype=param.dtype, device=param.device) * sigma)
        else:
            # Fallback for old configs without _meta
            for name, param in model.named_parameters():
                if param.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    eps = torch.randn(param.shape, generator=g, dtype=torch.float32)
                    param.add_(eps.to(dtype=param.dtype, device=param.device) * sigma)

    os.makedirs(temp_dir, exist_ok=True)
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    del model
    gc.collect()
    return temp_dir


def _perturbed_model_exists(temp_dir: str) -> bool:
    """Quick check: does the temp_dir hold a saved HF model?"""
    return os.path.exists(os.path.join(temp_dir, "config.json"))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Infer
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    temp_dir: str,
    harmful_prompts: List[str],
    utility_prompts: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Single vLLM session.  Returns (harmful_responses, utility_responses).
    Only called for prompts whose responses are not already cached.
    """
    # Filter out None values and empty strings to prevent vLLM errors
    harmful_clean = [p for p in harmful_prompts if p is not None and p.strip()]
    utility_clean = [p for p in utility_prompts if p is not None and p.strip()]
    
    logger = logging.getLogger(__name__)
    if len(harmful_clean) != len(harmful_prompts):
        logger.warning(f"Filtered {len(harmful_prompts) - len(harmful_clean)} invalid harmful prompts")
    if len(utility_clean) != len(utility_prompts):
        logger.warning(f"Filtered {len(utility_prompts) - len(utility_clean)} invalid utility prompts")
    
    try:
        llm = LLM(model=temp_dir, trust_remote_code=True)
        sp  = SamplingParams(temperature=0.0, max_tokens=512)
        logger.info(f"Successfully loaded model from {temp_dir}")
    except Exception as e:
        logger.error(f"Failed to load model from {temp_dir}: {e}")
        raise
    
    # Use cleaned prompts for generation
    harmful_resp = []
    utility_resp = []
    
    if harmful_clean:
        logger.info(f"Generating responses for {len(harmful_clean)} harmful prompts")
        try:
            harmful_outputs = llm.generate(harmful_clean, sp)
            harmful_resp = [o.outputs[0].text.strip() for o in harmful_outputs]
        except Exception as e:
            logger.error(f"Error generating harmful responses: {e}")
            harmful_resp = ["[ERROR]"] * len(harmful_clean)
    
    if utility_clean:
        logger.info(f"Generating responses for {len(utility_clean)} utility prompts")
        try:
            utility_outputs = llm.generate(utility_clean, sp)
            utility_resp = [o.outputs[0].text.strip() for o in utility_outputs]
        except Exception as e:
            logger.error(f"Error generating utility responses: {e}")
            utility_resp = ["[ERROR]"] * len(utility_clean)
    
    # Pad responses to match original length if needed
    while len(harmful_resp) < len(harmful_prompts):
        harmful_resp.append("[INVALID_PROMPT]")
    while len(utility_resp) < len(utility_prompts):
        utility_resp.append("[INVALID_PROMPT]")
    
    return harmful_resp, utility_resp


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helper
# ─────────────────────────────────────────────────────────────────────────────

def sample_items(items: List, test_num: Optional[int], seed: int = 42) -> List:
    """Sample items from a list, handling edge cases safely."""
    if not items:  # Handle empty lists
        return []
    if test_num is None or test_num <= 0 or test_num >= len(items):
        return items
    return random.Random(seed).sample(items, test_num)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(
    expert_id: Optional[int],
    config_file: str,
    base_model: str,
    temp_root: str,
    output_root: str,
    harmful_datasets: Sequence[str] = ("PolyGuardMix",),
    utility_datasets: Sequence[str] = ("gsm8k",),
    safety_test_num: Optional[int] = None,
    utility_test_num: Optional[int] = None,
    sample_seed: int = 42,
    safety_method: str = "substring_matching",
    utility_method: str = "regex",
    device: str = "cuda",
    clean_model_name: str = None,
) -> Dict:

    t_total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # ── Resolve structured paths ─────────────────────────────────────────────
    if expert_id is None:
        # Base model evaluation (no perturbation)
        meta = {}
        expert_dir = os.path.join(output_root, _model_shortname(base_model), "base_model")
        temp_dir = base_model  # Use original model path directly
        model_tag = "base model"
    else:
        # Expert evaluation (with perturbation)
        configs = load_config(config_file)
        meta = configs.get("_meta", {})
        expert_dir = get_expert_output_dir(output_root, base_model, meta, expert_id)
        temp_dir = get_expert_temp_dir(temp_root, base_model, meta, expert_id)
        model_tag = f"Expert {expert_id}"
    
    os.makedirs(expert_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{model_tag}")
    print(f"  output dir : {expert_dir}")
    print(f"  temp dir   : {temp_dir}")
    print(f"{'='*60}")

    # ── Load datasets ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    harmful_map: Dict[str, List[str]] = {}
    utility_map: Dict[str, List[str]] = {}
    gt_map:      Dict[str, List[str]] = {}

    for ds in harmful_datasets:
        prompts, _ = harmful_data_reader(dataset_name=ds, split="train")
        # Filter out None values from prompts
        prompts = [p for p in prompts if p is not None and str(p).strip()]
        logger.info(f"Loaded {len(prompts)} valid harmful prompts from {ds}")
        harmful_map[ds] = sample_items(prompts, safety_test_num, sample_seed)

    for ds in utility_datasets:
        trigger = "Solve the following problem:\n\n"
        questions, _, gt_answers = utility_data_reader(dataset_name=ds, split="test")
        questions = [trigger + q for q in questions]
        
        # Filter out None values from questions and answers
        valid_pairs = [(q, a) for q, a in zip(questions, gt_answers) 
                       if q is not None and a is not None and str(q).strip() and str(a).strip()]
        logger.info(f"Loaded {len(valid_pairs)} valid utility question-answer pairs from {ds}")
        
        # Sample pairs together to maintain question-answer correspondence
        sampled_pairs = sample_items(valid_pairs, utility_test_num, sample_seed)
        
        # Separate questions and answers after sampling
        utility_map[ds] = [pair[0] for pair in sampled_pairs]
        gt_map[ds] = [pair[1] for pair in sampled_pairs]
        
        # Validation: ensure lengths match
        assert len(utility_map[ds]) == len(gt_map[ds]), f"Length mismatch in {ds}: {len(utility_map[ds])} vs {len(gt_map[ds])}"
        
        logger.info(f"Sampled {len(sampled_pairs)} question-answer pairs for {ds}")
        
        # Debug: show first pair to verify correct pairing
        if sampled_pairs:
            first_q = utility_map[ds][0][:100] + "..." if len(utility_map[ds][0]) > 100 else utility_map[ds][0]
            first_a = gt_map[ds][0][:50] + "..." if len(str(gt_map[ds][0])) > 50 else str(gt_map[ds][0])
            logger.info(f"[{ds}] Sample Q-A pair: Q='{first_q}' -> A='{first_a}'")

    timings["data_loading"] = time.perf_counter() - t0
    logger.info("Step [data loading]  done in %.1fs", timings["data_loading"])

    # ── Check cache ──────────────────────────────────────────────────────────
    print("\nChecking response cache:")
    harmful_resp_map: Dict[str, List[str]] = {}
    utility_resp_map: Dict[str, List[str]] = {}

    harmful_missing = []
    utility_missing = []

    for ds in harmful_datasets:
        cached = _load_cached_responses(expert_dir, "safety", ds, len(harmful_map[ds]))
        if cached is not None:
            harmful_resp_map[ds] = cached
        else:
            harmful_missing.append(ds)

    for ds in utility_datasets:
        cached = _load_cached_responses(expert_dir, "utility", ds, len(utility_map[ds]))
        if cached is not None:
            utility_resp_map[ds] = cached
        else:
            utility_missing.append(ds)

    # ── Perturb + infer only for missing datasets ────────────────────────────
    if harmful_missing or utility_missing:
        if expert_id is None:
            # Base model evaluation: skip perturb, use original model directly
            print("Evaluating base model (no perturbation)")
            timings["perturb"] = 0.0
        else:
            # Expert evaluation: ensure perturbed model exists on disk
            if not _perturbed_model_exists(temp_dir):
                t0 = time.perf_counter()
                perturb(expert_id, config_file, base_model, temp_dir)
                timings["perturb"] = time.perf_counter() - t0
                logger.info("Step [perturb]       done in %.1fs", timings["perturb"])
            else:
                print(f"Perturbed model already exists at {temp_dir}, skipping perturb")
                timings["perturb"] = 0.0

        # Build flat prompt lists for missing datasets only
        harmful_infer: List[str] = []
        harmful_infer_slices: Dict[str, Tuple[int, int]] = {}
        for ds in harmful_missing:
            s = len(harmful_infer)
            harmful_infer.extend(harmful_map[ds])
            harmful_infer_slices[ds] = (s, len(harmful_infer))

        utility_infer: List[str] = []
        utility_infer_slices: Dict[str, Tuple[int, int]] = {}
        for ds in utility_missing:
            s = len(utility_infer)
            utility_infer.extend(utility_map[ds])
            utility_infer_slices[ds] = (s, len(utility_infer))

        print(f"\nRunning inference: "
              f"{len(harmful_infer)} safety prompts ({len(harmful_missing)} datasets), "
              f"{len(utility_infer)} utility prompts ({len(utility_missing)} datasets)")

        # Pad with an empty-string dummy if one side is empty (vLLM needs non-empty)
        harmful_run = harmful_infer or ["<placeholder>"]
        utility_run = utility_infer or ["<placeholder>"]

        t0 = time.perf_counter()
        harmful_resp_flat, utility_resp_flat = infer(temp_dir, harmful_run, utility_run)
        timings["inference"] = time.perf_counter() - t0
        logger.info("Step [inference]     done in %.1fs", timings["inference"])

        # Slice and save
        for ds in harmful_missing:
            s, e = harmful_infer_slices[ds]
            harmful_resp_map[ds] = harmful_resp_flat[s:e]
            _save_responses(expert_dir, "safety", ds, harmful_map[ds], harmful_resp_map[ds])

        for ds in utility_missing:
            s, e = utility_infer_slices[ds]
            utility_resp_map[ds] = utility_resp_flat[s:e]
            _save_responses(expert_dir, "utility", ds, utility_map[ds], utility_resp_map[ds])

    else:
        print("All responses loaded from cache — skipping perturb and inference.")
        timings["perturb"]   = 0.0
        timings["inference"] = 0.0

    # ── Compute rewards ──────────────────────────────────────────────────────
    harmful_for_reward = [
        (ds, harmful_map[ds], harmful_resp_map[ds])
        for ds in harmful_datasets
    ]
    utility_for_reward = [
        (ds, utility_map[ds], utility_resp_map[ds], gt_map[ds])
        for ds in utility_datasets
    ]

    t0 = time.perf_counter()
    print("\nSafety rewards:")
    safety_scores  = compute_safety_rewards(harmful_for_reward, safety_method, device)
    print("Utility rewards:")
    utility_scores = compute_utility_rewards(
        utility_for_reward, utility_method, clean_model_name
    )
    timings["reward"] = time.perf_counter() - t0
    logger.info("Step [reward]        done in %.1fs", timings["reward"])

    # ── Write scores back to config ──────────────────────────────────────────
    if os.path.exists(config_file):
        write_scores_to_config(
            config_file=config_file,
            expert_id=expert_id,  # None for base model, int for experts
            safety_scores=safety_scores,
            utility_scores=utility_scores,
        )
        if expert_id is None:
            logger.info(f"Base model scores written to config: {config_file}")
    else:
        if expert_id is None:
            logger.warning(f"Config file {config_file} not found, base model scores not written to config")
        else:
            logger.error(f"Config file {config_file} not found, expert scores not written to config")

    # ── Save reward record alongside the responses ───────────────────────────
    # Transform scores to dataset -> method -> {score, timestamp} format
    timestamp = datetime.now().isoformat()
    
    def transform_scores(score_dict):
        """Convert {dataset: {score: float, method: str}} to {dataset: {method: {score: float, timestamp: str}}}"""
        transformed = {}
        for dataset_name, score_entry in score_dict.items():
            method = score_entry["method"]
            score = score_entry["score"]
            transformed[dataset_name] = {
                method: {
                    "score": score,
                    "timestamp": timestamp,
                }
            }
        return transformed
    
    record = {
        "expert_id": expert_id,
        "model_type": "base_model" if expert_id is None else "expert",
        "scores": {
            "safety":  transform_scores(safety_scores),
            "utility": transform_scores(utility_scores),
        },
        "clean_model_name":   clean_model_name,
        "n_safety_prompts":   sum(len(v) for v in harmful_map.values()),
        "n_utility_prompts":  sum(len(v) for v in utility_map.values()),
        "timestamp":          timestamp,
    }
    with open(os.path.join(expert_dir, "reward.json"), "w") as f:
        json.dump(record, f, indent=4)

    timings["total"] = time.perf_counter() - t_total_start

    model_label = f"Expert {expert_id}" if expert_id is not None else "Base Model"
    logger.info(
        "\n%s\n%s  timing summary\n%s\n"
        "  data loading : %6.1fs\n"
        "  perturb      : %6.1fs\n"
        "  inference    : %6.1fs\n"
        "  reward       : %6.1fs\n"
        "  ─────────────────────\n"
        "  total        : %6.1fs\n%s",
        "=" * 40, model_label, "=" * 40,
        timings["data_loading"],
        timings["perturb"],
        timings["inference"],
        timings["reward"],
        timings["total"],
        "=" * 40,
    )

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.logging_utils import setup_logging
    
    parser = argparse.ArgumentParser()
    def parse_expert_id(value):
        if value.lower() == "none" or value.lower() == "base":
            return None
        return int(value)
    
    parser.add_argument("--expert_id", type=parse_expert_id, default=0,
                        help="Expert ID to evaluate, or 'base'/'none' for base model evaluation")
    parser.add_argument("--config_file",      type=str,
                        default="outputs/Llama-3.1-8B-Instruct.json")
    parser.add_argument("--base_model",       type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temp_root",        type=str,
                        default="/data2/xianglin/st/temp",
                        help="Root dir for perturbed model weights. "
                             "Actual path is {temp_root}/{model}/{run_tag}/expert_XXXX/")
    parser.add_argument("--output_root",      type=str,
                        default="/data2/xianglin/st/outputs",
                        help="Root dir for inference outputs and reward records. "
                             "Actual path is {output_root}/{model}/{run_tag}/expert_XXXX/")
    parser.add_argument("--harmful_datasets", type=str, nargs="+",
                        default=["PolyGuardMix"])
    parser.add_argument("--utility_datasets", type=str, nargs="+",
                        default=["gsm8k"])
    parser.add_argument("--safety_test_num",  type=int,   default=100,
                        help="Safety prompts sampled per dataset (0 = full dataset).")
    parser.add_argument("--utility_test_num", type=int,   default=100,
                        help="Utility prompts sampled per dataset (0 = full dataset).")
    parser.add_argument("--sample_seed",      type=int,   default=42)
    parser.add_argument("--safety_method",    type=str,   default="harmbench_cls",
                        choices=["substring_matching", "llamaguard3", "harmbench_cls"])
    parser.add_argument("--utility_method",   type=str,   default="llm",
                        choices=["regex", "llm"])
    parser.add_argument("--clean_model_name", type=str,   default="openai/gpt-4.1-nano",
                        help="Judge LLM for utility_method=llm. Set to openai/gpt-4.1-nano as default. ")
    parser.add_argument("--device",           type=str,   default="cuda:1")
    args = parser.parse_args()

    # Setup logging with run_id based on expert_id
    expert_str = "base" if args.expert_id is None else f"expert_{args.expert_id:04d}"
    setup_logging(
        task_name="main",
        log_level=logging.INFO,
        log_dir="logs",
        run_id=expert_str
    )

    # assertion
    if args.utility_method == "llm":
        assert args.clean_model_name is not None, "clean_model_name is required for utility_method=llm"

    main(
        expert_id=args.expert_id,
        config_file=args.config_file,
        base_model=args.base_model,
        temp_root=args.temp_root,
        output_root=args.output_root,
        harmful_datasets=args.harmful_datasets,
        utility_datasets=args.utility_datasets,
        safety_test_num=args.safety_test_num or None,
        utility_test_num=args.utility_test_num or None,
        sample_seed=args.sample_seed,
        safety_method=args.safety_method,
        utility_method=args.utility_method,
        clean_model_name=args.clean_model_name,
        device=args.device,
    )
