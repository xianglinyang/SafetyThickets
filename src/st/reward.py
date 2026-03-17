"""
Reward functions for the safety-thicket perturbation search.

Per-dataset score entries
─────────────────────────
  Each compute_*_rewards() returns Dict[dataset_name, ScoreEntry] where:

    ScoreEntry = {"score": float, "method": str}

  This lets every dataset record carry the method it was scored with, so
  results from different methods can coexist without ambiguity.

  safety_scores  : Dict[str, ScoreEntry]  — refusal rate (higher = safer)
  utility_scores : Dict[str, ScoreEntry]  — accuracy    (higher = more capable)

Combined scalar
───────────────
  combined = alpha * mean(safety_scores) + (1 - alpha) * mean(utility_scores)

Utility methods
───────────────
  "regex"   — fast regex-based answer extraction; no API call needed
  "llm"     — uses a judge LLM (default: openai/gpt-4.1-nano) to extract answers
              from the model's raw output before comparing to ground truth;
              mirrors the logic in src/evaluate/evaluate_common_reasoning.py

Write-back
──────────
  write_scores_to_config() atomically writes scores into the expert config file
  so SVD analysis can load everything from one place.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Each dataset's score entry: carries both the numeric score and the method used.
# Using a plain dict keeps JSON serialisation trivial.
#   {"score": 0.85, "method": "substring_matching"}
ScoreEntry = Dict[str, object]   # {"score": float, "method": str}

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run a coroutine from synchronous code using a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()



# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_final_response(text: str) -> str:
    """Strip the CoT preamble if the model used the '#### Response' format."""
    parts = text.split("#### Response")
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def _clean_pred(dataset_name: str, raw: str) -> str:
    """Normalise one raw answer string with the dataset-specific regex."""
    from src.data_utils.reasoning_datasets import answer_cleansing_with_regex
    try:
        return answer_cleansing_with_regex(dataset_name, raw)
    except Exception:
        return "[INVALID]"


# ─────────────────────────────────────────────────────────────────────────────
# Public multi-dataset API
# ─────────────────────────────────────────────────────────────────────────────

def compute_safety_rewards(
    datasets_data: List[Tuple[str, List[str], List[str]]],
    method: str = "harmbench_cls",
    device: str = "cuda",
) -> Dict[str, ScoreEntry]:
    """
    Merge all safety datasets → ONE guard-model call → slice results back.

    All safety datasets share the same guard model and the same scoring logic
    (refusal rate), so a single batch call is both correct and efficient.

    Args:
        datasets_data: [(dataset_name, prompts, responses), ...]
        method:        substring_matching | llamaguard3 | harmbench_cls
        device:        GPU device for classifier-based methods

    Returns:
        {dataset_name: {"score": refusal_rate ∈ [0,1], "method": str}}
    """
    if not datasets_data:
        return {}

    # ── Merge all datasets, track per-dataset slices ─────────────────────────
    all_prompts:   List[str] = []
    all_responses: List[str] = []
    slices: Dict[str, Tuple[int, int]] = {}   # name -> (start, end)

    for ds_name, prompts, responses in datasets_data:
        start = len(all_responses)
        all_prompts.extend(prompts)
        all_responses.extend(responses)
        slices[ds_name] = (start, len(all_responses))

    logger.info(
        f"Safety eval: {len(all_responses)} responses across "
        f"{len(datasets_data)} datasets → single {method} call"
    )

    # ── Single guard-model call for everything ────────────────────────────────
    if method == "substring_matching":
        from src.evaluate.evaluate_harmful import substring_matching_judge_fn
        batch_scores = [1 if substring_matching_judge_fn(r) else 0 for r in all_responses]

    elif method == "llamaguard3":
        from src.evaluate.evaluate_harmful import llamaguard3_judge_fn
        unsafe = llamaguard3_judge_fn(all_prompts, all_responses, device)
        batch_scores = [1 - s for s in unsafe]

    elif method == "harmbench_cls":
        from src.evaluate.evaluate_harmful import harmbench_judge_fn
        unsafe = harmbench_judge_fn(all_prompts, all_responses, device)
        batch_scores = [1 - s for s in unsafe]

    else:
        raise ValueError(
            f"Unknown safety method '{method}'. "
            "Choose: substring_matching | llamaguard3 | harmbench_cls"
        )

    # ── Slice back to per-dataset results ────────────────────────────────────
    scores: Dict[str, ScoreEntry] = {}
    for ds_name, (start, end) in slices.items():
        ds_scores = batch_scores[start:end]
        if not ds_scores:
            rate = 0.0
        else:
            rate = float(sum(ds_scores)) / len(ds_scores)
        scores[ds_name] = {"score": rate, "method": method}
        print(f"  [safety/{method}/{ds_name}]  {rate:.4f}  ({sum(ds_scores)}/{len(ds_scores)})")

    return scores


def compute_utility_rewards(
    datasets_data: List[Tuple[str, List[str], List[str], List[str]]],
    method: str = "regex",
    clean_model_name: str = "openai/gpt-4.1-nano",
) -> Dict[str, ScoreEntry]:
    """
    Per-dataset utility reward (accuracy).

    Unlike safety, utility datasets have DIFFERENT answer-cleansing rules
    (e.g. gsm8k extracts arabic numerals; arc-c/e extracts A-D letters).
    Therefore responses MUST be evaluated per-dataset with the matching cleansing
    logic; they cannot be merged into one homogeneous batch.

    For method="regex":  purely local, no API calls.
    For method="llm":    one async coroutine per dataset, all gathered in a
                         SINGLE event loop so we pay loop-startup cost only once.

    Args:
        datasets_data:    [(dataset_name, questions, responses, gt_answers), ...]
        method:           "regex" | "llm"
        clean_model_name: judge model for method="llm"

    Returns:
        {dataset_name: {"score": accuracy ∈ [0,1], "method": str}}
    """
    if not datasets_data:
        return {}

    if method not in ("regex", "llm"):
        raise ValueError(f"Unknown utility method '{method}'. Choose: regex | llm")

    from src.data_utils.reasoning_datasets import batch_gt_answer_cleansing

    # ── Build per-dataset (extracted_responses, clean_gt) pairs ──────────────
    # extraction (CoT stripping) is dataset-agnostic, so it's fine to do it here
    prepared: List[Tuple[str, List[str], List[str], List[str]]] = []
    for ds_name, questions, responses, gt_answers in datasets_data:
        extracted = [_extract_final_response(r) for r in responses]
        clean_gt  = batch_gt_answer_cleansing(ds_name, gt_answers)
        prepared.append((ds_name, questions, extracted, clean_gt))

    # ── Regex: purely synchronous, per-dataset ────────────────────────────────
    if method == "regex":
        scores: Dict[str, ScoreEntry] = {}
        for ds_name, _, extracted, clean_gt in prepared:
            preds = [_clean_pred(ds_name, r) for r in extracted]
            corrects = [p == g for p, g in zip(preds, clean_gt)]
            acc = float(sum(corrects)) / len(corrects) if corrects else 0.0
            scores[ds_name] = {"score": acc, "method": "regex"}
            print(f"  [utility/regex/{ds_name}]  {acc:.4f}  ({sum(corrects)}/{len(corrects)})")
        return scores

    # ── LLM: one event loop, all datasets run concurrently ───────────────────
    # Each dataset needs its own dataset-specific prompt trigger, so they
    # cannot be merged into one call.  We run them as concurrent coroutines
    # inside a single event loop to avoid repeated loop-startup overhead.
    from src.data_utils.reasoning_datasets import batch_answer_cleansing_with_llm

    async def _score_one(ds_name, questions, extracted, clean_gt):
        """Score a single dataset inside the shared event loop."""
        try:
            raw_preds = await batch_answer_cleansing_with_llm(
                ds_name, questions, extracted, clean_model_name
            )
            # Normalise the LLM's raw output with the same dataset-specific regex
            # so the comparison format matches clean_gt exactly.
            preds = [_clean_pred(ds_name, p) for p in raw_preds]
        except Exception as exc:
            logger.error(f"LLM cleansing failed for {ds_name}: {exc}, falling back to regex")
            preds = [_clean_pred(ds_name, r) for r in extracted]

        corrects = [p == g for p, g in zip(preds, clean_gt)]
        acc = float(sum(corrects)) / len(corrects) if corrects else 0.0
        logger.debug(
            f"  [{ds_name}] first 3: "
            + ", ".join(f"pred={p!r} gt={g!r}" for p, g in zip(preds[:3], clean_gt[:3]))
        )
        return ds_name, acc

    async def _score_all():
        tasks = [
            _score_one(ds_name, questions, extracted, clean_gt)
            for ds_name, questions, extracted, clean_gt in prepared
        ]
        return await asyncio.gather(*tasks)

    results = _run_async(_score_all())

    scores = {}
    gt_len = {ds_name: len(gt) for ds_name, _, _, gt in prepared}
    for ds_name, acc in results:
        scores[ds_name] = {"score": acc, "method": "llm"}
        n = gt_len[ds_name]
        print(f"  [utility/llm/{ds_name}]  {acc:.4f}  ({round(acc*n)}/{n})")

    return scores


def compute_combined_reward(
    safety_scores: Dict[str, ScoreEntry],
    utility_scores: Dict[str, ScoreEntry],
    alpha: float = 0.5,
) -> float:
    """
    Scalar combined reward = alpha * mean(safety) + (1-alpha) * mean(utility).
    Either dict may be empty; a missing half contributes 0.
    Extracts the numeric "score" field from each ScoreEntry.
    """
    avg_safety = (
        sum(float(e["score"]) for e in safety_scores.values()) / len(safety_scores)
        if safety_scores else 0.0
    )
    avg_utility = (
        sum(float(e["score"]) for e in utility_scores.values()) / len(utility_scores)
        if utility_scores else 0.0
    )
    combined = alpha * avg_safety + (1.0 - alpha) * avg_utility
    logger.info(
        f"[combined] α={alpha}  safety={avg_safety:.4f}  utility={avg_utility:.4f}  "
        f"combined={combined:.4f}"
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Write scores back into config file (with file locking)
# ─────────────────────────────────────────────────────────────────────────────

def write_scores_to_config(
    config_file: str,
    expert_id: Optional[int],
    safety_scores: Dict[str, ScoreEntry],
    utility_scores: Dict[str, ScoreEntry],
    max_retries: int = 5,
) -> bool:
    """
    Incrementally update scores in configs[str(expert_id)]["scores"].
    Uses fcntl exclusive locking so parallel expert jobs are safe.

    Structure supports multiple methods per dataset:
      "scores": {
        "safety": {
          "PolyGuardMix": {
            "substring_matching": {"score": 0.85, "timestamp": "2026-03-17T16:04:25"},
            "llamaguard3":        {"score": 0.92, "timestamp": "2026-03-17T16:10:15"}
          }
        },
        "utility": {
          "gsm8k": {
            "regex": {"score": 0.72, "timestamp": "2026-03-17T16:04:25"},
            "llm":   {"score": 0.74, "timestamp": "2026-03-17T16:08:30"}
          }
        }
      }

    This allows different methods to coexist without overwriting each other.
    """
    timestamp = datetime.now().isoformat()
    
    for attempt in range(max_retries):
        try:
            with open(config_file, "r+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    configs = json.load(f)
                    
                    # Handle base model (expert_id=None) vs regular experts
                    if expert_id is None:
                        expert_key = "base_model"
                        # Initialize base_model entry if it doesn't exist
                        if expert_key not in configs:
                            configs[expert_key] = {
                                "model_type": "base_model",
                                "description": "Original model without perturbation (baseline)"
                            }
                    else:
                        expert_key = str(expert_id)
                    
                    # Initialize scores structure if it doesn't exist
                    if "scores" not in configs[expert_key]:
                        configs[expert_key]["scores"] = {"safety": {}, "utility": {}}
                    if "safety" not in configs[expert_key]["scores"]:
                        configs[expert_key]["scores"]["safety"] = {}
                    if "utility" not in configs[expert_key]["scores"]:
                        configs[expert_key]["scores"]["utility"] = {}

                    # Update safety scores
                    for dataset_name, score_entry in safety_scores.items():
                        method = score_entry["method"]
                        score = score_entry["score"]
                        
                        if dataset_name not in configs[expert_key]["scores"]["safety"]:
                            configs[expert_key]["scores"]["safety"][dataset_name] = {}
                        
                        configs[expert_key]["scores"]["safety"][dataset_name][method] = {
                            "score": score,
                            "timestamp": timestamp,
                        }

                    # Update utility scores  
                    for dataset_name, score_entry in utility_scores.items():
                        method = score_entry["method"]
                        score = score_entry["score"]
                        
                        if dataset_name not in configs[expert_key]["scores"]["utility"]:
                            configs[expert_key]["scores"]["utility"][dataset_name] = {}
                        
                        configs[expert_key]["scores"]["utility"][dataset_name][method] = {
                            "score": score,
                            "timestamp": timestamp,
                        }

                    f.seek(0)
                    f.truncate()
                    json.dump(configs, f, indent=4)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            print(f"Scores written to config [expert {expert_id}] at {timestamp}")
            return True
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"Failed to write scores after {max_retries} attempts: {exc}")
                return False
            time.sleep(1)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible single-dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    prompts: List[str],
    harmful_responses: List[str],
    utility_questions: List[str],
    utility_responses: List[str],
    gt_answers: List[str],
    dataset_name: str,
    alpha: float = 0.5,
    safety_method: str = "substring_matching",
    utility_method: str = "regex",
    device: str = "cuda",
    clean_model_name: str = "openai/gpt-4.1-nano",
) -> Tuple[float, float, float]:
    """Single-dataset convenience wrapper. Returns (safety_score, utility_score, combined)."""
    safety_scores  = compute_safety_rewards(
        [(dataset_name, prompts, harmful_responses)], safety_method, device
    )
    utility_scores = compute_utility_rewards(
        [(dataset_name, utility_questions, utility_responses, gt_answers)],
        utility_method, clean_model_name,
    )
    combined = compute_combined_reward(safety_scores, utility_scores, alpha)
    return (
        float(safety_scores[dataset_name]["score"]),
        float(utility_scores[dataset_name]["score"]),
        combined,
    )
