#!/usr/bin/env python3
"""
Subspace Analysis for Safety Thickets

This module performs subspace analysis on model perturbations to understand 
the relationship between perturbation directions and safety/utility rewards.

Main workflow:
1. Reconstruct perturbations for all experts
2. Perform SVD decomposition on perturbations (per layer or combined)
3. Load reward scores from configuration files
4. Analyze correlation between subspace components and rewards
"""

import os
import json
import logging
import argparse
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_file, "r") as f:
        return json.load(f)


def get_expert(configs: Dict, expert_id: int) -> Dict:
    """Get configuration for a specific expert."""
    if isinstance(configs, list):
        return next(c for c in configs if c["expert_id"] == expert_id)
    key = str(expert_id)
    if key not in configs:
        raise KeyError(f"expert_id {expert_id} not found (keys: {list(configs.keys())})")
    return configs[key]


def reconstruct_perturbation(
    expert_id: int, 
    config_file: str, 
    base_model: str,
    device: str = "cpu"  # NOTE: Always uses CPU internally to match main.py
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct the perturbation (delta) for a specific expert.
    
    CRITICAL: This MUST produce identical perturbations as main.py perturb() function.
    All device handling, random number generation, and tensor operations are
    carefully matched to ensure perfect consistency.
    
    Args:
        expert_id: Expert ID to reconstruct
        config_file: Path to configuration file
        base_model: Path to base model
        device: Device parameter (IGNORED - always uses CPU for consistency)
    
    Returns:
        Dictionary mapping parameter names to their delta tensors
    """
    configs = load_config(config_file)
    meta = configs.get("_meta", {})
    config = get_expert(configs, expert_id)
    
    seed, sigma = config["seed"], config["sigma"]
    target_param_names: Optional[List[str]] = meta.get("target_param_names")
    
    logger.info(f"Reconstructing perturbation for expert {expert_id} (seed={seed}, sigma={sigma:.2e})")
    
    # Load model to get parameter shapes (we don't need the actual weights)
    # IMPORTANT: Use same device settings as main.py to ensure identical perturbations
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=torch.float16, 
        device_map="cpu"  # ✅ Match main.py: always use CPU
    )
    
    # Initialize generator with expert's seed - MUST use CPU to match main.py
    g = torch.Generator(device="cpu")  # ✅ Match main.py: always use CPU
    g.manual_seed(seed)
    
    deltas = {}
    named_params = dict(model.named_parameters())
    
    with torch.no_grad():
        if target_param_names:
            for name in target_param_names:
                if name not in named_params:
                    continue
                param = named_params[name]
                # ✅ Match main.py exactly: no device parameter in torch.randn
                eps = torch.randn(param.shape, generator=g, dtype=torch.float32)
                # ✅ Match main.py exactly: apply dtype/device conversion before scaling
                deltas[name] = eps.to(dtype=param.dtype, device=param.device) * sigma
        else:
            # Fallback for old configs without _meta
            for name, param in model.named_parameters():
                if param.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    # ✅ Match main.py exactly: no device parameter in torch.randn
                    eps = torch.randn(param.shape, generator=g, dtype=torch.float32)
                    # ✅ Match main.py exactly: apply dtype/device conversion before scaling
                    deltas[name] = eps.to(dtype=param.dtype, device=param.device) * sigma
    
    del model
    # No need for CUDA cache clearing since we always use CPU (matching main.py)
    
    logger.info(f"Reconstructed {len(deltas)} parameter deltas")
    return deltas


def verify_perturbation_consistency(
    expert_id: int,
    config_file: str,
    base_model: str,
    temp_dir: str,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that reconstructed perturbation matches the original perturbation
    by comparing against a freshly perturbed model.
    
    This is a validation function to ensure perfect consistency between
    main.py's perturb() and reconstruct_perturbation().
    """
    # Step 1: Create fresh perturbation using main.py logic (reimplemented for verification)
    configs = load_config(config_file)
    meta = configs.get("_meta", {})
    config = get_expert(configs, expert_id)
    
    seed, sigma = config["seed"], config["sigma"]
    target_param_names = meta.get("target_param_names")
    
    # Create perturbed model using exact main.py logic
    from transformers import AutoTokenizer
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
    
    # Save perturbed model
    import os
    os.makedirs(temp_dir, exist_ok=True)
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    
    # Step 2: Load original model and perturbed model
    original_model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu"
    )
    perturbed_model = AutoModelForCausalLM.from_pretrained(
        temp_dir, torch_dtype=torch.float16, device_map="cpu"
    )
    
    # Step 3: Calculate actual deltas
    actual_deltas = {}
    for name, orig_param in original_model.named_parameters():
        if name in perturbed_model.state_dict():
            pert_param = perturbed_model.state_dict()[name]
            actual_deltas[name] = pert_param - orig_param
    
    # Step 4: Get reconstructed deltas
    reconstructed_deltas = reconstruct_perturbation(expert_id, config_file, base_model)
    
    # Step 5: Compare
    all_match = True
    max_diff = 0.0
    
    for name, actual_delta in actual_deltas.items():
        if name in reconstructed_deltas:
            reconstructed_delta = reconstructed_deltas[name]
            diff = torch.abs(actual_delta - reconstructed_delta).max().item()
            max_diff = max(max_diff, diff)
            
            if diff > tolerance:
                logger.error(f"Mismatch in {name}: max_diff={diff:.2e} > tolerance={tolerance:.2e}")
                all_match = False
            else:
                logger.debug(f"✅ {name}: max_diff={diff:.2e}")
    
    # Cleanup
    del original_model, perturbed_model
    
    if all_match:
        logger.info(f"✅ Perturbation consistency verified! Max difference: {max_diff:.2e}")
    else:
        logger.error(f"❌ Perturbation consistency check failed! Max difference: {max_diff:.2e}")
    
    return all_match


def reconstruct_all_perturbations(
    config_file: str,
    base_model: str,
    device: str = "cpu"
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Reconstruct perturbations for all experts in the configuration.
    
    Returns:
        Dictionary mapping expert_id to their parameter deltas
    """
    configs = load_config(config_file)
    
    # Find all expert IDs
    expert_ids = []
    for key in configs.keys():
        if key != "_meta" and key != "base_model" and key.isdigit():
            expert_ids.append(int(key))
    
    logger.info(f"Found {len(expert_ids)} experts to analyze: {expert_ids}")
    
    all_deltas = {}
    for expert_id in expert_ids:
        try:
            deltas = reconstruct_perturbation(expert_id, config_file, base_model, device)
            all_deltas[expert_id] = deltas
        except Exception as e:
            logger.error(f"Failed to reconstruct expert {expert_id}: {e}")
            
    return all_deltas


def perform_svd_decomposition(
    all_deltas: Dict[int, Dict[str, torch.Tensor]],
    decompose_by_layer: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Perform SVD decomposition on perturbations.
    
    Args:
        all_deltas: Dictionary of expert_id -> parameter deltas
        decompose_by_layer: If True, decompose each parameter separately.
                          If False, concatenate all parameters for global decomposition.
    
    Returns:
        Dictionary with SVD results for each parameter (or "global" if combined)
    """
    logger.info(f"Performing SVD decomposition (by_layer={decompose_by_layer})")
    
    if not all_deltas:
        return {}
        
    # Get common parameter names across all experts
    expert_ids = list(all_deltas.keys())
    common_params = set(all_deltas[expert_ids[0]].keys())
    for expert_id in expert_ids[1:]:
        common_params &= set(all_deltas[expert_id].keys())
    
    logger.info(f"Found {len(common_params)} common parameters across {len(expert_ids)} experts")
    
    svd_results = {}
    
    if decompose_by_layer:
        # Decompose each parameter separately
        for param_name in common_params:
            logger.info(f"Decomposing parameter: {param_name}")
            
            # Stack perturbations from all experts for this parameter
            param_deltas = []
            for expert_id in expert_ids:
                delta = all_deltas[expert_id][param_name]
                param_deltas.append(delta.flatten().cpu().numpy())
            
            # Shape: (n_experts, n_parameters)
            X = np.array(param_deltas)
            
            # Perform SVD
            try:
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                
                # Calculate explained variance ratio
                explained_var = s**2 / np.sum(s**2)
                
                svd_results[param_name] = {
                    'U': U,  # (n_experts, n_components)
                    'singular_values': s,  # (n_components,)
                    'Vt': Vt,  # (n_components, n_parameters)
                    'explained_variance_ratio': explained_var,
                    'shape': all_deltas[expert_ids[0]][param_name].shape,
                    'n_experts': len(expert_ids),
                    'n_components': len(s)
                }
                
                logger.info(f"  {param_name}: {X.shape} -> {len(s)} components, "
                          f"top 3 explained var: {explained_var[:3]}")
                
            except Exception as e:
                logger.error(f"SVD failed for {param_name}: {e}")
                
    else:
        # Global decomposition: concatenate all parameters
        logger.info("Performing global SVD decomposition")
        
        # Concatenate all parameters for each expert
        expert_vectors = []
        param_info = []
        
        for expert_id in expert_ids:
            expert_vector = []
            if expert_id == expert_ids[0]:  # Record parameter info only once
                start_idx = 0
                
            for param_name in sorted(common_params):  # Sort for consistency
                delta = all_deltas[expert_id][param_name].flatten().cpu().numpy()
                expert_vector.append(delta)
                
                if expert_id == expert_ids[0]:
                    param_info.append({
                        'name': param_name,
                        'start_idx': start_idx,
                        'end_idx': start_idx + len(delta),
                        'shape': all_deltas[expert_id][param_name].shape
                    })
                    start_idx += len(delta)
            
            expert_vectors.append(np.concatenate(expert_vector))
        
        # Shape: (n_experts, total_parameters)
        X = np.array(expert_vectors)
        logger.info(f"Global perturbation matrix shape: {X.shape}")
        
        # Perform SVD
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            explained_var = s**2 / np.sum(s**2)
            
            svd_results['global'] = {
                'U': U,
                'singular_values': s,
                'Vt': Vt,
                'explained_variance_ratio': explained_var,
                'param_info': param_info,
                'n_experts': len(expert_ids),
                'n_components': len(s),
                'total_parameters': X.shape[1]
            }
            
            logger.info(f"Global SVD: {X.shape} -> {len(s)} components, "
                      f"top 5 explained var: {explained_var[:5]}")
            
        except Exception as e:
            logger.error(f"Global SVD failed: {e}")
    
    return svd_results


def load_rewards_from_config(config_file: str) -> Dict[int, Dict[str, float]]:
    """
    Load reward scores for all experts from configuration file.
    
    Returns:
        Dictionary mapping expert_id to flattened reward scores
        Format: {expert_id: {'safety_dataset1_method1': score, 'utility_dataset2_method2': score, ...}}
    """
    configs = load_config(config_file)
    rewards = {}
    
    # Find all expert IDs (excluding base_model)
    expert_ids = []
    for key in configs.keys():
        if key != "_meta" and key != "base_model" and key.isdigit():
            expert_ids.append(int(key))
    
    # Also check for base_model if it has scores
    if "base_model" in configs and "scores" in configs["base_model"]:
        expert_ids.append("base_model")
    
    logger.info(f"Loading rewards for {len(expert_ids)} experts/models")
    
    for expert_id in expert_ids:
        key = str(expert_id) if expert_id != "base_model" else "base_model"
        expert_config = configs[key]
        
        if "scores" not in expert_config:
            logger.warning(f"No scores found for expert {expert_id}")
            continue
            
        scores = expert_config["scores"]
        flattened_rewards = {}
        
        # Flatten nested score structure: safety/utility -> dataset -> method -> score
        for category in ["safety", "utility"]:
            if category not in scores:
                continue
                
            for dataset_name, dataset_scores in scores[category].items():
                for method_name, method_result in dataset_scores.items():
                    if isinstance(method_result, dict) and "score" in method_result:
                        score = method_result["score"]
                        key_name = f"{category}_{dataset_name}_{method_name}"
                        flattened_rewards[key_name] = float(score)
        
        if flattened_rewards:
            rewards[expert_id] = flattened_rewards
            logger.info(f"Expert {expert_id}: loaded {len(flattened_rewards)} reward scores")
        else:
            logger.warning(f"Expert {expert_id}: no valid reward scores found")
    
    return rewards


# TODO: Placeholder for additional analysis functions
def placeholder_advanced_analysis():
    """
    Placeholder for advanced subspace analysis functions.
    
    Future implementations might include:
    - Hierarchical clustering of experts in subspace
    - Principal Component Analysis (PCA) with interpretation
    - Correlation analysis between different subspace dimensions
    - Visualization of expert trajectories in reduced subspace
    - Statistical significance testing for correlations
    """
    pass


def analyze_subspace_reward_correlation(
    svd_results: Dict[str, Dict[str, Any]],
    rewards: Dict[int, Dict[str, float]],
    top_k_components: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze correlation between subspace components and reward scores.
    
    Args:
        svd_results: Results from SVD decomposition
        rewards: Reward scores for each expert
        top_k_components: Number of top components to analyze
    
    Returns:
        Dictionary with correlation analysis results
    """
    logger.info(f"Analyzing subspace-reward correlations (top {top_k_components} components)")
    
    if not svd_results or not rewards:
        logger.warning("No SVD results or rewards available for correlation analysis")
        return {}
    
    correlation_results = {}
    
    # Get expert IDs that have both SVD results and rewards
    expert_ids = set()
    for expert_id in rewards.keys():
        if expert_id != "base_model":  # Skip base_model for SVD analysis
            expert_ids.add(expert_id)
    expert_ids = sorted(list(expert_ids))
    
    if len(expert_ids) < 3:
        logger.warning(f"Too few experts ({len(expert_ids)}) for meaningful correlation analysis")
        return {}
    
    logger.info(f"Analyzing correlations for {len(expert_ids)} experts")
    
    # Get common reward metrics across all experts
    common_rewards = set(rewards[expert_ids[0]].keys())
    for expert_id in expert_ids[1:]:
        if expert_id in rewards:
            common_rewards &= set(rewards[expert_id].keys())
    
    logger.info(f"Found {len(common_rewards)} common reward metrics")
    
    for param_or_global in svd_results.keys():
        logger.info(f"Analyzing {param_or_global}")
        
        svd_data = svd_results[param_or_global]
        U = svd_data['U']  # (n_experts, n_components)
        n_components = min(top_k_components, svd_data['n_components'])
        
        param_correlations = {}
        
        for reward_name in common_rewards:
            # Extract reward values for experts in the same order as SVD
            reward_values = []
            for expert_id in expert_ids:
                if expert_id in rewards:
                    reward_values.append(rewards[expert_id][reward_name])
                else:
                    reward_values.append(np.nan)
            
            reward_values = np.array(reward_values)
            
            # Skip if too many missing values
            if np.sum(np.isnan(reward_values)) > len(expert_ids) * 0.3:
                continue
            
            component_correlations = []
            
            # Analyze correlation with each component
            for comp_idx in range(n_components):
                component_scores = U[:, comp_idx]
                
                # Filter out NaN values
                valid_mask = ~np.isnan(reward_values)
                if np.sum(valid_mask) < 3:
                    continue
                
                comp_filtered = component_scores[valid_mask]
                reward_filtered = reward_values[valid_mask]
                
                # Calculate Pearson and Spearman correlations
                try:
                    if SCIPY_AVAILABLE:
                        pearson_r, pearson_p = pearsonr(comp_filtered, reward_filtered)
                        spearman_r, spearman_p = spearmanr(comp_filtered, reward_filtered)
                    else:
                        # Fallback to numpy correlation
                        pearson_r = np.corrcoef(comp_filtered, reward_filtered)[0, 1]
                        pearson_p = 0.0  # Not available without scipy
                        spearman_r = pearson_r  # Use Pearson as fallback
                        spearman_p = 0.0
                    
                    component_correlations.append({
                        'component': comp_idx,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': len(comp_filtered),
                        'explained_variance': svd_data['explained_variance_ratio'][comp_idx]
                    })
                    
                except Exception as e:
                    logger.warning(f"Correlation failed for {param_or_global} component {comp_idx}: {e}")
            
            param_correlations[reward_name] = component_correlations
        
        correlation_results[param_or_global] = param_correlations
    
    return correlation_results


def visualize_correlations(
    correlation_results: Dict[str, Dict[str, Any]],
    output_dir: str = "subspace_analysis_plots"
):
    """
    Create visualizations for subspace-reward correlations.
    
    Args:
        correlation_results: Results from analyze_subspace_reward_correlation
        output_dir: Directory to save plots
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization packages not available. Skipping plots.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    for param_name, param_corrs in correlation_results.items():
        if not param_corrs:
            continue
            
        # Create heatmap of correlations
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Prepare data for heatmaps
        reward_names = list(param_corrs.keys())
        n_components = max(len(param_corrs[reward]) for reward in reward_names)
        
        pearson_matrix = np.full((len(reward_names), n_components), np.nan)
        spearman_matrix = np.full((len(reward_names), n_components), np.nan)
        
        for i, reward_name in enumerate(reward_names):
            for corr_data in param_corrs[reward_name]:
                comp_idx = corr_data['component']
                pearson_matrix[i, comp_idx] = corr_data['pearson_r']
                spearman_matrix[i, comp_idx] = corr_data['spearman_r']
        
        # Plot Pearson correlations
        sns.heatmap(pearson_matrix, 
                   xticklabels=[f'PC{i+1}' for i in range(n_components)],
                   yticklabels=reward_names,
                   center=0, cmap='RdBu_r', 
                   annot=True, fmt='.3f',
                   ax=axes[0])
        axes[0].set_title(f'Pearson Correlations - {param_name}')
        
        # Plot Spearman correlations  
        sns.heatmap(spearman_matrix,
                   xticklabels=[f'PC{i+1}' for i in range(n_components)],
                   yticklabels=reward_names,
                   center=0, cmap='RdBu_r',
                   annot=True, fmt='.3f', 
                   ax=axes[1])
        axes[1].set_title(f'Spearman Correlations - {param_name}')
        
        plt.tight_layout()
        
        # Clean parameter name for filename
        safe_param_name = param_name.replace('/', '_').replace('.', '_')
        plt.savefig(os.path.join(output_dir, f'correlations_{safe_param_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Correlation plots saved to {output_dir}")


def main():
    """Main function for subspace analysis."""
    parser = argparse.ArgumentParser(description="Subspace Analysis for Safety Thickets")
    parser.add_argument("--config_file", type=str, default="outputs/Meta-Llama-3-8B-Instruct.json",
                        help="Path to expert configuration file")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Path to base model")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device to use (cpu/cuda)")
    parser.add_argument("--decompose_by_layer", action="store_true", default=True,
                        help="Decompose each layer separately (default: global decomposition)")
    parser.add_argument("--top_k_components", type=int, default=5,
                        help="Number of top components to analyze")
    parser.add_argument("--output_dir", type=str, default="results/subspace_analysis",
                        help="Output directory for results and plots")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save intermediate results to disk")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Starting subspace analysis...")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Device: {args.device}")
    
    # Step 1: Reconstruct all perturbations
    logger.info("Step 1: Reconstructing perturbations for all experts")
    all_deltas = reconstruct_all_perturbations(args.config_file, args.base_model, args.device)
    
    if not all_deltas:
        logger.error("No perturbations could be reconstructed. Exiting.")
        return
    
    # Step 2: Perform SVD decomposition
    logger.info("Step 2: Performing SVD decomposition") 
    svd_results = perform_svd_decomposition(all_deltas, args.decompose_by_layer)
    
    if not svd_results:
        logger.error("SVD decomposition failed. Exiting.")
        return
    
    # Step 3: Load rewards
    logger.info("Step 3: Loading reward scores from configuration")
    rewards = load_rewards_from_config(args.config_file)
    
    if not rewards:
        logger.error("No rewards could be loaded. Exiting.")
        return
    
    # Step 4: Analyze correlations
    logger.info("Step 4: Analyzing subspace-reward correlations")
    correlation_results = analyze_subspace_reward_correlation(
        svd_results, rewards, args.top_k_components
    )
    
    # Step 5: Create visualizations
    if correlation_results:
        logger.info("Step 5: Creating visualizations")
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_correlations(correlation_results, 
                             os.path.join(args.output_dir, "plots"))
        
        # Save detailed results
        if args.save_results:
            results_file = os.path.join(args.output_dir, "correlation_results.json")
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for param_name, param_corrs in correlation_results.items():
                json_results[param_name] = param_corrs
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to {results_file}")
    
    logger.info("Subspace analysis completed!")


if __name__ == "__main__":
    main()