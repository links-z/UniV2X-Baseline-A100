#!/usr/bin/env python3
"""
G4-0: Offline Learned Fusion Reconstruction

This script implements the complete G4 evaluation pipeline:
1. Load G3 best model and predict on all candidate patches
2. Scan thresholds on Validation set to find optimal tau*
3. Reconstruct Learned Occupancy for Val/Test
4. Compare four strategies: Ego, Official OR, Learned, Oracle

Key equation:
    s_R = sigma(f_theta(X_R))
    a_R = 1[s_R > tau]
    O_R^learned = O_v,R OR (a_R AND O_i,R)

Only controls Infrastructure-added occupancy, preserves Ego occupancy.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import time

# Add G3 to path for model import
sys.path.insert(0, '/root/autodl-tmp/UniV2X/G0-G1-G2/G3')
from train_learnability_neighborhood_soft import NeighborhoodMLP

# Paths
G0_CACHE_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full')
G3_MODEL_PATH = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth')
G3_DATA_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G3/soft')
G4_RESULTS_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/results')
SAMPLE_IDX_MAPPING_PATH = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/sample_idx_to_export_idx.json')

# Load sample_idx to export_idx mapping
with open(SAMPLE_IDX_MAPPING_PATH) as f:
    SAMPLE_IDX_TO_EXPORT_IDX = json.load(f)

# Constants
PATCH_SIZE = 4
PATCH_GRID = 50  # 200x200 -> 50x50 patches
GRID_SIZE = 200
THRESHOLD_OCC = 0.1  # For binarizing Pv/Pi to Ov/Oi
HORIZONS = 5

def load_g3_model(model_path):
    """Load G3 best model."""
    print(f"Loading G3 model from {model_path}")

    model = NeighborhoodMLP()
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"  ✓ Model loaded")
    print(f"  - Best epoch: {ckpt['epoch']}")
    print(f"  - Val AUPRC: {ckpt['val_auprc']:.6f}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")

    return model, ckpt

def load_g3_dataset(split):
    """Load G3 dataset for a split."""
    path = G3_DATA_DIR / f'g3_neighborhood_{split}.npz'
    print(f"Loading G3 {split} dataset from {path}")

    data = np.load(path, allow_pickle=True)
    features = data['features']  # [N, 127]
    labels = data['labels']      # [N]
    metadata = data['metadata']  # [N] dict with sample_id, horizon, patch_row, patch_col

    print(f"  ✓ Loaded {len(labels)} samples")
    print(f"  - Positive rate: {labels.mean()*100:.2f}%")

    return features, labels, metadata

def predict_on_dataset(model, features, batch_size=2048):
    """
    Predict scores s_R = sigma(f_theta(X_R)) for all candidate patches.

    Returns:
        scores: [N] array of acceptance scores in [0, 1]
    """
    print(f"Predicting on {len(features)} samples (batch_size={batch_size})")

    model.eval()
    scores = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[i:i+batch_size]).float()
            logits = model(batch)
            probs = torch.sigmoid(logits).numpy().flatten()
            scores.append(probs)

    scores = np.concatenate(scores)

    print(f"  ✓ Prediction complete")
    print(f"  - Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  - Score mean: {scores.mean():.4f}")

    return scores

def load_cache(sample_idx):
    """Load G0 cache for a sample using sample_idx format (e.g., '002889')."""
    # Map sample_idx to export_idx
    if sample_idx not in SAMPLE_IDX_TO_EXPORT_IDX:
        raise FileNotFoundError(f"sample_idx {sample_idx} not in mapping")

    export_idx = SAMPLE_IDX_TO_EXPORT_IDX[sample_idx]
    cache_path = G0_CACHE_DIR / f'sample_{export_idx:05d}.npz'

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)

    # Extract occupancy grids
    pv = data['Pv']  # [H, 200, 200]
    pi = data['Pi_aligned']  # [H, 200, 200]

    # Squeeze if needed
    if pv.ndim == 4:
        pv = pv.squeeze(1)
    if pi.ndim == 4:
        pi = pi.squeeze(1)

    return pv, pi

def reconstruct_learned_occupancy(pv, pi, accept_decisions, metadata_subset):
    """
    Reconstruct O^learned = O_v OR (a_R AND O_i)

    Args:
        pv: [H, 200, 200] Ego probability
        pi: [H, 200, 200] Infra probability
        accept_decisions: [N] binary acceptance for each candidate patch
        metadata_subset: [N] metadata for this sample

    Returns:
        o_learned: [H, 200, 200] Learned occupancy (binary)
    """
    # Binarize to get base occupancies
    ov = (pv > THRESHOLD_OCC).astype(np.float32)  # [H, 200, 200]
    oi = (pi > THRESHOLD_OCC).astype(np.float32)  # [H, 200, 200]

    # Start with Ego occupancy
    o_learned = ov.copy()

    # For each accepted candidate patch, OR with Infrastructure
    for accept, meta in zip(accept_decisions, metadata_subset):
        if accept:
            h = meta['horizon']
            r = meta['patch_row']
            c = meta['patch_col']

            y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
            y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE

            # O_learned = O_v OR (a_R AND O_i)
            # Where a_R=1 here, so: O_learned = O_v OR O_i (for this patch)
            o_learned[h, y0:y1, x0:x1] = np.logical_or(
                ov[h, y0:y1, x0:x1],
                oi[h, y0:y1, x0:x1]
            ).astype(np.float32)

    return o_learned

def reconstruct_oracle_occupancy(pv, pi, labels, metadata_subset):
    """Reconstruct P4 Oracle occupancy (accept only if U_R > 0)."""
    ov = (pv > THRESHOLD_OCC).astype(np.float32)
    oi = (pi > THRESHOLD_OCC).astype(np.float32)

    o_oracle = ov.copy()

    for label, meta in zip(labels, metadata_subset):
        if label == 1:  # U_R > 0 (beneficial)
            h = meta['horizon']
            r = meta['patch_row']
            c = meta['patch_col']

            y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
            y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE

            o_oracle[h, y0:y1, x0:x1] = np.logical_or(
                ov[h, y0:y1, x0:x1],
                oi[h, y0:y1, x0:x1]
            ).astype(np.float32)

    return o_oracle

def compute_occupancy_metrics(pred, gt):
    """
    Compute IoU, Precision, Recall, F1 for occupancy grids.

    Args:
        pred: [H, 200, 200] predicted occupancy (binary)
        gt: [H, 200, 200] ground truth occupancy (binary)

    Returns:
        dict with iou, precision, recall, f1
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_pos = pred.sum()
    gt_pos = gt.sum()

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_pos if pred_pos > 0 else 0.0
    recall = intersection / gt_pos if gt_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'intersection': int(intersection),
        'union': int(union),
        'pred_positive': int(pred_pos),
        'gt_positive': int(gt_pos)
    }

def evaluate_threshold_on_split(model, split, threshold, device='cpu'):
    """
    Evaluate a single threshold on a split.

    Returns:
        metrics: dict with aggregated IoU, Precision, Recall, F1
        accept_rate: fraction of patches accepted
    """
    # Load G3 dataset
    features, labels, metadata = load_g3_dataset(split)

    # Predict scores
    scores = predict_on_dataset(model, features)

    # Apply threshold to get accept decisions
    accept_decisions = (scores > threshold).astype(int)
    accept_rate = accept_decisions.mean()

    # Group by sample_idx (not sample_id)
    sample_groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        sid = meta['sample_idx']
        sample_groups[sid].append(i)

    # Evaluate each sample
    all_metrics = []
    unique_samples = sorted(sample_groups.keys())

    print(f"  Evaluating {len(unique_samples)} unique samples...")

    for sid in unique_samples:
        indices = sample_groups[sid]

        # Load cache
        try:
            pv, pi = load_cache(sid)
        except FileNotFoundError:
            print(f"    Warning: Cache not found for {sid}, skipping")
            continue

        # Load GT occupancy from cache
        export_idx = SAMPLE_IDX_TO_EXPORT_IDX[sid]
        cache_path = G0_CACHE_DIR / f'sample_{export_idx:05d}.npz'
        cache_data = np.load(cache_path, allow_pickle=True)
        o_gt = cache_data['GT'].astype(np.float32)  # [H, 200, 200]

        # Reconstruct learned occupancy for this sample
        meta_subset = [metadata[i] for i in indices]
        accept_subset = accept_decisions[indices]

        o_learned = reconstruct_learned_occupancy(pv, pi, accept_subset, meta_subset)

        # Compute metrics
        metrics = compute_occupancy_metrics(o_learned, o_gt)
        all_metrics.append(metrics)

    # Aggregate metrics
    if len(all_metrics) == 0:
        return None, accept_rate

    aggregated = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'n_samples': len(all_metrics)
    }

    return aggregated, accept_rate

def evaluate_all_strategies_on_split(model, split, threshold, device='cpu'):
    """
    G4-2: Evaluate all four strategies on a split.

    Strategies:
      1. Ego: Only O_v
      2. Official OR: O_v OR O_i
      3. Learned: O_v OR (a_R AND O_i) with learned predictor
      4. Oracle: O_v OR (U_R>0 AND O_i) with GT labels

    Returns:
        strategy_results: dict mapping strategy_name -> metrics
    """
    print(f"\nEvaluating all strategies on {split} (threshold={threshold:.2f})")

    # Load G3 dataset
    features, labels, metadata = load_g3_dataset(split)

    # Predict scores
    scores = predict_on_dataset(model, features)

    # Apply threshold to get accept decisions
    accept_decisions = (scores > threshold).astype(int)

    # Group by sample_idx (not sample_id)
    sample_groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        sid = meta['sample_idx']
        sample_groups[sid].append(i)

    # Track metrics for each strategy
    strategy_metrics = {
        'ego': [],
        'official': [],
        'learned': [],
        'oracle': []
    }

    unique_samples = sorted(sample_groups.keys())
    print(f"  Processing {len(unique_samples)} unique samples...")

    for sid in unique_samples:
        indices = sample_groups[sid]

        # Load cache
        try:
            pv, pi = load_cache(sid)
            export_idx = SAMPLE_IDX_TO_EXPORT_IDX[sid]
            cache_path = G0_CACHE_DIR / f'sample_{export_idx:05d}.npz'
            cache_data = np.load(cache_path, allow_pickle=True)
            o_gt = cache_data['GT'].astype(np.float32)
        except (FileNotFoundError, KeyError) as e:
            print(f"    Warning: Could not load {sid}: {e}")
            continue

        # Get decisions for this sample
        meta_subset = [metadata[i] for i in indices]
        accept_subset = accept_decisions[indices]
        labels_subset = labels[indices]

        # Binarize
        ov = (pv > THRESHOLD_OCC).astype(np.float32)
        oi = (pi > THRESHOLD_OCC).astype(np.float32)

        # Strategy 1: Ego only
        o_ego = ov.copy()
        strategy_metrics['ego'].append(compute_occupancy_metrics(o_ego, o_gt))

        # Strategy 2: Official OR
        o_official = np.logical_or(ov, oi).astype(np.float32)
        strategy_metrics['official'].append(compute_occupancy_metrics(o_official, o_gt))

        # Strategy 3: Learned
        o_learned = reconstruct_learned_occupancy(pv, pi, accept_subset, meta_subset)
        strategy_metrics['learned'].append(compute_occupancy_metrics(o_learned, o_gt))

        # Strategy 4: Oracle
        o_oracle = reconstruct_oracle_occupancy(pv, pi, labels_subset, meta_subset)
        strategy_metrics['oracle'].append(compute_occupancy_metrics(o_oracle, o_gt))

    # Aggregate results
    strategy_results = {}
    for strategy, metrics_list in strategy_metrics.items():
        if len(metrics_list) == 0:
            continue

        strategy_results[strategy] = {
            'iou': np.mean([m['iou'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
            'n_samples': len(metrics_list)
        }

    return strategy_results

def scan_thresholds_on_val(model):
    """
    G4-1: Scan thresholds on Validation set to find optimal tau*.

    Returns:
        best_threshold: tau* that maximizes Val IoU
        threshold_results: dict mapping tau -> metrics
    """
    print("\n" + "="*80)
    print("G4-1: Threshold Scanning on Validation Set")
    print("="*80)

    thresholds = np.arange(0.05, 1.0, 0.05)
    threshold_results = {}

    for tau in thresholds:
        print(f"\nEvaluating threshold τ = {tau:.2f}")
        metrics, accept_rate = evaluate_threshold_on_split(model, 'val', tau)

        if metrics is None:
            print(f"  ✗ No valid samples")
            continue

        threshold_results[float(tau)] = {
            'metrics': metrics,
            'accept_rate': float(accept_rate)
        }

        print(f"  ✓ Val IoU: {metrics['iou']:.4f} | Accept rate: {accept_rate:.2%}")

    # Find best threshold by Val IoU
    best_tau = max(threshold_results.keys(),
                   key=lambda t: threshold_results[t]['metrics']['iou'])
    best_metrics = threshold_results[best_tau]['metrics']

    print(f"\n{'='*80}")
    print(f"✓ Best threshold: τ* = {best_tau:.2f}")
    print(f"  - Val IoU: {best_metrics['iou']:.4f}")
    print(f"  - Val Precision: {best_metrics['precision']:.4f}")
    print(f"  - Val Recall: {best_metrics['recall']:.4f}")
    print(f"  - Accept rate: {threshold_results[best_tau]['accept_rate']:.2%}")
    print(f"{'='*80}\n")

    return best_tau, threshold_results

def main():
    """Main G4 evaluation pipeline."""
    print("="*80)
    print("G4-0: Offline Learned Fusion Reconstruction")
    print("="*80)
    print()

    start_time = time.time()

    # Load G3 model
    model, ckpt = load_g3_model(G3_MODEL_PATH)

    # G4-1: Scan thresholds on Validation
    best_threshold, val_threshold_results = scan_thresholds_on_val(model)

    # Save threshold scanning results
    val_results_path = G4_RESULTS_DIR / 'g4_threshold_scan_val.json'
    with open(val_results_path, 'w') as f:
        json.dump({
            'best_threshold': best_threshold,
            'threshold_results': val_threshold_results
        }, f, indent=2)
    print(f"✓ Saved threshold scan results to {val_results_path}")

    # G4-2: Evaluate all strategies on Validation (with best threshold)
    print("\n" + "="*80)
    print("G4-2a: Strategy Comparison on Validation Set")
    print("="*80)
    val_strategy_results = evaluate_all_strategies_on_split(model, 'val', best_threshold)

    print(f"\nValidation Results (τ* = {best_threshold:.2f}):")
    print(f"{'Strategy':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for strategy in ['ego', 'official', 'learned', 'oracle']:
        if strategy in val_strategy_results:
            r = val_strategy_results[strategy]
            print(f"{strategy.capitalize():<15} {r['iou']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")

    # G4-2: Evaluate all strategies on Test (with frozen best threshold)
    print("\n" + "="*80)
    print("G4-2b: Strategy Comparison on Test Set (FROZEN THRESHOLD)")
    print("="*80)
    test_strategy_results = evaluate_all_strategies_on_split(model, 'test', best_threshold)

    print(f"\nTest Results (τ* = {best_threshold:.2f}, frozen from Val):")
    print(f"{'Strategy':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for strategy in ['ego', 'official', 'learned', 'oracle']:
        if strategy in test_strategy_results:
            r = test_strategy_results[strategy]
            print(f"{strategy.capitalize():<15} {r['iou']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")

    # Calculate recovery metrics
    if 'learned' in test_strategy_results and 'official' in test_strategy_results and 'oracle' in test_strategy_results:
        iou_learned = test_strategy_results['learned']['iou']
        iou_official = test_strategy_results['official']['iou']
        iou_oracle = test_strategy_results['oracle']['iou']
        iou_ego = test_strategy_results['ego']['iou']

        oracle_gap = iou_oracle - iou_official
        learned_gain = iou_learned - iou_official
        recovery = learned_gain / oracle_gap if oracle_gap > 0 else 0.0

        print(f"\n{'='*80}")
        print("Go/No-Go Analysis:")
        print(f"{'='*80}")
        print(f"IoU_Ego:      {iou_ego:.4f}")
        print(f"IoU_Official: {iou_official:.4f}")
        print(f"IoU_Learned:  {iou_learned:.4f}")
        print(f"IoU_Oracle:   {iou_oracle:.4f}")
        print()
        print(f"Learned vs Official: {'+' if learned_gain > 0 else ''}{learned_gain:.4f}")
        print(f"Learned vs Ego:      {'+' if iou_learned > iou_ego else ''}{iou_learned - iou_ego:.4f}")
        print(f"Oracle Gap Recovery: {recovery*100:.1f}%")
        print()

        # Go/No-Go verdict
        go_criterion_1 = iou_learned > iou_official
        go_criterion_2 = iou_learned > iou_ego

        if go_criterion_1 and go_criterion_2:
            verdict = "✅ GO - Learned Fusion beats both Official OR and Ego"
        elif go_criterion_1:
            verdict = "⚠️  CONDITIONAL GO - Learned beats Official but not Ego"
        else:
            verdict = "❌ NO-GO - Learned does not improve over Official OR"

        print(f"Verdict: {verdict}")
        print(f"{'='*80}")

    # Save all results
    final_results = {
        'best_threshold': best_threshold,
        'val_strategy_results': val_strategy_results,
        'test_strategy_results': test_strategy_results,
        'recovery_metrics': {
            'oracle_gap_recovery': float(recovery) if 'recovery' in locals() else None,
            'learned_vs_official': float(learned_gain) if 'learned_gain' in locals() else None,
            'learned_vs_ego': float(iou_learned - iou_ego) if 'iou_learned' in locals() and 'iou_ego' in locals() else None
        } if 'learned' in test_strategy_results else None
    }

    final_results_path = G4_RESULTS_DIR / 'g4_final_results.json'
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✓ Saved final results to {final_results_path}")

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"G4 Evaluation Complete")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
