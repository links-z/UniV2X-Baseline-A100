"""
G4-3: Mechanism Analysis and Simple Baselines

This script extends G4 evaluation with:
1. Accept rate breakdown by horizon (t_0 to t_4)
2. BA retention rate and HA admission rate for Learned predictor
3. Simple confidence baseline: Accept if P_i - P_v > delta
4. Comparison: Learned predictor vs confidence heuristic

Goal: Answer "Why Value Learning vs simple probability threshold?"
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Add G3 to path
sys.path.insert(0, '/root/autodl-tmp/UniV2X/G0-G1-G2/G3')
from train_learnability_neighborhood_soft import NeighborhoodMLP

# Paths
G0_CACHE_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full')
G3_MODEL_PATH = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth')
G3_DATA_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G3/soft')
G4_RESULTS_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/results')
SAMPLE_IDX_MAPPING_PATH = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/sample_idx_to_export_idx.json')

# Load mapping
with open(SAMPLE_IDX_MAPPING_PATH) as f:
    SAMPLE_IDX_TO_EXPORT_IDX = json.load(f)

# Constants
PATCH_SIZE = 4
THRESHOLD_OCC = 0.1
BEST_THRESHOLD = 0.70  # From G4-1

def load_g3_model(model_path):
    """Load G3 best model."""
    model = NeighborhoodMLP()
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt

def load_g3_dataset(split):
    """Load G3 dataset."""
    path = G3_DATA_DIR / f'g3_neighborhood_{split}.npz'
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels'], data['metadata']

def predict_scores(model, features, batch_size=2048):
    """Predict acceptance scores."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[i:i+batch_size]).float()
            logits = model(batch)
            probs = torch.sigmoid(logits).numpy().flatten()
            scores.append(probs)
    return np.concatenate(scores)

def load_cache(sample_idx):
    """Load G0 cache."""
    export_idx = SAMPLE_IDX_TO_EXPORT_IDX[sample_idx]
    cache_path = G0_CACHE_DIR / f'sample_{export_idx:05d}.npz'
    data = np.load(cache_path, allow_pickle=True)
    pv = data['Pv']
    pi = data['Pi_aligned']
    if pv.ndim == 4:
        pv = pv.squeeze(1)
    if pi.ndim == 4:
        pi = pi.squeeze(1)
    return pv, pi

def analyze_learned_predictor_decisions(model, split, threshold):
    """
    Analyze Learned predictor decisions on a split.

    Returns:
        - Per-horizon accept rates
        - BA retention rate (% of beneficial patches accepted)
        - HA admission rate (% of harmful patches accepted)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Learned Predictor on {split.upper()} (threshold={threshold:.2f})")
    print(f"{'='*80}\n")

    # Load data
    features, labels, metadata = load_g3_dataset(split)
    scores = predict_scores(model, features)
    accept_decisions = (scores > threshold).astype(int)

    # Overall statistics
    n_total = len(labels)
    n_beneficial = (labels == 1).sum()
    n_harmful = (labels == 0).sum()
    n_accepted = accept_decisions.sum()

    print(f"Overall Statistics:")
    print(f"  Total patches: {n_total}")
    print(f"  Beneficial (U>0): {n_beneficial} ({n_beneficial/n_total*100:.2f}%)")
    print(f"  Harmful (U≤0): {n_harmful} ({n_harmful/n_total*100:.2f}%)")
    print(f"  Accepted by predictor: {n_accepted} ({n_accepted/n_total*100:.2f}%)")

    # BA retention and HA admission
    beneficial_mask = (labels == 1)
    harmful_mask = (labels == 0)

    ba_accepted = (accept_decisions[beneficial_mask] == 1).sum()
    ha_accepted = (accept_decisions[harmful_mask] == 1).sum()

    ba_retention = ba_accepted / n_beneficial if n_beneficial > 0 else 0.0
    ha_admission = ha_accepted / n_harmful if n_harmful > 0 else 0.0

    print(f"\nDecision Quality:")
    print(f"  BA Retention (recall on beneficial): {ba_retention:.2%}")
    print(f"    → {ba_accepted}/{n_beneficial} beneficial patches accepted")
    print(f"  HA Admission (FPR on harmful): {ha_admission:.2%}")
    print(f"    → {ha_accepted}/{n_harmful} harmful patches leaked")

    # Per-horizon breakdown
    print(f"\nPer-Horizon Breakdown:")
    print(f"{'Horizon':<10} {'Total':<8} {'Beneficial':<12} {'Harmful':<10} {'Accept Rate':<12} {'BA Retention':<14} {'HA Admission'}")
    print("-" * 100)

    horizon_stats = {}
    for h in range(5):
        h_mask = np.array([m['horizon'] == h for m in metadata])
        h_total = h_mask.sum()
        h_beneficial = (labels[h_mask] == 1).sum()
        h_harmful = (labels[h_mask] == 0).sum()
        h_accepted = accept_decisions[h_mask].sum()
        h_accept_rate = h_accepted / h_total if h_total > 0 else 0.0

        h_ba_accepted = (accept_decisions[h_mask & beneficial_mask] == 1).sum()
        h_ha_accepted = (accept_decisions[h_mask & harmful_mask] == 1).sum()

        h_ba_retention = h_ba_accepted / h_beneficial if h_beneficial > 0 else 0.0
        h_ha_admission = h_ha_accepted / h_harmful if h_harmful > 0 else 0.0

        print(f"t={h} ({h*0.5:.1f}s) {h_total:<8} {h_beneficial:<12} {h_harmful:<10} "
              f"{h_accept_rate:<12.2%} {h_ba_retention:<14.2%} {h_ha_admission:.2%}")

        horizon_stats[h] = {
            'total': int(h_total),
            'beneficial': int(h_beneficial),
            'harmful': int(h_harmful),
            'accepted': int(h_accepted),
            'accept_rate': float(h_accept_rate),
            'ba_retention': float(h_ba_retention),
            'ha_admission': float(h_ha_admission)
        }

    return {
        'overall': {
            'n_total': int(n_total),
            'n_beneficial': int(n_beneficial),
            'n_harmful': int(n_harmful),
            'n_accepted': int(n_accepted),
            'accept_rate': float(n_accepted / n_total),
            'ba_retention': float(ba_retention),
            'ha_admission': float(ha_admission)
        },
        'per_horizon': horizon_stats
    }

def evaluate_confidence_baseline(split, delta_values, threshold_occ=0.1):
    """
    Evaluate simple confidence baseline: Accept if P_i - P_v > delta

    Returns dict mapping delta -> metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Confidence Baseline on {split.upper()}")
    print(f"{'='*80}\n")
    print(f"Rule: Accept patch if P_i - P_v > δ (where P_i, P_v are mean probabilities in patch)")

    # Load G3 dataset for labels and metadata
    _, labels, metadata = load_g3_dataset(split)

    # Group by sample
    sample_groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        sample_groups[meta['sample_idx']].append(i)

    # For each sample, extract P_i - P_v for all patches
    patch_confidence_diffs = []
    patch_labels = []
    patch_metadata = []

    print(f"Computing confidence differences for {len(sample_groups)} samples...")

    for sid in sorted(sample_groups.keys()):
        indices = sample_groups[sid]
        try:
            pv, pi = load_cache(sid)
        except (FileNotFoundError, KeyError):
            continue

        for idx in indices:
            meta = metadata[idx]
            h = meta['horizon']
            r = meta['patch_row']
            c = meta['patch_col']

            y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
            y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE

            pv_patch = pv[h, y0:y1, x0:x1]
            pi_patch = pi[h, y0:y1, x0:x1]

            # Mean probabilities in patch
            pv_mean = pv_patch.mean()
            pi_mean = pi_patch.mean()
            confidence_diff = pi_mean - pv_mean

            patch_confidence_diffs.append(confidence_diff)
            patch_labels.append(labels[idx])
            patch_metadata.append(meta)

    patch_confidence_diffs = np.array(patch_confidence_diffs)
    patch_labels = np.array(patch_labels)

    print(f"  ✓ Processed {len(patch_confidence_diffs)} patches")
    print(f"  - Confidence diff range: [{patch_confidence_diffs.min():.4f}, {patch_confidence_diffs.max():.4f}]")
    print(f"  - Confidence diff mean: {patch_confidence_diffs.mean():.4f}")

    # Evaluate each delta
    results = {}

    print(f"\nScanning δ values:")
    print(f"{'δ':<8} {'Accept Rate':<12} {'BA Retention':<14} {'HA Admission':<14} {'Precision':<12} {'Recall'}")
    print("-" * 80)

    for delta in delta_values:
        accept_decisions = (patch_confidence_diffs > delta).astype(int)

        n_total = len(patch_labels)
        n_beneficial = (patch_labels == 1).sum()
        n_harmful = (patch_labels == 0).sum()
        n_accepted = accept_decisions.sum()

        if n_accepted == 0:
            continue

        ba_accepted = (accept_decisions[patch_labels == 1] == 1).sum()
        ha_accepted = (accept_decisions[patch_labels == 0] == 1).sum()

        accept_rate = n_accepted / n_total
        ba_retention = ba_accepted / n_beneficial if n_beneficial > 0 else 0.0
        ha_admission = ha_accepted / n_harmful if n_harmful > 0 else 0.0
        precision = ba_accepted / n_accepted if n_accepted > 0 else 0.0
        recall = ba_retention

        print(f"{delta:<8.3f} {accept_rate:<12.2%} {ba_retention:<14.2%} {ha_admission:<14.2%} {precision:<12.2%} {recall:.2%}")

        results[float(delta)] = {
            'accept_rate': float(accept_rate),
            'ba_retention': float(ba_retention),
            'ha_admission': float(ha_admission),
            'precision': float(precision),
            'recall': float(recall),
            'n_accepted': int(n_accepted)
        }

    return results

def main():
    """G4-3: Mechanism Analysis and Simple Baselines."""
    print("="*80)
    print("G4-3: Mechanism Analysis and Simple Baselines")
    print("="*80)

    # Load G3 model
    print("\nLoading G3 model...")
    model, ckpt = load_g3_model(G3_MODEL_PATH)
    print(f"  ✓ Loaded (epoch {ckpt['epoch']}, Val AUPRC {ckpt['val_auprc']:.4f})")

    # Part 1: Analyze Learned predictor
    val_learned_analysis = analyze_learned_predictor_decisions(model, 'val', BEST_THRESHOLD)
    test_learned_analysis = analyze_learned_predictor_decisions(model, 'test', BEST_THRESHOLD)

    # Part 2: Evaluate confidence baseline
    delta_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                    0.12, 0.15, 0.20, 0.25, 0.30]

    val_confidence_results = evaluate_confidence_baseline('val', delta_values)
    test_confidence_results = evaluate_confidence_baseline('test', delta_values)

    # Save results
    output = {
        'learned_predictor': {
            'threshold': BEST_THRESHOLD,
            'val': val_learned_analysis,
            'test': test_learned_analysis
        },
        'confidence_baseline': {
            'val': val_confidence_results,
            'test': test_confidence_results
        }
    }

    output_path = G4_RESULTS_DIR / 'g4_mechanism_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Mechanism analysis complete")
    print(f"✓ Results saved to {output_path}")
    print(f"{'='*80}")

    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print("Test Set Performance:")
    print(f"\nLearned Predictor (τ={BEST_THRESHOLD}):")
    print(f"  Accept Rate: {test_learned_analysis['overall']['accept_rate']:.2%}")
    print(f"  BA Retention: {test_learned_analysis['overall']['ba_retention']:.2%}")
    print(f"  HA Admission: {test_learned_analysis['overall']['ha_admission']:.2%}")

    # Find best confidence baseline by BA retention
    best_conf_delta = max(test_confidence_results.keys(),
                         key=lambda d: test_confidence_results[d]['ba_retention'])
    best_conf = test_confidence_results[best_conf_delta]

    print(f"\nBest Confidence Baseline (δ={best_conf_delta}):")
    print(f"  Accept Rate: {best_conf['accept_rate']:.2%}")
    print(f"  BA Retention: {best_conf['ba_retention']:.2%}")
    print(f"  HA Admission: {best_conf['ha_admission']:.2%}")

    # Comparison
    learned_ba = test_learned_analysis['overall']['ba_retention']
    learned_ha = test_learned_analysis['overall']['ha_admission']
    conf_ba = best_conf['ba_retention']
    conf_ha = best_conf['ha_admission']

    print(f"\nLearned vs Best Confidence:")
    print(f"  BA Retention: {learned_ba:.2%} vs {conf_ba:.2%} "
          f"({'better' if learned_ba > conf_ba else 'worse'})")
    print(f"  HA Admission: {learned_ha:.2%} vs {conf_ha:.2%} "
          f"({'worse' if learned_ha > conf_ha else 'better'})")

    if learned_ba > conf_ba and learned_ha < conf_ha:
        print(f"\n✅ Learned predictor dominates: Higher BA retention AND lower HA admission")
    elif learned_ba > conf_ba:
        print(f"\n✅ Learned predictor advantage: Higher BA retention")
    else:
        print(f"\n⚠️  Confidence baseline competitive or better")

if __name__ == '__main__':
    main()
