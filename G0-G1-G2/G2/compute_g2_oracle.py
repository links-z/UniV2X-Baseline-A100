#!/usr/bin/env python3
"""
G2-B: Patch Accept / Fallback Oracle Analysis

比较不同融合策略的 Occupancy 性能：
1. Ego: 只用车端预测
2. Official OR: 当前 UniV2X 方案 (Ov OR Oi)
3. Cell Oracle: 逐格子最优选择 (Ov OR BA)
4. Patch-4 Oracle: 4×4 patch Accept/Fallback
5. Patch-8 Oracle: 8×8 patch Accept/Fallback
6. Patch-16 Oracle: 16×16 patch Accept/Fallback (padding to 208×208)

Oracle 决策规则：
对于每个 patch R:
  U_R = BA_R - HA_R
  if U_R > 0: Accept (添加路侧新增)
  else: Fallback to Ego (保留车端预测)

目标：量化选择性融合的理论收益空间
"""

import os
import glob
import numpy as np
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
CACHE_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full"
OUT_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/G2"

BEV_H = 200
BEV_W = 200

# Patch sizes to evaluate
PATCH_SIZES = [4, 8, 16]

print("=" * 80)
print("G2-B: Patch Accept / Fallback Oracle Analysis")
print("=" * 80)
print()

# ============================================================
# Step 1: 加载所有样本
# ============================================================
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

if len(files) == 0:
    raise RuntimeError(f"No cache files found in {CACHE_DIR}")

print(f"Total cache files: {len(files)}")
print()

# ============================================================
# Step 2: 筛选有效样本
# ============================================================
valid_samples = []
invalid_samples = []

for path in files:
    with np.load(path) as d:
        future_valid = d["future_valid_mask"]

        if future_valid.all():
            valid_samples.append(path)
        else:
            invalid_samples.append(path)

print(f"Valid samples: {len(valid_samples)}")
print(f"Invalid samples: {len(invalid_samples)}")
print()

# ============================================================
# Step 3: 初始化统计
# ============================================================
# Horizon-wise statistics
horizon_stats = {
    "ego": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    "official": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    "cell_oracle": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
}

for ps in PATCH_SIZES:
    horizon_stats[f"patch_{ps}"] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

# Oracle 决策统计
oracle_decision_stats = {}
for ps in PATCH_SIZES:
    oracle_decision_stats[f"patch_{ps}"] = {
        "patch_with_add": 0,
        "accepted_patch": 0,
        "rejected_patch": 0,
        "accepted_ADD": 0,
        "accepted_BA": 0,
        "accepted_HA": 0,
        "rejected_ADD": 0,
        "rejected_BA": 0,
        "rejected_HA": 0,
    }

# Sample-wise statistics
sample_stats = []

# G1 consistency check
total_add = 0
total_ha = 0
total_ba = 0

# ============================================================
# Step 4: Oracle 函数定义
# ============================================================
def compute_cell_oracle(ov, oi, gt, valid_mask):
    """
    Cell Oracle: Ov OR BA

    只在新增区域接受 GT=1 的格子，拒绝 GT=0 的格子
    """
    add_mask = (ov == 0) & (oi == 1) & valid_mask
    ba_mask = add_mask & (gt == 1)

    # Start from Ego
    oracle = ov.copy()

    # Accept BA (beneficial additions)
    oracle[ba_mask] = 1

    # Sanity check: never suppress Ego
    assert np.all(oracle[ov == 1] == 1), "Cell Oracle suppressed Ego occupied cells"

    return oracle


def compute_patch_oracle(ov, oi, gt, valid_mask, patch_size):
    """
    Patch Oracle: Accept/Fallback based on U_R = BA_R - HA_R

    对于每个 patch:
      - 计算 BA_R (beneficial additions)
      - 计算 HA_R (harmful additions)
      - if BA_R > HA_R: Accept (添加该 patch 的所有路侧新增)
      - else: Fallback to Ego (保留车端预测)
    """
    h, w = ov.shape

    # Padding to ensure patch_size divides dimensions
    if patch_size == 16:
        # 200 -> 208 (13 patches of size 16)
        pad_h = 208 - h
        pad_w = 208 - w

        ov_padded = np.pad(ov, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        oi_padded = np.pad(oi, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        gt_padded = np.pad(gt, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        valid_padded = np.pad(valid_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)

        h_padded, w_padded = ov_padded.shape
    else:
        ov_padded = ov
        oi_padded = oi
        gt_padded = gt
        valid_padded = valid_mask
        h_padded, w_padded = h, w

    # Start from Ego
    oracle = ov_padded.copy()

    # Patch decision statistics
    patch_with_add = 0
    accepted_patch = 0
    rejected_patch = 0

    accepted_ADD = 0
    accepted_BA = 0
    accepted_HA = 0

    rejected_ADD = 0
    rejected_BA = 0
    rejected_HA = 0

    # 遍历所有 patch
    for y in range(0, h_padded, patch_size):
        for x in range(0, w_padded, patch_size):
            y_end = y + patch_size
            x_end = x + patch_size

            # Extract patch
            patch_ov = ov_padded[y:y_end, x:x_end]
            patch_oi = oi_padded[y:y_end, x:x_end]
            patch_gt = gt_padded[y:y_end, x:x_end]
            patch_valid = valid_padded[y:y_end, x:x_end]

            # Compute add_mask for this patch
            patch_add_mask = (patch_ov == 0) & (patch_oi == 1) & patch_valid

            if patch_add_mask.sum() == 0:
                # No additions in this patch, skip
                continue

            patch_with_add += 1

            # Compute BA and HA
            ba_count = ((patch_add_mask) & (patch_gt == 1)).sum()
            ha_count = ((patch_add_mask) & (patch_gt == 0)).sum()

            # Decision: U_R = BA_R - HA_R
            utility = ba_count - ha_count

            if utility > 0:
                # Accept: 添加该 patch 的所有路侧新增
                oracle[y:y_end, x:x_end][patch_add_mask] = 1

                accepted_patch += 1
                accepted_ADD += patch_add_mask.sum()
                accepted_BA += ba_count
                accepted_HA += ha_count
            else:
                # Fallback to Ego: 什么都不做
                rejected_patch += 1
                rejected_ADD += patch_add_mask.sum()
                rejected_BA += ba_count
                rejected_HA += ha_count

    # Crop back to original size
    if patch_size == 16:
        oracle = oracle[:h, :w]

    # Sanity check: never suppress Ego
    assert np.all(oracle[ov == 1] == 1), f"Patch-{patch_size} Oracle suppressed Ego occupied cells"

    decision_stats = {
        "patch_with_add": patch_with_add,
        "accepted_patch": accepted_patch,
        "rejected_patch": rejected_patch,
        "accepted_ADD": accepted_ADD,
        "accepted_BA": accepted_BA,
        "accepted_HA": accepted_HA,
        "rejected_ADD": rejected_ADD,
        "rejected_BA": rejected_BA,
        "rejected_HA": rejected_HA,
    }

    return oracle, decision_stats


# ============================================================
# Step 5: 遍历所有有效样本
# ============================================================
print("Processing valid samples...")
print()

for i, path in enumerate(valid_samples):
    with np.load(path, allow_pickle=True) as d:
        Ov = d["Ov"]
        Oi = d["Oi"]
        Oofficial = d["Oofficial"]
        GT = d["GT"]

        gt_cell_valid = d["gt_cell_valid_mask"]
        warp_valid = d["warp_valid_mask"]

        sample_idx = str(d["sample_idx"].item())
        scene_token = str(d["scene_token"].item())

    # Sample-level accumulators
    sample_results = {
        "sample_idx": sample_idx,
        "scene_token": scene_token,
    }

    for strategy in ["ego", "official", "cell_oracle"] + [f"patch_{ps}" for ps in PATCH_SIZES]:
        sample_results[strategy] = {"tp": 0, "fp": 0, "fn": 0}

    # Process each horizon
    for t in range(5):
        valid_mask = (gt_cell_valid[t] == 1) & (warp_valid == 1)

        ov_t = Ov[t]
        oi_t = Oi[t]
        gt_t = GT[t]
        oofficial_t = Oofficial[t]

        # 1. Ego
        ego_pred = ov_t
        ego_tp = ((ego_pred == 1) & (gt_t == 1) & valid_mask).sum()
        ego_fp = ((ego_pred == 1) & (gt_t == 0) & valid_mask).sum()
        ego_fn = ((ego_pred == 0) & (gt_t == 1) & valid_mask).sum()

        horizon_stats["ego"][t]["tp"] += int(ego_tp)
        horizon_stats["ego"][t]["fp"] += int(ego_fp)
        horizon_stats["ego"][t]["fn"] += int(ego_fn)

        sample_results["ego"]["tp"] += int(ego_tp)
        sample_results["ego"]["fp"] += int(ego_fp)
        sample_results["ego"]["fn"] += int(ego_fn)

        # 2. Official OR
        official_pred = oofficial_t
        off_tp = ((official_pred == 1) & (gt_t == 1) & valid_mask).sum()
        off_fp = ((official_pred == 1) & (gt_t == 0) & valid_mask).sum()
        off_fn = ((official_pred == 0) & (gt_t == 1) & valid_mask).sum()

        horizon_stats["official"][t]["tp"] += int(off_tp)
        horizon_stats["official"][t]["fp"] += int(off_fp)
        horizon_stats["official"][t]["fn"] += int(off_fn)

        sample_results["official"]["tp"] += int(off_tp)
        sample_results["official"]["fp"] += int(off_fp)
        sample_results["official"]["fn"] += int(off_fn)

        # 3. Cell Oracle
        cell_oracle_pred = compute_cell_oracle(ov_t, oi_t, gt_t, valid_mask)
        cell_tp = ((cell_oracle_pred == 1) & (gt_t == 1) & valid_mask).sum()
        cell_fp = ((cell_oracle_pred == 1) & (gt_t == 0) & valid_mask).sum()
        cell_fn = ((cell_oracle_pred == 0) & (gt_t == 1) & valid_mask).sum()

        horizon_stats["cell_oracle"][t]["tp"] += int(cell_tp)
        horizon_stats["cell_oracle"][t]["fp"] += int(cell_fp)
        horizon_stats["cell_oracle"][t]["fn"] += int(cell_fn)

        sample_results["cell_oracle"]["tp"] += int(cell_tp)
        sample_results["cell_oracle"]["fp"] += int(cell_fp)
        sample_results["cell_oracle"]["fn"] += int(cell_fn)

        # Accumulate add_mask for G1 consistency check
        add_mask = (ov_t == 0) & (oi_t == 1) & valid_mask
        ha_mask = add_mask & (gt_t == 0)
        ba_mask = add_mask & (gt_t == 1)

        total_add += int(add_mask.sum())
        total_ha += int(ha_mask.sum())
        total_ba += int(ba_mask.sum())

        # 4. Patch Oracles
        for ps in PATCH_SIZES:
            patch_oracle_pred, decision_stats = compute_patch_oracle(ov_t, oi_t, gt_t, valid_mask, ps)
            patch_tp = ((patch_oracle_pred == 1) & (gt_t == 1) & valid_mask).sum()
            patch_fp = ((patch_oracle_pred == 1) & (gt_t == 0) & valid_mask).sum()
            patch_fn = ((patch_oracle_pred == 0) & (gt_t == 1) & valid_mask).sum()

            strategy_name = f"patch_{ps}"
            horizon_stats[strategy_name][t]["tp"] += int(patch_tp)
            horizon_stats[strategy_name][t]["fp"] += int(patch_fp)
            horizon_stats[strategy_name][t]["fn"] += int(patch_fn)

            sample_results[strategy_name]["tp"] += int(patch_tp)
            sample_results[strategy_name]["fp"] += int(patch_fp)
            sample_results[strategy_name]["fn"] += int(patch_fn)

            # Accumulate oracle decision statistics
            for key in decision_stats:
                oracle_decision_stats[strategy_name][key] += decision_stats[key]

    # Compute sample-level IoU
    for strategy in ["ego", "official", "cell_oracle"] + [f"patch_{ps}" for ps in PATCH_SIZES]:
        tp = sample_results[strategy]["tp"]
        fp = sample_results[strategy]["fp"]
        fn = sample_results[strategy]["fn"]
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else np.nan
        sample_results[f"{strategy}_iou"] = iou

    sample_stats.append(sample_results)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(valid_samples)} samples")

print(f"Processed all {len(valid_samples)} samples")
print()

# ============================================================
# Step 5.1: G1 Consistency Check
# ============================================================
print("=" * 80)
print("G2-B SANITY CHECK: Verify consistency with G1")
print("=" * 80)
print(f"Total ADD : {total_add:,}")
print(f"Total HA  : {total_ha:,}")
print(f"Total BA  : {total_ba:,}")
print()
print("Expected from G1:")
print("Total ADD : 799,179")
print("Total HA  : 646,667")
print("Total BA  : 152,512")
print()

if total_add == 799179 and total_ha == 646667 and total_ba == 152512:
    print("✓ PASS: G2-B statistics match G1 exactly")
else:
    print("✗ FAIL: G2-B statistics do NOT match G1")
    print("        Check valid_mask and add_mask definitions!")

print()
print("Verifying ADD = HA + BA:")
if total_add == total_ha + total_ba:
    print(f"✓ PASS: {total_add:,} = {total_ha:,} + {total_ba:,}")
else:
    print(f"✗ FAIL: {total_add:,} ≠ {total_ha:,} + {total_ba:,}")

print("=" * 80)
print()

# ============================================================
# Step 6: 计算 Horizon-wise 指标
# ============================================================
print("=" * 80)
print("Horizon-wise Results")
print("=" * 80)
print()

strategies = ["ego", "official", "cell_oracle"] + [f"patch_{ps}" for ps in PATCH_SIZES]

horizon_metrics = {}

for strategy in strategies:
    horizon_metrics[strategy] = {
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for t in range(5):
        tp = horizon_stats[strategy][t]["tp"]
        fp = horizon_stats[strategy][t]["fp"]
        fn = horizon_stats[strategy][t]["fn"]

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        denom_prec = tp + fp
        precision = tp / denom_prec if denom_prec > 0 else 0.0

        denom_rec = tp + fn
        recall = tp / denom_rec if denom_rec > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        horizon_metrics[strategy]["iou"].append(iou)
        horizon_metrics[strategy]["precision"].append(precision)
        horizon_metrics[strategy]["recall"].append(recall)
        horizon_metrics[strategy]["f1"].append(f1)

# Print IoU table
print(f"{'Horizon':<10} {'Ego':<10} {'Official':<10} {'Cell':<10} {'Patch-4':<10} {'Patch-8':<10} {'Patch-16':<10}")
print("-" * 80)

for t in range(5):
    row = f"t={t:<8}"
    for strategy in strategies:
        iou = horizon_metrics[strategy]["iou"][t]
        row += f" {iou:.4f}    "
    print(row)

# Total
print("-" * 80)
row = "Total    "
for strategy in strategies:
    total_tp = sum(horizon_stats[strategy][t]["tp"] for t in range(5))
    total_fp = sum(horizon_stats[strategy][t]["fp"] for t in range(5))
    total_fn = sum(horizon_stats[strategy][t]["fn"] for t in range(5))
    total_denom = total_tp + total_fp + total_fn
    total_iou = total_tp / total_denom if total_denom > 0 else 0.0
    row += f" {total_iou:.4f}    "
print(row)
print()

# ============================================================
# Step 7: 计算总体统计
# ============================================================
print("=" * 80)
print("Overall Statistics")
print("=" * 80)
print()

total_stats = {}

for strategy in strategies:
    total_tp = sum(horizon_stats[strategy][t]["tp"] for t in range(5))
    total_fp = sum(horizon_stats[strategy][t]["fp"] for t in range(5))
    total_fn = sum(horizon_stats[strategy][t]["fn"] for t in range(5))

    denom_iou = total_tp + total_fp + total_fn
    iou = total_tp / denom_iou if denom_iou > 0 else 0.0

    denom_prec = total_tp + total_fp
    precision = total_tp / denom_prec if denom_prec > 0 else 0.0

    denom_rec = total_tp + total_fn
    recall = total_tp / denom_rec if denom_rec > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_stats[strategy] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print(f"{strategy:<15}: IoU = {iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

print()

# Compute improvements
ego_iou = total_stats["ego"]["iou"]
official_iou = total_stats["official"]["iou"]
cell_oracle_iou = total_stats["cell_oracle"]["iou"]

print(f"Official vs Ego:        Δ = {official_iou - ego_iou:+.4f}")
print(f"Cell Oracle vs Ego:     Δ = {cell_oracle_iou - ego_iou:+.4f}")
print(f"Cell Oracle vs Official: Δ = {cell_oracle_iou - official_iou:+.4f}")
print()

for ps in PATCH_SIZES:
    patch_iou = total_stats[f"patch_{ps}"]["iou"]
    print(f"Patch-{ps} vs Ego:       Δ = {patch_iou - ego_iou:+.4f}")
    print(f"Patch-{ps} vs Official:  Δ = {patch_iou - official_iou:+.4f}")
    print()

# ============================================================
# Step 8: Sample-wise 分布分析
# ============================================================
print("=" * 80)
print("Sample-wise Distribution")
print("=" * 80)
print()

for strategy in strategies:
    ious = np.array([s[f"{strategy}_iou"] for s in sample_stats])
    valid_ious = ious[np.isfinite(ious)]

    if len(valid_ious) > 0:
        print(f"{strategy:<15}:")
        print(f"  mean   = {valid_ious.mean():.4f}")
        print(f"  std    = {valid_ious.std():.4f}")
        print(f"  median = {np.median(valid_ious):.4f}")
        print(f"  min    = {valid_ious.min():.4f}")
        print(f"  max    = {valid_ious.max():.4f}")
        print()

# ============================================================
# Step 9: Oracle Decision Statistics
# ============================================================
print("=" * 80)
print("Oracle Decision Statistics")
print("=" * 80)
print()

for ps in PATCH_SIZES:
    stats = oracle_decision_stats[f"patch_{ps}"]
    print(f"Patch-{ps} Oracle:")
    print(f"  Total patches with additions:    {stats['patch_with_add']:,}")
    print(f"  Accepted patches (U > 0):         {stats['accepted_patch']:,} ({100*stats['accepted_patch']/stats['patch_with_add']:.1f}%)" if stats['patch_with_add'] > 0 else f"  Accepted patches (U > 0):         {stats['accepted_patch']:,}")
    print(f"  Rejected patches (U ≤ 0):         {stats['rejected_patch']:,} ({100*stats['rejected_patch']/stats['patch_with_add']:.1f}%)" if stats['patch_with_add'] > 0 else f"  Rejected patches (U ≤ 0):         {stats['rejected_patch']:,}")
    print()
    print(f"  Accepted patch additions:")
    print(f"    Total ADD: {stats['accepted_ADD']:,}")
    print(f"    BA:        {stats['accepted_BA']:,}")
    print(f"    HA:        {stats['accepted_HA']:,}")
    print(f"    WAR:       {100*stats['accepted_HA']/stats['accepted_ADD']:.2f}%" if stats['accepted_ADD'] > 0 else "    WAR:       N/A")
    print()
    print(f"  Rejected patch additions:")
    print(f"    Total ADD: {stats['rejected_ADD']:,}")
    print(f"    BA:        {stats['rejected_BA']:,}")
    print(f"    HA:        {stats['rejected_HA']:,}")
    print(f"    WAR:       {100*stats['rejected_HA']/stats['rejected_ADD']:.2f}%" if stats['rejected_ADD'] > 0 else "    WAR:       N/A")
    print()
    print("-" * 80)
    print()

# ============================================================
# Step 10: 保存结果
# ============================================================
out_path = os.path.join(OUT_DIR, "g2_oracle_stats.npz")

# Prepare horizon arrays
horizon_arrays = {}
for strategy in strategies:
    horizon_arrays[f"horizon_{strategy}_iou"] = np.array(horizon_metrics[strategy]["iou"])
    horizon_arrays[f"horizon_{strategy}_precision"] = np.array(horizon_metrics[strategy]["precision"])
    horizon_arrays[f"horizon_{strategy}_recall"] = np.array(horizon_metrics[strategy]["recall"])
    horizon_arrays[f"horizon_{strategy}_f1"] = np.array(horizon_metrics[strategy]["f1"])

# Prepare sample arrays
sample_arrays = {}
for strategy in strategies:
    sample_arrays[f"sample_{strategy}_iou"] = np.array([s[f"{strategy}_iou"] for s in sample_stats])

# Prepare total stats arrays
total_stats_arrays = {}
for strategy in strategies:
    total_stats_arrays[f"total_{strategy}_iou"] = total_stats[strategy]["iou"]
    total_stats_arrays[f"total_{strategy}_precision"] = total_stats[strategy]["precision"]
    total_stats_arrays[f"total_{strategy}_recall"] = total_stats[strategy]["recall"]
    total_stats_arrays[f"total_{strategy}_f1"] = total_stats[strategy]["f1"]

# Prepare oracle decision stats
oracle_decision_arrays = {}
for ps in PATCH_SIZES:
    key = f"patch_{ps}"
    stats = oracle_decision_stats[key]
    for stat_name, value in stats.items():
        oracle_decision_arrays[f"{key}_{stat_name}"] = value

np.savez_compressed(
    out_path,
    # Horizon-wise
    **horizon_arrays,

    # Sample-wise
    sample_idx=np.array([s["sample_idx"] for s in sample_stats], dtype=object),
    scene_token=np.array([s["scene_token"] for s in sample_stats], dtype=object),
    **sample_arrays,

    # Total statistics
    **total_stats_arrays,

    # Oracle decision statistics
    **oracle_decision_arrays,

    # G1 consistency check
    total_add=total_add,
    total_ha=total_ha,
    total_ba=total_ba,

    # Metadata
    num_valid_samples=len(valid_samples),
    num_invalid_samples=len(invalid_samples),
)

print(f"G2-B oracle statistics saved to: {out_path}")
print("=" * 80)
