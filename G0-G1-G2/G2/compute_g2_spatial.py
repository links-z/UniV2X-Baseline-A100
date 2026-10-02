#!/usr/bin/env python3
"""
G2-A: Spatial Diagnosis

分析路侧新增 Occupancy 的空间分布特征：
1. Spatial Map: HA/BA/ADD 的空间分布
2. WAR vs Ego Distance: 不同距离下的 WAR
3. WAR vs Warp Boundary Distance: 边界距离与 WAR 的关系
4. Sample Overlap Analysis: warp overlap ratio 与 WAR/Delta_IoU 的相关性
"""

import os
import glob
import numpy as np
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

# ============================================================
# Configuration
# ============================================================
CACHE_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full"
OUT_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/G2"

BEV_H = 200
BEV_W = 200

# BEV range
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
CELL_SIZE = 0.512  # meters

# Ego at BEV center (world coordinates)
EGO_X_WORLD = 0.0
EGO_Y_WORLD = 0.0

# Distance binning (meters)
# Max distance from ego to corner: sqrt(51.2^2 + 51.2^2) ≈ 72.4m
DISTANCE_BINS = np.arange(0.0, 75.0 + 5.0, 5.0)  # 0-80m, 每 5m 一个 bin

# Boundary distance bins (cells)
# 动态到最大可能值，但先用合理上限
BOUNDARY_BINS = np.arange(0, 52, 2)  # 0-52 cells, 每 2 cells 一个 bin，最后一个是 overflow

print("=" * 80)
print("G2-A: Spatial Diagnosis")
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
# Step 3: 初始化累计 maps
# ============================================================
overall_add_map = np.zeros((BEV_H, BEV_W), dtype=np.int64)
overall_ha_map = np.zeros((BEV_H, BEV_W), dtype=np.int64)
overall_ba_map = np.zeros((BEV_H, BEV_W), dtype=np.int64)

horizon_add_map = np.zeros((5, BEV_H, BEV_W), dtype=np.int64)
horizon_ha_map = np.zeros((5, BEV_H, BEV_W), dtype=np.int64)
horizon_ba_map = np.zeros((5, BEV_H, BEV_W), dtype=np.int64)

# Distance-wise statistics
distance_stats = defaultdict(lambda: {"ADD": 0, "HA": 0, "BA": 0})

# Boundary distance statistics
boundary_stats = defaultdict(lambda: {"ADD": 0, "HA": 0, "BA": 0})

# Sample-level statistics
sample_stats = []

# ============================================================
# Step 4: 计算每个格子到 Ego 的距离
# ============================================================
# 构建真实世界坐标系
x_coords_world = PC_RANGE[0] + (np.arange(BEV_W) + 0.5) * CELL_SIZE  # [200]
y_coords_world = PC_RANGE[1] + (np.arange(BEV_H) + 0.5) * CELL_SIZE  # [200]

xx, yy = np.meshgrid(x_coords_world, y_coords_world)  # [200, 200]

# Distance to ego (in meters)
dist_to_ego_meters = np.sqrt(
    (xx - EGO_X_WORLD) ** 2 +
    (yy - EGO_Y_WORLD) ** 2
)

# ============================================================
# Step 5: 遍历所有有效样本
# ============================================================
print("Processing valid samples...")
print()

for i, path in enumerate(valid_samples):
    with np.load(path, allow_pickle=True) as d:
        Pv = d["Pv"]
        Pi = d["Pi_aligned"]

        Ov = d["Ov"]
        Oi = d["Oi"]
        Oofficial = d["Oofficial"]

        GT = d["GT"]

        gt_cell_valid = d["gt_cell_valid_mask"]
        warp_valid = d["warp_valid_mask"]

        sample_idx = str(d["sample_idx"].item())
        scene_token = str(d["scene_token"].item())
        timestamp = float(d["timestamp"])

    # Sample overlap ratio
    overlap_ratio = float(warp_valid.mean())

    # Compute boundary distance for each warp_valid cell using distance_transform_edt
    # Pad with False to ensure boundary detection at edges
    padded_warp = np.pad(warp_valid.astype(bool), 1, mode='constant', constant_values=False)
    dist_to_boundary_padded = distance_transform_edt(padded_warp)
    dist_to_boundary_map = dist_to_boundary_padded[1:-1, 1:-1].astype(np.float32)

    # 对于 warp_valid=0 的位置设为 NaN
    dist_to_boundary_map[warp_valid == 0] = np.nan

    # Sample-level accumulators
    sample_add = 0
    sample_ha = 0
    sample_ba = 0
    sample_ego_tp = 0
    sample_ego_fp = 0
    sample_ego_fn = 0
    sample_off_tp = 0
    sample_off_fp = 0
    sample_off_fn = 0

    # Process each horizon
    for t in range(5):
        valid_mask = (gt_cell_valid[t] == 1) & (warp_valid == 1)

        ov_t = Ov[t]
        oi_t = Oi[t]
        gt_t = GT[t]
        oofficial_t = Oofficial[t]

        add_mask = (ov_t == 0) & (oi_t == 1) & valid_mask
        ha_mask = add_mask & (gt_t == 0)
        ba_mask = add_mask & (gt_t == 1)

        # Accumulate spatial maps
        overall_add_map += add_mask.astype(np.int64)
        overall_ha_map += ha_mask.astype(np.int64)
        overall_ba_map += ba_mask.astype(np.int64)

        horizon_add_map[t] += add_mask.astype(np.int64)
        horizon_ha_map[t] += ha_mask.astype(np.int64)
        horizon_ba_map[t] += ba_mask.astype(np.int64)

        # Distance-wise statistics
        for bin_idx in range(len(DISTANCE_BINS) - 1):
            bin_min = DISTANCE_BINS[bin_idx]
            bin_max = DISTANCE_BINS[bin_idx + 1]

            in_bin = (dist_to_ego_meters >= bin_min) & (dist_to_ego_meters < bin_max)

            distance_stats[bin_idx]["ADD"] += int((add_mask & in_bin).sum())
            distance_stats[bin_idx]["HA"] += int((ha_mask & in_bin).sum())
            distance_stats[bin_idx]["BA"] += int((ba_mask & in_bin).sum())

        # Boundary distance statistics
        for bin_idx in range(len(BOUNDARY_BINS) - 1):
            bin_min = BOUNDARY_BINS[bin_idx]
            bin_max = BOUNDARY_BINS[bin_idx + 1]

            in_bin = (dist_to_boundary_map >= bin_min) & (dist_to_boundary_map < bin_max)
            in_bin = np.nan_to_num(in_bin, nan=False)

            boundary_stats[bin_idx]["ADD"] += int((add_mask & in_bin).sum())
            boundary_stats[bin_idx]["HA"] += int((ha_mask & in_bin).sum())
            boundary_stats[bin_idx]["BA"] += int((ba_mask & in_bin).sum())

        # Sample-level IoU components
        sample_add += int(add_mask.sum())
        sample_ha += int(ha_mask.sum())
        sample_ba += int(ba_mask.sum())

        sample_ego_tp += int(((ov_t == 1) & (gt_t == 1) & valid_mask).sum())
        sample_ego_fp += int(((ov_t == 1) & (gt_t == 0) & valid_mask).sum())
        sample_ego_fn += int(((ov_t == 0) & (gt_t == 1) & valid_mask).sum())

        sample_off_tp += int(((oofficial_t == 1) & (gt_t == 1) & valid_mask).sum())
        sample_off_fp += int(((oofficial_t == 1) & (gt_t == 0) & valid_mask).sum())
        sample_off_fn += int(((oofficial_t == 0) & (gt_t == 1) & valid_mask).sum())

    # Compute sample-level metrics
    sample_war = sample_ha / sample_add if sample_add > 0 else np.nan

    ego_denom = sample_ego_tp + sample_ego_fp + sample_ego_fn
    off_denom = sample_off_tp + sample_off_fp + sample_off_fn

    sample_ego_iou = sample_ego_tp / ego_denom if ego_denom > 0 else np.nan
    sample_off_iou = sample_off_tp / off_denom if off_denom > 0 else np.nan
    sample_delta_iou = sample_off_iou - sample_ego_iou if (ego_denom > 0 and off_denom > 0) else np.nan

    sample_stats.append({
        "sample_idx": sample_idx,
        "scene_token": scene_token,
        "timestamp": timestamp,
        "overlap_ratio": overlap_ratio,
        "ADD": sample_add,
        "HA": sample_ha,
        "BA": sample_ba,
        "WAR": sample_war,
        "ego_iou": sample_ego_iou,
        "off_iou": sample_off_iou,
        "delta_iou": sample_delta_iou,
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(valid_samples)} samples")

print(f"Processed all {len(valid_samples)} samples")
print()

# ============================================================
# Step 6: 计算 WAR maps
# ============================================================
print("Computing WAR maps...")

overall_war_map = np.full((BEV_H, BEV_W), np.nan, dtype=np.float64)
valid_cells = overall_add_map > 0
overall_war_map[valid_cells] = (
    overall_ha_map[valid_cells].astype(np.float64) /
    overall_add_map[valid_cells].astype(np.float64)
)

horizon_war_map = np.full((5, BEV_H, BEV_W), np.nan, dtype=np.float64)
for t in range(5):
    valid_cells = horizon_add_map[t] > 0
    horizon_war_map[t][valid_cells] = (
        horizon_ha_map[t][valid_cells].astype(np.float64) /
        horizon_add_map[t][valid_cells].astype(np.float64)
    )

print("WAR maps computed")
print()

# Sanity check: 验证与 G1 总体统计一致
total_add = int(overall_add_map.sum())
total_ha = int(overall_ha_map.sum())
total_ba = int(overall_ba_map.sum())
total_war = total_ha / total_add if total_add > 0 else np.nan

print("=" * 80)
print("G2-A SANITY CHECK: Verify consistency with G1")
print("=" * 80)
print(f"Total ADD : {total_add:,}")
print(f"Total HA  : {total_ha:,}")
print(f"Total BA  : {total_ba:,}")
print(f"Total WAR : {total_war:.4f}")
print()
print("Expected from G1:")
print("Total ADD : 799,179")
print("Total HA  : 646,667")
print("Total BA  : 152,512")
print("Total WAR : 0.8092")
print()

if total_add == 799179 and total_ha == 646667 and total_ba == 152512:
    print("✓ PASS: G2-A spatial statistics match G1 exactly")
else:
    print("✗ FAIL: G2-A spatial statistics do NOT match G1")
    print("        Check valid_mask and add_mask definitions!")

print("=" * 80)
print()

# ============================================================
# Step 7: 整理 distance-wise statistics
# ============================================================
distance_add = np.array([distance_stats[i]["ADD"] for i in range(len(DISTANCE_BINS) - 1)])
distance_ha = np.array([distance_stats[i]["HA"] for i in range(len(DISTANCE_BINS) - 1)])
distance_ba = np.array([distance_stats[i]["BA"] for i in range(len(DISTANCE_BINS) - 1)])

distance_war = np.full(len(DISTANCE_BINS) - 1, np.nan)
valid_bins = distance_add > 0
distance_war[valid_bins] = distance_ha[valid_bins].astype(np.float64) / distance_add[valid_bins].astype(np.float64)

# ============================================================
# Step 8: 整理 boundary distance statistics
# ============================================================
boundary_add = np.array([boundary_stats[i]["ADD"] for i in range(len(BOUNDARY_BINS) - 1)])
boundary_ha = np.array([boundary_stats[i]["HA"] for i in range(len(BOUNDARY_BINS) - 1)])
boundary_ba = np.array([boundary_stats[i]["BA"] for i in range(len(BOUNDARY_BINS) - 1)])

boundary_war = np.full(len(BOUNDARY_BINS) - 1, np.nan)
valid_bins = boundary_add > 0
boundary_war[valid_bins] = boundary_ha[valid_bins].astype(np.float64) / boundary_add[valid_bins].astype(np.float64)

# ============================================================
# Step 9: Sample overlap correlation analysis
# ============================================================
print("Computing sample-level correlations...")

overlap_ratios = np.array([s["overlap_ratio"] for s in sample_stats])
wars = np.array([s["WAR"] for s in sample_stats])
delta_ious = np.array([s["delta_iou"] for s in sample_stats])

# Remove NaN values for correlation
valid_war_mask = np.isfinite(wars)
valid_delta_mask = np.isfinite(delta_ious)

if valid_war_mask.sum() > 1:
    spearman_overlap_war, pval_overlap_war = spearmanr(
        overlap_ratios[valid_war_mask],
        wars[valid_war_mask]
    )
else:
    spearman_overlap_war = np.nan
    pval_overlap_war = np.nan

if valid_delta_mask.sum() > 1:
    spearman_overlap_delta, pval_overlap_delta = spearmanr(
        overlap_ratios[valid_delta_mask],
        delta_ious[valid_delta_mask]
    )
else:
    spearman_overlap_delta = np.nan
    pval_overlap_delta = np.nan

print(f"Spearman(overlap, WAR): {spearman_overlap_war:.4f} (p={pval_overlap_war:.4e})")
print(f"Spearman(overlap, Delta_IoU): {spearman_overlap_delta:.4f} (p={pval_overlap_delta:.4e})")
print()

# ============================================================
# Step 10: 保存结果
# ============================================================
out_path = os.path.join(OUT_DIR, "g2_spatial_stats.npz")

np.savez_compressed(
    out_path,
    # Spatial maps
    overall_add_map=overall_add_map,
    overall_ha_map=overall_ha_map,
    overall_ba_map=overall_ba_map,
    overall_war_map=overall_war_map,

    horizon_add_map=horizon_add_map,
    horizon_ha_map=horizon_ha_map,
    horizon_ba_map=horizon_ba_map,
    horizon_war_map=horizon_war_map,

    # Distance statistics
    distance_bin_edges=DISTANCE_BINS,
    distance_add=distance_add,
    distance_ha=distance_ha,
    distance_ba=distance_ba,
    distance_war=distance_war,

    # Boundary distance statistics
    boundary_bin_edges=BOUNDARY_BINS,
    boundary_add=boundary_add,
    boundary_ha=boundary_ha,
    boundary_ba=boundary_ba,
    boundary_war=boundary_war,

    # Sample-level statistics
    sample_idx=np.array([s["sample_idx"] for s in sample_stats], dtype=object),
    scene_token=np.array([s["scene_token"] for s in sample_stats], dtype=object),
    sample_timestamp=np.array([s["timestamp"] for s in sample_stats]),
    sample_overlap_ratio=np.array([s["overlap_ratio"] for s in sample_stats]),
    sample_add=np.array([s["ADD"] for s in sample_stats]),
    sample_ha=np.array([s["HA"] for s in sample_stats]),
    sample_ba=np.array([s["BA"] for s in sample_stats]),
    sample_war=np.array([s["WAR"] for s in sample_stats]),
    sample_ego_iou=np.array([s["ego_iou"] for s in sample_stats]),
    sample_off_iou=np.array([s["off_iou"] for s in sample_stats]),
    sample_delta_iou=np.array([s["delta_iou"] for s in sample_stats]),

    # Correlation statistics
    spearman_overlap_war=spearman_overlap_war,
    spearman_overlap_delta_iou=spearman_overlap_delta,
    pval_overlap_war=pval_overlap_war,
    pval_overlap_delta_iou=pval_overlap_delta,

    # Summary
    num_total_samples=len(files),
    num_valid_samples=len(valid_samples),
    num_invalid_samples=len(invalid_samples),
)

print(f"G2-A spatial statistics saved to: {out_path}")
print("=" * 80)
