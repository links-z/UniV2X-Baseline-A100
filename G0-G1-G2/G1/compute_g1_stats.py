import os
import glob
import numpy as np
from collections import defaultdict

CACHE_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full"
OUT_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/G1"

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

if not files:
    raise RuntimeError("No .npz files found in cache_full")

print("=" * 80)
print("G1: Negative Cooperation Analysis")
print("=" * 80)
print(f"Total samples: {len(files)}")
print()

# ============================================================
# Step 1: 筛选出与 baseline Occupancy 评估一致的 549 个样本
# ============================================================
valid_samples = []
invalid_samples = []

for i, path in enumerate(files):
    with np.load(path) as d:
        future_valid = d["future_valid_mask"]

        if future_valid.all():
            valid_samples.append((i, path))
        else:
            invalid_samples.append((i, path))

print(f"Valid samples (occ_to_eval): {len(valid_samples)}")
print(f"Invalid samples (occ_has_invalid_frame): {len(invalid_samples)}")
print()

# ============================================================
# Step 2: 对 549 个有效样本，逐个统计 Harmful/Beneficial Addition
# ============================================================
# Definitions:
# - GT cell: 被标注为 occupied 的格子（GT == 1）
# - Harmful Addition (HA): 路侧添加了，但 GT 说不该有
#   HA[t,i,j] = (Oi[t,i,j] == 1) AND (GT[t,i,j] == 0) AND (Ov[t,i,j] == 0)
#
# - Beneficial Addition (BA): 路侧添加了，GT 也确实有
#   BA[t,i,j] = (Oi[t,i,j] == 1) AND (GT[t,i,j] == 1) AND (Ov[t,i,j] == 0)
#
# - False Detection (FD): 车端自己误检的
#   FD[t,i,j] = (Ov[t,i,j] == 1) AND (GT[t,i,j] == 0)

# Horizon-wise statistics
horizon_stats = defaultdict(lambda: {
    "HA": 0,
    "BA": 0,
    "ADD": 0,
    "ego_tp": 0,
    "ego_fp": 0,
    "ego_fn": 0,
    "off_tp": 0,
    "off_fp": 0,
    "off_fn": 0,
})

# Sequence-wise statistics
seq_stats = []

for idx, (sample_idx, path) in enumerate(valid_samples):
    with np.load(path) as d:
        Pv = d["Pv"]  # [5, 200, 200]
        Pi = d["Pi_aligned"]  # [5, 200, 200]

        Ov = d["Ov"]  # [5, 200, 200]
        Oi = d["Oi"]  # [5, 200, 200]
        Oofficial = d["Oofficial"]  # [5, 200, 200]

        GT = d["GT"]  # [5, 200, 200]

        gt_cell_valid = d["gt_cell_valid_mask"]  # [5, 200, 200]
        future_valid = d["future_valid_mask"]  # [5]
        warp_valid = d["warp_valid_mask"]  # [200, 200]

        threshold = float(d["test_seg_thresh"])

    # Sanity checks
    assert Ov.shape == (5, 200, 200)
    assert Oi.shape == (5, 200, 200)
    assert GT.shape == (5, 200, 200)
    assert future_valid.all(), "Should only process valid samples"

    # 对每个 horizon 统计
    seq_ha = 0
    seq_ba = 0
    seq_add = 0
    seq_ego_tp = 0
    seq_ego_fp = 0
    seq_ego_fn = 0
    seq_off_tp = 0
    seq_off_fp = 0
    seq_off_fn = 0

    for t in range(5):
        # 只在 GT 有效 AND warp 有效的格子上统计
        valid_mask = (gt_cell_valid[t] == 1) & (warp_valid == 1)

        ov_t = Ov[t]
        oi_t = Oi[t]
        gt_t = GT[t]
        oofficial_t = Oofficial[t]

        # Infrastructure Addition: 路侧新增（车端没有，路侧有）
        add_mask = (ov_t == 0) & (oi_t == 1) & valid_mask

        # Harmful Addition: 路侧新增但 GT 说没有
        ha_mask = add_mask & (gt_t == 0)
        ha_count = int(ha_mask.sum())

        # Beneficial Addition: 路侧新增且 GT 确实有
        ba_mask = add_mask & (gt_t == 1)
        ba_count = int(ba_mask.sum())

        # Total Addition
        add_count = int(add_mask.sum())
        assert add_count == ha_count + ba_count, "Addition count mismatch"

        # Ego IoU components
        ego_tp = int(((ov_t == 1) & (gt_t == 1) & valid_mask).sum())
        ego_fp = int(((ov_t == 1) & (gt_t == 0) & valid_mask).sum())
        ego_fn = int(((ov_t == 0) & (gt_t == 1) & valid_mask).sum())

        # Official IoU components
        off_tp = int(((oofficial_t == 1) & (gt_t == 1) & valid_mask).sum())
        off_fp = int(((oofficial_t == 1) & (gt_t == 0) & valid_mask).sum())
        off_fn = int(((oofficial_t == 0) & (gt_t == 1) & valid_mask).sum())

        # Accumulate horizon-wise
        horizon_stats[t]["HA"] += ha_count
        horizon_stats[t]["BA"] += ba_count
        horizon_stats[t]["ADD"] += add_count
        horizon_stats[t]["ego_tp"] += ego_tp
        horizon_stats[t]["ego_fp"] += ego_fp
        horizon_stats[t]["ego_fn"] += ego_fn
        horizon_stats[t]["off_tp"] += off_tp
        horizon_stats[t]["off_fp"] += off_fp
        horizon_stats[t]["off_fn"] += off_fn

        # Accumulate sequence-wise
        seq_ha += ha_count
        seq_ba += ba_count
        seq_add += add_count
        seq_ego_tp += ego_tp
        seq_ego_fp += ego_fp
        seq_ego_fn += ego_fn
        seq_off_tp += off_tp
        seq_off_fp += off_fp
        seq_off_fn += off_fn

    # Compute sequence-level IoU
    ego_iou = seq_ego_tp / (seq_ego_tp + seq_ego_fp + seq_ego_fn) if (seq_ego_tp + seq_ego_fp + seq_ego_fn) > 0 else 0.0
    off_iou = seq_off_tp / (seq_off_tp + seq_off_fp + seq_off_fn) if (seq_off_tp + seq_off_fp + seq_off_fn) > 0 else 0.0
    delta_iou = off_iou - ego_iou

    seq_stats.append({
        "sample_idx": sample_idx,
        "HA": seq_ha,
        "BA": seq_ba,
        "ADD": seq_add,
        "ego_tp": seq_ego_tp,
        "ego_fp": seq_ego_fp,
        "ego_fn": seq_ego_fn,
        "off_tp": seq_off_tp,
        "off_fp": seq_off_fp,
        "off_fn": seq_off_fn,
        "ego_iou": ego_iou,
        "off_iou": off_iou,
        "delta_iou": delta_iou,
    })

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(valid_samples)} samples...")

print()
print("=" * 80)
print("G1 Statistics: Horizon-wise")
print("=" * 80)
print()
print(f"{'Horizon':<10} {'ADD':<10} {'HA':<10} {'BA':<10} {'WAR':<10} {'BAR_add':<10} {'Ego_IoU':<10} {'Off_IoU':<10} {'Δ_IoU':<10}")
print("-" * 100)

total_add = 0
total_ha = 0
total_ba = 0
total_ego_tp = 0
total_ego_fp = 0
total_ego_fn = 0
total_off_tp = 0
total_off_fp = 0
total_off_fn = 0

for t in range(5):
    stats = horizon_stats[t]
    add = stats["ADD"]
    ha = stats["HA"]
    ba = stats["BA"]

    ego_tp = stats["ego_tp"]
    ego_fp = stats["ego_fp"]
    ego_fn = stats["ego_fn"]

    off_tp = stats["off_tp"]
    off_fp = stats["off_fp"]
    off_fn = stats["off_fn"]

    war = ha / add if add > 0 else np.nan
    bar_add = ba / add if add > 0 else np.nan

    ego_iou = ego_tp / (ego_tp + ego_fp + ego_fn) if (ego_tp + ego_fp + ego_fn) > 0 else 0.0
    off_iou = off_tp / (off_tp + off_fp + off_fn) if (off_tp + off_fp + off_fn) > 0 else 0.0
    delta_iou = off_iou - ego_iou

    print(f"t={t:<8} {add:<10} {ha:<10} {ba:<10} {war:<10.4f} {bar_add:<10.4f} {ego_iou:<10.4f} {off_iou:<10.4f} {delta_iou:<10.4f}")

    total_add += add
    total_ha += ha
    total_ba += ba
    total_ego_tp += ego_tp
    total_ego_fp += ego_fp
    total_ego_fn += ego_fn
    total_off_tp += off_tp
    total_off_fp += off_fp
    total_off_fn += off_fn

print("-" * 100)

total_war = total_ha / total_add if total_add > 0 else np.nan
total_bar_add = total_ba / total_add if total_add > 0 else np.nan
total_ego_iou = total_ego_tp / (total_ego_tp + total_ego_fp + total_ego_fn)
total_off_iou = total_off_tp / (total_off_tp + total_off_fp + total_off_fn)
total_delta_iou = total_off_iou - total_ego_iou

print(f"{'Total':<10} {total_add:<10} {total_ha:<10} {total_ba:<10} {total_war:<10.4f} {total_bar_add:<10.4f} {total_ego_iou:<10.4f} {total_off_iou:<10.4f} {total_delta_iou:<10.4f}")
print()

# Verify WAR + BAR_add = 1
if total_add > 0:
    sum_check = total_war + total_bar_add
    print(f"Verification: WAR + BAR_add = {sum_check:.6f} (should be 1.0)")
    print()

print("=" * 80)
print("G1 Statistics: Sequence-wise Distribution")
print("=" * 80)
print()

# 统计每个样本的 WAR, BAR_add, Delta_IoU
war_list = []
bar_add_list = []
delta_iou_list = []

for s in seq_stats:
    if s["ADD"] > 0:
        war_list.append(s["HA"] / s["ADD"])
        bar_add_list.append(s["BA"] / s["ADD"])
    delta_iou_list.append(s["delta_iou"])

war_arr = np.array(war_list)
bar_add_arr = np.array(bar_add_list)
delta_iou_arr = np.array(delta_iou_list)

print(f"WAR: mean={war_arr.mean():.4f}, std={war_arr.std():.4f}, "
      f"median={np.median(war_arr):.4f}, max={war_arr.max():.4f}")
print(f"BAR_add: mean={bar_add_arr.mean():.4f}, std={bar_add_arr.std():.4f}, "
      f"median={np.median(bar_add_arr):.4f}, max={bar_add_arr.max():.4f}")
print(f"Δ_IoU: mean={delta_iou_arr.mean():.6f}, std={delta_iou_arr.std():.6f}, "
      f"median={np.median(delta_iou_arr):.6f}, min={delta_iou_arr.min():.6f}, max={delta_iou_arr.max():.6f}")
print()

# 统计有多少样本是负合作（Delta_IoU < 0）
negative_samples = sum(1 for s in seq_stats if s["delta_iou"] < 0)
positive_samples = sum(1 for s in seq_stats if s["delta_iou"] > 0)
neutral_samples = sum(1 for s in seq_stats if np.isclose(s["delta_iou"], 0))

print(f"Negative cooperation samples (Δ_IoU < 0): {negative_samples} ({negative_samples/len(seq_stats)*100:.1f}%)")
print(f"Positive cooperation samples (Δ_IoU > 0): {positive_samples} ({positive_samples/len(seq_stats)*100:.1f}%)")
print(f"Neutral samples (Δ_IoU ≈ 0): {neutral_samples} ({neutral_samples/len(seq_stats)*100:.1f}%)")
print()

# 统计 WAR > 0.5 的样本（有害添加占主导）
war_dominant_samples = sum(1 for s in seq_stats if s["ADD"] > 0 and s["HA"] > s["BA"])
print(f"Samples with HA > BA (harmful dominant): {war_dominant_samples} ({war_dominant_samples/len(seq_stats)*100:.1f}%)")
print()

# ============================================================
# Step 3: 保存详细结果
# ============================================================
out_path = os.path.join(OUT_DIR, "g1_stats.npz")

np.savez_compressed(
    out_path,
    # Horizon-wise
    horizon_add=np.array([horizon_stats[t]["ADD"] for t in range(5)]),
    horizon_ha=np.array([horizon_stats[t]["HA"] for t in range(5)]),
    horizon_ba=np.array([horizon_stats[t]["BA"] for t in range(5)]),
    horizon_ego_tp=np.array([horizon_stats[t]["ego_tp"] for t in range(5)]),
    horizon_ego_fp=np.array([horizon_stats[t]["ego_fp"] for t in range(5)]),
    horizon_ego_fn=np.array([horizon_stats[t]["ego_fn"] for t in range(5)]),
    horizon_off_tp=np.array([horizon_stats[t]["off_tp"] for t in range(5)]),
    horizon_off_fp=np.array([horizon_stats[t]["off_fp"] for t in range(5)]),
    horizon_off_fn=np.array([horizon_stats[t]["off_fn"] for t in range(5)]),

    # Sequence-wise
    seq_sample_idx=np.array([s["sample_idx"] for s in seq_stats]),
    seq_add=np.array([s["ADD"] for s in seq_stats]),
    seq_ha=np.array([s["HA"] for s in seq_stats]),
    seq_ba=np.array([s["BA"] for s in seq_stats]),
    seq_ego_tp=np.array([s["ego_tp"] for s in seq_stats]),
    seq_ego_fp=np.array([s["ego_fp"] for s in seq_stats]),
    seq_ego_fn=np.array([s["ego_fn"] for s in seq_stats]),
    seq_off_tp=np.array([s["off_tp"] for s in seq_stats]),
    seq_off_fp=np.array([s["off_fp"] for s in seq_stats]),
    seq_off_fn=np.array([s["off_fn"] for s in seq_stats]),
    seq_ego_iou=np.array([s["ego_iou"] for s in seq_stats]),
    seq_off_iou=np.array([s["off_iou"] for s in seq_stats]),
    seq_delta_iou=np.array([s["delta_iou"] for s in seq_stats]),

    # Summary
    total_add=total_add,
    total_ha=total_ha,
    total_ba=total_ba,
    total_ego_tp=total_ego_tp,
    total_ego_fp=total_ego_fp,
    total_ego_fn=total_ego_fn,
    total_off_tp=total_off_tp,
    total_off_fp=total_off_fp,
    total_off_fn=total_off_fn,
    total_ego_iou=total_ego_iou,
    total_off_iou=total_off_iou,
    total_delta_iou=total_delta_iou,

    num_valid_samples=len(valid_samples),
    num_invalid_samples=len(invalid_samples),
)

print(f"G1 statistics saved to: {out_path}")
print("=" * 80)
