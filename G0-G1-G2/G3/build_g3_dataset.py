#!/usr/bin/env python3
"""
G3-0: Build Local Cooperation Value Dataset

Constructs patch-level dataset for cooperation utility prediction:
- Input: X_R = [Pv, Pi, Pi-Pv, |Pi-Pv|, M^warp] for each 4×4 patch
- Label: y_R = 1(U_R > 0), where U_R = BA_R - HA_R
- Split: 13/4/4 scenes train/val/test, SEED=2026, split by scene to prevent data leakage
- Only include patches with ADD_R > 0

Sanity check: Must reproduce G2-B statistics exactly before training:
  - Valid samples: 549
  - Addition-containing patches: 119,448
  - Oracle accepted patches: 17,805
  - Acceptance rate: 14.9%
"""

import os
import glob
import numpy as np
import pickle
from collections import defaultdict

# Paths
CACHE_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full"
VALID_SAMPLES_FILE = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3/cache_valid_samples.txt"
OUT_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3"

# Parameters
PATCH_SIZE = 4
GRID_SIZE = 200  # Occupancy grid is 200×200
N_HORIZONS = 5  # t=0,1,2,3,4
SEED = 2026

# Scene split (13 train / 4 val / 4 test)
np.random.seed(SEED)

print("=" * 70)
print("G3-0: BUILD LOCAL COOPERATION VALUE DATASET")
print("=" * 70)
print()

# Step 1: Load valid sample list
print("Step 1: Loading valid sample list...")
valid_sample_ids = set()
scene_to_samples = defaultdict(list)

with open(VALID_SAMPLES_FILE, 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            sample_idx = parts[0]
            scene_token = parts[1]
            valid_sample_ids.add(sample_idx)
            scene_to_samples[scene_token].append(sample_idx)

print(f"Loaded {len(valid_sample_ids)} valid samples from {len(scene_to_samples)} scenes")
print()

# Step 2: Split scenes into train/val/test
print("Step 2: Splitting scenes...")
all_scenes = sorted(scene_to_samples.keys())
n_scenes = len(all_scenes)

np.random.shuffle(all_scenes)
train_scenes = set(all_scenes[:13])
val_scenes = set(all_scenes[13:17])
test_scenes = set(all_scenes[17:21])

print(f"Total scenes: {n_scenes}")
print(f"Train scenes: {len(train_scenes)}")
print(f"Val scenes:   {len(val_scenes)}")
print(f"Test scenes:  {len(test_scenes)}")
print()

# Step 3: Process all valid samples and extract patch-level features
print("Step 3: Extracting patch-level features...")

train_data = []
val_data = []
test_data = []

# Global statistics for sanity check
total_patches_with_add = 0
total_accepted_patches = 0
total_rejected_patches = 0

# Track accepted/rejected additions
accepted_ADD = 0
accepted_BA = 0
accepted_HA = 0
rejected_ADD = 0
rejected_BA = 0
rejected_HA = 0

cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

for cache_file in cache_files:
    with np.load(cache_file, allow_pickle=True) as data:
        sample_idx = str(data["sample_idx"].item() if data["sample_idx"].ndim == 0 else data["sample_idx"])

        # Only process valid samples
        if sample_idx not in valid_sample_ids:
            continue

        scene_token = str(data["scene_token"].item() if data["scene_token"].ndim == 0 else data["scene_token"])
        timestamp = float(data["timestamp"].reshape(-1)[0])

        # Determine split
        if scene_token in train_scenes:
            target_list = train_data
        elif scene_token in val_scenes:
            target_list = val_data
        elif scene_token in test_scenes:
            target_list = test_data
        else:
            continue  # Should not happen

        # Load occupancy data: shape [5, 200, 200]
        Ov = data["Ov"].astype(bool)                      # [5, 200, 200]
        Oi = data["Oi"].astype(bool)                      # [5, 200, 200]
        GT = data["GT"].astype(bool)                      # [5, 200, 200]
        M_gt = data["gt_cell_valid_mask"].astype(bool)    # [5, 200, 200]

        # Warp valid mask is [200, 200], broadcast to [5, 200, 200]
        warp_valid_2d = data["warp_valid_mask"].astype(bool)  # [200, 200]
        M_warp = np.broadcast_to(warp_valid_2d, (5, 200, 200))  # [5, 200, 200]

        # Candidate domain mask
        M_valid = M_gt & M_warp  # [5, 200, 200]

        # Compute addition masks
        ADD = (~Ov) & Oi & M_valid  # [5, 200, 200]
        BA = ADD & GT
        HA = ADD & (~GT)

        # Process each horizon
        for horizon in range(N_HORIZONS):
            Ov_h = Ov[horizon]  # [200, 200]
            Oi_h = Oi[horizon]  # [200, 200]
            M_warp_h = M_warp[horizon]  # [200, 200]
            ADD_h = ADD[horizon]  # [200, 200]
            BA_h = BA[horizon]  # [200, 200]
            HA_h = HA[horizon]  # [200, 200]

            # Convert to float for feature extraction
            Pv_h = Ov_h.astype(np.float32)
            Pi_h = Oi_h.astype(np.float32)

            # Iterate over 4×4 patches
            n_patches = GRID_SIZE // PATCH_SIZE  # 200 / 4 = 50
            for patch_row in range(n_patches):
                for patch_col in range(n_patches):
                    # Extract patch region
                    r_start = patch_row * PATCH_SIZE
                    r_end = r_start + PATCH_SIZE
                    c_start = patch_col * PATCH_SIZE
                    c_end = c_start + PATCH_SIZE

                    # Extract patch data
                    ADD_patch = ADD_h[r_start:r_end, c_start:c_end]

                    # Only include patches with ADD > 0
                    ADD_count = ADD_patch.sum()
                    if ADD_count == 0:
                        continue

                    total_patches_with_add += 1

                    # Compute GT utility
                    BA_patch = BA_h[r_start:r_end, c_start:c_end]
                    HA_patch = HA_h[r_start:r_end, c_start:c_end]

                    BA_count = BA_patch.sum()
                    HA_count = HA_patch.sum()
                    utility = BA_count - HA_count

                    # Oracle decision
                    accept_label = 1 if utility > 0 else 0

                    if accept_label == 1:
                        total_accepted_patches += 1
                        accepted_ADD += ADD_count
                        accepted_BA += BA_count
                        accepted_HA += HA_count
                    else:
                        total_rejected_patches += 1
                        rejected_ADD += ADD_count
                        rejected_BA += BA_count
                        rejected_HA += HA_count

                    # Extract input features: X_R = [Pv, Pi, Pi-Pv, |Pi-Pv|, M^warp]
                    Pv_patch = Pv_h[r_start:r_end, c_start:c_end]  # [4, 4]
                    Pi_patch = Pi_h[r_start:r_end, c_start:c_end]  # [4, 4]
                    diff_patch = Pi_patch - Pv_patch  # [4, 4]
                    abs_diff_patch = np.abs(diff_patch)  # [4, 4]
                    warp_valid_patch = M_warp_h[r_start:r_end, c_start:c_end].astype(np.float32)  # [4, 4]

                    # Stack into feature tensor: [5, 4, 4]
                    features = np.stack([
                        Pv_patch,
                        Pi_patch,
                        diff_patch,
                        abs_diff_patch,
                        warp_valid_patch
                    ], axis=0)

                    # Create data record
                    record = {
                        'features': features,  # [5, 4, 4]
                        'label': accept_label,  # 0 or 1
                        'utility': utility,  # BA - HA
                        'ADD': ADD_count,
                        'BA': BA_count,
                        'HA': HA_count,
                        'horizon': horizon,
                        'patch_row': patch_row,
                        'patch_col': patch_col,
                        'sample_idx': sample_idx,
                        'scene_token': scene_token,
                        'timestamp': timestamp,
                    }

                    target_list.append(record)

print(f"Processed {len(cache_files)} cache files")
print(f"Extracted {len(train_data) + len(val_data) + len(test_data)} patch samples")
print()

# Step 4: Sanity check - must reproduce G2-B statistics exactly
print("=" * 70)
print("SANITY CHECK: G2-B STATISTICS REPRODUCTION")
print("=" * 70)
print()

acceptance_rate = (total_accepted_patches / total_patches_with_add * 100) if total_patches_with_add > 0 else 0.0
accepted_WAR = (accepted_HA / accepted_ADD * 100) if accepted_ADD > 0 else 0.0
rejected_WAR = (rejected_HA / rejected_ADD * 100) if rejected_ADD > 0 else 0.0

print(f"Valid samples                    : {len(valid_sample_ids)}")
print(f"Patch size                       : {PATCH_SIZE} × {PATCH_SIZE}")
print(f"Addition-containing patches      : {total_patches_with_add}")
print(f"Oracle accepted patches          : {total_accepted_patches}")
print(f"Oracle rejected patches          : {total_rejected_patches}")
print(f"Acceptance rate                  : {acceptance_rate:.1f}%")
print()

print("Accepted additions:")
print(f"  ADD = {accepted_ADD}")
print(f"  BA  = {accepted_BA}")
print(f"  HA  = {accepted_HA}")
print(f"  WAR = {accepted_WAR:.2f}%")
print()

print("Rejected additions:")
print(f"  ADD = {rejected_ADD}")
print(f"  BA  = {rejected_BA}")
print(f"  HA  = {rejected_HA}")
print(f"  WAR = {rejected_WAR:.2f}%")
print()

# Expected values from G2-B
EXPECTED_PATCHES = 119448
EXPECTED_ACCEPTED = 17805
EXPECTED_ACCEPTANCE_RATE = 14.9

# Verify
check_passed = True
if total_patches_with_add != EXPECTED_PATCHES:
    print(f"❌ MISMATCH: Expected {EXPECTED_PATCHES} patches, got {total_patches_with_add}")
    check_passed = False

if total_accepted_patches != EXPECTED_ACCEPTED:
    print(f"❌ MISMATCH: Expected {EXPECTED_ACCEPTED} accepted, got {total_accepted_patches}")
    check_passed = False

if abs(acceptance_rate - EXPECTED_ACCEPTANCE_RATE) > 0.1:
    print(f"❌ MISMATCH: Expected {EXPECTED_ACCEPTANCE_RATE}% acceptance rate, got {acceptance_rate:.1f}%")
    check_passed = False

if check_passed:
    print("✓ SANITY CHECK PASSED")
    print()
    print("G2-B statistics reproduced exactly. Proceeding to save dataset.")
else:
    print()
    print("❌ SANITY CHECK FAILED")
    print()
    print("CRITICAL: Statistics do not match G2-B.")
    print("DO NOT proceed to training. Debug the mismatch first.")
    import sys
    sys.exit(1)

print()
print("=" * 70)

# Step 5: Save dataset
print("Step 5: Saving dataset...")
print()

train_features = np.array([x['features'] for x in train_data], dtype=np.float32)
train_labels = np.array([x['label'] for x in train_data], dtype=np.int32)
train_metadata = [{k: v for k, v in x.items() if k not in ['features', 'label']} for x in train_data]

val_features = np.array([x['features'] for x in val_data], dtype=np.float32)
val_labels = np.array([x['label'] for x in val_data], dtype=np.int32)
val_metadata = [{k: v for k, v in x.items() if k not in ['features', 'label']} for x in val_data]

test_features = np.array([x['features'] for x in test_data], dtype=np.float32)
test_labels = np.array([x['label'] for x in test_data], dtype=np.int32)
test_metadata = [{k: v for k, v in x.items() if k not in ['features', 'label']} for x in test_data]

# Save
np.savez_compressed(
    os.path.join(OUT_DIR, 'g3_dataset_train.npz'),
    features=train_features,
    labels=train_labels,
    metadata=train_metadata
)

np.savez_compressed(
    os.path.join(OUT_DIR, 'g3_dataset_val.npz'),
    features=val_features,
    labels=val_labels,
    metadata=val_metadata
)

np.savez_compressed(
    os.path.join(OUT_DIR, 'g3_dataset_test.npz'),
    features=test_features,
    labels=test_labels,
    metadata=test_metadata
)

# Save scene splits for reference
with open(os.path.join(OUT_DIR, 'scene_splits.txt'), 'w') as f:
    f.write("SEED: 2026\n\n")
    f.write("Train scenes:\n")
    for scene in sorted(train_scenes):
        f.write(f"  {scene}\n")
    f.write("\nVal scenes:\n")
    for scene in sorted(val_scenes):
        f.write(f"  {scene}\n")
    f.write("\nTest scenes:\n")
    for scene in sorted(test_scenes):
        f.write(f"  {scene}\n")

print(f"Train: {len(train_data)} samples from {len(train_scenes)} scenes")
print(f"Val:   {len(val_data)} samples from {len(val_scenes)} scenes")
print(f"Test:  {len(test_data)} samples from {len(test_scenes)} scenes")
print()

# Dataset statistics
train_pos = train_labels.sum()
val_pos = val_labels.sum()
test_pos = test_labels.sum()

print("Class distribution:")
print(f"  Train: {train_pos} / {len(train_labels)} positive ({train_pos / len(train_labels) * 100:.1f}%)")
print(f"  Val:   {val_pos} / {len(val_labels)} positive ({val_pos / len(val_labels) * 100:.1f}%)")
print(f"  Test:  {test_pos} / {len(test_labels)} positive ({test_pos / len(test_labels) * 100:.1f}%)")
print()

print("Dataset saved:")
print(f"  {os.path.join(OUT_DIR, 'g3_dataset_train.npz')}")
print(f"  {os.path.join(OUT_DIR, 'g3_dataset_val.npz')}")
print(f"  {os.path.join(OUT_DIR, 'g3_dataset_test.npz')}")
print(f"  {os.path.join(OUT_DIR, 'scene_splits.txt')}")
print()

print("=" * 70)
print("G3-0: DATASET CONSTRUCTION COMPLETE")
print("=" * 70)
print()
print("Next step: G3-1 (Logistic Regression / MLP baseline)")
print("Only proceed after confirming sanity check passed.")
