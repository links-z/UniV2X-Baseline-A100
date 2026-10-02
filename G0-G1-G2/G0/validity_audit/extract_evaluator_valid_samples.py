#!/usr/bin/env python3
"""
G3: Extract Evaluator Valid Samples (Offline - Direct from PKL)

Directly read from the dataset pkl file and replicate the evaluator's
occ_has_invalid_frame logic without needing full config/dataset initialization.
"""

import os
import pickle
import numpy as np

# Paths - using val set as test set
PKL_PATH = "/root/autodl-tmp/UniV2X/data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl"
OUT_FILE = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3/evaluator_valid_samples.txt"

print("=" * 70)
print("EVALUATOR OCC VALIDITY EXTRACTION (OFFLINE)")
print("=" * 70)
print()

# Load dataset pkl
print(f"Loading dataset from: {PKL_PATH}")
with open(PKL_PATH, 'rb') as f:
    data_infos = pickle.load(f)

print(f"Dataset loaded: {len(data_infos['infos'])} samples")
print()

infos = data_infos['infos']

# Parameters from config (based on UniV2X occ_receptive_field = 3)
OCC_RECEPTIVE_FIELD = 3
PAST_STEPS = OCC_RECEPTIVE_FIELD - 1  # 3 - 1 = 2 past frames: [t-2, t-1]
OCC_ONLY_TOTAL_FRAMES = 7  # total evaluator window: 2 past + 1 current + 4 future

valid_records = []
invalid_records = []

for idx, info in enumerate(infos):
    sample_idx = info['token']
    scene_token = info.get('scene_token', '')
    timestamp = info.get('timestamp', 0) / 1e6  # Convert to seconds

    # Replicate get_prev_indices and get_future_indices logic
    # These functions check if samples are in the same sequence

    # For prev_indices: go back PAST_STEPS frames
    prev_indices = []
    for i in range(1, PAST_STEPS + 1):
        candidate_idx = idx - i
        if candidate_idx >= 0:
            candidate_info = infos[candidate_idx]
            # Check if in same scene
            if candidate_info.get('scene_token', '') == scene_token:
                prev_indices.append(candidate_idx)
            else:
                prev_indices.append(-1)  # Invalid frame marker
        else:
            prev_indices.append(-1)
    prev_indices = prev_indices[::-1]  # Reverse to chronological order

    # For future_indices: go forward (OCC_ONLY_TOTAL_FRAMES - PAST_STEPS - 1) frames
    # OCC_ONLY_TOTAL_FRAMES = 7, PAST_STEPS = 4, so future = 7 - 4 - 1 = 2 frames
    future_steps = OCC_ONLY_TOTAL_FRAMES - PAST_STEPS - 1
    future_indices = []
    for i in range(1, future_steps + 1):
        candidate_idx = idx + i
        if candidate_idx < len(infos):
            candidate_info = infos[candidate_idx]
            # Check if in same scene
            if candidate_info.get('scene_token', '') == scene_token:
                future_indices.append(candidate_idx)
            else:
                future_indices.append(-1)  # Invalid frame marker
        else:
            future_indices.append(-1)

    # Build all_frames
    all_frames = prev_indices + [idx] + future_indices

    # The critical check: whether invalid frames (-1) exist in occ frames
    has_invalid_frame = -1 in all_frames[:OCC_ONLY_TOTAL_FRAMES]

    # Evaluator logic: occ_to_eval = not occ_has_invalid_frame
    occ_to_eval = not has_invalid_frame

    record = (sample_idx, scene_token, timestamp)

    if occ_to_eval:
        valid_records.append(record)
    else:
        invalid_records.append(record)

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} / {len(infos)} samples...")

print(f"Processed all {len(infos)} samples")
print()

# Sort by scene_token, timestamp, sample_idx
valid_records = sorted(valid_records, key=lambda x: (x[1], x[2], x[0]))

print("=" * 70)
print("EVALUATOR OCC VALIDITY STATISTICS")
print("=" * 70)
print(f"Total samples   : {len(infos)}")
print(f"Valid samples   : {len(valid_records)}")
print(f"Invalid samples : {len(invalid_records)}")
if len(infos) > 0:
    print(f"Valid ratio     : {len(valid_records) / len(infos) * 100:.2f}%")
print()

# Save to file
with open(OUT_FILE, "w") as f:
    f.write("sample_idx\tscene_token\ttimestamp\n")
    for sample_idx, scene_token, timestamp in valid_records:
        f.write(f"{sample_idx}\t{scene_token}\t{timestamp:.6f}\n")

print(f"Saved valid-sample list to: {OUT_FILE}")
print()
print("First 10 valid samples:")
for x in valid_records[:10]:
    print(x)
