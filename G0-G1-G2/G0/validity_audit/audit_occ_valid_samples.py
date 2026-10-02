#!/usr/bin/env python3
"""
G3: Cache OCC Validity Audit

Export the list of valid occupancy samples from cache based on future_valid_mask.all().
This will be compared against evaluator's occ_to_eval list to verify identity consistency.
"""

import os
import glob
import numpy as np

CACHE_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full"
OUT_FILE = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3/cache_valid_samples.txt"

files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

if len(files) == 0:
    raise RuntimeError(f"No cache files found in {CACHE_DIR}")

valid_records = []
invalid_records = []

for path in files:
    with np.load(path, allow_pickle=True) as data:
        future_valid = np.asarray(data["future_valid_mask"]).astype(bool)

        sample_idx = data["sample_idx"]
        if isinstance(sample_idx, np.ndarray):
            sample_idx = sample_idx.item() if sample_idx.ndim == 0 else str(sample_idx)
        sample_idx = str(sample_idx)

        scene_token = data["scene_token"]
        if isinstance(scene_token, np.ndarray):
            scene_token = scene_token.item() if scene_token.ndim == 0 else str(scene_token)
        scene_token = str(scene_token)

        timestamp = float(np.asarray(data["timestamp"]).reshape(-1)[0])

        record = (
            sample_idx,
            scene_token,
            timestamp,
            os.path.basename(path),
        )

        if future_valid.all():
            valid_records.append(record)
        else:
            invalid_records.append(record)

valid_records = sorted(valid_records, key=lambda x: (x[1], x[2], x[0]))

print("=" * 70)
print("CACHE OCC VALIDITY AUDIT")
print("=" * 70)
print(f"Total samples   : {len(files)}")
print(f"Valid samples   : {len(valid_records)}")
print(f"Invalid samples : {len(invalid_records)}")
if len(files) > 0:
    print(f"Valid ratio     : {len(valid_records) / len(files) * 100:.2f}%")

with open(OUT_FILE, "w") as f:
    f.write("sample_idx\tscene_token\ttimestamp\tcache_file\n")
    for sample_idx, scene_token, timestamp, filename in valid_records:
        f.write(
            f"{sample_idx}\t{scene_token}\t"
            f"{timestamp:.6f}\t{filename}\n"
        )

print()
print(f"Saved valid-sample list to: {OUT_FILE}")
print()
print("First 10 valid samples:")
for x in valid_records[:10]:
    print(x)
