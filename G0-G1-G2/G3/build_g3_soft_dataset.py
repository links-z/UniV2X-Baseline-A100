import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


PATCH_SIZE = 4
GRID_SIZE = 200
NUM_HORIZONS = 5
THRESHOLD = 0.1


def canonical_sample_idx(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    s = str(x)

    if s.isdigit():
        return str(int(s))

    return s


def scalar_from_npz(arr):
    arr = np.asarray(arr)

    if arr.size != 1:
        raise ValueError(
            f"Expected scalar metadata, got shape={arr.shape}"
        )

    return arr.reshape(-1)[0].item()


def squeeze_occ(arr, name):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.shape != (NUM_HORIZONS, GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"{name} shape={arr.shape}, "
            f"expected {(NUM_HORIZONS, GRID_SIZE, GRID_SIZE)}"
        )

    return arr.astype(np.float32, copy=False)


def squeeze_warp(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"warp_valid_mask shape={arr.shape}, "
            f"expected {(GRID_SIZE, GRID_SIZE)}"
        )

    return arr.astype(np.float32, copy=False)


def metadata_equal(a, b):
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if x != y:
            return False

    return True


def build_cache_index(cache_dir):
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.rglob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No npz files found under {cache_dir}"
        )

    index = {}
    skipped = 0

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as d:

                required = {
                    "sample_idx",
                    "Pv",
                    "Pi_aligned",
                    "warp_valid_mask",
                }

                if not required.issubset(set(d.files)):
                    skipped += 1
                    continue

                sid = canonical_sample_idx(
                    scalar_from_npz(d["sample_idx"])
                )

                if sid in index:
                    raise RuntimeError(
                        f"Duplicate sample_idx={sid}\n"
                        f"A: {index[sid]}\n"
                        f"B: {path}"
                    )

                index[sid] = str(path)

        except Exception as e:
            print(f"[WARN] skip {path}: {e}")
            skipped += 1

    print("=" * 72)
    print("G0 SOFT CACHE INDEX")
    print("=" * 72)
    print(f"Search dir      : {cache_dir}")
    print(f"NPZ discovered : {len(files)}")
    print(f"Valid caches    : {len(index)}")
    print(f"Skipped         : {skipped}")

    if not index:
        raise RuntimeError(
            "No valid G0 cache with Pv/Pi_aligned was found."
        )

    return index


def load_cache(path):
    with np.load(path, allow_pickle=True) as d:
        pv = squeeze_occ(d["Pv"], "Pv")
        pi = squeeze_occ(
            d["Pi_aligned"],
            "Pi_aligned",
        )
        warp = squeeze_warp(
            d["warp_valid_mask"]
        )

    return pv, pi, warp


def build_soft_patch(pv, pi, warp, h, r, c):
    y0 = r * PATCH_SIZE
    x0 = c * PATCH_SIZE

    pv_patch = pv[
        h,
        y0:y0 + PATCH_SIZE,
        x0:x0 + PATCH_SIZE,
    ]

    pi_patch = pi[
        h,
        y0:y0 + PATCH_SIZE,
        x0:x0 + PATCH_SIZE,
    ]

    warp_patch = warp[
        y0:y0 + PATCH_SIZE,
        x0:x0 + PATCH_SIZE,
    ]

    if pv_patch.shape != (4, 4):
        raise RuntimeError(
            f"Pv patch shape error: {pv_patch.shape}"
        )

    if pi_patch.shape != (4, 4):
        raise RuntimeError(
            f"Pi patch shape error: {pi_patch.shape}"
        )

    if warp_patch.shape != (4, 4):
        raise RuntimeError(
            f"Warp patch shape error: {warp_patch.shape}"
        )

    diff = pi_patch - pv_patch

    soft = np.stack(
        [
            pv_patch,
            pi_patch,
            diff,
            np.abs(diff),
            warp_patch,
        ],
        axis=0,
    ).astype(np.float32)

    return soft


def soft_to_expected_binary(soft):
    """
    Reconstruct the existing Binary G3 representation from Soft input.

    soft[0] = Pv
    soft[1] = Pi_aligned
    soft[4] = warp mask

    Returns Binary feature [5, 4, 4]:
        Ov = 1[Pv > 0.1]
        Oi = 1[Pi > 0.1]
        Oi - Ov
        |Oi - Ov|
        warp mask
    """
    ov = (soft[0] > THRESHOLD).astype(np.float32)
    oi = (soft[1] > THRESHOLD).astype(np.float32)

    diff = oi - ov

    binary = np.stack(
        [
            ov,
            oi,
            diff,
            np.abs(diff),
            soft[4],  # warp mask unchanged
        ],
        axis=0,
    ).astype(np.float32)

    return binary


def process_split(
    split,
    binary_dir,
    cache_index,
    out_dir,
):
    binary_path = os.path.join(
        binary_dir,
        f"g3_dataset_{split}.npz",
    )

    print("\n" + "=" * 72)
    print(f"PROCESS SPLIT: {split.upper()}")
    print("=" * 72)
    print(f"Binary source: {binary_path}")

    d = np.load(binary_path, allow_pickle=True)

    binary_features = d["features"]
    labels = d["labels"].copy()
    metadata = d["metadata"].copy()

    n = len(labels)

    print(f"Samples       : {n}")
    print(f"Positive      : {int(labels.sum())}")
    print(f"Positive rate : {labels.mean():.6f}")

    if binary_features.shape != (n, 5, 4, 4):
        raise ValueError(
            f"Binary feature shape error: "
            f"{binary_features.shape}"
        )

    soft_features = np.empty_like(
        binary_features,
        dtype=np.float32,
    )

    # Group by sample for efficient cache loading
    groups = defaultdict(list)

    for i, m in enumerate(metadata):
        sid = canonical_sample_idx(m["sample_idx"])
        groups[sid].append(i)

    print(f"Unique samples: {len(groups)}")

    missing = [
        sid for sid in groups
        if sid not in cache_index
    ]

    if missing:
        print("\nMissing sample_idx:")
        for sid in missing[:20]:
            print(f"  {sid}")

        raise RuntimeError(
            f"{len(missing)} samples not found in G0 cache"
        )

    # Consistency check counters
    max_soft_binary_diff = 0.0
    checked_patches = 0
    mismatched_patches = 0

    for group_no, (sid, indices) in enumerate(
        groups.items(),
        start=1,
    ):
        cache_path = cache_index[sid]

        pv, pi, warp = load_cache(cache_path)

        for i in indices:
            m = metadata[i]

            h = int(m["horizon"])
            r = int(m["patch_row"])
            c = int(m["patch_col"])

            if not (0 <= h < NUM_HORIZONS):
                raise ValueError(
                    f"Invalid horizon={h}, index={i}"
                )

            if not (0 <= r < 50 and 0 <= c < 50):
                raise ValueError(
                    f"Invalid patch coords (r={r}, c={c}), "
                    f"index={i}"
                )

            soft = build_soft_patch(pv, pi, warp, h, r, c)

            # Verify: Soft -> Binary should match existing Binary
            binary_reconstructed = soft_to_expected_binary(soft)

            binary_existing = binary_features[i]

            diff = float(
                np.max(
                    np.abs(
                        binary_reconstructed - binary_existing
                    )
                )
            )

            max_soft_binary_diff = max(
                max_soft_binary_diff,
                diff,
            )

            if diff > 1e-5:
                mismatched_patches += 1

            checked_patches += 1

            soft_features[i] = soft

        if group_no % 50 == 0 or group_no == len(groups):
            print(
                f"Processed samples: {group_no}/{len(groups)}"
            )

    print("\nSoft -> Binary consistency check:")
    print(f"  Checked patches      : {checked_patches}")
    print(f"  Max abs diff         : {max_soft_binary_diff:.10f}")
    print(f"  Mismatched (>1e-5)   : {mismatched_patches}")

    if max_soft_binary_diff > 1e-5:
        raise RuntimeError(
            "Soft probability thresholding does NOT reproduce "
            "existing Binary G3 features.\n"
            f"max_diff={max_soft_binary_diff}\n"
            "This indicates G0 cache or Binary G3 dataset "
            "may be inconsistent."
        )

    # Verify labels/metadata identity
    if not np.array_equal(labels, d["labels"]):
        raise RuntimeError("Labels mismatch")

    if not metadata_equal(metadata, d["metadata"]):
        raise RuntimeError("Metadata mismatch")

    out_path = os.path.join(
        out_dir,
        f"g3_dataset_{split}.npz",
    )

    np.savez_compressed(
        out_path,
        features=soft_features,
        labels=labels,
        metadata=metadata,
    )

    print("\nOutput:")
    print(f"  {out_path}")
    print(f"  features : {soft_features.shape}")
    print(f"  labels   : {labels.shape}")
    print(f"  metadata : {metadata.shape}")
    print(f"  Labels identity   : PASS")
    print(f"  Metadata identity : PASS")
    print(f"  Soft->Binary audit: PASS")

    # Verify soft is NOT binary
    pv_channel = soft_features[:, 0, :, :]
    non_binary_ratio = (
        (pv_channel != 0.0) & (pv_channel != 1.0)
    ).mean()

    print(f"  Pv non-binary ratio: {non_binary_ratio:.6f}")

    if non_binary_ratio < 0.5:
        raise RuntimeError(
            f"Soft features appear to be binary! "
            f"non_binary_ratio={non_binary_ratio}"
        )

    return {
        "split": split,
        "n": n,
        "positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "max_diff": float(max_soft_binary_diff),
        "mismatched": mismatched_patches,
        "non_binary_ratio": float(non_binary_ratio),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="G0 cache directory with Pv/Pi_aligned",
    )

    parser.add_argument(
        "--binary-dir",
        type=str,
        default="G0-G1-G2/G3",
        help="Directory containing existing Binary G3 dataset",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="G0-G1-G2/G3/soft",
        help="Output directory for Soft G3 dataset",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache_index = build_cache_index(args.cache_dir)

    results = []

    for split in ["train", "val", "test"]:
        results.append(
            process_split(
                split,
                args.binary_dir,
                cache_index,
                args.out_dir,
            )
        )

    print("\n" + "=" * 72)
    print("G3 SOFT DATASET SUMMARY")
    print("=" * 72)

    for r in results:
        print(
            f"{r['split']:5s} | "
            f"N={r['n']:6d} | "
            f"pos={r['positive']:5d} | "
            f"rate={r['positive_rate']:.6f} | "
            f"diff={r['max_diff']:.3e} | "
            f"mismatch={r['mismatched']:4d} | "
            f"non_bin={r['non_binary_ratio']:.4f}"
        )

    expected = {
        "train": 86143,
        "val": 18555,
        "test": 14750,
    }

    for r in results:
        if r["n"] != expected[r["split"]]:
            raise RuntimeError(
                f"{r['split']} N mismatch: "
                f"{r['n']} != {expected[r['split']]}"
            )

    print("\nExpected sample counts : PASS")
    print("Labels identity        : PASS")
    print("Metadata identity      : PASS")
    print("Soft->Binary audit     : PASS")
    print("Soft non-binary check  : PASS")

    print("\n" + "=" * 72)
    print("G3 SOFT DATASET CONSTRUCTION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
