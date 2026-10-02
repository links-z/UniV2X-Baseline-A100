import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


PATCH_SIZE = 4
GRID_SIZE = 200
PATCH_GRID = GRID_SIZE // PATCH_SIZE   # 50
NUM_HORIZONS = 5

# 固定邻居顺序
NEIGHBOR_OFFSETS = [
    (-1, -1),  # NW
    (-1,  0),  # N
    (-1,  1),  # NE
    ( 0, -1),  # W
    ( 0,  1),  # E
    ( 1, -1),  # SW
    ( 1,  0),  # S
    ( 1,  1),  # SE
]


def canonical_sample_idx(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    s = str(x)

    if s.isdigit():
        return str(int(s))

    return s


def scalar_from_npz(arr):
    arr = np.asarray(arr)

    if arr.size == 1:
        return arr.reshape(-1)[0].item()

    raise ValueError(
        f"Expected scalar metadata, got shape={arr.shape}"
    )


def get_key(npz, candidates, required=True):
    for k in candidates:
        if k in npz.files:
            return k

    if required:
        raise KeyError(
            f"Cannot find any key from {candidates}. "
            f"Available keys: {npz.files}"
        )

    return None


def squeeze_occ(arr, name):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.shape != (NUM_HORIZONS, GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"{name} shape mismatch: {arr.shape}, "
            f"expected {(NUM_HORIZONS, GRID_SIZE, GRID_SIZE)}"
        )

    return arr.astype(np.float32, copy=False)


def squeeze_warp(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"warp_valid_mask shape mismatch: {arr.shape}, "
            f"expected {(GRID_SIZE, GRID_SIZE)}"
        )

    return arr.astype(np.float32, copy=False)


def build_cache_index(cache_dir):
    cache_dir = Path(cache_dir)

    files = sorted(cache_dir.rglob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No .npz files found under: {cache_dir}"
        )

    index = {}

    skipped = 0

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as d:
                if "sample_idx" not in d.files:
                    skipped += 1
                    continue

                pv_key = get_key(
                    d,
                    ["Pv", "Pveh"],
                    required=False,
                )

                pi_key = get_key(
                    d,
                    ["Pi_aligned", "Pinfra_aligned"],
                    required=False,
                )

                warp_key = get_key(
                    d,
                    ["warp_valid_mask"],
                    required=False,
                )

                if pv_key is None or pi_key is None or warp_key is None:
                    skipped += 1
                    continue

                sample_idx = canonical_sample_idx(
                    scalar_from_npz(d["sample_idx"])
                )

                if sample_idx in index:
                    raise RuntimeError(
                        f"Duplicate sample_idx={sample_idx}\n"
                        f"  A: {index[sample_idx]}\n"
                        f"  B: {path}\n"
                    )

                index[sample_idx] = str(path)

        except Exception as e:
            print(f"[WARN] Skip {path}: {e}")
            skipped += 1

    print("=" * 72)
    print("G0 CACHE INDEX")
    print("=" * 72)
    print(f"Search dir      : {cache_dir}")
    print(f"NPZ discovered : {len(files)}")
    print(f"Valid cache    : {len(index)}")
    print(f"Skipped        : {skipped}")

    if len(index) == 0:
        raise RuntimeError(
            "No valid G0 Occupancy cache files were indexed."
        )

    return index


def load_full_cache(path):
    with np.load(path, allow_pickle=True) as d:
        # Use Soft Pv/Pi
        pv_key = get_key(d, ["Pv", "Pveh"])
        pi_key = get_key(d, ["Pi_aligned", "Pinfra_aligned"])
        warp_key = get_key(d, ["warp_valid_mask"])

        pv = squeeze_occ(d[pv_key], pv_key)
        pi = squeeze_occ(d[pi_key], pi_key)
        warp = squeeze_warp(d[warp_key])

    return pv, pi, warp


def make_full_feature(pv, pi, warp, horizon):
    """
    Return: [5, 200, 200]

    Channel order:
        0 Pv (soft probability ego)
        1 Pi_aligned (soft probability infra)
        2 Pi_aligned - Pv
        3 abs(Pi_aligned - Pv)
        4 warp_valid_mask
    """
    pvh = pv[horizon].astype(np.float32)
    pih = pi[horizon].astype(np.float32)

    diff = pih - pvh

    feat = np.stack(
        [
            pvh,
            pih,
            diff,
            np.abs(diff),
            warp,
        ],
        axis=0,
    ).astype(np.float32)

    return feat


def patch_means(full_feature):
    """
    [5, 200, 200] -> [5, 50, 50]

    每个 4x4 Patch 对每个 channel 求 mean。
    """
    x = full_feature.reshape(
        5,
        PATCH_GRID,
        PATCH_SIZE,
        PATCH_GRID,
        PATCH_SIZE,
    )

    return x.mean(axis=(2, 4))


def extract_neighbor_stats(mean_grid, patch_row, patch_col):
    """
    mean_grid: [5, 50, 50]

    8 neighbors × 5 channels = 40 dimensions.

    边界外邻居使用全 0。
    """
    out = []

    for dr, dc in NEIGHBOR_OFFSETS:
        nr = patch_row + dr
        nc = patch_col + dc

        if (
            nr < 0
            or nr >= PATCH_GRID
            or nc < 0
            or nc >= PATCH_GRID
        ):
            vals = np.zeros(5, dtype=np.float32)
        else:
            vals = mean_grid[:, nr, nc].astype(
                np.float32,
                copy=False,
            )

        out.append(vals)

    out = np.concatenate(out, axis=0)

    assert out.shape == (40,)

    return out


def make_horizon_onehot(h):
    x = np.zeros(NUM_HORIZONS, dtype=np.float32)
    x[h] = 1.0
    return x


def make_spatial_position(r, c):
    """
    normalized patch coordinates (u, v) in approximately [-0.98, 0.98]
    """
    u = 2.0 * ((c + 0.5) / PATCH_GRID) - 1.0
    v = 2.0 * ((r + 0.5) / PATCH_GRID) - 1.0

    return np.array([u, v], dtype=np.float32)


def metadata_equal(a, b):
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if x != y:
            return False

    return True


def process_split(split, g3_dir, out_dir, cache_index):
    src_path = os.path.join(
        g3_dir,
        f"g3_dataset_{split}.npz",
    )

    print("\n" + "=" * 72)
    print(f"PROCESS SPLIT: {split.upper()}")
    print("=" * 72)
    print(f"Source: {src_path}")

    d = np.load(src_path, allow_pickle=True)

    local = d["features"].astype(np.float32)
    labels = d["labels"].copy()
    metadata = d["metadata"].copy()

    if local.ndim != 4 or local.shape[1:] != (5, 4, 4):
        raise ValueError(
            f"Unexpected G3 feature shape: {local.shape}"
        )

    n = len(local)

    if len(labels) != n or len(metadata) != n:
        raise ValueError(
            "features / labels / metadata length mismatch"
        )

    # 最终 127-dim
    # 80 local + 5 horizon + 2 spatial + 40 neighborhood
    features_out = np.empty(
        (n, 127),
        dtype=np.float32,
    )

    neighbors_out = np.empty(
        (n, 40),
        dtype=np.float32,
    )

    # 按 sample 分组，只加载一次 G0 cache
    groups = defaultdict(list)

    for i, m in enumerate(metadata):
        sid = canonical_sample_idx(m["sample_idx"])
        groups[sid].append(i)

    print(f"Samples represented: {len(groups)}")
    print(f"Candidate patches   : {n}")

    missing_samples = [
        sid for sid in groups
        if sid not in cache_index
    ]

    if missing_samples:
        print("\nMissing sample_idx examples:")
        for sid in missing_samples[:20]:
            print(" ", sid)

        raise RuntimeError(
            f"{len(missing_samples)} sample_idx values "
            f"cannot be found in G0 cache."
        )

    max_center_diff = 0.0
    checked_centers = 0

    for group_no, (sid, indices) in enumerate(
        groups.items(),
        start=1,
    ):
        cache_path = cache_index[sid]

        pv, pi, warp = load_full_cache(cache_path)

        # 同一个 sample 最多需要构造 5 个 horizon
        full_feature_by_h = {}
        mean_grid_by_h = {}

        for i in indices:
            m = metadata[i]

            h = int(m["horizon"])
            r = int(m["patch_row"])
            c = int(m["patch_col"])

            if not (0 <= h < NUM_HORIZONS):
                raise ValueError(
                    f"Invalid horizon={h}, index={i}"
                )

            if not (
                0 <= r < PATCH_GRID
                and 0 <= c < PATCH_GRID
            ):
                raise ValueError(
                    f"Invalid patch coordinates "
                    f"(r={r}, c={c}), index={i}"
                )

            if h not in full_feature_by_h:
                full_feat = make_full_feature(
                    pv,
                    pi,
                    warp,
                    h,
                )

                full_feature_by_h[h] = full_feat
                mean_grid_by_h[h] = patch_means(full_feat)

            full_feat = full_feature_by_h[h]
            mean_grid = mean_grid_by_h[h]

            # patch_row/patch_col are 50x50 patch-grid indices
            # Pixel start = index * 4
            y0 = r * PATCH_SIZE
            x0 = c * PATCH_SIZE

            center_from_cache = full_feat[
                :,
                y0:y0 + PATCH_SIZE,
                x0:x0 + PATCH_SIZE,
            ]

            if center_from_cache.shape != (5, 4, 4):
                raise RuntimeError(
                    f"Center patch extraction failed: "
                    f"{center_from_cache.shape}"
                )

            # Soft G3 local feature 与 G0 cache 必须一致
            diff = float(
                np.max(
                    np.abs(
                        center_from_cache.astype(np.float32)
                        -
                        local[i].astype(np.float32)
                    )
                )
            )

            max_center_diff = max(
                max_center_diff,
                diff,
            )
            checked_centers += 1

            neighbor = extract_neighbor_stats(
                mean_grid,
                r,
                c,
            )

            horizon = make_horizon_onehot(h)
            spatial = make_spatial_position(r, c)

            local_flat = local[i].reshape(-1)

            feature = np.concatenate(
                [
                    local_flat,  # 80
                    horizon,     # 5
                    spatial,     # 2
                    neighbor,    # 40
                ],
                axis=0,
            ).astype(np.float32)

            if feature.shape != (127,):
                raise RuntimeError(
                    f"Feature dim error: {feature.shape}"
                )

            features_out[i] = feature
            neighbors_out[i] = neighbor

        if group_no % 50 == 0 or group_no == len(groups):
            print(
                f"Processed samples: "
                f"{group_no}/{len(groups)}"
            )

    print("\nCenter-patch consistency:")
    print(f"Checked patches  : {checked_centers}")
    print(f"Max abs diff     : {max_center_diff:.10f}")

    # 如果这里不一致，不允许继续
    if max_center_diff > 1e-5:
        raise RuntimeError(
            "G0 cache center feature does not match "
            "existing Soft G3 local feature.\n"
            f"max_abs_diff={max_center_diff}\n"
            "Stop here and audit feature definitions."
        )

    # 基本一致性
    assert features_out.shape == (n, 127)
    assert neighbors_out.shape == (n, 40)

    # labels / metadata 不重新生成，直接继承 Soft G3
    labels_check = labels.copy()
    metadata_check = metadata.copy()

    assert np.array_equal(
        labels_check,
        labels,
    )

    assert metadata_equal(
        metadata_check,
        metadata,
    )

    out_path = os.path.join(
        out_dir,
        f"g3_neighborhood_{split}.npz",
    )

    np.savez_compressed(
        out_path,
        features=features_out,
        neighbor_features=neighbors_out,
        labels=labels,
        metadata=metadata,
    )

    print("\nOutput:")
    print(f"  {out_path}")
    print(f"  features          : {features_out.shape}")
    print(f"  neighbor_features : {neighbors_out.shape}")
    print(f"  labels            : {labels.shape}")
    print(f"  metadata          : {metadata.shape}")
    print(f"  positive          : {int(labels.sum())}")
    print(f"  positive rate     : {labels.mean():.6f}")

    # 回读验证
    saved = np.load(out_path, allow_pickle=True)

    if not np.array_equal(
        saved["labels"],
        labels,
    ):
        raise RuntimeError(
            "Saved labels differ from Soft G3 labels."
        )

    if not metadata_equal(
        saved["metadata"],
        metadata,
    ):
        raise RuntimeError(
            "Saved metadata differ from Soft G3 metadata."
        )

    print("  labels identity   : PASS")
    print("  metadata identity : PASS")
    print("  center consistency: PASS")

    return {
        "split": split,
        "n": n,
        "positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "max_center_diff": float(max_center_diff),
        "out_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory containing G0 full-cache .npz files",
    )

    parser.add_argument(
        "--g3-dir",
        type=str,
        default="/root/autodl-tmp/UniV2X/G0-G1-G2/G3/soft",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/autodl-tmp/UniV2X/G0-G1-G2/G3/soft",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache_index = build_cache_index(
        args.cache_dir
    )

    results = []

    for split in ["train", "val", "test"]:
        results.append(
            process_split(
                split,
                args.g3_dir,
                args.out_dir,
                cache_index,
            )
        )

    print("\n" + "=" * 72)
    print("G3-1D SOFT NEIGHBORHOOD DATASET SUMMARY")
    print("=" * 72)

    for r in results:
        print(
            f"{r['split']:5s} | "
            f"N={r['n']:6d} | "
            f"positive={r['positive']:5d} | "
            f"rate={r['positive_rate']:.6f} | "
            f"center_diff={r['max_center_diff']:.3e}"
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

    print("\nExpected sample counts: PASS")
    print("Labels identity       : PASS")
    print("Metadata identity     : PASS")
    print("Center feature audit  : PASS")

    print("\n" + "=" * 72)
    print("G3-1D SOFT DATASET CONSTRUCTION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
