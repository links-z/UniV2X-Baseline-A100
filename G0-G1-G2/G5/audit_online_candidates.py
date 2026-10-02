#!/usr/bin/env python3

from pathlib import Path
import numpy as np

ROOT = Path("/root/autodl-tmp/UniV2X")
CACHE_DIR = ROOT / "G0-G1-G2/cache_full"
TEST_NPZ = ROOT / "G0-G1-G2/G3/soft/g3_neighborhood_test.npz"

PATCH_SIZE = 4
PATCH_GRID = 50
H = 5


def canonical_sample_idx(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def squeeze_occ(x):
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    return x


def patch_any(mask):
    # [5, 200, 200] -> [5, 50, 50]
    x = mask.reshape(
        H,
        PATCH_GRID,
        PATCH_SIZE,
        PATCH_GRID,
        PATCH_SIZE,
    )
    return x.any(axis=(2, 4))


# ------------------------------------------------------------
# 1. Load held-out Test sample IDs from G3 metadata
# ------------------------------------------------------------
dtest = np.load(TEST_NPZ, allow_pickle=True)
metadata = dtest["metadata"]

test_sample_ids = sorted(
    {
        canonical_sample_idx(m["sample_idx"])
        for m in metadata
    }
)

print("=" * 72)
print("G5-0A ONLINE CANDIDATE CONSISTENCY AUDIT")
print("=" * 72)
print("Test samples from G3 metadata:", len(test_sample_ids))
print("Offline metadata candidate patches:", len(metadata))
print()


# ------------------------------------------------------------
# 2. Build sample_idx -> cache path mapping
# ------------------------------------------------------------
cache_index = {}

for p in sorted(CACHE_DIR.glob("*.npz")):
    with np.load(p, allow_pickle=True) as d:
        sid = canonical_sample_idx(d["sample_idx"])
    cache_index[sid] = p


# ------------------------------------------------------------
# 3. Compare offline vs inference-safe candidate sets
# ------------------------------------------------------------
offline_total = 0
online_total = 0
extra_online_total = 0
missing_online_total = 0

exact_samples = 0
mismatch_samples = 0

offline_h = np.zeros(H, dtype=np.int64)
online_h = np.zeros(H, dtype=np.int64)
extra_h = np.zeros(H, dtype=np.int64)

for sid in test_sample_ids:

    if sid not in cache_index:
        raise FileNotFoundError(f"Cache not found for sample_idx={sid}")

    with np.load(cache_index[sid], allow_pickle=True) as d:
        Ov = squeeze_occ(d["Ov"]).astype(bool)
        Oi = squeeze_occ(d["Oi"]).astype(bool)
        M_gt = squeeze_occ(d["gt_cell_valid_mask"]).astype(bool)
        M_warp_2d = np.asarray(d["warp_valid_mask"]).astype(bool)

    assert Ov.shape == (5, 200, 200), (sid, Ov.shape)
    assert Oi.shape == (5, 200, 200), (sid, Oi.shape)
    assert M_gt.shape == (5, 200, 200), (sid, M_gt.shape)
    assert M_warp_2d.shape == (200, 200), (sid, M_warp_2d.shape)

    M_warp = np.broadcast_to(
        M_warp_2d[None],
        (5, 200, 200)
    )

    # G3/G4 offline definition
    add_offline = (
        (~Ov)
        & Oi
        & M_gt
        & M_warp
    )

    # Inference-safe definition
    add_online = (
        (~Ov)
        & Oi
        & M_warp
    )

    c_offline = patch_any(add_offline)
    c_online = patch_any(add_online)

    extra_online = c_online & (~c_offline)
    missing_online = c_offline & (~c_online)

    n_offline = int(c_offline.sum())
    n_online = int(c_online.sum())
    n_extra = int(extra_online.sum())
    n_missing = int(missing_online.sum())

    offline_total += n_offline
    online_total += n_online
    extra_online_total += n_extra
    missing_online_total += n_missing

    offline_h += c_offline.sum(axis=(1, 2))
    online_h += c_online.sum(axis=(1, 2))
    extra_h += extra_online.sum(axis=(1, 2))

    if np.array_equal(c_offline, c_online):
        exact_samples += 1
    else:
        mismatch_samples += 1


# ------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------
print("========== Overall ==========")
print("Offline candidate patches :", offline_total)
print("Online candidate patches  :", online_total)
print("Extra online patches      :", extra_online_total)
print("Missing online patches    :", missing_online_total)

if offline_total > 0:
    print(
        "Online / Offline ratio   :",
        f"{online_total / offline_total:.6f}"
    )

if online_total > 0:
    print(
        "Extra-online ratio       :",
        f"{extra_online_total / online_total:.6%}"
    )

print()
print("Exact-match samples       :", exact_samples)
print("Mismatch samples          :", mismatch_samples)
print("Total test samples        :", len(test_sample_ids))

print()
print("========== Horizon-wise ==========")

for h in range(H):
    extra_ratio = (
        extra_h[h] / online_h[h]
        if online_h[h] > 0
        else 0.0
    )

    print(
        f"t={h}: "
        f"offline={offline_h[h]:6d} | "
        f"online={online_h[h]:6d} | "
        f"extra={extra_h[h]:6d} | "
        f"extra_ratio={extra_ratio:.4%}"
    )

print()
print("========== Sanity ==========")

# Offline count should reproduce G3 Test candidate count
print(
    "Metadata candidate count :", len(metadata)
)

if offline_total == len(metadata):
    print("Offline candidate reproduction: PASS")
else:
    print(
        "Offline candidate reproduction: FAIL",
        offline_total,
        "!=",
        len(metadata)
    )

if missing_online_total == 0:
    print("Offline ⊆ Online: PASS")
else:
    print(
        "Offline ⊆ Online: FAIL, missing =",
        missing_online_total
    )

print("=" * 72)
