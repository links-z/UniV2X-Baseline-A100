#!/usr/bin/env python3

from pathlib import Path
import importlib.util

import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

ROOT = Path("/root/autodl-tmp/UniV2X")

TEST_NPZ = ROOT / "G0-G1-G2/G3/soft/g3_neighborhood_test.npz"
CACHE_FILE = ROOT / "G0-G1-G2/cache_full/sample_00471.npz"
CKPT = ROOT / "G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth"
STCV_FILE = ROOT / "projects/mmdet3d_plugin/univ2x/dense_heads/stcv_occ.py"

SAMPLE_IDX = "013326"
TAU = 0.70
SOFT_TOL = 1e-5


# ============================================================
# Utilities
# ============================================================

def canonical_sample_idx(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def load_stcv_module(path):
    """
    Load stcv_occ.py directly, avoiding unrelated package side effects.
    """
    spec = importlib.util.spec_from_file_location(
        "stcv_occ_audit_module",
        str(path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diff_stats(a, b):
    """
    Numerical difference statistics.
    """
    assert a.shape == b.shape, (a.shape, b.shape)

    if a.numel() == 0:
        return 0.0, 0.0

    diff = torch.abs(a.float() - b.float())
    return float(diff.max().item()), float(diff.mean().item())


def manual_selective_fusion(ov, oi, candidate_indices, accept_mask):
    """
    Independent G4-style reconstruction.

    O_learned starts from Ov.
    Only accepted patches receive Ov OR Oi.
    """
    out = ov.clone()

    for idx, (h, r, c) in enumerate(candidate_indices):
        if bool(accept_mask[idx].item()):
            rs = slice(4 * r, 4 * r + 4)
            cs = slice(4 * c, 4 * c + 4)

            out[h, rs, cs] = torch.bitwise_or(
                out[h, rs, cs],
                oi[h, rs, cs],
            )

    return out


# ============================================================
# 1. Load actual STCV implementation
# ============================================================

stcv_module = load_stcv_module(STCV_FILE)

NeighborhoodMLP = stcv_module.NeighborhoodMLP
STCVOcc = stcv_module.STCVOcc


# ============================================================
# 2. Load G3 stored Test reference
# ============================================================

dtest = np.load(TEST_NPZ, allow_pickle=True)

g3_features_all = dtest["features"].astype(np.float32)
metadata = dtest["metadata"]

sample_rows = [
    i
    for i, m in enumerate(metadata)
    if canonical_sample_idx(m["sample_idx"]) == SAMPLE_IDX
]

if len(sample_rows) == 0:
    raise RuntimeError(
        f"No G3 Test rows found for sample_idx={SAMPLE_IDX}"
    )

offline_indices = [
    (
        int(metadata[i]["horizon"]),
        int(metadata[i]["patch_row"]),
        int(metadata[i]["patch_col"]),
    )
    for i in sample_rows
]

offline_features = torch.from_numpy(
    g3_features_all[sample_rows]
).float()


# ============================================================
# 3. Load corresponding G0 cache
# ============================================================

with np.load(CACHE_FILE, allow_pickle=True) as d:
    cache_sid = canonical_sample_idx(d["sample_idx"])

    pv = torch.from_numpy(
        np.asarray(d["Pv"], dtype=np.float32)
    )

    pi = torch.from_numpy(
        np.asarray(d["Pi_aligned"], dtype=np.float32)
    )

    ov = torch.from_numpy(
        np.asarray(d["Ov"])
    ).long()

    oi = torch.from_numpy(
        np.asarray(d["Oi"])
    ).long()

    warp = torch.from_numpy(
        np.asarray(d["warp_valid_mask"])
    )

assert cache_sid == SAMPLE_IDX, (
    f"Cache sample_idx mismatch: {cache_sid} != {SAMPLE_IDX}"
)

assert pv.shape == (5, 200, 200), pv.shape
assert pi.shape == (5, 200, 200), pi.shape
assert ov.shape == (5, 200, 200), ov.shape
assert oi.shape == (5, 200, 200), oi.shape
assert warp.shape == (200, 200), warp.shape


# ============================================================
# 4. Offline reference predictor
# ============================================================

offline_model = NeighborhoodMLP()

ckpt = torch.load(CKPT, map_location="cpu")
offline_model.load_state_dict(
    ckpt["model"],
    strict=True,
)
offline_model.eval()

n_params = sum(
    p.numel()
    for p in offline_model.parameters()
)

assert n_params == 73985, n_params

with torch.no_grad():
    offline_logits = offline_model(offline_features)
    offline_scores = torch.sigmoid(offline_logits)
    offline_accept = offline_scores > TAU


# ============================================================
# 5. New STCV online module path
# ============================================================

stcv = STCVOcc(
    checkpoint_path=str(CKPT),
    threshold=TAU,
)
stcv.eval()

online_indices, online_features = (
    stcv.extract_candidates_and_features(
        pv=pv,
        pi=pi,
        ov=ov,
        oi=oi,
        warp=warp,
    )
)

with torch.no_grad():
    online_logits = stcv.predictor(online_features)
    online_scores = torch.sigmoid(online_logits)
    online_accept = online_scores > TAU


# ============================================================
# 6. Candidate consistency
# ============================================================

candidate_exact = (
    offline_indices == online_indices
)

offline_n = len(offline_indices)
online_n = len(online_indices)


# ============================================================
# 7. Feature / logit / score consistency
# ============================================================

if offline_features.shape == online_features.shape:
    feature_max, feature_mean = diff_stats(
        offline_features,
        online_features,
    )
else:
    feature_max = float("inf")
    feature_mean = float("inf")

if offline_logits.shape == online_logits.shape:
    logit_max, logit_mean = diff_stats(
        offline_logits,
        online_logits,
    )
else:
    logit_max = float("inf")
    logit_mean = float("inf")

if offline_scores.shape == online_scores.shape:
    score_max, score_mean = diff_stats(
        offline_scores,
        online_scores,
    )
else:
    score_max = float("inf")
    score_mean = float("inf")


# ============================================================
# 8. Accept consistency
# ============================================================

accept_shape_exact = (
    offline_accept.shape == online_accept.shape
)

if accept_shape_exact:
    accept_equal = (
        offline_accept == online_accept
    )

    accept_match_count = int(
        accept_equal.sum().item()
    )

    accept_mismatch_count = int(
        (~accept_equal).sum().item()
    )

    accept_exact = bool(
        torch.equal(
            offline_accept,
            online_accept,
        )
    )
else:
    accept_match_count = 0
    accept_mismatch_count = max(
        offline_accept.numel(),
        online_accept.numel(),
    )
    accept_exact = False


# ============================================================
# 9. Independent offline G4-style reconstruction
# ============================================================

offline_occ = manual_selective_fusion(
    ov=ov,
    oi=oi,
    candidate_indices=offline_indices,
    accept_mask=offline_accept,
)


# ============================================================
# 10. New STCV selective_fusion reconstruction
# ============================================================

online_occ = stcv.selective_fusion(
    ov=ov,
    oi=oi,
    candidate_indices=online_indices,
    accept_mask=online_accept,
)

occ_exact = bool(
    torch.equal(
        offline_occ,
        online_occ,
    )
)

occ_diff = (
    offline_occ != online_occ
)

occ_diff_cells = int(
    occ_diff.sum().item()
)

occ_diff_h = [
    int(occ_diff[h].sum().item())
    for h in range(5)
]


# ============================================================
# 11. Test production _forward_single()
# ============================================================

with torch.no_grad():
    forward_single_occ = stcv._forward_single(
        pv=pv,
        pi=pi,
        ov=ov,
        oi=oi,
        warp=warp,
    )

forward_single_exact = bool(
    torch.equal(
        online_occ,
        forward_single_occ,
    )
)


# ============================================================
# 12. Test production batch forward()
# ============================================================

with torch.no_grad():
    batch_occ = stcv(
        pv=pv.unsqueeze(0),
        pi=pi.unsqueeze(0),
        ov=ov.unsqueeze(0),
        oi=oi.unsqueeze(0),
        warp=warp.unsqueeze(0),
    )[0]

batch_forward_exact = bool(
    torch.equal(
        online_occ,
        batch_occ,
    )
)


# ============================================================
# 13. Threshold-margin diagnostic
# ============================================================

margin = torch.abs(
    offline_scores - TAU
)

nearest_idx = int(
    torch.argmin(margin).item()
)

nearest_score = float(
    offline_scores[nearest_idx].item()
)

nearest_candidate = offline_indices[nearest_idx]

nearest_margin = float(
    margin[nearest_idx].item()
)


# ============================================================
# 14. Final decision
# ============================================================

soft_pass = (
    feature_max < SOFT_TOL
    and logit_max < SOFT_TOL
    and score_max < SOFT_TOL
)

hard_pass = (
    candidate_exact
    and accept_exact
    and occ_exact
    and forward_single_exact
    and batch_forward_exact
)

final_pass = (
    soft_pass
    and hard_pass
)


# ============================================================
# 15. Report
# ============================================================

print("=" * 88)
print("G5-1A-1 SINGLE-SAMPLE MODULE-LEVEL EXACT AUDIT")
print("=" * 88)

print()
print("========== Sample ==========")
print(f"sample_idx                     : {SAMPLE_IDX}")
print(f"cache                          : {CACHE_FILE.name}")
print(f"G3 metadata rows               : {sample_rows[0]} ... {sample_rows[-1]}")

print()
print("========== Candidate ==========")
print(f"Offline candidates             : {offline_n}")
print(f"Online candidates              : {online_n}")
print(f"Candidate indices exact        : {'PASS' if candidate_exact else 'FAIL'}")

if not candidate_exact:
    max_show = min(
        max(offline_n, online_n),
        20,
    )

    print("First candidate mismatches:")

    for i in range(max_show):
        off = (
            offline_indices[i]
            if i < offline_n
            else None
        )

        on = (
            online_indices[i]
            if i < online_n
            else None
        )

        if off != on:
            print(
                f"  idx={i:4d} "
                f"offline={off} "
                f"online={on}"
            )

print()
print("========== 127-d Feature ==========")
print(f"Offline shape                  : {tuple(offline_features.shape)}")
print(f"Online shape                   : {tuple(online_features.shape)}")
print(f"Feature max_abs_diff           : {feature_max:.10e}")
print(f"Feature mean_abs_diff          : {feature_mean:.10e}")
print(
    f"Feature tolerance ({SOFT_TOL:.1e})      : "
    f"{'PASS' if feature_max < SOFT_TOL else 'FAIL'}"
)

print()
print("========== Predictor Logit ==========")
print(f"Logit max_abs_diff             : {logit_max:.10e}")
print(f"Logit mean_abs_diff            : {logit_mean:.10e}")
print(
    f"Logit tolerance ({SOFT_TOL:.1e})        : "
    f"{'PASS' if logit_max < SOFT_TOL else 'FAIL'}"
)

print()
print("========== Predictor Score ==========")
print(f"Score max_abs_diff             : {score_max:.10e}")
print(f"Score mean_abs_diff            : {score_mean:.10e}")
print(
    f"Score tolerance ({SOFT_TOL:.1e})        : "
    f"{'PASS' if score_max < SOFT_TOL else 'FAIL'}"
)

print()
print("========== Accept Decision ==========")
print(
    f"Offline accepted               : "
    f"{int(offline_accept.sum().item())}/{offline_n}"
)
print(
    f"Online accepted                : "
    f"{int(online_accept.sum().item())}/{online_n}"
)
print(
    f"Accept exact                   : "
    f"{accept_match_count}/{offline_n}"
)
print(f"Accept mismatches              : {accept_mismatch_count}")
print(
    f"Accept mask exact              : "
    f"{'PASS' if accept_exact else 'FAIL'}"
)

print()
print("========== Threshold Margin ==========")
print(f"Closest candidate              : {nearest_candidate}")
print(f"Closest score to tau           : {nearest_score:.10f}")
print(f"|score - tau|                  : {nearest_margin:.10e}")

print()
print("========== Learned Occupancy ==========")
print(f"Differing cells total          : {occ_diff_cells}")

for h, n in enumerate(occ_diff_h):
    print(
        f"  t={h} differing cells        : {n}"
    )

print(
    f"Learned occupancy exact        : "
    f"{'PASS' if occ_exact else 'FAIL'}"
)

print()
print("========== Production Path ==========")
print(
    f"_forward_single exact          : "
    f"{'PASS' if forward_single_exact else 'FAIL'}"
)
print(
    f"batch forward exact            : "
    f"{'PASS' if batch_forward_exact else 'FAIL'}"
)

print()
print("========== Final ==========")
print(
    f"Soft numerical consistency     : "
    f"{'PASS' if soft_pass else 'FAIL'}"
)
print(
    f"Hard exact consistency         : "
    f"{'PASS' if hard_pass else 'FAIL'}"
)

print()
print(
    "FINAL RESULT                   : "
    + ("PASS" if final_pass else "FAIL")
)

print("=" * 88)


# ============================================================
# 16. Exit code
# ============================================================

if not final_pass:
    raise SystemExit(1)

