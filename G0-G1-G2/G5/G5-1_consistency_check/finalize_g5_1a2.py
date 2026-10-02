#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/UniV2X")
sys.path.insert(0, str(ROOT))

from projects.mmdet3d_plugin.univ2x.dense_heads.stcv_occ import STCVOcc


REF_PATH = ROOT / "G0-G1-G2/cache_full/sample_00471.npz"

RUNTIME_PATH = ROOT / (
    "G0-G1-G2/G5/G5-1_consistency_check/"
    "runtime_seq_013326/sample_00000.npz"
)

RUNTIME_OFUSED_PATH = ROOT / (
    "G0-G1-G2/G5/G5-1_consistency_check/"
    "runtime_seq_013326/Ofused.npy"
)

CKPT_PATH = ROOT / (
    "G0-G1-G2/G3/checkpoints_soft/"
    "g3_neighborhood_best.pth"
)

TAU = 0.70
EXPECTED_CANDIDATES = 405
EXPECTED_ACCEPTED = 151
SOFT_TOL = 1e-5


def max_mean_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    assert a.shape == b.shape, (
        a.shape,
        b.shape,
    )

    d = np.abs(
        a.astype(np.float32)
        - b.astype(np.float32)
    )

    return float(d.max()), float(d.mean())


def normalize_output(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.shape == (1, 5, 200, 200):
        x = x[0]

    if x.shape == (1, 5, 1, 200, 200):
        x = x[0, :, 0]

    if x.shape != (5, 200, 200):
        raise RuntimeError(
            f"Unexpected fused occupancy shape: {x.shape}"
        )

    return x.astype(np.uint8)


# ============================================================
# 1. Load
# ============================================================

ref = np.load(
    REF_PATH,
    allow_pickle=True,
)

runtime = np.load(
    RUNTIME_PATH,
    allow_pickle=True,
)

actual_ofused = np.load(
    RUNTIME_OFUSED_PATH,
)


# ============================================================
# 2. Reconfirm pre-STCV identity
# ============================================================

pv_max, pv_mean = max_mean_diff(
    ref["Pv"],
    runtime["Pv"],
)

pi_max, pi_mean = max_mean_diff(
    ref["Pi_aligned"],
    runtime["Pi_aligned"],
)

hard_pre = {}

for k in [
    "Ov",
    "Oi",
    "warp_valid_mask",
    "Oofficial",
]:
    hard_pre[k] = np.array_equal(
        ref[k],
        runtime[k],
    )


# ============================================================
# 3. Load exact production STCV module
# ============================================================

stcv = STCVOcc(
    checkpoint_path=str(CKPT_PATH),
    threshold=TAU,
)

stcv.eval()


# ============================================================
# 4. Tensor conversion
# ============================================================

def make_inputs(d):
    pv = torch.from_numpy(
        d["Pv"].astype(np.float32)
    )

    pi = torch.from_numpy(
        d["Pi_aligned"].astype(np.float32)
    )

    ov = torch.from_numpy(
        d["Ov"].astype(np.int64)
    )

    oi = torch.from_numpy(
        d["Oi"].astype(np.int64)
    )

    warp = torch.from_numpy(
        d["warp_valid_mask"].astype(np.uint8)
    )

    return pv, pi, ov, oi, warp


ref_inputs = make_inputs(ref)
run_inputs = make_inputs(runtime)


# ============================================================
# 5. Candidate + feature extraction
# ============================================================

with torch.no_grad():

    ref_idx, ref_feat = \
        stcv.extract_candidates_and_features(
            pv=ref_inputs[0],
            pi=ref_inputs[1],
            ov=ref_inputs[2],
            oi=ref_inputs[3],
            warp=ref_inputs[4],
        )

    run_idx, run_feat = \
        stcv.extract_candidates_and_features(
            pv=run_inputs[0],
            pi=run_inputs[1],
            ov=run_inputs[2],
            oi=run_inputs[3],
            warp=run_inputs[4],
        )


candidate_exact = (
    ref_idx == run_idx
)

ref_feat_np = (
    ref_feat.detach().cpu().numpy()
)

run_feat_np = (
    run_feat.detach().cpu().numpy()
)

feat_max, feat_mean = max_mean_diff(
    ref_feat_np,
    run_feat_np,
)


# ============================================================
# 6. Predictor
# ============================================================

with torch.no_grad():

    ref_logits = stcv.predictor(
        ref_feat
    )

    run_logits = stcv.predictor(
        run_feat
    )

    ref_scores = torch.sigmoid(
        ref_logits
    )

    run_scores = torch.sigmoid(
        run_logits
    )

    ref_accept = (
        ref_scores > TAU
    )

    run_accept = (
        run_scores > TAU
    )


logit_max, logit_mean = max_mean_diff(
    ref_logits.detach().cpu().numpy(),
    run_logits.detach().cpu().numpy(),
)

score_max, score_mean = max_mean_diff(
    ref_scores.detach().cpu().numpy(),
    run_scores.detach().cpu().numpy(),
)

ref_accept_np = (
    ref_accept.detach().cpu().numpy()
)

run_accept_np = (
    run_accept.detach().cpu().numpy()
)

accept_exact = np.array_equal(
    ref_accept_np,
    run_accept_np,
)

n_candidates = len(
    run_idx
)

n_accepted = int(
    run_accept_np.sum()
)


# ============================================================
# 7. Production STCV forward
# ============================================================

with torch.no_grad():

    expected_ref = stcv(
        pv=ref_inputs[0].unsqueeze(0),
        pi=ref_inputs[1].unsqueeze(0),
        ov=ref_inputs[2].unsqueeze(0),
        oi=ref_inputs[3].unsqueeze(0),
        warp=ref_inputs[4].unsqueeze(0),
    )

    expected_runtime = stcv(
        pv=run_inputs[0].unsqueeze(0),
        pi=run_inputs[1].unsqueeze(0),
        ov=run_inputs[2].unsqueeze(0),
        oi=run_inputs[3].unsqueeze(0),
        warp=run_inputs[4].unsqueeze(0),
    )


expected_ref = normalize_output(
    expected_ref
)

expected_runtime = normalize_output(
    expected_runtime
)

actual_ofused = normalize_output(
    actual_ofused
)


# ============================================================
# 8. Final occupancy checks
# ============================================================

production_ref_runtime_exact = \
    np.array_equal(
        expected_ref,
        expected_runtime,
    )

actual_expected_exact = \
    np.array_equal(
        actual_ofused,
        expected_runtime,
    )

actual_expected_diff = (
    actual_ofused
    != expected_runtime
)

diff_total = int(
    actual_expected_diff.sum()
)

diff_h = [
    int(
        actual_expected_diff[h].sum()
    )
    for h in range(5)
]


# ============================================================
# 9. PASS
# ============================================================

soft_pass = (
    pv_max < SOFT_TOL
    and pi_max < SOFT_TOL
    and feat_max < SOFT_TOL
    and logit_max < SOFT_TOL
    and score_max < SOFT_TOL
)

prehard_pass = all(
    hard_pre.values()
)

stcv_hard_pass = (
    candidate_exact
    and accept_exact
    and n_candidates == EXPECTED_CANDIDATES
    and n_accepted == EXPECTED_ACCEPTED
    and production_ref_runtime_exact
    and actual_expected_exact
)

final_pass = (
    soft_pass
    and prehard_pass
    and stcv_hard_pass
)


# ============================================================
# 10. Report
# ============================================================

print("=" * 88)
print("G5-1A-2 FINAL RUNTIME INTEGRATION AUDIT")
print("=" * 88)

print()
print("========== Runtime Input Reproduction ==========")

print(
    f"Pv max_abs_diff               : "
    f"{pv_max:.10e}"
)

print(
    f"Pv mean_abs_diff              : "
    f"{pv_mean:.10e}"
)

print(
    f"Pi_aligned max_abs_diff       : "
    f"{pi_max:.10e}"
)

print(
    f"Pi_aligned mean_abs_diff      : "
    f"{pi_mean:.10e}"
)

for k, ok in hard_pre.items():
    print(
        f"{k:30s}: "
        f"{'PASS' if ok else 'FAIL'}"
    )


print()
print("========== Candidate / Feature ==========")

print(
    f"reference candidates          : "
    f"{len(ref_idx)}"
)

print(
    f"runtime candidates            : "
    f"{n_candidates}"
)

print(
    f"candidate indices exact       : "
    f"{'PASS' if candidate_exact else 'FAIL'}"
)

print(
    f"feature max_abs_diff          : "
    f"{feat_max:.10e}"
)

print(
    f"feature mean_abs_diff         : "
    f"{feat_mean:.10e}"
)


print()
print("========== Predictor ==========")

print(
    f"logit max_abs_diff            : "
    f"{logit_max:.10e}"
)

print(
    f"logit mean_abs_diff           : "
    f"{logit_mean:.10e}"
)

print(
    f"score max_abs_diff            : "
    f"{score_max:.10e}"
)

print(
    f"score mean_abs_diff           : "
    f"{score_mean:.10e}"
)

print(
    f"reference accepted            : "
    f"{int(ref_accept_np.sum())}/{len(ref_accept_np)}"
)

print(
    f"runtime accepted              : "
    f"{n_accepted}/{n_candidates}"
)

print(
    f"accept exact                  : "
    f"{'PASS' if accept_exact else 'FAIL'}"
)


print()
print("========== Production Output ==========")

print(
    "reference STCV vs runtime STCV: "
    + (
        "PASS"
        if production_ref_runtime_exact
        else "FAIL"
    )
)

print(
    f"runtime Ofused differing cells: "
    f"{diff_total}"
)

for h, n in enumerate(diff_h):
    print(
        f"  t={h} differing cells       : "
        f"{n}"
    )

print(
    "runtime Ofused exact         : "
    + (
        "PASS"
        if actual_expected_exact
        else "FAIL"
    )
)


print()
print("========== Final ==========")

print(
    f"Soft consistency              : "
    f"{'PASS' if soft_pass else 'FAIL'}"
)

print(
    f"Pre-STCV hard consistency     : "
    f"{'PASS' if prehard_pass else 'FAIL'}"
)

print(
    f"STCV hard consistency         : "
    f"{'PASS' if stcv_hard_pass else 'FAIL'}"
)

print()

print(
    "FINAL RESULT                  : "
    + (
        "PASS"
        if final_pass
        else "FAIL"
    )
)

print("=" * 88)


if not final_pass:
    raise SystemExit(1)
