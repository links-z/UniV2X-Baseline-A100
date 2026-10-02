#!/usr/bin/env python3
"""
G5-1A-2: Runtime Integration Audit
Target: sample_idx=013326 (export_idx=00471)
Goal: Verify runtime Ofused == offline Olearned
"""

import os
import sys
import shutil
import importlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/UniV2X")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import import_modules_from_strings

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent


# ============================================================
# Configuration
# ============================================================

TARGET_SAMPLE = "013326"

STCV_CONFIG = ROOT / \
    "projects/configs_e2e_univ2x/univ2x_coop_e2e_stcv.py"

OFFICIAL_CKPT = ROOT / \
    "ckpts/univ2x_coop_e2e_stg2.pth"

REF_CACHE = ROOT / \
    "G0-G1-G2/cache_full/sample_00471.npz"

G3_TEST = ROOT / \
    "G0-G1-G2/G3/soft/g3_neighborhood_test.npz"

RUNTIME_DIR = ROOT / \
    "G0-G1-G2/G5/G5-1_consistency_check/runtime_013326"

TAU = 0.70
SOFT_TOL = 1e-5

EXPECTED_CANDIDATES = 405
EXPECTED_ACCEPTED = 151


# ============================================================
# Utilities
# ============================================================

def canon(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()

    s = str(x)

    # SPD sample_idx in this experiment is six-digit numeric string
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)

    return s


def token_from_info(info):
    if not isinstance(info, dict):
        return None

    for key in ("token", "sample_idx"):
        if key in info:
            return canon(info[key])

    return None


def locate_dataset_index(dataset, target):
    """
    Locate target sample while preserving original full dataset.
    Supports ordinary dataset and common wrapper/concat forms.
    """

    if hasattr(dataset, "data_infos"):
        matches = []

        for i, info in enumerate(dataset.data_infos):
            if token_from_info(info) == target:
                matches.append(i)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple dataset indices found for {target}: {matches}"
            )

    # Concat-style dataset
    if hasattr(dataset, "datasets"):
        offset = 0

        for child in dataset.datasets:
            result = locate_dataset_index(child, target)

            if result is not None:
                return offset + result

            offset += len(child)

    # Wrapper-style dataset
    if hasattr(dataset, "dataset"):
        return locate_dataset_index(dataset.dataset, target)

    return None


class SingleIndexDataset:
    """
    Exposes exactly one sample while internally retaining the complete
    original dataset, so temporal/context logic still uses base[index].
    """

    def __init__(self, base, index):
        self.base = base
        self.index = index

        if hasattr(base, "flag"):
            flag = np.asarray(base.flag)
            self.flag = np.asarray(
                [flag[index]],
                dtype=flag.dtype,
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)

        return self.base[self.index]

    def __getattr__(self, name):
        return getattr(self.base, name)


def soft_stats(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape:
        return float("inf"), float("inf")

    diff = np.abs(a - b)

    return float(diff.max()), float(diff.mean())


def exact_np(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return (
        a.shape == b.shape
        and np.array_equal(a, b)
    )


def manual_selective_fusion(
    ov,
    oi,
    candidate_indices,
    accept_mask,
):
    out = ov.copy()

    for i, (h, r, c) in enumerate(candidate_indices):
        if not bool(accept_mask[i]):
            continue

        rs = slice(4 * r, 4 * r + 4)
        cs = slice(4 * c, 4 * c + 4)

        out[h, rs, cs] = np.logical_or(
            out[h, rs, cs],
            oi[h, rs, cs],
        ).astype(out.dtype)

    return out


# ============================================================
# 0. Prepare clean runtime capture directory
# ============================================================

if RUNTIME_DIR.exists():
    shutil.rmtree(RUNTIME_DIR)

RUNTIME_DIR.mkdir(parents=True)

# IMPORTANT:
# OccHead reads these environment variables during __init__.
os.environ["UNIV2X_G012_EXPORT_DIR"] = str(RUNTIME_DIR)
os.environ["UNIV2X_G012_EXPORT_LIMIT"] = "1"


# ============================================================
# 1. Load config and register plugin
# ============================================================

cfg = Config.fromfile(str(STCV_CONFIG))

if cfg.get("custom_imports", None):
    import_modules_from_strings(**cfg["custom_imports"])

# Explicit UniV2X plugin registration
importlib.import_module("projects.mmdet3d_plugin")

if hasattr(cfg, "plugin") and cfg.plugin:
    if hasattr(cfg, "plugin_dir"):
        plugin_dir = cfg.plugin_dir.rstrip("/")
        module_path = plugin_dir.replace("/", ".")
        importlib.import_module(module_path)

cfg.model_ego_agent.pretrained = None


# ============================================================
# 2. Build complete validation dataset
# ============================================================

if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True

elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True

dataset = build_dataset(cfg.data.test)

target_dataset_idx = locate_dataset_index(
    dataset,
    TARGET_SAMPLE,
)

if target_dataset_idx is None:
    raise RuntimeError(
        f"Could not locate sample_idx={TARGET_SAMPLE} "
        f"in validation dataset"
    )

print("=" * 90)
print("G5-1A-2 TARGET SAMPLE LOCATED")
print("=" * 90)
print("sample_idx        :", TARGET_SAMPLE)
print("dataset index     :", target_dataset_idx)
print("dataset length    :", len(dataset))
print("reference cache   :", REF_CACHE.name)
print("=" * 90)


# ============================================================
# 3. Create one-sample view
# ============================================================

single_dataset = SingleIndexDataset(
    dataset,
    target_dataset_idx,
)

nonshuffler_sampler = cfg.data.get(
    "nonshuffler_sampler",
    None,
)

data_loader = build_dataloader(
    single_dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
    nonshuffler_sampler=nonshuffler_sampler,
)

assert len(single_dataset) == 1


# ============================================================
# 4. Build infrastructure agent(s)
# ============================================================

other_agent_names = []

for key in cfg.keys():
    if "model_other_agent" in key:
        other_agent_names.append(key)

model_other_agents = {}

for other_agent_name in other_agent_names:

    agent_cfg = cfg.get(other_agent_name)
    agent_cfg.train_cfg = None

    model_other_agent = build_model(
        agent_cfg,
        test_cfg=cfg.get("test_cfg"),
    )

    load_from = agent_cfg.load_from

    if load_from:
        load_checkpoint(
            model_other_agent,
            load_from,
            map_location="cpu",
            revise_keys=[
                (r"^model_ego_agent\.", "")
            ],
        )

    model_other_agents[other_agent_name] = \
        model_other_agent


# ============================================================
# 5. Build ego agent
# ============================================================

cfg.model_ego_agent.train_cfg = None

model_ego_agent = build_model(
    cfg.model_ego_agent,
    test_cfg=cfg.get("test_cfg"),
)

load_from = cfg.model_ego_agent.load_from

if load_from:
    load_checkpoint(
        model_ego_agent,
        load_from,
        map_location="cpu",
        revise_keys=[
            (r"^model_ego_agent\.", "")
        ],
    )


# ============================================================
# 6. Build complete MultiAgent model
# ============================================================

model_multi_agents = MultiAgent(
    model_ego_agent,
    model_other_agents,
)

fp16_cfg = cfg.get("fp16", None)

if fp16_cfg is not None:
    wrap_fp16_model(model_multi_agents)

load_checkpoint(
    model_multi_agents,
    str(OFFICIAL_CKPT),
    map_location="cpu",
    strict=False,
)

model = MMDataParallel(
    model_multi_agents.cuda(),
    device_ids=[0],
)

model.eval()


# ============================================================
# 7. Execute exactly ONE real UniV2X forward
# ============================================================

data = next(iter(data_loader))

print()
print("=" * 90)
print("RUNNING REAL UNIV2X FORWARD")
print("=" * 90)

with torch.no_grad():
    result = model(
        return_loss=False,
        rescale=True,
        **data,
    )

print("Forward completed.")


# ============================================================
# 8. Verify runtime G0-style export exists
# ============================================================

runtime_files = sorted(
    RUNTIME_DIR.glob("sample_*.npz")
)

if len(runtime_files) != 1:
    raise RuntimeError(
        f"Expected exactly 1 runtime export, "
        f"found {len(runtime_files)}: "
        f"{runtime_files}"
    )

runtime_file = runtime_files[0]

runtime = np.load(
    runtime_file,
    allow_pickle=True,
)

runtime_sid = canon(
    runtime["sample_idx"]
)

if runtime_sid != TARGET_SAMPLE:
    raise RuntimeError(
        f"Wrong runtime sample: "
        f"{runtime_sid} != {TARGET_SAMPLE}"
    )


# ============================================================
# 9. Load frozen G0 reference
# ============================================================

ref = np.load(
    REF_CACHE,
    allow_pickle=True,
)

ref_sid = canon(
    ref["sample_idx"]
)

assert ref_sid == TARGET_SAMPLE, (
    ref_sid,
    TARGET_SAMPLE,
)

ref_export_idx = int(
    np.asarray(
        ref["export_idx"]
    ).item()
)


# ============================================================
# 10. Extract actual fused output from real model result
# ============================================================

if "occ" not in result[0]:
    raise RuntimeError(
        "Runtime result has no occ output"
    )

seg_out = result[0]["occ"]["seg_out"]

if torch.is_tensor(seg_out):
    seg_out = (
        seg_out
        .detach()
        .cpu()
        .numpy()
    )

# Expected:
# [B, 5, 1, 200, 200]
if seg_out.shape == (1, 5, 1, 200, 200):
    runtime_ofused = \
        seg_out[0, :, 0].astype(np.uint8)

elif seg_out.shape == (1, 5, 200, 200):
    runtime_ofused = \
        seg_out[0].astype(np.uint8)

else:
    raise RuntimeError(
        f"Unexpected seg_out shape: "
        f"{seg_out.shape}"
    )


# ============================================================
# 11. Compare runtime pre-fusion tensors against G0
# ============================================================

pv_max, pv_mean = soft_stats(
    runtime["Pv"],
    ref["Pv"],
)

pi_max, pi_mean = soft_stats(
    runtime["Pi_aligned"],
    ref["Pi_aligned"],
)

ov_exact = exact_np(
    runtime["Ov"],
    ref["Ov"],
)

oi_exact = exact_np(
    runtime["Oi"],
    ref["Oi"],
)

warp_exact = exact_np(
    runtime["warp_valid_mask"],
    ref["warp_valid_mask"],
)

official_exact = exact_np(
    runtime["Oofficial"],
    ref["Oofficial"],
)


# ============================================================
# 12. Load G3 Test reference metadata/features
# ============================================================

g3 = np.load(
    G3_TEST,
    allow_pickle=True,
)

metadata = g3["metadata"]
g3_features_all = \
    g3["features"].astype(np.float32)

sample_rows = [
    i
    for i, m in enumerate(metadata)
    if canon(m["sample_idx"]) == TARGET_SAMPLE
]

if not sample_rows:
    raise RuntimeError(
        "Target sample not found in G3 Test NPZ"
    )

offline_indices = [
    (
        int(metadata[i]["horizon"]),
        int(metadata[i]["patch_row"]),
        int(metadata[i]["patch_col"]),
    )
    for i in sample_rows
]

offline_features = \
    g3_features_all[sample_rows]


# ============================================================
# 13. Obtain STCV module actually used by runtime model
# ============================================================

occ_head = (
    model.module
    .model_ego_agent
    .occ_head
)

if not occ_head.use_stcv_occ:
    raise RuntimeError(
        "Runtime ego OccHead has use_stcv_occ=False"
    )

stcv = occ_head.stcv_occ

if stcv is None:
    raise RuntimeError(
        "Runtime STCV module is None"
    )

device = next(
    stcv.predictor.parameters()
).device


# ============================================================
# 14. Reconstruct candidate/features from RUNTIME tensors
# ============================================================

rt_pv = torch.from_numpy(
    runtime["Pv"].astype(np.float32)
).to(device)

rt_pi = torch.from_numpy(
    runtime["Pi_aligned"].astype(np.float32)
).to(device)

rt_ov = torch.from_numpy(
    runtime["Ov"].astype(np.int64)
).to(device)

rt_oi = torch.from_numpy(
    runtime["Oi"].astype(np.int64)
).to(device)

rt_warp = torch.from_numpy(
    runtime["warp_valid_mask"].astype(np.uint8)
).to(device)

runtime_indices, runtime_features = \
    stcv.extract_candidates_and_features(
        pv=rt_pv,
        pi=rt_pi,
        ov=rt_ov,
        oi=rt_oi,
        warp=rt_warp,
    )

with torch.no_grad():
    runtime_logits = \
        stcv.predictor(runtime_features)

    runtime_scores = \
        torch.sigmoid(runtime_logits)

    runtime_accept = \
        runtime_scores > TAU


# ============================================================
# 15. Offline reference predictor on stored G3 features
# ============================================================

offline_features_t = torch.from_numpy(
    offline_features
).to(device)

with torch.no_grad():
    offline_logits = \
        stcv.predictor(offline_features_t)

    offline_scores = \
        torch.sigmoid(offline_logits)

    offline_accept = \
        offline_scores > TAU


# ============================================================
# 16. Feature / logit / score consistency
# ============================================================

runtime_features_np = (
    runtime_features
    .detach()
    .cpu()
    .numpy()
)

runtime_logits_np = (
    runtime_logits
    .detach()
    .cpu()
    .numpy()
)

runtime_scores_np = (
    runtime_scores
    .detach()
    .cpu()
    .numpy()
)

runtime_accept_np = (
    runtime_accept
    .detach()
    .cpu()
    .numpy()
)

offline_logits_np = (
    offline_logits
    .detach()
    .cpu()
    .numpy()
)

offline_scores_np = (
    offline_scores
    .detach()
    .cpu()
    .numpy()
)

offline_accept_np = (
    offline_accept
    .detach()
    .cpu()
    .numpy()
)

feat_max, feat_mean = soft_stats(
    runtime_features_np,
    offline_features,
)

logit_max, logit_mean = soft_stats(
    runtime_logits_np,
    offline_logits_np,
)

score_max, score_mean = soft_stats(
    runtime_scores_np,
    offline_scores_np,
)


# ============================================================
# 17. Candidate + accept exact checks
# ============================================================

candidate_exact = (
    runtime_indices == offline_indices
)

accept_exact = (
    runtime_accept_np.shape
    == offline_accept_np.shape
    and np.array_equal(
        runtime_accept_np,
        offline_accept_np,
    )
)

runtime_n_candidates = len(
    runtime_indices
)

runtime_n_accept = int(
    runtime_accept_np.sum()
)


# ============================================================
# 18. Build independent offline learned occupancy
# ============================================================

offline_learned = manual_selective_fusion(
    ov=ref["Ov"].astype(np.uint8),
    oi=ref["Oi"].astype(np.uint8),
    candidate_indices=offline_indices,
    accept_mask=offline_accept_np,
)

ofused_exact = exact_np(
    runtime_ofused,
    offline_learned,
)

ofused_diff = (
    runtime_ofused != offline_learned
)

ofused_diff_total = int(
    ofused_diff.sum()
)

ofused_diff_h = [
    int(
        ofused_diff[h].sum()
    )
    for h in range(5)
]


# ============================================================
# 19. Final PASS criteria
# ============================================================

soft_pass = (
    pv_max < SOFT_TOL
    and pi_max < SOFT_TOL
    and feat_max < SOFT_TOL
    and logit_max < SOFT_TOL
    and score_max < SOFT_TOL
)

hard_pass = (
    runtime_sid == TARGET_SAMPLE
    and ov_exact
    and oi_exact
    and warp_exact
    and official_exact
    and candidate_exact
    and accept_exact
    and ofused_exact
    and runtime_n_candidates == EXPECTED_CANDIDATES
    and runtime_n_accept == EXPECTED_ACCEPTED
)

final_pass = (
    soft_pass
    and hard_pass
)


# ============================================================
# 20. Report
# ============================================================

print()
print("=" * 90)
print("G5-1A-2 RUNTIME INTEGRATION AUDIT")
print("=" * 90)

print()
print("========== Identity ==========")
print(f"sample_idx                    : {runtime_sid}")
print(f"runtime dataset index         : {target_dataset_idx}")
print(f"G0 reference export_idx       : {ref_export_idx}")
print(f"runtime capture               : {runtime_file.name}")

print()
print("========== Runtime vs G0 ==========")
print(f"Pv max_abs_diff               : {pv_max:.10e}")
print(f"Pv mean_abs_diff              : {pv_mean:.10e}")
print(f"Pi_aligned max_abs_diff       : {pi_max:.10e}")
print(f"Pi_aligned mean_abs_diff      : {pi_mean:.10e}")
print(f"Ov exact                      : {'PASS' if ov_exact else 'FAIL'}")
print(f"Oi exact                      : {'PASS' if oi_exact else 'FAIL'}")
print(f"warp exact                    : {'PASS' if warp_exact else 'FAIL'}")
print(f"Oofficial exact               : {'PASS' if official_exact else 'FAIL'}")

print()
print("========== STCV Candidate ==========")
print(f"offline candidates            : {len(offline_indices)}")
print(f"runtime candidates            : {runtime_n_candidates}")
print(f"candidate exact               : {'PASS' if candidate_exact else 'FAIL'}")

print()
print("========== STCV Feature ==========")
print(f"feature max_abs_diff          : {feat_max:.10e}")
print(f"feature mean_abs_diff         : {feat_mean:.10e}")
print(f"logit max_abs_diff            : {logit_max:.10e}")
print(f"logit mean_abs_diff           : {logit_mean:.10e}")
print(f"score max_abs_diff            : {score_max:.10e}")
print(f"score mean_abs_diff           : {score_mean:.10e}")

print()
print("========== STCV Decision ==========")
print(
    f"offline accepted              : "
    f"{int(offline_accept_np.sum())}/{len(offline_accept_np)}"
)
print(
    f"runtime accepted              : "
    f"{runtime_n_accept}/{runtime_n_candidates}"
)
print(f"accept exact                  : {'PASS' if accept_exact else 'FAIL'}")

print()
print("========== Final Occupancy ==========")
print(f"Ofused differing cells        : {ofused_diff_total}")

for h, n in enumerate(ofused_diff_h):
    print(
        f"  t={h} differing cells       : {n}"
    )

print(f"Ofused exact                  : {'PASS' if ofused_exact else 'FAIL'}")

print()
print("========== Final ==========")
print(f"Soft consistency              : {'PASS' if soft_pass else 'FAIL'}")
print(f"Hard consistency              : {'PASS' if hard_pass else 'FAIL'}")
print(
    "FINAL RESULT                  : "
    + ("PASS" if final_pass else "FAIL")
)

print("=" * 90)


# ============================================================
# 21. Exit status
# ============================================================

if not final_pass:
    raise SystemExit(1)