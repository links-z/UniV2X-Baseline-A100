#!/usr/bin/env python3

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

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent


TARGET_SAMPLE = "013326"

CONFIG = ROOT / (
    "projects/configs_e2e_univ2x/"
    "univ2x_coop_e2e_stcv.py"
)

CHECKPOINT = ROOT / "ckpts/univ2x_coop_e2e_stg2.pth"

OUT_DIR = ROOT / (
    "G0-G1-G2/G5/G5-1_consistency_check/"
    "runtime_seq_013326"
)


def canon(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()

    s = str(x)

    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)

    return s


def find_index(dataset, target):
    if hasattr(dataset, "data_infos"):
        matches = []

        for i, info in enumerate(dataset.data_infos):
            token = info.get(
                "token",
                info.get("sample_idx", None),
            )

            if token is not None and canon(token) == target:
                matches.append(i)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple target indices: {matches}"
            )

    if hasattr(dataset, "datasets"):
        offset = 0

        for child in dataset.datasets:
            idx = find_index(child, target)

            if idx is not None:
                return offset + idx

            offset += len(child)

    if hasattr(dataset, "dataset"):
        return find_index(dataset.dataset, target)

    return None


# ------------------------------------------------------------
# Clean output
# ------------------------------------------------------------

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(parents=True)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

set_random_seed(
    0,
    deterministic=False,
)


# ------------------------------------------------------------
# Config / plugin
# ------------------------------------------------------------

cfg = Config.fromfile(str(CONFIG))

cfg.model_ego_agent.pretrained = None

importlib.import_module(
    "projects.mmdet3d_plugin"
)


# ------------------------------------------------------------
# Dataset: SAME complete validation set
# ------------------------------------------------------------

samples_per_gpu = 1

if isinstance(cfg.data.test, dict):

    cfg.data.test.test_mode = True

    samples_per_gpu = cfg.data.test.pop(
        "samples_per_gpu",
        1,
    )

    if samples_per_gpu > 1:
        cfg.data.test.pipeline = \
            replace_ImageToTensor(
                cfg.data.test.pipeline
            )

elif isinstance(cfg.data.test, list):

    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True

    samples_per_gpu = max(
        [
            ds_cfg.pop(
                "samples_per_gpu",
                1,
            )
            for ds_cfg in cfg.data.test
        ]
    )

    if samples_per_gpu > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = \
                replace_ImageToTensor(
                    ds_cfg.pipeline
                )


dataset = build_dataset(
    cfg.data.test
)

target_idx = find_index(
    dataset,
    TARGET_SAMPLE,
)

if target_idx is None:
    raise RuntimeError(
        f"Could not locate {TARGET_SAMPLE}"
    )


print("=" * 80)
print("SEQUENTIAL RUNTIME AUDIT")
print("=" * 80)
print("target sample_idx :", TARGET_SAMPLE)
print("target dataset idx:", target_idx)
print("dataset length    :", len(dataset))
print("=" * 80)


# IMPORTANT:
# Full sequential dataloader.
# Do NOT reduce dataset to one sample.
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
)


# ------------------------------------------------------------
# Build other-agent model(s)
# ------------------------------------------------------------

other_agent_names = []

for key in cfg.keys():
    if "model_other_agent" in key:
        other_agent_names.append(key)


model_other_agents = {}

for name in other_agent_names:

    agent_cfg = cfg.get(name)
    agent_cfg.train_cfg = None

    agent = build_model(
        agent_cfg,
        test_cfg=cfg.get("test_cfg"),
    )

    if agent_cfg.load_from:

        load_checkpoint(
            agent,
            agent_cfg.load_from,
            map_location="cpu",
            revise_keys=[
                (r"^model_ego_agent\.", "")
            ],
        )

    model_other_agents[name] = agent


# ------------------------------------------------------------
# Build ego model
# ------------------------------------------------------------

cfg.model_ego_agent.train_cfg = None

model_ego_agent = build_model(
    cfg.model_ego_agent,
    test_cfg=cfg.get("test_cfg"),
)

if cfg.model_ego_agent.load_from:

    load_checkpoint(
        model_ego_agent,
        cfg.model_ego_agent.load_from,
        map_location="cpu",
        revise_keys=[
            (r"^model_ego_agent\.", "")
        ],
    )


# ------------------------------------------------------------
# MultiAgent + official Stage2 checkpoint
# ------------------------------------------------------------

model_multi_agents = MultiAgent(
    model_ego_agent,
    model_other_agents,
)

fp16_cfg = cfg.get(
    "fp16",
    None,
)

if fp16_cfg is not None:
    wrap_fp16_model(
        model_multi_agents
    )


load_checkpoint(
    model_multi_agents,
    str(CHECKPOINT),
    map_location="cpu",
    strict=False,
)


model = MMDataParallel(
    model_multi_agents.cuda(),
    device_ids=[0],
)

model.eval()


# ------------------------------------------------------------
# Runtime OccHead
# ------------------------------------------------------------

occ_head = (
    model.module
    .model_ego_agent
    .occ_head
)

assert occ_head.stcv_occ is not None

print()
print("STCV module loaded   : YES")
print("STCV tau             :", occ_head.stcv_threshold)


# ------------------------------------------------------------
# Critical:
# Warm-up must reproduce ORIGINAL official fusion.
#
# G0 was generated before STCV integration.
# Therefore use official OR for frames 0 ... target-1.
#
# Enable STCV ONLY on target frame.
# ------------------------------------------------------------

occ_head.use_stcv_occ = False

# Disable G0 exporter during warm-up
occ_head._g012_export_dir = ""
occ_head._g012_export_limit = -1
occ_head._g012_export_idx = 0


# ------------------------------------------------------------
# Sequential forward
# ------------------------------------------------------------

target_result = None

for i, data in enumerate(data_loader):

    if i > target_idx:
        break

    if i == target_idx:

        print()
        print("=" * 80)
        print("TARGET FRAME REACHED")
        print("=" * 80)
        print("dataset index:", i)

        # Enable STCV ONLY for audited target
        occ_head.use_stcv_occ = True

        # Enable one-shot G0-style runtime capture
        occ_head._g012_export_dir = str(
            OUT_DIR
        )

        occ_head._g012_export_limit = 1
        occ_head._g012_export_idx = 0

    with torch.no_grad():

        result = model(
            return_loss=False,
            rescale=True,
            **data,
        )

    if i % 25 == 0 or i == target_idx:

        print(
            f"[warm-up] "
            f"{i}/{target_idx}"
        )

    if i == target_idx:

        runtime_token = canon(
            result[0].get(
                "token",
                "",
            )
        )

        print(
            "runtime token:",
            runtime_token,
        )

        if runtime_token != TARGET_SAMPLE:
            raise RuntimeError(
                f"Wrong target token: "
                f"{runtime_token}"
            )

        if "occ" not in result[0]:
            raise RuntimeError(
                "No occ output"
            )

        seg_out = result[0]["occ"][
            "seg_out"
        ]

        if torch.is_tensor(seg_out):
            seg_out = (
                seg_out
                .detach()
                .cpu()
                .numpy()
            )

        # [1,5,1,200,200]
        if seg_out.shape == (
            1,
            5,
            1,
            200,
            200,
        ):

            ofused = seg_out[
                0,
                :,
                0,
            ]

        elif seg_out.shape == (
            1,
            5,
            200,
            200,
        ):

            ofused = seg_out[0]

        else:
            raise RuntimeError(
                f"Unexpected seg_out "
                f"shape={seg_out.shape}"
            )

        np.save(
            OUT_DIR / "Ofused.npy",
            ofused.astype(np.uint8),
        )

        target_result = result

        print(
            "Saved runtime Ofused:",
            OUT_DIR / "Ofused.npy",
        )

        break


if target_result is None:
    raise RuntimeError(
        "Target frame was not reached"
    )


# ------------------------------------------------------------
# Verify runtime capture
# ------------------------------------------------------------

captures = sorted(
    OUT_DIR.glob(
        "sample_*.npz"
    )
)

print()
print("=" * 80)
print("CAPTURE SUMMARY")
print("=" * 80)

print(
    "npz captures:",
    len(captures),
)

for p in captures:
    print(" ", p)


if len(captures) != 1:
    raise RuntimeError(
        f"Expected one capture, "
        f"found {len(captures)}"
    )


d = np.load(
    captures[0],
    allow_pickle=True,
)

captured_sid = canon(
    d["sample_idx"]
)

print(
    "captured sample_idx:",
    captured_sid,
)

if captured_sid != TARGET_SAMPLE:
    raise RuntimeError(
        f"Captured wrong sample "
        f"{captured_sid}"
    )


print()
print(
    "SEQUENTIAL RUNTIME CAPTURE: PASS"
)
print("=" * 80)
