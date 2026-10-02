#!/usr/bin/env python3

import os
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Models
# ============================================================

class SpatialOnlyMLP(nn.Module):
    """Baseline: predict from spatial position (u,v) only"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class LocalHorizonSpatialMLP(nn.Module):
    """Local (80) + Horizon (5) + Spatial (2) = 87"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(87, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# Data
# ============================================================

def load_split(path, mode):
    """
    Load dataset and construct input based on mode:
    - 'spatial': only 2-dim normalized patch coordinates (u,v)
    - 'mlp_hsp': local (80) + horizon (5) + spatial (2) = 87-dim
    """
    d = np.load(path, allow_pickle=True)

    x = d["features"].astype(np.float32)  # [N, 5, 4, 4]
    y = d["labels"].astype(np.float32)    # [N]
    metadata = d["metadata"]

    assert x.ndim == 4
    assert x.shape[1:] == (5, 4, 4)
    assert len(x) == len(y)

    # --------------------------------------------------------
    # Local features
    # --------------------------------------------------------
    x_local = x.reshape(len(x), -1)  # [N, 80]

    # --------------------------------------------------------
    # Horizon one-hot
    # --------------------------------------------------------
    horizons = np.array(
        [int(m["horizon"]) for m in metadata],
        dtype=np.int64,
    )
    assert horizons.min() >= 0
    assert horizons.max() <= 4

    horizon_onehot = np.eye(5, dtype=np.float32)[horizons]  # [N, 5]

    # --------------------------------------------------------
    # Normalized patch coordinates
    # Patch-4 on 200x200 grid -> 50x50 patch grid
    # --------------------------------------------------------
    patch_rows = np.array(
        [int(m["patch_row"]) for m in metadata],
        dtype=np.int64,
    )
    patch_cols = np.array(
        [int(m["patch_col"]) for m in metadata],
        dtype=np.int64,
    )

    # Normalize to [-1, 1]
    u = 2.0 * ((patch_cols + 0.5) / 50.0) - 1.0  # [N]
    v = 2.0 * ((patch_rows + 0.5) / 50.0) - 1.0  # [N]

    spatial_coords = np.stack([u, v], axis=1).astype(np.float32)  # [N, 2]

    if mode == "spatial":
        # Spatial-only baseline: [N, 2]
        x_out = spatial_coords

    elif mode == "mlp_hsp":
        # Local + Horizon + Spatial: [N, 80] + [N, 5] + [N, 2] → [N, 87]
        x_out = np.concatenate([x_local, horizon_onehot, spatial_coords], axis=1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return x_out, y


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        logits = model(x)

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = 1.0 / (1.0 + np.exp(-logits))

    return labels.astype(np.int32), probs


def find_best_f1_threshold(labels, probs):
    best_threshold = 0.5
    best_f1 = -1.0

    # threshold must be selected on validation set only
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (probs >= threshold).astype(np.int32)

        score = f1_score(
            labels,
            pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def compute_metrics(labels, probs, threshold):
    pred = (probs >= threshold).astype(np.int32)

    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(
            precision_score(labels, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, pred, zero_division=0)
        ),
        "threshold": float(threshold),
        "positive_rate_gt": float(labels.mean()),
        "positive_rate_pred": float(pred.mean()),
    }


def print_metrics(name, metrics):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
        else:
            print(f"{key:20s}: {value}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["spatial", "mlp_hsp"],
        required=True,
        help="spatial: Spatial-only baseline; mlp_hsp: Local + Horizon + Spatial MLP"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    root = "G0-G1-G2/G3"

    train_file = os.path.join(root, "g3_dataset_train.npz")
    val_file = os.path.join(root, "g3_dataset_val.npz")
    test_file = os.path.join(root, "g3_dataset_test.npz")

    ckpt_dir = os.path.join(root, "checkpoints")
    result_dir = os.path.join(root, "results")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    x_train, y_train = load_split(train_file, args.model)
    x_val, y_val = load_split(val_file, args.model)
    x_test, y_test = load_split(test_file, args.model)

    print("=" * 70)
    print("G3-1C SPATIAL POSITION CONTEXT LEARNABILITY")
    print("=" * 70)

    print(f"Model      : {args.model}")
    print(f"Train      : {len(y_train)}")
    print(f"Val        : {len(y_val)}")
    print(f"Test       : {len(y_test)}")

    print(
        f"Train pos  : {int(y_train.sum())} "
        f"({y_train.mean() * 100:.2f}%)"
    )

    print(
        f"Val pos    : {int(y_val.sum())} "
        f"({y_val.mean() * 100:.2f}%)"
    )

    print(
        f"Test pos   : {int(y_test.sum())} "
        f"({y_test.mean() * 100:.2f}%)"
    )

    print()
    print(
        f"Random test AUPRC reference ≈ "
        f"{y_test.mean():.4f}"
    )

    # --------------------------------------------------------
    # PyTorch datasets
    # --------------------------------------------------------

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )

    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
    )

    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device     : {device}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if args.model == "spatial":
        model = SpatialOnlyMLP()
    elif args.model == "mlp_hsp":
        model = LocalHorizonSpatialMLP()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters : {n_params:,}")

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - y_train.sum())

    pos_weight_value = n_neg / max(n_pos, 1.0)

    print(f"pos_weight : {pos_weight_value:.4f}")

    pos_weight = torch.tensor(
        [pos_weight_value],
        device=device,
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    best_val_auprc = -1.0
    best_epoch = -1
    bad_epochs = 0

    ckpt_path = os.path.join(
        ckpt_dir,
        f"g3_{args.model}_best.pth",
    )

    for epoch in range(1, args.epochs + 1):

        model.train()

        running_loss = 0.0
        n_seen = 0

        for x, y in train_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()

            batch_size = len(y)

            running_loss += (
                loss.item() * batch_size
            )

            n_seen += batch_size

        train_loss = running_loss / n_seen

        # Validation ranking performance
        val_labels, val_probs = predict(
            model,
            val_loader,
            device,
        )

        val_auroc = roc_auc_score(
            val_labels,
            val_probs,
        )

        val_auprc = average_precision_score(
            val_labels,
            val_probs,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.6f} | "
            f"val_AUROC={val_auroc:.6f} | "
            f"val_AUPRC={val_auprc:.6f}"
        )

        # checkpoint by validation AUPRC
        if val_auprc > best_val_auprc:

            best_val_auprc = val_auprc
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_auprc": val_auprc,
                    "args": vars(args),
                },
                ckpt_path,
            )

        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print()
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}"
            )
            break

    # --------------------------------------------------------
    # Reload best model
    # --------------------------------------------------------

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    print()
    print(
        f"Loaded best checkpoint: "
        f"epoch={ckpt['epoch']}, "
        f"val_AUPRC={ckpt['val_auprc']:.6f}"
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    val_labels, val_probs = predict(
        model,
        val_loader,
        device,
    )

    best_threshold, val_best_f1 = \
        find_best_f1_threshold(
            val_labels,
            val_probs,
        )

    val_metrics = compute_metrics(
        val_labels,
        val_probs,
        best_threshold,
    )

    # --------------------------------------------------------
    # Test — threshold frozen from validation
    # --------------------------------------------------------

    test_labels, test_probs = predict(
        model,
        test_loader,
        device,
    )

    test_metrics = compute_metrics(
        test_labels,
        test_probs,
        best_threshold,
    )

    print_metrics(
        "VALIDATION",
        val_metrics,
    )

    print_metrics(
        "TEST",
        test_metrics,
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {
        "model": args.model,
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
        "validation_threshold": float(best_threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "random_test_auprc_reference": float(
            y_test.mean()
        ),
        "num_parameters": int(n_params),
        "seed": int(args.seed),
    }

    result_file = os.path.join(
        result_dir,
        f"g3_{args.model}_learnability.json",
    )

    with open(result_file, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(f"Saved result: {result_file}")
    print(f"Saved ckpt  : {ckpt_path}")


if __name__ == "__main__":
    main()