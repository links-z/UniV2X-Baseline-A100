import os
import glob
import numpy as np

CACHE = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full"

files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))

print("=" * 80)
print("G0 CACHE STRUCTURAL CHECK")
print("=" * 80)
print("Cache directory:", CACHE)
print("Number of npz files:", len(files))

if len(files) == 0:
    raise RuntimeError("No .npz files found.")

required_keys = {
    "Pv",
    "Pi_aligned",
    "Ov",
    "Oi",
    "Oofficial",
    "GT",
    "gt_cell_valid_mask",
    "future_valid_mask",
    "warp_valid_mask",
    "test_seg_thresh",
    "export_idx",
}

errors = []

pv_min_all = []
pv_max_all = []
pi_min_all = []
pi_max_all = []

warp_ratios = []
ov_ratios = []
oi_ratios = []
official_ratios = []

for i, path in enumerate(files):
    name = os.path.basename(path)

    with np.load(path) as d:
        missing = required_keys - set(d.files)

        if missing:
            errors.append(
                f"{name}: missing keys {sorted(missing)}"
            )
            continue

        Pv = d["Pv"]
        Pi = d["Pi_aligned"]

        Ov = d["Ov"]
        Oi = d["Oi"]
        Oofficial = d["Oofficial"]

        GT = d["GT"]

        gt_valid = d["gt_cell_valid_mask"]
        future_valid = d["future_valid_mask"]
        warp_valid = d["warp_valid_mask"]

        threshold = float(d["test_seg_thresh"])

        expected = (5, 200, 200)

        for key, arr in [
            ("Pv", Pv),
            ("Pi_aligned", Pi),
            ("Ov", Ov),
            ("Oi", Oi),
            ("Oofficial", Oofficial),
            ("GT", GT),
            ("gt_cell_valid_mask", gt_valid),
        ]:
            if arr.shape != expected:
                errors.append(
                    f"{name}: {key} shape={arr.shape}, "
                    f"expected={expected}"
                )

        if future_valid.shape != (5,):
            errors.append(
                f"{name}: future_valid_mask shape="
                f"{future_valid.shape}, expected=(5,)"
            )

        if warp_valid.shape != (200, 200):
            errors.append(
                f"{name}: warp_valid_mask shape="
                f"{warp_valid.shape}, expected=(200,200)"
            )

        if not np.isclose(threshold, 0.1, atol=1e-6):
            errors.append(
                f"{name}: threshold={threshold}, expected 0.1"
            )

        for key, arr in [
            ("Pv", Pv),
            ("Pi_aligned", Pi),
        ]:
            if not np.isfinite(arr).all():
                errors.append(
                    f"{name}: {key} contains NaN or Inf"
                )

            if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
                errors.append(
                    f"{name}: {key} outside [0,1], "
                    f"range=[{arr.min()}, {arr.max()}]"
                )

        for key, arr in [
            ("Ov", Ov),
            ("Oi", Oi),
            ("Oofficial", Oofficial),
            ("gt_cell_valid_mask", gt_valid),
            ("future_valid_mask", future_valid),
            ("warp_valid_mask", warp_valid),
        ]:
            unique = set(np.unique(arr).tolist())

            if not unique.issubset({0, 1}):
                errors.append(
                    f"{name}: {key} not binary: {sorted(unique)}"
                )

        Ov_check = (Pv > threshold).astype(np.uint8)
        Oi_check = (Pi > threshold).astype(np.uint8)

        if not np.array_equal(Ov, Ov_check):
            mismatch = np.count_nonzero(Ov != Ov_check)
            errors.append(
                f"{name}: Ov != (Pv > threshold), "
                f"mismatch={mismatch}"
            )

        if not np.array_equal(Oi, Oi_check):
            mismatch = np.count_nonzero(Oi != Oi_check)
            errors.append(
                f"{name}: Oi != (Pi_aligned > threshold), "
                f"mismatch={mismatch}"
            )

        official_check = np.maximum(Ov, Oi)

        if not np.array_equal(Oofficial, official_check):
            mismatch = np.count_nonzero(
                Oofficial != official_check
            )
            errors.append(
                f"{name}: Oofficial != Ov OR Oi, "
                f"mismatch={mismatch}"
            )

        gt_valid_check = (GT != 255).astype(np.uint8)

        if not np.array_equal(gt_valid, gt_valid_check):
            errors.append(
                f"{name}: gt_cell_valid_mask inconsistent "
                f"with GT != 255"
            )

        pv_min_all.append(float(Pv.min()))
        pv_max_all.append(float(Pv.max()))

        pi_min_all.append(float(Pi.min()))
        pi_max_all.append(float(Pi.max()))

        warp_ratios.append(float(warp_valid.mean()))

        ov_ratios.append(float(Ov.mean()))
        oi_ratios.append(float(Oi.mean()))
        official_ratios.append(float(Oofficial.mean()))

        if i < 5:
            print(
                f"{name} | "
                f"thr={threshold:.3f} | "
                f"Pv=[{Pv.min():.6f},{Pv.max():.6f}] | "
                f"Pi=[{Pi.min():.6f},{Pi.max():.6f}] | "
                f"warp={warp_valid.mean():.4f} | "
                f"Ov={Ov.mean():.4f} | "
                f"Oi={Oi.mean():.4f} | "
                f"Official={Oofficial.mean():.4f}"
            )

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Pv global range:",
    min(pv_min_all),
    max(pv_max_all)
)

print(
    "Pi_aligned global range:",
    min(pi_min_all),
    max(pi_max_all)
)

print(
    "warp_valid ratio mean/min/max:",
    np.mean(warp_ratios),
    np.min(warp_ratios),
    np.max(warp_ratios)
)

print(
    "Ov occupied ratio mean:",
    np.mean(ov_ratios)
)

print(
    "Oi occupied ratio mean:",
    np.mean(oi_ratios)
)

print(
    "Official occupied ratio mean:",
    np.mean(official_ratios)
)

print()

if errors:
    print("=" * 80)
    print(f"G0 STRUCTURAL CHECK: FAILED ({len(errors)} errors)")
    print("=" * 80)

    for err in errors:
        print("[ERROR]", err)

    raise SystemExit(1)

else:
    print("=" * 80)
    print("G0 STRUCTURAL CHECK: PASS")
    print("=" * 80)