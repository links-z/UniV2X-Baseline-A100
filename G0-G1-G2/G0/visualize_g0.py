import os
import glob
import numpy as np
import matplotlib.pyplot as plt

CACHE_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/cache_smoke"
OUT_DIR = "/root/autodl-tmp/UniV2X/G0-G1-G2/G0/visualization"

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

if not files:
    raise RuntimeError("No npz files found.")

# ============================================================
# 1. 统计每个样本 warp valid ratio
# ============================================================
infos = []

for path in files:
    with np.load(path) as d:
        warp = d["warp_valid_mask"]
        ratio = float(warp.mean())

    infos.append((path, ratio))

print("Warp valid ratio:")
for path, ratio in infos:
    print(f"{os.path.basename(path)}: {ratio:.6f}")

# ============================================================
# 2. 自动选 4 个代表样本
#
#   A: warp = 0 或最小
#   B: 最小的非零 warp
#   C: 非零样本中的中间水平
#   D: warp 最大
# ============================================================
infos_sorted = sorted(infos, key=lambda x: x[1])

sample_min = infos_sorted[0]

positive = [x for x in infos_sorted if x[1] > 0]

if len(positive) >= 3:
    sample_low = positive[0]
    sample_mid = positive[len(positive) // 2]
    sample_max = positive[-1]

    selected = [
        sample_min,
        sample_low,
        sample_mid,
        sample_max,
    ]
else:
    selected = infos_sorted[:4]

# 去重
unique_selected = []
seen = set()

for item in selected:
    if item[0] not in seen:
        unique_selected.append(item)
        seen.add(item[0])

print("\nSelected samples:")
for path, ratio in unique_selected:
    print(
        os.path.basename(path),
        f"warp_ratio={ratio:.6f}"
    )


# ============================================================
# 3. 画图函数
# ============================================================
def visualize_sample(path, warp_ratio):

    with np.load(path) as d:
        Pv = d["Pv"]
        Pi = d["Pi_aligned"]

        Ov = d["Ov"]
        Oi = d["Oi"]
        Oofficial = d["Oofficial"]

        GT = d["GT"]
        warp = d["warp_valid_mask"]

        threshold = float(d["test_seg_thresh"])

    sample_name = os.path.splitext(
        os.path.basename(path)
    )[0]

    # -----------------------------------------------
    # 一些额外 sanity checks
    # -----------------------------------------------
    outside = (warp == 0)

    # Pi shape = [5,200,200]
    outside_nonzero = np.count_nonzero(
        (np.abs(Pi) > 1e-8) &
        outside[None, :, :]
    )

    outside_total = int(
        outside.sum() * Pi.shape[0]
    )

    print(
        f"\n[{sample_name}] "
        f"warp={warp_ratio:.6f}, "
        f"Pi nonzero outside warp="
        f"{outside_nonzero}/{outside_total}"
    )

    # -----------------------------------------------
    # 5 horizons × 7 fields
    # -----------------------------------------------
    fig, axes = plt.subplots(
        5,
        7,
        figsize=(21, 15),
        constrained_layout=True
    )

    column_titles = [
        "Pv",
        "Pi_aligned",
        "Ov",
        "Oi",
        "Oofficial",
        "GT",
        "warp_valid_mask",
    ]

    for c, title in enumerate(column_titles):
        axes[0, c].set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

    for t in range(5):

        # GT 中的 255 不作为正常 occupancy 展示
        gt_t = np.ma.masked_equal(
            GT[t],
            255
        )

        data_list = [
            Pv[t],
            Pi[t],
            Ov[t],
            Oi[t],
            Oofficial[t],
            gt_t,
            warp,
        ]

        for c, data in enumerate(data_list):

            ax = axes[t, c]

            if c in [0, 1]:
                # soft probability
                ax.imshow(
                    data,
                    origin="lower",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest"
                )

            else:
                # binary / GT
                ax.imshow(
                    data,
                    origin="lower",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest"
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if c == 0:
                ax.set_ylabel(
                    f"t={t}",
                    fontsize=11,
                    fontweight="bold"
                )

    fig.suptitle(
        f"{sample_name} | "
        f"threshold={threshold:.3f} | "
        f"warp_valid_ratio={warp_ratio:.4f}",
        fontsize=15
    )

    out_path = os.path.join(
        OUT_DIR,
        f"{sample_name}_G0.png"
    )

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", out_path)


# ============================================================
# 4. 绘制
# ============================================================
for path, ratio in unique_selected:
    visualize_sample(path, ratio)

print("\n========================================")
print("G0 visualization finished.")
print("Output directory:")
print(OUT_DIR)
print("========================================")