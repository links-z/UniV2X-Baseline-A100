# G3-1C：Spatial Position Context 增量可学习性验证

## 1. 实验目的

G3-1B 已验证，加入 Horizon Context 后，Local + Horizon MLP 在测试集上的 AUPRC 从 0.1873 提升至 0.2089，并超过此前表现最好的 Local Linear（0.1941），说明预测 Horizon 能够提供 Local Occupancy Evidence 之外的增量信息。

G3-1C 进一步研究：

\[
\boxed{
\text{绝对空间位置是否能够在 Local + Horizon 基础上继续提供增量信息？}
}
\]

为避免同时改变空间位置表示和感受野范围，本阶段仅加入当前 Patch 的归一化二维位置坐标，不引入 neighboring patches。

---

## 2. Spatial Position 表示

当前 Occupancy BEV grid 大小为：

\[
200\times200
\]

采用 Patch-4 后，对应：

\[
50\times50
\]

个 Patch。

对于 Patch 的行列索引：

\[
r=\texttt{patch\_row}
\]

\[
c=\texttt{patch\_col}
\]

将 Patch 中心归一化到：

\[
[-1,1]
\]

定义：

\[
u_R
=
2\frac{c+0.5}{50}-1
\]

\[
v_R
=
2\frac{r+0.5}{50}-1
\]

其中：

\[
(u_R,v_R)
\]

表示归一化 Patch 坐标。

当前阶段暂不将其直接称为物理 BEV \((x,y)\)，以避免在未进一步确认 row/column 与 BEV 坐标轴方向对应关系前引入不必要假设。

---

## 3. 坐标 Sanity Check

归一化坐标检查结果：

```text
row: 0 49
col: 0 49

u: -0.98 0.98
v: -0.98 0.98
```

说明 Patch index 覆盖完整的 \(50\times50\) patch grid，归一化结果符合预期。

因此：

\[
\boxed{
\text{Spatial Position Encoding Sanity Check: PASS}
}
\]

---

## 4. 实验模型

### 4.1 Spatial-only Baseline

仅输入二维归一化空间坐标：

\[
X_S=[u_R,v_R]
\]

输入维度为：

\[
2
\]

模型用于回答：

\[
\boxed{
\text{单独依赖绝对位置能够预测多少 Cooperation Utility？}
}
\]

---

### 4.2 Local + Horizon + Spatial

G3-1B 输入为：

\[
X_{LH}
=
[
X_{local},
e_h
]
\]

维度：

\[
80+5=85
\]

加入空间坐标后：

\[
X_{LHS}
=
[
X_{local},
e_h,
u_R,
v_R
]
\]

因此：

\[
X_{LHS}\in\mathbb{R}^{87}
\]

模型结构保持为：

```text
87
↓
128
↓
64
↓
1
```

除输入维度外，其余训练设置与 G3-1B 保持一致。

---

## 5. 实验协议

G3-1C 与前述实验保持相同：

```text
Scene split:
Train = 13 scenes
Val   = 4 scenes
Test  = 4 scenes

Seed          = 2026
Batch size    = 2048
Learning rate = 1e-3
Weight decay  = 1e-4
Optimizer     = AdamW
Loss          = BCEWithLogitsLoss(pos_weight)
Early stopping= 10
Best model    = Validation AUPRC
Threshold     = selected on Validation set
Test          = evaluated with frozen threshold
```

Test 集正类比例为：

\[
0.1412
\]

因此随机 AUPRC 参考水平约为：

\[
0.1412
\]

---

## 6. 实验结果

| 模型 | 输入 | Test AUROC | Test AUPRC | ΔAUPRC vs Random |
|---|---|---:|---:|---:|
| Local Linear | Local | 0.6067 | 0.1941 | +0.0529 |
| Local MLP | Local | 0.6066 | 0.1873 | +0.0461 |
| Horizon-only | H | 0.5845 | 0.1705 | +0.0293 |
| Local + H | Local + H | 0.6300 | 0.2089 | +0.0677 |
| Spatial-only | \((u,v)\) | 0.5051 | 0.1453 | +0.0041 |
| **Local + H + S** | **Local + H + \((u,v)\)** | **0.6501** | **0.2118** | **+0.0706** |

---

## 7. Spatial-only 分析

Spatial-only Baseline 的测试结果为：

\[
AUROC=0.5051
\]

\[
AUPRC=0.1453
\]

与随机水平相比：

\[
0.1453-0.1412
=
0.0041
\]

说明单独依赖二维绝对位置几乎无法有效完成局部 Cooperation Utility Prediction。

因此不能据此声称：

\[
\text{Absolute Spatial Position alone has strong predictive ability}
\]

更准确的结论是：

\[
\boxed{
\text{Absolute Position Alone 提供的独立判别信息非常有限}
}
\]

---

## 8. Local + Horizon + Spatial 增量分析

Local + Horizon 模型：

\[
AUROC=0.6300
\]

\[
AUPRC=0.2089
\]

加入 Spatial Position 后：

\[
AUROC=0.6501
\]

\[
AUPRC=0.2118
\]

因此：

\[
\Delta AUROC
=
0.6501-0.6300
=
+0.0201
\]

\[
\Delta AUPRC
=
0.2118-0.2089
=
+0.0029
\]

AUPRC 相对提升约：

\[
\frac{0.0029}{0.2089}\times100\%
\approx1.4\%
\]

因此，加入归一化空间位置后，模型在测试集上观察到小幅增益。

同时：

\[
AUROC
\]

提升幅度明显大于：

\[
AUPRC
\]

提升幅度，说明 Spatial Position 对整体排序能力的改善相对更明显，而对正类稀疏条件下 Precision-Recall 表现的改善较有限。

---

## 9. G3-1C 结论

G3-1C 得到以下结论：

1. Spatial-only Baseline 的性能接近随机水平，说明简单的二维绝对位置本身几乎不足以完成 Cooperation Utility Prediction；
2. 在 Local + Horizon 基础上加入 Spatial Position 后，Test AUROC 从 0.6300 提升至 0.6501；
3. Test AUPRC 从 0.2089 提升至 0.2118；
4. AUPRC 绝对增量为 0.0029，相对提升约 1.4%；
5. 因此，绝对空间位置可能提供一定的补充信息，但增量有限；
6. 简单的 \((u,v)\) 坐标不足以充分刻画 G2-A 中观察到的空间异质性。

综合判断：

\[
\boxed{
\text{G3-1C: GO, but weak incremental evidence}
}
\]

需要注意，当前实验仅采用单个固定随机种子和 Scene split，因此不应使用“统计显著提升”或“稳定显著提升”等表述。

---

# 10. 论文中可直接使用的表述

## 10.1 Spatial Position 增量实验

为进一步分析空间位置是否能够改善局部协同价值预测，本文在 Local + Horizon 输入基础上加入当前 Patch 的二维归一化位置坐标。结果显示，模型的测试集 AUROC 由 0.6300 提升至 0.6501，AUPRC 由 0.2089 提升至 0.2118。

其中 AUPRC 的绝对提升为 0.0029，约对应 1.4% 的相对增幅。相比之下，仅使用二维空间位置的 Spatial-only Baseline 仅取得 0.5051 AUROC 和 0.1453 AUPRC，接近随机参考水平。

该结果表明，绝对空间位置本身并不足以可靠预测局部协同价值，但在结合局部 Occupancy Evidence 和 Horizon Context 后，可以提供一定的补充信息。

---

## 10.2 对空间异质性的解释

G2-A 已观察到路侧新增 Occupancy 的错误比例在空间上具有明显异质性。然而，G3-1C 表明，仅使用二维绝对位置坐标只能带来有限的 AUPRC 增益。

这说明：

\[
\boxed{
\text{空间异质性并不能被简单的绝对位置坐标充分刻画}
}
\]

局部 Cooperation Utility 可能进一步依赖周围区域 Occupancy 结构、局部一致性以及 Ego/Infrastructure 之间的空间关系。

因此，后续实验需要进一步验证 Neighborhood Context 是否能够提供更强的空间增量信息。

---

# 11. 与 STCV-Occ 方法设计的衔接

当前可学习性证据链为：

```text
Local Evidence
      ↓
存在基础可学习信号
      ↓
+ Horizon Context
      ↓
获得明显增量
      ↓
+ Absolute Spatial Position
      ↓
获得小幅额外增量
      ↓
Absolute Position 不足以充分刻画空间异质性
      ↓
进一步测试 Neighborhood Spatial Context
```

因此，STCV-Occ 的空间部分不应简单等价为二维绝对坐标，而应进一步考虑局部邻域中的空间结构信息。

---

# 12. 当前 G3 状态

```text
G3-0   PASS
G3-1A  Weak GO
G3-1B  GO
G3-1C  GO, weak incremental evidence

Current Best:
Local + Horizon + Spatial

Test AUROC = 0.6501
Test AUPRC = 0.2118

Next:
G3-1D Neighborhood Context
```

\[
\boxed{
\text{G3-1C COMPLETE}
}
\]