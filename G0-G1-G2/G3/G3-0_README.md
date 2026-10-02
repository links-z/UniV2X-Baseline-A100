# G3-0：局部协同价值学习数据集构建与实验协议确认

## 1. G3-0 的目的

G3-0 用于构建后续局部协同价值学习所需的数据集，并验证数据构建过程与 G2-B 中 Patch-4 Oracle 的定义保持一致。

G3-0 不属于最终方法性能实验，其主要作用包括：

1. 确认 Occupancy 有效样本口径；
2. 构建 Patch-4 局部协同决策样本；
3. 根据 G2-B 中 BA/HA 定义生成局部协同价值标签；
4. 按 Scene 划分 Train / Val / Test，避免时序数据泄漏；
5. 严格复现 G2-B Patch-4 Oracle 的统计结果；
6. 为后续 G3-1 Cooperation Utility Learnability 实验提供固定数据基础。

---

# 2. Occupancy 有效样本审计

当前验证集共包含：

- Total samples：675
- Occupancy valid samples：549
- Occupancy invalid samples：126
- Scene 数量：21

通过身份一致性审计确认，在当前 UniV2X 代码、配置和验证集条件下：

\[
\boxed{
\texttt{future\_valid\_mask.all()}
\Longleftrightarrow
\texttt{occ\_to\_eval}
}
\]

对全部 675 个样本逐样本完全成立。

审计结果：

| 项目 | 数量 |
|---|---:|
| Cache valid samples | 549 |
| Evaluator valid samples | 549 |
| Intersection | 549 |
| Cache only | 0 |
| Evaluator only | 0 |

因此：

\[
\boxed{
\text{G1/G2/G3 使用的 549 个有效样本
与当前 UniV2X Occupancy evaluator 的有效样本集合完全一致}
}
\]

---

## 2.1 Occupancy 有效时间窗口

当前配置：

\[
\texttt{occ\_receptive\_field}=3
\]

\[
\texttt{occ\_n\_future\_only\_occ}=4
\]

因此 Occupancy evaluator 对一个样本要求的完整时间窗口为：

\[
[t-2,\ t-1,\ t,\ t+1,\ t+2,\ t+3,\ t+4]
\]

即：

- 2 个过去帧；
- 1 个当前帧；
- 4 个未来帧。

每个 Scene 的：

- 前 2 个样本无效；
- 后 4 个样本无效。

因此每个 Scene 有：

\[
2+4=6
\]

个不能进入完整 Occupancy evaluation 的样本。

当前共有 21 个 Scene，因此：

\[
21\times6=126
\]

最终：

\[
675-126=549
\]

与 evaluator 实际输出完全一致。

---

# 3. Patch-4 局部协同候选定义

G3 沿用 G2-B 中的 Patch-4 局部决策设置。

Patch size：

\[
\boxed{4\times4}
\]

车辆端和路侧端二值 Occupancy 分别为：

\[
O_v=\mathbf{1}[P_v>0.1]
\]

\[
O_i=\mathbf{1}[P_i^{aligned}>0.1]
\]

其中：

- \(P_v\)：OccFusion 前 Ego soft occupancy probability；
- \(P_i^{aligned}\)：空间对齐后的 Infrastructure soft occupancy probability。

定义有效区域：

\[
M_t^{valid}
=
M_t^{GT}
\land
M^{warp}
\]

定义路侧新增 Occupancy：

\[
M^{add}
=
(O_v=0)
\land
(O_i=1)
\land
M_t^{valid}
\]

进一步定义：

\[
HA
=
M^{add}\land(GT=0)
\]

\[
BA
=
M^{add}\land(GT=1)
\]

其中：

- HA：Harmful Addition，错误新增 Occupancy；
- BA：Beneficial Addition，正确新增 Occupancy。

---

# 4. Patch-level Cooperation Utility

对于局部 Patch \(R\)，定义：

\[
BA_R=\sum_R BA
\]

\[
HA_R=\sum_R HA
\]

局部协同效用定义为：

\[
\boxed{
U_R=BA_R-HA_R
}
\]

进一步构造 Accept/Fallback 标签：

\[
\boxed{
y_R=
\begin{cases}
1,&U_R>0\\
0,&U_R\leq0
\end{cases}
}
\]

其中：

- \(y_R=1\)：Accept；
- \(y_R=0\)：Fallback。

仅保留满足：

\[
ADD_R=BA_R+HA_R>0
\]

的 Patch。

原因是当 Patch 内不存在路侧新增 Occupancy 时，不存在实际的 Accept/Fallback 决策问题。

---

# 5. G3 输入特征

每个 Patch 的模型输入仅包含推理阶段可以获得的信息：

\[
\boxed{
X_R=
[
P_v,\;
P_i^{aligned},\;
P_i^{aligned}-P_v,\;
|P_i^{aligned}-P_v|,\;
M^{warp}
]
}
\]

对应 Shape：

```text
[5, 4, 4]
```

5 个通道分别为：

1. \(P_v\)
2. \(P_i^{aligned}\)
3. \(P_i^{aligned}-P_v\)
4. \(|P_i^{aligned}-P_v|\)
5. \(M^{warp}\)

需要强调：

\[
\boxed{
GT\text{ 仅用于生成训练监督标签，不作为模型输入}
}
\]

因此后续预测过程不依赖 GT。

---

# 6. G3-0 总体数据统计

G3-0 最终获得：

\[
\boxed{119,448}
\]

个满足：

\[
ADD_R>0
\]

的 Patch-4 候选区域。

标签统计：

| 类型 | 数量 | 比例 |
|---|---:|---:|
| Accept | 17,805 | 14.9% |
| Fallback | 101,643 | 85.1% |
| Total | 119,448 | 100% |

总体正类比例：

\[
\frac{17,805}{119,448}
\approx14.9\%
\]

说明后续任务存在明显类别不平衡。

因此模型评价不能仅使用 Accuracy，应重点使用：

- AUROC
- AUPRC
- Precision
- Recall
- F1

---

# 7. G2-B Patch-4 Sanity Check

G3-0 数据构建结果严格复现 G2-B Patch-4 Oracle。

## 7.1 Patch 数量

| 项目 | 数值 |
|---|---:|
| Addition-containing patches | 119,448 |
| Accepted patches | 17,805 |
| Rejected patches | 101,643 |
| Acceptance rate | 14.9% |

---

## 7.2 Accepted Patch

| 指标 | 数值 |
|---|---:|
| Accepted ADD | 133,206 |
| Accepted BA | 112,239 |
| Accepted HA | 20,967 |
| Accepted WAR | 15.74% |

其中：

\[
WAR_{accepted}
=
\frac{20,967}{133,206}
=
15.74\%
\]

---

## 7.3 Rejected Patch

| 指标 | 数值 |
|---|---:|
| Rejected ADD | 665,973 |
| Rejected BA | 40,273 |
| Rejected HA | 625,700 |
| Rejected WAR | 93.95% |

其中：

\[
WAR_{rejected}
=
\frac{625,700}{665,973}
=
93.95\%
\]

因此：

\[
\boxed{
\text{G3-0 Sanity Check: PASS}
}
\]

说明 G3 数据构建过程与 G2-B Patch-4 Oracle 定义完全一致。

---

# 8. Scene-level 数据划分

为避免同一 Scene 内相邻时序帧和高度相关 Patch 同时进入训练集与测试集，G3 不采用随机 Patch 划分，而是按照：

```text
scene_token
```

进行 Scene-level split。

当前划分：

```text
Train : 13 scenes
Val   : 4 scenes
Test  : 4 scenes
```

统计如下：

| Split | Scenes | Patches | Positive Rate |
|---|---:|---:|---:|
| Train | 13 | 86,143 | 14.77% |
| Val | 4 | 18,555 | 16.18% |
| Test | 4 | 14,750 | 14.12% |
| Total | 21 | 119,448 | 14.90% |

精确正类比例：

```text
Train = 0.1476614467
Val   = 0.1618431690
Test  = 0.1411525424
```

同一个 Scene 中的：

- Frame
- Horizon
- Patch

不会跨 Train / Val / Test。

该策略用于降低由于相邻帧高度相关造成的数据泄漏风险。

---

# 9. 数据文件结构

当前 G3-0 数据集：

```text
G3/
├── build_g3_dataset.py
├── g3_dataset_train.npz
├── g3_dataset_val.npz
├── g3_dataset_test.npz
├── scene_splits.txt
├── checkpoints/
├── models/
├── results/
└── README.md
```

三个 Dataset 文件结构如下。

## Train

```text
features    shape=(86143, 5, 4, 4)   dtype=float32
labels      shape=(86143,)           dtype=int32
metadata    shape=(86143,)           dtype=object
```

## Validation

```text
features    shape=(18555, 5, 4, 4)   dtype=float32
labels      shape=(18555,)           dtype=int32
metadata    shape=(18555,)           dtype=object
```

## Test

```text
features    shape=(14750, 5, 4, 4)   dtype=float32
labels      shape=(14750,)           dtype=int32
metadata    shape=(14750,)           dtype=object
```

其中：

- `features`：推理阶段可获得的局部 Occupancy 特征；
- `labels`：由 \(U_R>0\) 构造的 Accept/Fallback 标签；
- `metadata`：Scene、Sample、Horizon、Patch 位置等相关信息。

---

# 10. G3-0 最终结论

G3-0 成功完成 Patch-level Cooperation Utility Dataset 的构建，并完成数据与实验协议验证。

主要结果为：

| 项目 | 结果 |
|---|---:|
| Total validation samples | 675 |
| Valid Occupancy samples | 549 |
| Scene number | 21 |
| Patch size | \(4\times4\) |
| ADD > 0 patches | 119,448 |
| Accept patches | 17,805 |
| Fallback patches | 101,643 |
| Acceptance rate | 14.9% |
| Accepted WAR | 15.74% |
| Rejected WAR | 93.95% |
| Train patches | 86,143 |
| Val patches | 18,555 |
| Test patches | 14,750 |

综合来看：

1. G3 使用的 549 个有效样本与当前 UniV2X Occupancy evaluator 使用的有效样本集合逐样本完全一致；
2. G3-0 严格复现 G2-B Patch-4 Oracle 的 Patch 数量与 Accept/Fallback 统计；
3. Accepted 与 Rejected Patch 的 BA、HA、ADD 和 WAR 均与 G2-B 完全一致；
4. 数据按照 Scene 进行划分，减少相邻时序数据泄漏风险；
5. GT 仅用于构造局部 Cooperation Utility 监督标签，不进入预测模型输入。

因此：

\[
\boxed{
\text{G3-0: PASS}
}
\]

---

# 11. G3-0 在整个研究流程中的位置

当前实验逻辑为：

```text
G0
│
├── Occupancy Cache 与变量正确性验证
├── Occupancy 有效样本身份审计
│
▼
G1
│
├── Negative Cooperation 存在性验证
│
▼
G2-A
│
├── BA / HA 空间结构诊断
│
▼
G2-B
│
├── Patch-level Accept/Fallback Oracle
├── 验证局部选择性融合存在性能空间
│
▼
G3-0
│
├── 构建 Cooperation Utility Dataset
├── Scene-level Train/Val/Test Split
├── 复现 Patch-4 Oracle 统计
│
▼
G3-1
│
└── Cooperation Utility Learnability
```

因此 G3-0 的作用是：

\[
\boxed{
\text{将 G2 中基于 GT 的 Oracle 决策问题，
转化为一个可训练、可验证的推理期局部协同价值预测问题}
}
\]

---

# 12. 论文中可直接使用的实验设置描述

## 12.1 数据构建描述

为进一步验证局部协同价值能否由推理阶段可观测信息预测，本文基于前述 Patch-level Oracle 分析构建局部协同价值学习数据集。对于每个 \(4\times4\) BEV 局部区域，首先统计路侧新增 Occupancy 中的 Beneficial Addition（BA）与 Harmful Addition（HA），并定义局部协同效用为

\[
U_R=BA_R-HA_R.
\]

当 \(U_R>0\) 时，将该区域标记为 Accept，否则标记为 Fallback。为避免无实际融合决策的区域干扰学习，仅保留包含至少一个路侧新增 Occupancy 的 Patch。

最终，在 549 个有效 Occupancy 样本上共获得 119,448 个候选区域，其中 17,805 个区域具有正协同效用，占总样本的 14.9%。

---

## 12.2 数据划分描述

考虑到同一场景中相邻帧及其局部区域具有较强时序相关性，若直接随机划分 Patch，可能造成训练集与测试集之间的信息泄漏。因此，本文按照 `scene_token` 对数据进行场景级划分，将 21 个场景划分为 13 个训练场景、4 个验证场景和 4 个测试场景。

最终训练集、验证集和测试集分别包含 86,143、18,555 和 14,750 个 Patch，对应正类比例分别为 14.77%、16.18% 和 14.12%。

---

## 12.3 输入特征描述

在不使用 Ground Truth 的条件下，局部协同价值预测器仅使用推理阶段可获得的信息作为输入。对于每个 \(4\times4\) Patch，构建由 Ego soft occupancy probability、对齐后的 Infrastructure soft occupancy probability、二者的概率差、绝对概率差以及 Warp 有效区域组成的五通道输入：

\[
X_R=
[
P_v,
P_i^{aligned},
P_i^{aligned}-P_v,
|P_i^{aligned}-P_v|,
M^{warp}
].
\]

因此，每个局部样本的输入维度为 \(5\times4\times4\)。GT 信息仅用于构造训练监督标签，不参与模型推理。

---

## 12.4 数据一致性描述

为保证后续可学习性分析与 UniV2X Occupancy 评估协议保持一致，本文进一步对 Cache 中的有效样本与当前 UniV2X evaluator 的实际评价样本进行逐样本身份审计。结果显示，两种方式均获得 549 个有效样本，且 549 个样本完全重合，无额外或缺失样本。

此外，G3 数据构建过程能够严格复现 G2-B 中 Patch-4 Oracle 的候选 Patch 数量、Accept/Fallback 数量以及 Accepted/Rejected WAR，验证了数据构建过程与前述 Oracle 分析的一致性。

---

# 13. 论文结果表建议

论文中可以使用如下表格描述 G3 数据集。

| Dataset Split | Scenes | Patches | Positive Patches | Positive Rate |
|---|---:|---:|---:|---:|
| Train | 13 | 86,143 | 12,720 | 14.77% |
| Validation | 4 | 18,555 | 3,003 | 16.18% |
| Test | 4 | 14,750 | 2,082 | 14.12% |
| Total | 21 | 119,448 | 17,805 | 14.90% |

另可报告 Patch-4 Oracle 数据分布：

| Decision | Patch Number | ADD | BA | HA | WAR |
|---|---:|---:|---:|---:|---:|
| Accept | 17,805 | 133,206 | 112,239 | 20,967 | 15.74% |
| Fallback | 101,643 | 665,973 | 40,273 | 625,700 | 93.95% |

---

# 14. 写论文时必须保留的口径说明

需要注意：

1. G3-0 的 549 个样本与当前 UniV2X Occupancy evaluator 的评价样本集合一致；
2. 但 G1/G2/G3 中部分诊断指标是在 Candidate Domain 与 Warp-valid 区域内计算；
3. 因此这些诊断 IoU、WAR 等指标不能直接等价为 UniV2X 官方 Occupancy IoU；
4. 官方 Occupancy benchmark 与本文局部协同诊断指标应在论文中分开报告；
5. G3-0 属于数据构建与实验协议验证，不应表述为最终方法性能提升；
6. 真正的 G3 方法证据来自后续 Cooperation Utility Learnability 与 Learned Fusion 实验。

---

# 15. 当前状态

```text
G3-0 Dataset Construction      PASS
G3-0 Evaluator Identity Audit PASS
G3-0 G2-B Reproduction        PASS
G3-0 Scene-level Split        PASS

Status:
READY FOR G3-1
```

\[
\boxed{
\text{G3-0 COMPLETE}
}
\]


为分析预测时域对局部协同价值的影响，进一步统计不同预测 horizon 下正协同效用 Patch 的比例。结果表明，各 horizon 的正类比例存在明显差异，例如测试集中由 \(h=1\) 的 19.11% 下降至 \(h=4\) 的 8.19%，说明局部协同价值具有显著的预测时域异质性。同时，不同场景划分下各 horizon 的正类分布并非严格单调一致，表明 horizon 信息应作为协同价值预测的上下文因素，而不宜直接作为固定规则进行决策。
这段可以直接作为后面 STCV-Occ 的 temporal motivation。
所以现在的判断是：
\[
\boxed{\text{G3-1B 值得做，而且建议加入 Horizon-only baseline}}
\]
下一步可以直接实现 Horizon-only + Local+Horizon MLP，并保持 seed、split、loss、optimizer、threshold selection 全部与 G3-1A 一致。

结果-1
============================================================
TRAIN
============================================================
h=0: N= 18022, positive= 3042, rate=0.1688
h=1: N= 18446, positive= 3102, rate=0.1682
h=2: N= 18448, positive= 2638, rate=0.1430
h=3: N= 17008, positive= 2160, rate=0.1270
h=4: N= 14219, positive= 1778, rate=0.1250

============================================================
VAL
============================================================
h=0: N=  3747, positive=  535, rate=0.1428
h=1: N=  3700, positive=  640, rate=0.1730
h=2: N=  3902, positive=  682, rate=0.1748
h=3: N=  3902, positive=  638, rate=0.1635
h=4: N=  3304, positive=  508, rate=0.1538

============================================================
TEST
============================================================
h=0: N=  3073, positive=  492, rate=0.1601
h=1: N=  2857, positive=  546, rate=0.1911
h=2: N=  3111, positive=  478, rate=0.1536
h=3: N=  3120, positive=  354, rate=0.1135
h=4: N=  2589, positive=  212, rate=0.0819