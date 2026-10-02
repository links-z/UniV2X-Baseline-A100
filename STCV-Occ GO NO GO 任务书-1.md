# STCV-Occ GO / NO-GO 阶段性决策任务书

## 一、任务目标

本任务用于判断第二项研究工作 **STCV-Occ** 是否值得继续，以及当前设想的 **Cooperation Value Learning** 是否具有可学习性和方法必要性。

整个 GO / NO-GO 分为两个阶段：

### 第一阶段：方向可行性筛选

包括：

- G0：数据流正确性；
- G1：负合作是否真实存在；
- G2：结构级选择是否具有 Oracle 提升空间。

该阶段回答：

> **UniV2X Occupancy Fusion 是否确实存在值得研究的负合作问题，并且这种问题是否能够通过区域级选择进行改善？**

### 第二阶段：方法可行性筛选

包括：

- G3：Cooperation Utility 是否能够被模型学习；
- G4：Value Learning 是否具有独立的方法必要性。

该阶段回答：

> **即使研究问题成立，当前提出的 STCV-Occ / Cooperation Value Learning 是否真正有效且值得作为核心方法？**

---

# 二、当前基础状态

目前已经完成：

- UniV2X 官方代码复现；
- 官方 Cooperative Planning 模型正常运行；
- 官方 checkpoint 正常加载；
- 官方验证流程跑通。

因此，本阶段不重新训练原始 UniV2X。

固定使用：

```text
Config:
projects/configs_e2e_univ2x/univ2x_coop_e2e.py

Checkpoint:
ckpts/univ2x_coop_e2e_stg2.pth
```

所有实验尽量保持：

```text
同一 checkpoint
同一数据划分
同一 evaluation pipeline
```

---

# 三、总体流程

```text
Official UniV2X
      │
      ▼
G0：数据流是否正确？
      │
      ├── FAIL → 修正导出代码
      │
      ▼
G1：官方 Occupancy Fusion 是否存在稳定负合作？
      │
      ├── NO-GO → 停止该方向
      │
      ▼
G2：Patch / Component 级 Oracle 是否存在明显空间？
      │
      ├── NO-GO → 停止该方向
      │
      ▼
方向可行
      │
      ▼
正式开发 Candidate + Predictor
      │
      ▼
G3：Utility 是否可学习？
      │
      ├── NO-GO → 尝试简单 Gate / Classification
      │              │
      │              └── 仍失败 → 停止该方向
      ▼
G4：Value Learning 是否具有必要性？
      │
      ├── NO-GO → 保留结构选择，弱化 Value Learning
      │
      ▼
STCV-Occ 成立
      │
      ▼
消融 / 鲁棒性 / 下游实验 / 论文
```

---

# 四、官方 Occupancy Fusion 定义

根据当前 UniV2X 官方代码：

车端产生 Soft Occupancy Probability：

$$
P_v\in[0,1]
$$

路端经过空间对齐后得到：

$$
P_i\in[0,1]
$$

当前 Occupancy 测试阈值：

$$
\eta=0.1
$$

因此：

$$
O_v=\mathbf{1}[P_v>0.1]
$$

$$
O_i=\mathbf{1}[P_i>0.1]
$$

官方 Occupancy Fusion 为：

$$
\boxed{
O_{\mathrm{official}}
=
O_v\lor O_i
}
$$

即 Binary Max / OR。

必须明确区分：

### Official Fusion

$$
O_{\mathrm{official}}
=
\mathbf{1}[P_v>0.1]
\lor
\mathbf{1}[P_i>0.1]
$$

### Soft Max Diagnostic

$$
P_{\mathrm{soft}}
=
\max(P_v,P_i)
$$

其中 Soft Max 仅用于后续连续 Cooperation Utility 分析，**不是官方原始 OccFusion**。

---

# 五、G0：数据流正确性 Gate

## 5.1 目标

确认后续实验使用的核心变量正确：

$$
P_v,\quad
P_i,\quad
O_v,\quad
O_i,\quad
O_{\mathrm{official}},\quad
Y
$$

---

## 5.2 需要导出的变量

主要文件：

```text
projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py
```

需要保存：

```text
Pv
Pi_aligned
Ov
Oi
Oofficial
GT
valid_mask
sample_id
sequence_id
frame_id
horizon
```

其中：

$$
O_v=\mathbf{1}[P_v>0.1]
$$

$$
O_i=\mathbf{1}[P_i>0.1]
$$

$$
O_{\mathrm{official}}=O_v\lor O_i
$$

---

## 5.3 数值检查

检查：

```text
shape
dtype
min
max
mean
occupied_ratio
NaN
Inf
```

应满足：

$$
P_v,P_i\in[0,1]
$$

当前 Occupancy 网格应为：

$$
200\times200
$$

当前：

$$
n_{\mathrm{future}}=4
$$

因此包含：

$$
t=0,1,2,3,4
$$

共 5 个 Horizon。

---

## 5.4 可视化检查

随机抽取若干样本绘制：

```text
Pv
Pi_aligned
Ov
Oi
Official OR
GT
```

重点检查：

- 路端空间对齐是否正确；
- 是否存在整体偏移；
- Horizon 是否对应；
- GT 是否对应；
- Binary OR 是否与官方输出一致。

---

## 5.5 G0 判定

若全部正确：

$$
\boxed{
G0=PASS
}
$$

若存在关键数据错误：

$$
\boxed{
G0=FAIL
}
$$

先修正数据流，不进入 G1。

---

# 六、G1：Negative Cooperation Gate

## 6.1 核心问题

回答：

> **UniV2X 官方 Occupancy OR 是否会稳定地向 Ego Occupancy 中加入错误 Occupancy？**

---

## 6.2 官方新增区域

官方真正修改 Ego Occupancy 的位置：

$$
O_v(x,t)=0
$$

且：

$$
O_i(x,t)=1
$$

因此：

$$
\boxed{
M_t^{add}(x)
=
M_t^{valid}(x)
\land
[O_v(x,t)=0]
\land
[O_i(x,t)=1]
}
$$

---

## 6.3 Beneficial Addition

如果：

$$
M^{add}=1
$$

且：

$$
Y=1
$$

则为：

$$
\boxed{
Beneficial\ Addition
}
$$

表示 Infrastructure 修复了 Ego 的 False Negative。

---

## 6.4 Harmful Addition

如果：

$$
M^{add}=1
$$

且：

$$
Y=0
$$

则为：

$$
\boxed{
Harmful\ Addition
}
$$

表示 Infrastructure 新增了错误 Occupancy。

---

## 6.5 核心指标

新增数量：

$$
N_{add}
=
\sum M^{add}
$$

Beneficial：

$$
N_{beneficial}
=
\sum M^{add}Y
$$

Harmful：

$$
N_{harmful}
=
\sum M^{add}(1-Y)
$$

定义 Harmful Addition Ratio：

$$
\boxed{
HAR
=
\frac{
N_{harmful}
}{
N_{add}+\epsilon
}
}
$$

定义 Beneficial Addition Ratio：

$$
\boxed{
BAR
=
\frac{
N_{beneficial}
}{
N_{add}+\epsilon
}
}
$$

---

## 6.6 Ego 与 Official 对比

分别计算：

$$
IoU(O_v,Y)
$$

和：

$$
IoU(O_{\mathrm{official}},Y)
$$

同时报告：

```text
IoU
Precision
Recall
F1
```

定义：

$$
\Delta IoU
=
IoU_{\mathrm{official}}
-
IoU_v
$$

---

## 6.7 Frame-Level Negative Cooperation

对每个 Frame / Horizon：

$$
\Delta IoU_f
=
IoU_f(O_{\mathrm{official}},Y)
-
IoU_f(O_v,Y)
$$

统计：

$$
\boxed{
FDR
=
\frac{
N(\Delta IoU_f<0)
}{
N_{frames}
}
}
$$

即 Official Fusion 导致性能下降的 Frame 比例。

---

## 6.8 Horizon-wise 统计

分别统计：

$$
t=0,1,2,3,4
$$

输出：

| Horizon | Added | Beneficial | Harmful |  HAR | Ego IoU | Official IoU |
| ------- | ----: | ---------: | ------: | ---: | ------: | -----------: |
| t0      |       |            |         |      |         |              |
| t1      |       |            |         |      |         |              |
| t2      |       |            |         |      |         |              |
| t3      |       |            |         |      |         |              |
| t4      |       |            |         |      |         |              |
| All     |       |            |         |      |         |              |

---

## 6.9 Sequence-wise 检查

至少统计：

```text
sequence_id
added_cells
beneficial_cells
harmful_cells
HAR
ego_iou
official_iou
delta_iou
```

目的：

> 判断 Harmful Cooperation 是否只是集中在极少数异常序列。

---

## 6.10 G1 GO

若：

- Harmful Addition 稳定存在；
- 多个 Sequence 可观察；
- 多个 Horizon 可观察；
- 不只是极少数异常帧；
- 存在一定比例的 Performance-Drop Frames；
- Harmful Addition 数量具有实际规模；

则：

$$
\boxed{
G1=GO
}
$$

---

## 6.11 G1 NO-GO

若：

- Infrastructure 新增几乎全部正确；
- Harmful Addition 极少；
- 错误主要来自少数异常样本；
- Official 几乎不存在 Frame-Level 性能下降；

则：

$$
\boxed{
G1=NO\text{-}GO
}
$$

停止 STCV-Occ 路线。

可考虑切换：

```text
Pose-/Latency-Aware Occupancy Alignment
```

---

# 七、G2：Structured Oracle Gate

## 7.1 核心问题

Cell Oracle 可以利用 GT 对每个 Cell 单独选择，因此过于理想。

G2 真正回答：

> **如果一个区域必须统一 Accept 或 Fallback，是否仍然存在明显提升空间？**

---

# 八、Cell Oracle

Cell Oracle 仅作为 Loose Upper Bound。

对于：

$$
M^{add}=1
$$

若：

$$
Y=1
$$

则接受 Infrastructure。

若：

$$
Y=0
$$

则回退 Ego。

得到：

$$
O_{\mathrm{CellOracle}}
$$

计算：

$$
IoU_{\mathrm{CellOracle}}
$$

注意：

> **Cell Oracle 提升很大并不能单独证明第二工作值得继续。**

---

# 九、Patch Oracle

第一阶段采用固定非重叠 Patch：

$$
4\times4
$$

$$
8\times8
$$

$$
16\times16
$$

分别记为：

```text
Patch-4
Patch-8
Patch-16
```

---

## 9.1 Patch 动作

每个 Patch 只允许两个动作。

### Fallback

保持：

$$
O_v
$$

### Accept

使用官方动作：

$$
O_v\lor O_i
$$

---

## 9.2 Patch Utility

设 Patch 为 $R$。

正确新增：

$$
N_R^+
=
\sum_{x\in R}
M^{add}(x)Y(x)
$$

错误新增：

$$
N_R^-
=
\sum_{x\in R}
M^{add}(x)(1-Y(x))
$$

定义：

$$
\boxed{
U_R=N_R^+-N_R^-
}
$$

若：

$$
U_R>0
$$

则接受该 Patch 的 Infrastructure 信息。

若：

$$
U_R\le0
$$

则保持 Ego。

---

# 十、G2 结果表

| Method          |  IoU | Precision | Recall |   F1 | ΔIoU vs Official |
| --------------- | ---: | --------: | -----: | ---: | ---------------: |
| Ego             |      |           |        |      |                  |
| Official OR     |      |           |        |      |                0 |
| Cell Oracle     |      |           |        |      |                  |
| Patch-4 Oracle  |      |           |        |      |                  |
| Patch-8 Oracle  |      |           |        |      |                  |
| Patch-16 Oracle |      |           |        |      |                  |

---

# 十一、Oracle Gap

Cell Oracle Gap：

$$
Gap_{cell}
=
IoU_{CellOracle}
-
IoU_{Official}
$$

Patch Oracle Gap：

$$
Gap_{patch}
=
IoU_{PatchOracle}
-
IoU_{Official}
$$

定义 Oracle Retention Ratio：

$$
\boxed{
R_{patch}
=
\frac{
Gap_{patch}
}{
Gap_{cell}+\epsilon
}
}
$$

用于观察 Patch 还能保留多少 Cell Oracle 的理论收益。

---

# 十二、G2 判定

## Strong GO

如果某个 Patch：

$$
Gap_{patch}\gtrsim1.0
$$

IoU point，

且在多个 Horizon / Sequence 上稳定：

$$
\boxed{
G2=STRONG\ GO
}
$$

---

## GO

若：

$$
Gap_{patch}\approx0.5\sim1.0
$$

且提升稳定：

$$
\boxed{
G2=GO
}
$$

---

## GRAY

若：

$$
Gap_{patch}\approx0.2\sim0.5
$$

则：

$$
\boxed{
G2=GRAY
}
$$

只补做：

```text
Connected Component Oracle
```

然后再决定。

---

## NO-GO

如果：

$$
CellOracle\gg Official
$$

但：

$$
Patch4
\approx
Patch8
\approx
Patch16
\approx
Official
$$

则说明理论收益主要依赖 GT 逐 Cell 选择。

此时：

$$
\boxed{
G2=NO\text{-}GO
}
$$

停止该方向。

---

# 十三、G0-G2 第一阶段最终判定

若：

$$
G0=PASS
$$

$$
G1=GO
$$

且：

$$
G2=GO
$$

则：

$$
\boxed{
\text{研究方向可行}
}
$$

意味着：

> UniV2X Occupancy Fusion 存在真实负合作，并且区域级选择具有可利用空间。

但此时还不能证明：

```text
Cooperation Value Learning
```

一定成立。

---

# 十四、G0-G2 通过后的正式开发

进入：

```text
Patch / Component Candidate
        ↓
Continuous Cooperation Utility
        ↓
Value Predictor
        ↓
Benefit Classification
        ↓
Same-Capacity SoftGate
```

随后进入 G3 和 G4。

---

# 十五、G3：Learnability Gate

## 15.1 核心问题

G2 证明的是：

> 使用 GT 可以选择出更好的合作区域。

G3 需要回答：

> **在推理阶段没有 GT 时，仅根据模型可获得的信息，能否预测哪些 Candidate 值得接受？**

---

## 15.2 Soft Candidate Action

用于连续 Utility 分析：

$$
P_{\max}
=
\max(P_v,P_i)
$$

注意：

> $P_{\max}$ 是研究用概率域 Candidate Action，不是官方 OccFusion。

---

## 15.3 Cooperation Utility

采用 Brier Loss：

$$
\ell(P,Y)
=
(P-Y)^2
$$

Cell-level Utility：

$$
\Delta\ell_t(x)
=
(P_v-Y)^2
-
(P_{\max}-Y)^2
$$

对于 Candidate $C_{k,t}$：

$$
\boxed{
V^{mean}_{k,t}
=
\frac{1}{|C_{k,t}|}
\sum_{x\in C_{k,t}}
\Delta\ell_t(x)
}
$$

同时计算：

$$
V^{total}_{k,t}
=
\sum_{x\in C_{k,t}}
\Delta\ell_t(x)
$$

---

## 15.4 Predictor

输入可包含：

```text
Pv
Pi
Pi - Pv
|Pi - Pv|
valid mask
candidate area
mean / max probability
distance
horizon embedding
```

学习：

$$
\hat V_{k,t}
=
g_\theta(C_{k,t})
$$

最终：

$$
G_{k,t}
=
\mathbf{1}[\hat V_{k,t}>\tau]
$$

---

## 15.5 G3 评价指标

至少报告：

```text
Value MAE
Spearman Correlation
Top-K Actual Value
Benefit Classification AUC
Final Occupancy IoU
HCAR
BCR
NCG
```

其中：

### BCR

$$
BCR
=
\frac{
\text{Accepted Beneficial}
}{
\text{All Beneficial}
}
$$

### HCAR

$$
HCAR
=
\frac{
\text{Accepted Harmful}
}{
\text{All Harmful}
}
$$

### Net Cooperation Gain

$$
NCG
=
\sum_{k,t}
G_{k,t}V^{total}_{k,t}
$$

---

## 15.6 G3 GO

若：

- $\hat V$ 与真实 Value 存在稳定相关；
- Spearman 明显高于随机；
- Top-K Candidate 实际收益明显更高；
- Harmful Acceptance 明显下降；
- 最终 Occupancy 指标稳定高于 Official；

则：

$$
\boxed{
G3=GO
}
$$

---

## 15.7 G3 NO-GO

若：

- Value Prediction 接近随机；
- Spearman 接近 0；
- Top-K 无排序能力；
- 验证集无法复现训练集趋势；
- 最终性能基本等于 Official；

则：

$$
\boxed{
G3=NO\text{-}GO
}
$$

此时依次尝试：

```text
Value Regression
↓
Benefit Classification
↓
Simple Patch Gate
↓
Same-Capacity SoftGate
```

若这些方法仍无法利用 G2 中的 Oracle Gap，则：

$$
\boxed{
\text{理论空间存在，但实际不可学习}
}
$$

此时停止 STCV-Occ 路线。

---

# 十六、G4：Method Necessity Gate

## 16.1 核心问题

即使 G3 证明可以学习，还必须回答：

> **为什么需要 Cooperation Value，而不是直接做 Accept / Reject Classification 或 SoftGate？**

---

## 16.2 强基线

至少比较：

```text
Official Fusion
Confidence Gate
Probability Gap Gate
Proposal Validity Classification
Benefit Binary Classification
Same-Capacity SoftGate
Value Regression
```

学习型方法尽可能保持：

```text
相同 Candidate
相同输入特征
相同 Backbone
相同 Pooling
相同 Temporal Module
相近参数量
相同训练数据
```

仅改变：

```text
Supervision Target
Output Head
Decision Form
```

---

## 16.3 G4 GO

如果 Value Regression 在以下方面表现出稳定优势：

- 更好的 Value Ranking；
- 更高的 Spearman；
- 更好的 Top-K Actual Gain；
- 更优的 HCAR-BCR trade-off；
- 更高的 NCG；
- 更好的风险控制；
- 最终 Occupancy 性能具有竞争力；

则：

$$
\boxed{
G4=GO
}
$$

此时可以保留：

```text
Cooperation Value Learning
```

作为 STCV-Occ 的核心方法。

---

## 16.4 G4 CONDITIONAL GO

如果：

- Value Regression 有效；
- 但 Benefit Classification 或 SoftGate 与其性能接近；
- Value 在排序、风险分析或可解释性方面仍有额外价值；

则：

$$
\boxed{
G4=CONDITIONAL\ GO
}
$$

可以保留 Value，但降低论文中的绝对表述。

核心表述调整为：

> Structure-aware selective occupancy cooperation with explicit cooperation utility modeling.

而不是宣称：

> Value Regression 是唯一有效方案。

---

## 16.5 G4 NO-GO

如果：

```text
Benefit Classification
或
Same-Capacity SoftGate
```

在主要指标上持续优于 Value Regression，

且 Value 在：

```text
Ranking
Risk Control
Calibration
Interpretability
```

方面也没有明显额外收益，

则：

$$
\boxed{
G4=NO\text{-}GO
}
$$

此时不需要放弃整个第二研究方向。

改为：

$$
\boxed{
\text{Structure-Aware Selective Occupancy Fusion}
}
$$

保留：

- Patch / Component；
- Horizon-specific Decision；
- Selective Cooperation；
- Harmful Cooperation Reduction；

删除或弱化：

```text
Cooperation Value Learning
```

作为核心创新。

---

# 十七、四个 Gate 的含义

| Gate | 回答的问题                   | NO-GO 后处理               |
| ---- | ---------------------------- | -------------------------- |
| G0   | 数据是否正确？               | 修代码                     |
| G1   | 问题是否真实存在？           | 换研究方向                 |
| G2   | 问题是否有结构化可利用空间？ | 换研究方向                 |
| G3   | Oracle 空间能否被模型学习？  | 换建模方式，仍失败则换方向 |
| G4   | Value Learning 是否有必要？  | 保留问题，换方法表述       |

---

# 十八、最终决策矩阵

## 情况 A

```text
G0 PASS
G1 GO
G2 GO
G3 GO
G4 GO
```

最终：

$$
\boxed{
STCV\text{-}Occ
}
$$

继续完整实验。

---

## 情况 B

```text
G0 PASS
G1 GO
G2 GO
G3 GO
G4 NO-GO
```

说明：

> 研究问题成立，也能够学习，但 Cooperation Value 并不是必要的核心监督。

最终转为：

$$
\boxed{
Structure\text{-}Aware\ Selective\ Occupancy\ Fusion
}
$$

---

## 情况 C

```text
G0 PASS
G1 GO
G2 GO
G3 NO-GO
```

说明：

> Oracle 空间存在，但当前输入或模型无法可靠预测。

先尝试：

```text
Benefit Classification
Simple Gate
SoftGate
```

如果仍失败：

$$
\boxed{
STOP
}
$$

---

## 情况 D

```text
G1 NO-GO
```

说明：

> 官方 Occupancy Fusion 本身不存在足够稳定的 Harmful Cooperation。

直接：

$$
\boxed{
STOP
}
$$

换第二研究方向。

---

## 情况 E

```text
G1 GO
G2 NO-GO
```

说明：

> Cell-Level 负合作存在，但区域级统一选择没有明显空间。

即：

$$
\boxed{
\text{问题存在，但不适合当前结构化选择路线}
}
$$

停止 STCV-Occ。

---

# 十九、第一阶段最小实验输出

在正式开发任何网络之前，只完成以下三张表。

## Table 1：Ego vs Official

| Method      |  IoU | Precision | Recall |   F1 |
| ----------- | ---: | --------: | -----: | ---: |
| Ego         |      |           |        |      |
| Official OR |      |           |        |      |

---

## Table 2：Negative Cooperation

| Horizon | Added | Beneficial | Harmful |  HAR |
| ------- | ----: | ---------: | ------: | ---: |
| t0      |       |            |         |      |
| t1      |       |            |         |      |
| t2      |       |            |         |      |
| t3      |       |            |         |      |
| t4      |       |            |         |      |
| All     |       |            |         |      |

---

## Table 3：Oracle

| Method      |  IoU | ΔIoU vs Official | Retention |
| ----------- | ---: | ---------------: | --------: |
| Official OR |      |                0 |         — |
| Cell Oracle |      |                  |      100% |
| Patch-4     |      |                  |           |
| Patch-8     |      |                  |           |
| Patch-16    |      |                  |           |

完成上述三张表后，先做：

$$
\boxed{
G0-G2\ Direction\ Decision
}
$$

不要提前训练 STCV-Occ。

---

# 二十、第一阶段必须生成的可视化

至少生成：

### Figure 1：Occupancy Data Flow

```text
Pv
Pi_aligned
Ov
Oi
Official OR
GT
```

### Figure 2：Infrastructure Added Occupancy

显示：

```text
Ego Occupancy
Infrastructure Added Occupancy
GT
```

### Figure 3：Beneficial / Harmful Map

```text
Beneficial Addition
Harmful Addition
```

### Figure 4：Horizon-wise HAR

横轴：

$$
t=0,1,2,3,4
$$

纵轴：

$$
HAR
$$

### Figure 5：Oracle Granularity

横轴：

```text
Official
Cell
Patch-4
Patch-8
Patch-16
```

纵轴：

```text
IoU
```

---

# 二十一、推荐实际执行顺序

```text
Step 1
导出 Pv / Pi / Oofficial / GT
        │
        ▼
Step 2
数值检查 + 可视化
        │
        ▼
G0
        │
        ▼
Step 3
统计 Official Added Mask
        │
        ▼
Step 4
统计 Beneficial / Harmful Addition
        │
        ▼
Step 5
Horizon / Sequence / Frame 分析
        │
        ▼
G1
        │
        ├── NO-GO → STOP
        │
        ▼
Step 6
Cell Oracle
        │
        ▼
Step 7
Patch-4 / 8 / 16 Oracle
        │
        ▼
G2
        │
        ├── NO-GO → STOP
        ├── GRAY → Component Oracle
        └── GO
              │
              ▼
Step 8
Candidate / Component 构造
              │
              ▼
Step 9
Value Predictor
              │
              ▼
G3
              │
              ├── NO-GO → 简单 Gate / Classification
              │
              ▼
Step 10
与 BenefitCls / SoftGate 公平比较
              │
              ▼
G4
              │
              ├── GO → STCV-Occ
              └── NO-GO → Structure-Aware Selective Fusion
```

---

# 二十二、核心原则

本 GO / NO-GO 不是为了证明 STCV-Occ 一定成立。

它的目的在于：

$$
\boxed{
\text{尽可能早、尽可能低成本地淘汰不成立的研究假设}
}
$$

因此：

- G1 不成立，则换问题；
- G2 不成立，则停止结构化选择路线；
- G3 不成立，则说明 Oracle 空间难以学习；
- G4 不成立，不代表问题失败，只代表 Value Learning 不是必要方案；
- 不因为已经投入时间就强行继续；
- 不因为 Cell Oracle 很高就认为方法一定成立；
- 不因为最终 IoU 有提升就忽略强基线比较；
- 不混淆 Official Binary OR 与 Soft Max；
- 不使用测试集决定阈值和方法结构。

---

# 二十三、最终目标

理想情况下，研究链条应形成：

$$
\boxed{
\text{Negative Cooperation Exists}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{Structured Oracle Gap Exists}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{Utility Is Learnable}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{Value Supervision Provides Additional Benefit}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{STCV-Occ}
}
$$

如果链条在任一阶段断裂，则按照对应 Gate 的规则及时调整方法或停止该方向。