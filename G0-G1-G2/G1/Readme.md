# G1: Negative Cooperation Analysis

## 1. 目的

G1 用于验证 UniV2X Occupancy 协同融合过程中是否存在稳定的负合作（Negative Cooperation）现象。

当前 UniV2X Occupancy 融合采用车端与路侧二值 Occupancy 的 OR 融合：

\[
O_{\mathrm{official}} = O_v \lor O_i
\]

其中：

- \(O_v\)：OccFusion 前 ego soft occupancy 经阈值二值化后的结果；
- \(O_i\)：路侧 Occupancy 经空间对齐后再经阈值二值化的结果；
- \(O_{\mathrm{official}}\)：UniV2X 当前 Occupancy 融合结果。

G1 重点研究：

> 当车端原本预测为空闲，而路侧新增 Occupancy 时，这些新增信息中有多少是真正有益的，又有多少属于错误新增；这些新增是否会进一步造成最终 Occupancy IoU 下降。

G1 不进行模型训练，只基于 G0 已导出的 Occupancy cache 进行离线统计。

---

## 2. 数据来源

输入缓存目录：

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full
```

缓存样本总数：

```text
675
```

每个 `.npz` 文件包含主要变量：

```text
Pv
Pi_aligned
Ov
Oi
Oofficial
GT
gt_cell_valid_mask
future_valid_mask
warp_valid_mask
scene_token
sample_idx
timestamp
test_seg_thresh
```

当前 Occupancy 二值化阈值：

```text
test_seg_thresh = 0.1
```

---

## 3. Occupancy 有效样本筛选

首先使用：

```python
future_valid_mask.all()
```

筛选具有完整 Occupancy 时序 GT 的样本。

统计结果：

```text
Total samples   : 675
Valid samples   : 549
Invalid samples : 126
```

549 个有效样本的数量与当前 UniV2X baseline Occupancy evaluator 中实际参与 Occupancy 指标计算的样本数量一致。

需要注意：

G1 的核心分析进一步限制在 GT 有效且路侧 warp 有效的空间区域，因此 G1 中计算的 Candidate-domain IoU 与 UniV2X 官方 evaluator 输出的 Occupancy IoU 不属于完全相同的空间统计口径。

---

## 4. Candidate-domain 定义

对于每个预测 horizon \(t\)，定义有效分析区域：

\[
M_t^{valid}
=
M_t^{GT}
\land
M^{warp}
\]

代码实现：

```python
valid_mask = (
    (gt_cell_valid[t] == 1)
    & (warp_valid == 1)
)
```

其中：

- \(M_t^{GT}\)：GT 有效格子；
- \(M^{warp}\)：路侧 Occupancy 经过 BEV warp 后有效的空间区域。

因此，G1 重点分析：

> 路侧真正能够参与 Occupancy 协同的有效空间区域。

---

## 5. 路侧新增 Occupancy 定义

定义路侧独有新增区域：

\[
M_t^{add}
=
(\neg O_v)
\land
O_i
\land
M_t^{valid}
\]

代码实现：

```python
add_mask = (
    (ov_t == 0)
    & (oi_t == 1)
    & valid_mask
)
```

即：

> 车端原本预测为空闲，但路侧预测为占用的格子。

---

## 6. HA 与 BA 定义

### 6.1 Harmful Addition（HA）

错误新增定义为：

\[
HA
=
M^{add}
\land
(GT=0)
\]

表示：

> 路侧新增了 Occupancy，但 GT 表明该位置实际上为空闲。

代码：

```python
ha_mask = add_mask & (gt_t == 0)
```

---

### 6.2 Beneficial Addition（BA）

正确新增定义为：

\[
BA
=
M^{add}
\land
(GT=1)
\]

表示：

> 车端原本未预测为占用，而路侧新增的 Occupancy 与 GT 一致。

代码：

```python
ba_mask = add_mask & (gt_t == 1)
```

因此：

\[
ADD = HA + BA
\]

---

## 7. 新增可靠性指标

### 7.1 Wrong Addition Ratio（WAR）

定义错误新增比例：

\[
WAR
=
\frac{HA}{HA+BA}
=
\frac{HA}{ADD}
\]

WAR 越高，表示路侧独有新增 Occupancy 中错误信息占比越高。

---

### 7.2 Beneficial Addition Ratio（BAR_add）

定义正确新增比例：

\[
BAR_{add}
=
\frac{BA}{HA+BA}
=
\frac{BA}{ADD}
\]

因此：

\[
WAR + BAR_{add}=1
\]

---

## 8. Negative Cooperation 定义

仅使用：

\[
HA > BA
\]

不能严格定义性能意义上的负合作。

因此，G1 进一步直接比较 Ego Occupancy 与 Official OR Fusion 的 IoU。

定义：

\[
\Delta IoU
=
IoU_{\mathrm{official}}
-
IoU_{\mathrm{ego}}
\]

其中：

### 负合作

\[
\Delta IoU < 0
\]

表示加入路侧 Occupancy 后，IoU 下降。

### 正合作

\[
\Delta IoU > 0
\]

表示加入路侧 Occupancy 后，IoU 提升。

### 中性合作

\[
\Delta IoU \approx 0
\]

表示加入路侧 Occupancy 前后性能基本不变。

这里的 IoU 均在：

\[
M^{GT}\cap M^{warp}
\]

Candidate-domain 内统计。

因此，该 IoU 是用于 G1 负合作诊断的局部统计指标，不直接等价于 UniV2X 官方 evaluator 报告的 Occupancy IoU。

---

## 9. G1 总体结果

有效样本数：

```text
549 / 675
```

总体新增统计：

```text
ADD = 799,179
HA  = 646,667
BA  = 152,512
```

错误新增与正确新增数量之比：

\[
\frac{HA}{BA}
=
4.24
\]

即：

> 错误新增格子数量约为正确新增格子数量的 4.24 倍。

总体新增可靠性：

```text
WAR     = 80.92%
BAR_add = 19.08%
```

即：

> 在 Candidate-domain 内，路侧独有新增 Occupancy 中约 80.9% 为错误新增，仅约 19.1% 为正确新增。

---

## 10. Horizon-wise Results

| Horizon | ADD | HA | BA | WAR | BAR_add | Ego IoU | Official IoU | ΔIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t=0 | 150,615 | 117,854 | 32,761 | 0.7825 | 0.2175 | 0.3756 | 0.3525 | -0.0231 |
| t=1 | 161,620 | 125,468 | 36,152 | 0.7763 | 0.2237 | 0.2932 | 0.2919 | -0.0013 |
| t=2 | 175,622 | 141,803 | 33,819 | 0.8074 | 0.1926 | 0.2536 | 0.2492 | -0.0044 |
| t=3 | 173,047 | 144,582 | 28,465 | 0.8355 | 0.1645 | 0.2419 | 0.2278 | -0.0141 |
| t=4 | 138,275 | 116,960 | 21,315 | 0.8459 | 0.1541 | 0.2387 | 0.2232 | -0.0155 |
| Total | 799,179 | 646,667 | 152,512 | 0.8092 | 0.1908 | 0.2842 | 0.2710 | -0.0132 |

WAR 随预测时域总体呈上升趋势：

```text
t=0 : 78.25%
t=1 : 77.63%
t=2 : 80.74%
t=3 : 83.55%
t=4 : 84.59%
```

可以观察到：

- t=0 与 t=1 的 WAR 相对较低；
- 从 t=2 开始 WAR 明显升高；
- t=4 的 WAR 达到 84.59%。

说明随着预测时域延长，路侧独有新增 Occupancy 的可靠性总体呈下降趋势。

但是：

\[
\Delta IoU
\]

并没有随 horizon 严格单调下降。

各 horizon 的 \(\Delta IoU\) 为：

```text
t=0 : -0.0231
t=1 : -0.0013
t=2 : -0.0044
t=3 : -0.0141
t=4 : -0.0155
```

因此不能简单得出：

> 预测时间越远，最终 IoU 一定下降得越严重。

更合理的解释是：

> 协同收益不仅取决于错误新增数量，还同时受到正确新增数量、Ego 原始 Occupancy 状态以及新增区域空间分布等因素影响。

---

## 11. Sample-wise Results

### 11.1 WAR 分布

```text
mean   = 0.8295
std    = 0.1193
median = 0.8393
max    = 1.0000
```

说明从样本级来看，大多数样本的路侧新增 Occupancy 中均存在较高比例的错误新增。

---

### 11.2 BAR_add 分布

```text
mean   = 0.1705
std    = 0.1193
median = 0.1607
max    = 0.6223
```

---

### 11.3 ΔIoU 分布

```text
mean   = -0.006933
std    =  0.068029
median = -0.015415
min    = -0.186186
max    =  0.337393
```

其中：

```text
median ΔIoU = -0.015415
```

说明超过一半样本的 Candidate-domain IoU 在加入路侧 Occupancy 后出现下降。

同时，部分样本仍能从路侧信息中获得明显收益：

```text
max ΔIoU = +0.337393
```

这说明路侧 Occupancy 并非整体无效，而是具有明显的样本依赖性和局部价值差异。

这也说明后续更适合研究：

> 哪些路侧新增值得接受，哪些新增应该拒绝或回退 Ego。

而不是简单取消路侧协同。

---

## 12. Negative Cooperation Statistics

按照：

\[
\Delta IoU < 0
\]

严格定义性能意义上的负合作。

统计结果：

```text
Negative cooperation : 347 / 549 = 63.2%
Positive cooperation : 181 / 549 = 33.0%
Neutral              :  21 / 549 =  3.8%
```

因此：

> 在 63.2% 的有效样本中，直接加入路侧 Occupancy 后 Candidate-domain IoU 出现下降。

另外：

```text
HA > BA samples:
521 / 549 = 94.9%
```

即：

> 94.9% 的有效样本中，错误新增数量高于正确新增数量。

需要注意：

`HA > BA` 只能说明错误新增在数量上占主导，不能单独作为性能意义上的 Negative Cooperation 判据。

因此：

- `HA > BA`：描述新增数量结构；
- `ΔIoU < 0`：用于严格判断性能是否发生负合作。

---

## 13. G1 核心结果汇总

```text
Total samples                  = 675
Valid Occupancy samples        = 549
Invalid Occupancy samples      = 126

ADD                            = 799,179
HA                             = 646,667
BA                             = 152,512

HA / BA                        = 4.24

WAR                            = 80.92%
BAR_add                        = 19.08%

Candidate-domain Ego IoU       = 0.2842
Candidate-domain Official IoU  = 0.2710
Candidate-domain ΔIoU          = -0.0132

Negative cooperation samples   = 347 / 549 = 63.2%
Positive cooperation samples   = 181 / 549 = 33.0%
Neutral samples                = 21 / 549  = 3.8%

HA > BA samples                = 521 / 549 = 94.9%
```

---

## 14. G1 主要结论

G1 表明，UniV2X 当前直接 OR Occupancy Fusion 中存在明显的低可靠路侧新增现象。

在 549 个有效 Occupancy 样本的 Candidate-domain 内：

\[
WAR = 80.92\%
\]

说明约 80.9% 的路侧独有新增 Occupancy 与 GT 不一致。

同时：

\[
63.2\%
\]

的有效样本在加入路侧 Occupancy 后出现 Candidate-domain IoU 下降，说明性能意义上的负合作在多数有效样本中存在。

另外：

\[
94.9\%
\]

的有效样本满足：

\[
HA > BA
\]

说明错误新增数量占主导并非由少量异常样本造成，而是在绝大多数样本中普遍存在。

与此同时，仍有：

```text
181 / 549 = 33.0%
```

的样本在加入路侧 Occupancy 后获得正收益。

因此，当前结果并不支持简单删除路侧 Occupancy，而是支持进一步研究：

> 如何识别不同空间位置、不同预测时域和不同样本中的协同价值，并有选择地接受真正有益的路侧 Occupancy 信息。

因此 G1 判定为：

```text
G1: GO
```

---

## 15. 与 UniV2X 官方 Occupancy 指标的区别

需要特别注意：

G1 中得到：

```text
Ego IoU      = 0.2842
Official IoU = 0.2710
Delta IoU    = -0.0132
```

这些 IoU 是在：

\[
M^{GT}\cap M^{warp}
\]

Candidate-domain 内重新计算得到的诊断指标。

因此，它们不能直接与 UniV2X baseline evaluator 输出的：

```text
Occupancy IoU: 22.6 / 25.9
```

进行数值比较。

两者作用不同。

### UniV2X Official Evaluator

用于报告整体 Occupancy benchmark 性能。

### G1 Candidate-domain Analysis

用于分析：

- 路侧独有新增信息的可靠性；
- Harmful Addition；
- Beneficial Addition；
- Wrong Addition Ratio；
- 路侧融合是否造成局部负合作。

因此，G1 的 Candidate-domain IoU 是诊断指标，而不是替代官方 Occupancy benchmark 指标。

---

## 16. 文件说明

### 16.1 G1 分析脚本

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/G1/compute_g1_stats.py
```

作用：

- 读取 `cache_full`;
- 筛选有效 Occupancy 样本；
- 计算 ADD / HA / BA；
- 计算 WAR / BAR_add；
- 计算 Candidate-domain Ego IoU；
- 计算 Candidate-domain Official IoU；
- 计算 ΔIoU；
- 统计样本级负合作比例。

---

### 16.2 G1 最终统计结果

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/G1/g1_stats.npz
```

用于：

- 后续 G1 绘图；
- G2 对比；
- 论文结果整理；
- 实验复现。

---

### 16.3 共用 Occupancy Cache

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full/
```

该缓存由 G0 导出，并由 G1、G2 共用。

不需要在 G1 目录中重复复制。

---

## 17. G1 运行方法

进入 UniV2X 项目根目录：

```bash
cd /root/autodl-tmp/UniV2X
```

运行：

```bash
python G0-G1-G2/G1/compute_g1_stats.py
```

该过程仅进行离线 NumPy 统计：

```text
不重新加载模型
不重新执行神经网络推理
不重新运行 675 个样本的 baseline evaluation
不进行模型训练
```

---

## 18. 下一阶段：G2

G1 已经确认：

```text
大量路侧新增 Occupancy 为错误新增；
性能意义上的 Negative Cooperation 在多数样本中存在；
但部分样本仍能明显受益于路侧协同。
```

因此下一步进入：

```text
G2: Patch Accept / Fallback Oracle Analysis
```

G2 核心问题：

> 如果能够局部决定“接受路侧新增”还是“回退 Ego”，理论上能够获得多大的 Occupancy 性能提升？

计划比较：

```text
Ego
Official OR
Cell Oracle
Patch-4 Oracle
Patch-8 Oracle
Patch-16 Oracle
```

G2 用于回答：

> 当前负合作问题是否存在足够大的可利用性能空间，从而值得进一步设计 STCV-Occ 等可学习的协同价值预测与选择性融合方法。

---

## 19. 当前阶段结论

```text
G0: PASS
G1: GO
G2: Pending
```

当前研究逻辑：

\[
\text{G0：确认研究测量链路可信}
\]

\[
\Downarrow
\]

\[
\text{G1：确认负合作现象真实且普遍存在}
\]

\[
\Downarrow
\]

\[
\text{G2：判断选择性融合的理论收益空间}
\]

\[
\Downarrow
\]

\[
\text{Method：学习空间—时间协同价值并进行可靠 Occupancy 融合}
\]