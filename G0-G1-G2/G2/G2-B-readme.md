# G2-B: Patch Accept/Fallback Oracle Analysis

## 1. 目的

G2-B 用于评估在 UniV2X Occupancy 协同融合中，如果能够根据局部协同价值选择性决定：

```text
Accept infrastructure additions
或
Fallback to Ego
```

理论上能够获得多大的 Occupancy 性能提升空间。

G1 已表明，当前 Official OR Fusion 中存在大量 Harmful Addition（HA），并且多数有效样本出现负合作现象。

G2-A 进一步表明，路侧新增可靠性具有明显空间异质性，但 Ego distance、warp-boundary distance 和 overall overlap ratio 等单一几何指标均不足以充分刻画最终协同价值。

因此 G2-B 进一步回答：

> 如果能够利用 GT 完美或近似完美地判断局部路侧新增是否值得接受，选择性融合能够获得多大的性能收益？

G2-B 为 Oracle 分析，不进行模型训练，也不代表实际部署性能。

---

# 2. 数据与分析范围

输入数据来自 G0 导出的 Occupancy cache：

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full/
```

总样本数：

```text
675
```

满足完整 Occupancy 时序有效性要求的样本数：

```text
549
```

与 G1 使用的有效样本集合保持一致。

G2-B 继续采用与 G1 相同的 Candidate-domain：

\[
M_t^{valid}
=
M_t^{GT}
\land
M^{warp}
\]

其中：

- \(M_t^{GT}\)：GT 有效区域；
- \(M^{warp}\)：路侧 Occupancy warp 后的有效区域。

因此，本实验中报告的 IoU、Precision、Recall 和 F1 均属于 Candidate-domain 诊断指标。

这些指标不能直接与 UniV2X 官方 evaluator 输出的：

```text
Occupancy IoU: 22.6 / 25.9
```

进行数值比较。

---

# 3. 路侧新增定义

路侧新增 Occupancy 定义为：

\[
M_t^{add}
=
(O_v=0)
\land
(O_i=1)
\land
M_t^{valid}
\]

即：

> Ego 原本没有预测为 occupied，但经过路侧 Occupancy 融合后新增为 occupied 的位置。

其中：

\[
HA
=
M^{add}\land(GT=0)
\]

表示 Harmful Addition，即错误新增；

\[
BA
=
M^{add}\land(GT=1)
\]

表示 Beneficial Addition，即正确新增。

满足：

\[
ADD=HA+BA
\]

---

# 4. Oracle Decision Rule

## 4.1 Cell Oracle

Cell Oracle 对每一个路侧新增 cell 单独使用 GT 判断是否接受。

最终结果为：

\[
O_{\mathrm{Cell}}
=
O_v\lor BA
\]

即：

- 如果路侧新增位置对应 \(GT=1\)，则接受；
- 如果路侧新增位置对应 \(GT=0\)，则拒绝；
- Ego 原本已经预测为 occupied 的位置始终保留。

因此 Cell Oracle 仅控制：

```text
是否接受路侧新增
```

不会删除 Ego 原有 Occupancy。

Cell Oracle 可视为当前 addition-only selective fusion 设定下的 cell-level Oracle 上界。

---

## 4.2 Patch Oracle

对于空间 patch \(R\)，统计：

\[
BA_R
=
\sum_{x\in R} BA(x)
\]

以及：

\[
HA_R
=
\sum_{x\in R} HA(x)
\]

定义局部净效用：

\[
U_R
=
BA_R-HA_R
\]

Oracle 决策规则：

```text
if U_R > 0:
    Accept
else:
    Fallback to Ego
```

即：

\[
U_R>0
\Rightarrow
\text{Accept infrastructure additions}
\]

\[
U_R\leq0
\Rightarrow
\text{Fallback to Ego}
\]

该 Oracle 优化的是 patch 内新增区域的二值 cell-error utility，而不是直接声明全局 IoU 最优。

最终仍使用 IoU、Precision、Recall 和 F1 对产生的预测结果进行评价。

---

# 5. 对比策略

G2-B 比较以下 6 种策略：

```text
1. Ego
2. Official OR
3. Cell Oracle
4. Patch-4 Oracle
5. Patch-8 Oracle
6. Patch-16 Oracle
```

其中：

### Ego

\[
O_{\mathrm{Ego}}=O_v
\]

完全不接受 OccFusion 中的路侧新增。

### Official OR

\[
O_{\mathrm{Official}}
=
O_v\lor O_i
\]

即当前 UniV2X Occupancy Fusion 的直接 OR 策略。

### Cell Oracle

逐 cell 判断是否接受路侧新增。

### Patch-4 Oracle

按照 \(4\times4\) patch 统一执行 Accept/Fallback。

### Patch-8 Oracle

按照 \(8\times8\) patch 统一执行 Accept/Fallback。

### Patch-16 Oracle

按照 \(16\times16\) patch 统一执行 Accept/Fallback。

---

# 6. G1 Consistency Verification

G2-B 首先重新统计 Candidate-domain 中的路侧新增信息。

结果：

```text
Total ADD : 799,179
Total HA  : 646,667
Total BA  : 152,512
```

与 G1 完全一致：

```text
G1:
ADD = 799,179
HA  = 646,667
BA  = 152,512
```

因此：

\[
WAR
=
\frac{646667}{799179}
\approx80.92\%
\]

G2-B 与 G1 的统计口径保持一致。

状态：

```text
G1 Consistency Check: PASS
```

---

# 7. Overall Performance

总体结果如下：

| Strategy | IoU | Precision | Recall | F1 | Δ vs Ego | Δ vs Official |
|---|---:|---:|---:|---:|---:|---:|
| Ego | 0.2842 | 0.4856 | 0.4067 | 0.4427 | - | - |
| Official OR | 0.2710 | 0.3554 | 0.5332 | 0.4265 | -0.0132 | - |
| Cell Oracle | 0.3726 | 0.5531 | 0.5332 | 0.5430 | +0.0884 | +0.1016 |
| Patch-4 | 0.3451 | 0.5272 | 0.4998 | 0.5131 | +0.0609 | +0.0741 |
| Patch-8 | 0.3329 | 0.5144 | 0.4854 | 0.4995 | +0.0486 | +0.0618 |
| Patch-16 | 0.3213 | 0.5040 | 0.4700 | 0.4864 | +0.0371 | +0.0503 |

---

# 8. Official OR 与 Ego

Ego IoU：

\[
IoU_{\mathrm{Ego}}=0.2842
\]

Official OR IoU：

\[
IoU_{\mathrm{Official}}=0.2710
\]

因此：

\[
\Delta IoU_{\mathrm{Official-Ego}}
=
0.2710-0.2842
=
-0.0132
\]

说明在 Candidate-domain 内，无条件接受全部路侧新增后，整体 IoU 下降了：

```text
0.0132 absolute IoU
```

与此同时：

```text
Ego Recall     = 0.4067
Official Recall = 0.5332
```

Official OR 明显提高 Recall，但 Precision 从：

```text
0.4856
```

下降至：

```text
0.3554
```

说明无条件接受路侧新增虽然能够恢复部分 Ego 漏检，但同时引入了大量错误 Occupancy，从而导致总体 IoU 和 F1 下降。

---

# 9. Cell Oracle

Cell Oracle 得到：

```text
IoU       = 0.3726
Precision = 0.5531
Recall    = 0.5332
F1        = 0.5430
```

相比 Ego：

\[
0.3726-0.2842
=
+0.0884
\]

相比 Official OR：

\[
0.3726-0.2710
=
+0.1016
\]

因此，Cell Oracle 展示的最大 addition-selection headroom 为：

```text
+0.0884 absolute IoU vs Ego
+0.1016 absolute IoU vs Official OR
```

需要注意：

Cell Oracle 使用 GT，因此上述结果属于 Oracle 上界分析，不代表可部署模型能够直接达到该性能。

---

# 10. Patch Oracle Performance

## 10.1 Patch-4

Patch-4 Oracle：

```text
IoU       = 0.3451
Precision = 0.5272
Recall    = 0.4998
F1        = 0.5131
```

相比 Ego：

\[
0.3451-0.2842
=
+0.0609
\]

相比 Official OR：

\[
0.3451-0.2710
=
+0.0741
\]

---

## 10.2 Patch-8

Patch-8 Oracle：

```text
IoU       = 0.3329
Precision = 0.5144
Recall    = 0.4854
F1        = 0.4995
```

相比 Ego：

\[
0.3329-0.2842
=
+0.0486
\]

相比 Official OR：

\[
0.3329-0.2710
=
+0.0618
\]

---

## 10.3 Patch-16

Patch-16 Oracle：

```text
IoU       = 0.3213
Precision = 0.5040
Recall    = 0.4700
F1        = 0.4864
```

相比 Ego：

\[
0.3213-0.2842
=
+0.0371
\]

相比 Official OR：

\[
0.3213-0.2710
=
+0.0503
\]

---

# 11. Spatial Granularity Analysis

Patch Oracle 性能随空间粒度变粗而逐渐下降：

```text
Cell Oracle : 0.3726
Patch-4     : 0.3451
Patch-8     : 0.3329
Patch-16    : 0.3213
```

对应：

\[
IoU_{\mathrm{Cell}}
>
IoU_{\mathrm{Patch4}}
>
IoU_{\mathrm{Patch8}}
>
IoU_{\mathrm{Patch16}}
\]

在本次实验中，更细粒度的空间决策能够获得更高的 Oracle 性能。

这表明：

> 局部不同区域中的路侧新增价值存在明显差异，更细粒度的 Accept/Fallback 决策能够更有效地区分 Beneficial Addition 与 Harmful Addition。

---

# 12. Patch-4 vs Cell Oracle Improvement

以 Official OR 为基准，Cell Oracle 提供：

\[
0.3726-0.2710
=
0.1016
\]

的 IoU 改善空间。

Patch-4 提供：

\[
0.3451-0.2710
=
0.0741
\]

因此 Patch-4 恢复了：

\[
\frac{0.0741}{0.1016}
\approx72.9\%
\]

的 Cell-Oracle improvement over Official OR。

即：

```text
Patch-4 recovers approximately 72.9%
of the Cell-Oracle gain over Official OR.
```

如果以 Ego 为基准，则：

\[
\frac{0.0609}{0.0884}
\approx68.9\%
\]

因此必须明确比较基准，不能写成模糊的“Patch-4 达到 Cell Oracle 的 xx%”。

---

# 13. Patch-4 Decision Statistics

Patch-4 是三个 Patch Oracle 中性能最高的方案。

包含路侧新增的 patch 总数：

```text
119,448
```

接受 patch：

```text
17,805
```

因此 Acceptance Rate 为：

\[
\frac{17805}{119448}
\approx14.9\%
\]

即：

```text
Acceptance rate = 14.9%
```

其余约：

```text
85.1%
```

的含新增 patch 被 Oracle 选择 Fallback to Ego。

这里的 85.1% 仅表示 Oracle 的拒绝比例，不表示分类准确率。

---

# 14. Patch-4 Accepted Additions

Patch-4 接受区域中：

```text
ADD = 133,206
BA  = 112,239
HA  = 20,967
```

因此：

\[
WAR_{\mathrm{accepted}}
=
\frac{20967}{133206}
=
15.74\%
\]

即：

```text
Accepted WAR = 15.74%
```

相比所有路侧新增的总体：

```text
Overall WAR = 80.92%
```

Patch-4 Oracle 接受区域中的新增可靠性显著更高。

---

# 15. Patch-4 Rejected Additions

被 Patch-4 Oracle 拒绝的区域中：

```text
ADD = 665,973
BA  = 40,273
HA  = 625,700
```

对应：

\[
WAR_{\mathrm{rejected}}
=
\frac{625700}{665973}
=
93.95\%
\]

即：

```text
Rejected WAR = 93.95%
```

这说明按照 GT-defined utility rule，被拒绝区域主要由 Harmful Addition 构成。

---

# 16. Patch-4 BA Retention

全部 Beneficial Addition：

```text
152,512
```

Patch-4 接受：

```text
112,239
```

因此 BA retention rate 为：

\[
\frac{112239}{152512}
\approx73.6\%
\]

即：

```text
Patch-4 retains approximately 73.6% of all BA.
```

---

# 17. Patch-4 HA Admission

全部 Harmful Addition：

```text
646,667
```

Patch-4 接受：

```text
20,967
```

因此 HA admission rate 为：

\[
\frac{20967}{646667}
\approx3.24\%
\]

即：

```text
Patch-4 admits approximately 3.24% of all HA.
```

因此，Patch-4 Oracle 的核心特征可以概括为：

```text
保留约 73.6% 的 Beneficial Addition
同时仅接受约 3.24% 的 Harmful Addition
```

该结果表明局部路侧新增中存在明显的可分结构。

---

# 18. Patch-8 Decision Statistics

Patch-8：

```text
Accepted patches : 7,777 / 58,829
Acceptance rate  : 13.2%
```

Accepted additions：

```text
ADD = 127,761
BA  = 94,866
HA  = 32,895
WAR = 25.75%
```

Rejected additions：

```text
ADD = 671,418
BA  = 57,646
HA  = 613,772
WAR = 91.41%
```

BA retention：

\[
\frac{94866}{152512}
\approx62.2\%
\]

HA admission：

\[
\frac{32895}{646667}
\approx5.09\%
\]

即：

```text
BA retention ≈ 62.2%
HA admission ≈ 5.09%
```

---

# 19. Patch-16 Decision Statistics

Patch-16：

```text
Accepted patches : 3,674 / 33,490
Acceptance rate  : 11.0%
```

Accepted additions：

```text
ADD = 114,503
BA  = 76,291
HA  = 38,212
WAR = 33.37%
```

Rejected additions：

```text
ADD = 684,676
BA  = 76,221
HA  = 608,455
WAR = 88.87%
```

BA retention：

\[
\frac{76291}{152512}
\approx50.0\%
\]

HA admission：

\[
\frac{38212}{646667}
\approx5.91\%
\]

即：

```text
BA retention ≈ 50.0%
HA admission ≈ 5.91%
```

---

# 20. Patch Granularity Comparison

| Strategy | Accepted WAR | BA Retention | HA Admission |
|---|---:|---:|---:|
| Patch-4 | 15.74% | 73.6% | 3.24% |
| Patch-8 | 25.75% | 62.2% | 5.09% |
| Patch-16 | 33.37% | 50.0% | 5.91% |

结果显示，在本次 Oracle 实验中：

- Patch-4 保留的 BA 最多；
- Patch-4 接受的 HA 最少；
- Patch-4 Accepted WAR 最低；
- Patch-4 最终 IoU 最高。

因此，更细粒度的空间决策在当前实验中能够更有效地区分局部协同价值。

---

# 21. Horizon-wise Results

不同预测时域结果如下：

| Horizon | Ego | Official | Cell | Patch-4 | Patch-8 | Patch-16 |
|---|---:|---:|---:|---:|---:|---:|
| t=0 | 0.3756 | 0.3525 | 0.4615 | 0.4334 | 0.4219 | 0.4138 |
| t=1 | 0.2932 | 0.2919 | 0.3905 | 0.3615 | 0.3482 | 0.3366 |
| t=2 | 0.2536 | 0.2492 | 0.3513 | 0.3215 | 0.3075 | 0.2924 |
| t=3 | 0.2419 | 0.2278 | 0.3318 | 0.3031 | 0.2902 | 0.2762 |
| t=4 | 0.2387 | 0.2232 | 0.3075 | 0.2860 | 0.2770 | 0.2675 |

可以看到：

```text
Cell Oracle
Patch-4
Patch-8
Patch-16
```

在五个预测 horizon 上均高于 Official OR。

说明局部选择性接受路侧新增所带来的潜在收益并非只存在于某一个特定预测时域。

---

# 22. Key Findings

## Finding 1: Official OR 存在负合作

Official OR：

```text
IoU = 0.2710
```

低于 Ego：

```text
IoU = 0.2842
```

即：

\[
\Delta IoU=-0.0132
\]

说明无条件接受全部路侧新增会降低 Candidate-domain Occupancy 性能。

---

## Finding 2: Cell-level selective fusion 存在较大 Oracle headroom

Cell Oracle：

```text
IoU = 0.3726
```

相比：

```text
Ego      : +0.0884 absolute IoU
Official : +0.1016 absolute IoU
```

说明大量路侧新增虽然总体可靠性较低，但其中仍包含具有显著价值的 Beneficial Addition。

关键问题不是：

```text
是否应该使用路侧信息
```

而是：

```text
应该接受哪些路侧信息
```

---

## Finding 3: Patch-level selective fusion 仍具有显著结构化收益空间

即使使用统一 patch 决策：

```text
Patch-4  : IoU = 0.3451
Patch-8  : IoU = 0.3329
Patch-16 : IoU = 0.3213
```

仍全部超过：

```text
Ego      : 0.2842
Official : 0.2710
```

其中 Patch-4 相比 Official 提升：

```text
+0.0741 absolute IoU
```

并恢复 Cell-Oracle over Official 增益的约：

```text
72.9%
```

---

## Finding 4: Finer spatial granularity is more effective

在当前 Oracle 条件下：

```text
Cell > Patch-4 > Patch-8 > Patch-16
```

随着 patch 尺度增大，选择能力下降。

这说明 Beneficial Addition 与 Harmful Addition 在局部空间内具有细粒度混合特征。

---

## Finding 5: Patch-4 can retain most BA while suppressing most HA

Patch-4：

```text
BA retention = 73.6%
HA admission = 3.24%
Accepted WAR = 15.74%
```

相比全部 ADD：

```text
WAR = 80.92%
```

表明基于局部协同价值进行选择具有明显的理论潜力。

---

# 23. Interpretation

G2-B 并不能说明：

```text
当前已经存在一个真实模型能够达到 Oracle 性能
```

因为所有 Oracle 决策均使用了 GT。

G2-B 真正证明的是：

> Beneficial Addition 和 Harmful Addition 在局部区域中具有足够明显的结构化可分空间，因此值得进一步研究如何在推理阶段根据可观测信息预测这种局部协同价值。

因此，Oracle 是：

```text
diagnostic upper bound / structured opportunity analysis
```

而不是：

```text
deployable fusion method
```

---

# 24. G2-B Conclusion

G2-B 结果表明，UniV2X 当前 Official OR Fusion 的主要问题并非路侧 Occupancy 完全无效，而是无条件接受所有路侧新增。

在 Candidate-domain 内：

```text
Ego IoU      = 0.2842
Official IoU = 0.2710
```

无条件融合使 IoU 下降：

```text
-0.0132
```

而在使用 GT Oracle 选择路侧新增后：

```text
Cell Oracle IoU = 0.3726
```

相较 Official OR 提升：

```text
+0.1016 absolute IoU
```

即使将决策限制在 patch 级：

```text
Patch-4 IoU = 0.3451
```

仍相较 Official OR 提升：

```text
+0.0741 absolute IoU
```

并恢复约：

```text
72.9%
```

的 Cell-Oracle improvement over Official OR。

Patch-4 进一步表现出：

```text
73.6% BA retention
3.24% HA admission
15.74% accepted WAR
```

说明局部选择性融合具有明显的结构化收益空间。

因此：

```text
G2-B Status: GO
```

---

# 25. Research Implication

结合 G1、G2-A 和 G2-B：

```text
G1:
负合作现象真实存在，大量路侧新增为 Harmful Addition。

G2-A:
路侧新增可靠性具有空间异质性，但简单单一几何指标不足以刻画最终协同价值。

G2-B:
局部 Accept/Fallback Oracle 能够显著提高 Occupancy 性能，
说明 Beneficial / Harmful Addition 具有可利用的局部结构。
```

因此后续工作的核心问题转变为：

> 能否在不使用 GT 的推理阶段，根据车端与路侧 Occupancy 概率、局部空间结构和预测时域演化，预测某个路侧新增区域的实际协同价值？

该问题构成后续 STCV-Occ 方法设计的直接动机。

---

# 26. 文件说明

G2-B 分析脚本：

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/G2/compute_g2_oracle.py
```

统计输出：

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/G2/g2_oracle_stats.npz
```

实验记录：

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/G2/G2-B-readme.md
```

共用 cache：

```text
/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full/
```

---

# 27. 运行方法

进入项目根目录：

```bash
cd /root/autodl-tmp/UniV2X
```

运行：

```bash
python G0-G1-G2/G2/compute_g2_oracle.py
```

该分析仅使用 G0 已导出的 cache 进行离线统计：

```text
不重新训练模型
不重新加载 checkpoint 推理
不重新执行 675 个样本的 baseline evaluation
```

---

# 28. 当前阶段状态

```text
G0   : PASS
G1   : GO
G2-A : PASS
G2-B : GO
```

下一阶段：

```text
STCV-Occ Method Design
```

目标：

> 在不访问 GT 的情况下预测局部 Cooperation Value，并利用预测值完成可靠的 Accept/Fallback Occupancy Fusion。