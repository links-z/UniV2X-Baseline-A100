G4：Learned Selective Fusion 离线闭环评估记录
项目：UniV2X Occupancy Reliability / STCV-Occ
阶段：G4 — Learned Fusion Evaluation
状态：GO
用途：固定 G4 的实验设计、关键结果、机制分析、论文写作口径与后续集成依据。

1. G4 的研究目的
G1–G3 已分别回答：
\[
\text{G1：Negative Cooperation 是否存在？}
\]
\[
\text{G2：Patch-level selective fusion 是否具有 Oracle headroom？}
\]
\[
\text{G3：不用 GT，Cooperation Utility 是否可学习？}
\]
G4 进一步回答：
\[
\boxed{
\text{学习得到的 Cooperation Utility 真正用于融合后，能否改善 Occupancy？}
}
\]
因此，G4 不再只评价 AUROC / AUPRC，而是直接评价 Learned Selective Fusion 对下游 Occupancy IoU 的影响。
完整闭环：
Negative Cooperation
        ↓
Patch-level Oracle Headroom
        ↓
Cooperation Utility Learnability
        ↓
Learned Selective Fusion
        ↓
Downstream Occupancy Improvement
2. G4 使用的 Predictor
G4 使用 G3 最终最佳 Cooperation Utility Predictor。
输入：
\[
127
=
80_{\text{Local}}
+
5_{\text{Horizon}}
+
2_{\text{Spatial}}
+
40_{\text{Neighborhood}}
\]
其中：
- Local：Soft Occupancy 特征；
- Horizon：5 维 prediction horizon one-hot；
- Spatial：归一化 Patch 坐标 \((u,v)\)；
- Neighborhood：8-neighbor 局部统计。
模型：
\[
127 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 1
\]
参数量：
\[
\boxed{73,985}
\]
G3 Test：
\[
AUPRC=0.2712,\qquad AUROC=0.7035
\]
3. Learned Selective Fusion 定义
对候选 Patch \(R\)，Predictor 输出：
\[
s_R=\sigma(f_\theta(X_R))
\]
根据部署阈值 \(\tau\)：
\[
a_R=\mathbf{1}[s_R>\tau]
\]
其中：
- \(a_R=1\)：Accept Infrastructure-added Occupancy；
- \(a_R=0\)：Fallback to Ego。
最终：
\[
\boxed{
O_R^{learned}
=
O_{v,R}
\lor
(a_R\land O_{i,R})
}
\]
该方法只控制路侧新增 Occupancy 是否被接受，不主动删除 Ego Occupancy，因此属于：
\[
\boxed{\text{Accept/Fallback Selective Fusion}}
\]
4. G4 实验组成
G4-0  Offline Learned Fusion Reconstruction
G4-1  Validation Threshold Selection
G4-2  Held-out Test Evaluation
G4-3  Mechanism / Necessity Analysis
5. G4-0：离线重建与 sample_idx 映射修复
G4 采用 G0 cache + G3 Predictor 进行离线 Learned Fusion 重建，不修改 UniV2X 主代码。
评估脚本初版存在 sample_idx 与 G0 cache export_idx 映射错误。例如：
sample_idx = '002889'
不能直接用于加载：
sample_002889.npz
必须先通过：
SAMPLE_IDX_TO_EXPORT_IDX
映射得到对应 export_idx，例如：
export_idx = 74
再加载：
sample_00074.npz
修复后，所有样本能够正确对应到原 G0 cache，并完成 Ego / Official / Learned / Oracle 四种策略评估。
6. G4-1：Validation 阈值选择
仅在 Validation scenes 上扫描：
\[
\tau \in \{0.05,0.10,\ldots,0.95\}
\]
共 19 个阈值。
选择标准：
\[
\boxed{
\tau^*
=
\arg\max_{\tau} IoU_{\text{Val}}
}
\]
最终：
\[
\boxed{\tau^*=0.70}
\]
之后将该阈值冻结，用于 held-out Test scenes，不再在 Test 上调参。
论文表述：
The deployment threshold was selected exclusively on the validation scenes according to downstream occupancy IoU and was kept fixed for the held-out test evaluation.

7. Validation Set 结果
Validation Set：
\[
N=94
\]
Strategy	IoU
Ego	0.3018
Official OR	0.3029
Learned	0.3235
Patch-level Oracle	0.3650


Learned 相比 Official：
\[
0.3235-0.3029
=
\boxed{+0.0206}
\]
相对提升约：
\[
\boxed{6.8\%}
\]
8. G4-2：Held-out Test 结果
Test Set：
\[
N=86
\]
固定：
\[
\boxed{\tau^*=0.70}
\]
最终结果：
Strategy	IoU	Relative vs Official	Relative vs Ego
Official OR	0.2526	—	-5.4%
Ego	0.2671	+5.7%	—
Learned	0.2732	+8.2%	+2.3%
Patch-level Oracle	0.3077	+21.8%	+15.2%


核心排序：
\[
\boxed{
IoU_{\text{Learned}}
>
IoU_{\text{Ego}}
>
IoU_{\text{Official}}
}
\]
即：
\[
\boxed{
0.2732
>
0.2671
>
0.2526
}
\]
9. Learned vs Official
绝对增益：
\[
\Delta IoU_{\text{Learned-Official}}
=
0.2732-0.2526
=
\boxed{+0.0206}
\]
相对 Official OR：
\[
\boxed{+8.2\%}
\]
说明 Learned Selective Fusion 能够修复 Official OR 的负协同问题。
10. Learned vs Ego
绝对增益：
\[
\Delta IoU_{\text{Learned-Ego}}
=
0.2732-0.2671
\approx
\boxed{+0.0062}
\]
相对 Ego：
\[
\boxed{+2.3\%}
\]
这一结果说明 Learned 并非简单拒绝路侧信息，而是在过滤部分有害 Infrastructure additions 的同时，仍然保留了能带来净收益的有益协同信息。
11. Patch-level Oracle Headroom Recovery
Test：
\[
IoU_{\text{Official}}=0.2526
\]
\[
IoU_{\text{Learned}}=0.2732
\]
\[
IoU_{\text{Oracle}}=0.3077
\]
Patch-level Oracle 相对 Official 的 headroom：
\[
0.3077-0.2526
=
0.0551
\]
Learned 实际恢复：
\[
0.2732-0.2526
=
0.0206
\]
因此：
\[
\boxed{
\text{Patch-level Oracle Headroom Recovery}
=
\frac{0.0206}{0.0551}
\approx37.4\%
}
\]
注意必须写：
Patch-level Oracle headroom recovery

不能写成：
“恢复了 37.4% 的理论最优性能”

因为 Oracle 只是 Patch-4 Accept/Fallback Oracle，不是全局最优 Occupancy Oracle。
Learned 与 Oracle 仍有：
\[
0.3077-0.2732
=
\boxed{0.0345}
\]
绝对 IoU gap。
12. G4-3：机制分析
Test、\(\tau=0.70\)：
\[
\boxed{\text{Accept Rate}=18.94\%}
\]
\[
\boxed{\text{BA Retention}=37.75\%}
\]
\[
\boxed{\text{HA Admission}=15.84\%}
\]
其中：
- BA Retention：有益 Patch 被接受的比例；
- HA Admission：有害 Patch 被错误接受的比例。
整体表现为相对保守的协同策略。
13. Horizon-wise 机制
Test 中：
Accept Rate:
t=0 : 22.65%
...
t=4 : 12.78%
BA Retention 大致保持在：
32% – 44%
HA Admission 由约：
18.87% (t=0)
下降至：
10.01% (t=4)
因此可以描述为：
Predictor 在当前 Test split 上随 prediction horizon 增长表现出更保守的接受行为。

不能写成：
“越远 horizon 一定越不可靠”

因为此前 horizon heterogeneity 在不同 split 上并非严格单调。
14. 与简单 Confidence Baseline 对比
在相似 Accept Rate 范围下：
Metric	Learned (\(\tau=0.70\))	Confidence (\(\delta=0.30\))	Difference
Accept Rate	18.94%	24.90%	-5.97 pp
BA Retention	37.75%	38.18%	-0.43 pp
HA Admission	15.84%	22.72%	-6.88 pp
Precision	28.14%	21.64%	+6.50 pp


注：Confidence heuristic 的精确定义以 scripts/fair_comparison.py 的最终实现为准。

BA Retention 几乎相同：
\[
37.75\% \quad vs \quad 38.18\%
\]
仅相差：
\[
0.43\text{个百分点}
\]
但 HA Admission：
\[
15.84\%
\quad vs \quad
22.72\%
\]
绝对降低：
\[
\boxed{6.88\text{个百分点}}
\]
相对减少约：
\[
\boxed{30\%}
\]
Precision：
\[
28.14\%
\quad vs \quad
21.64\%
\]
提升：
\[
\boxed{6.50\text{个百分点}}
\]
因此：
\[
\boxed{
\text{Learned Predictor 并非简单减少接受数量，
而是在近似 BA Retention 下更有效地抑制 HA Admission}
}
\]
这构成 Value Learning 相对于简单 confidence filtering 的必要性证据。
15. G2-B 与 G4 的数值不能直接混用
G2-B 使用全部 549 个有效 Occupancy samples 进行 Oracle diagnosis：
\[
IoU_{Ego}=0.2842
\]
\[
IoU_{Official}=0.2710
\]
\[
IoU_{P4Oracle}=0.3451
\]
G4 使用 Scene-level split 后的 held-out scenes，其中 Test 为 86 个样本，因此：
\[
0.2671,\ 0.2526,\ 0.2732,\ 0.3077
\]
不能与 G2-B 的绝对数值直接比较。
正确解释：
G2-B reports Oracle headroom over the complete valid occupancy set, whereas G4 evaluates learned selective fusion on held-out scenes under the scene-level split.

16. G4 Go/No-Go 判定
条件 1
\[
IoU_{\text{Learned}}>IoU_{\text{Official}}
\]
实际：
\[
0.2732>0.2526
\]
PASS。
条件 2
\[
IoU_{\text{Learned}}>IoU_{\text{Ego}}
\]
实际：
\[
0.2732>0.2671
\]
PASS。
条件 3
恢复非零 Patch-level Oracle headroom：
\[
37.4\%
\]
PASS。
条件 4
不能被简单 confidence heuristic 等价替代：
- BA Retention 近似一致；
- HA Admission 降低 6.88 pp；
- 相对减少约 30%；
- Precision 提高 6.50 pp。
PASS。
因此：
\[
\boxed{\text{G4: GO}}
\]
17. G0–G4 完整证据链
G0
Measurement / Protocol Calibration
        ↓
G1
Negative Cooperation Exists
        ↓
G2
Selective Fusion Has Patch-level Oracle Headroom
        ↓
G3
Cooperation Utility Is Learnable
        ↓
G4
Learned Utility Improves Actual Fusion
核心逻辑：
\[
\boxed{
\text{问题发现}
\rightarrow
\text{性能空间验证}
\rightarrow
\text{协同价值学习}
\rightarrow
\text{学习式选择融合}
}
\]
18. G4 的科学意义
G4 最重要的结论不是：
“Predictor AUPRC 较高。”

而是：
\[
\boxed{
\text{学习得到的局部 Cooperation Utility 能够实际转化为下游 Occupancy Fusion 性能增益}
}
\]
具体表现为：
1. Learned Fusion 修复 Official OR 的负协同；
2. Learned Fusion 超过 Ego-only baseline；
3. 恢复 37.4% 的 Patch-level Oracle headroom；
4. 在近似 BA Retention 下明显降低 HA Admission；
5. Validation 与 held-out Test 呈现一致的总体增益趋势。
19. 论文中可直接使用：实验目的
前述实验已经证明，车路协同 Occupancy 融合中存在负协同现象，且 Patch-level selective fusion 具有可利用的 Oracle 性能空间。同时，局部 Cooperation Utility 可以通过推理阶段可获得的 soft occupancy confidence、预测时域、空间位置及邻域上下文进行预测。然而，较高的 Utility Prediction 性能并不必然意味着最终 Occupancy Fusion 性能能够改善。因此，本节进一步将学习得到的 Cooperation Utility Predictor 应用于实际的 Accept/Fallback 融合决策，以验证局部协同价值能否真正转化为下游 Occupancy 性能增益。

20. 论文中可直接使用：方法描述
对于局部候选区域 \(R\)，首先利用 Cooperation Utility Predictor 输出局部协同价值分数 \(s_R=\sigma(f_\theta(X_R))\)。随后，根据验证集确定的部署阈值 \(\tau\)，构造 Patch-level 接受决策 \(a_R=\mathbf{1}[s_R>\tau]\)。当 \(a_R=1\) 时接受对应区域的 Infrastructure-added Occupancy，否则退回 Ego prediction。最终 Learned Selective Fusion 表示为
\[
O_R^{learned}
=
O_{v,R}
\lor
(a_R\land O_{i,R}).
\]
该机制仅控制路侧新增 Occupancy 是否进入最终预测，而不主动删除 Ego Occupancy，从而在保留单车基础感知能力的前提下抑制潜在有害协同。

21. 论文中可直接使用：阈值选择
为避免 Test 信息泄漏，本文仅在 Validation scenes 上扫描部署阈值，并以下游 Occupancy IoU 最大化为选择标准。最终得到 \(\tau^*=0.70\)。阈值确定后，在 held-out Test scenes 上保持固定，不再进行测试集调参。

22. 论文中可直接使用：主结果分析
在 held-out Test scenes 上，Official OR、Ego-only、Learned Selective Fusion 和 Patch-level Oracle 的 Occupancy IoU 分别为 0.2526、0.2671、0.2732 和 0.3077。Official OR 的 IoU 低于 Ego-only baseline，说明无条件接受路侧新增 Occupancy 会产生负协同。采用 Learned Selective Fusion 后，IoU 提升至 0.2732，相比 Official OR 获得 0.0206 的绝对 IoU 增益，对应约 8.2% 的相对提升；同时相比 Ego-only baseline 仍获得约 0.0062 的绝对增益。
值得注意的是，Learned Fusion 不仅恢复了 Official OR 所产生的性能下降，而且进一步超过 Ego-only baseline。这说明所学习的 Cooperation Utility Predictor 并非简单倾向于拒绝路侧信息，而是能够在抑制部分有害 Infrastructure additions 的同时，保留具有补充价值的协同 Occupancy 信息。

23. 论文中可直接使用：Oracle 分析
Patch-level Oracle 在相同 Test scenes 上取得 0.3077 IoU，相比 Official OR 提供 0.0551 的可利用 headroom。Learned Selective Fusion 相比 Official OR 获得 0.0206 的实际 IoU 增益，对应约 37.4% 的 Patch-level Oracle headroom recovery。该结果表明，当前轻量 Cooperation Utility Predictor 已能够恢复部分理想 Patch-level Accept/Fallback 决策的性能潜力，但 Learned 与 Oracle 之间仍存在 0.0345 的绝对 IoU gap，说明局部协同价值估计仍具有进一步优化空间。

24. 论文中可直接使用：机制分析
为进一步验证性能增益并非由简单置信度过滤造成，本文在相近接受比例下比较 Learned Predictor 与 confidence heuristic。两种方法的 BA Retention 分别为 37.75% 和 38.18%，差异仅为 0.43 个百分点；然而 Learned Predictor 的 HA Admission 从 22.72% 降低至 15.84%，绝对降低 6.88 个百分点，相对减少约 30%。同时，其接受 Precision 由 21.64% 提升至 28.14%。
上述结果表明，Learned Predictor 的优势并非简单减少路侧信息接受数量，而是在维持近似有益信息保留能力的同时，更有效地区分有益与有害的 Infrastructure additions。这进一步支持了基于局部 Cooperation Utility 建模而非固定 confidence threshold 进行选择性融合的必要性。

25. 论文中可直接使用：Val/Test 泛化描述
在 Validation scenes 上，Official OR、Ego、Learned Fusion 和 Patch-level Oracle 的 IoU 分别为 0.3029、0.3018、0.3235 和 0.3650。基于 Validation IoU 选择 \(\tau^*=0.70\) 后，将该阈值冻结并直接应用于 held-out Test scenes，Learned Fusion 仍保持高于 Ego 和 Official OR 的性能排序。Validation 与 Test 上一致的总体趋势表明，所学习的局部协同价值能够在未参与阈值优化的场景中继续转化为 Occupancy fusion 增益。

26. 论文摘要/结论推荐表述
针对车路协同 Occupancy 融合中路侧新增信息可能引发负协同的问题，本文提出基于局部协同价值估计的选择性 Occupancy 融合方法。该方法利用车端与路侧 soft occupancy confidence，并联合预测时域、空间位置和邻域上下文，估计 Patch-level Cooperation Utility，据此对 Infrastructure-added Occupancy 实施 Accept/Fallback 决策。在 held-out Test scenes 上，Learned Selective Fusion 的 IoU 达到 0.2732，高于 Official OR 的 0.2526 和 Ego-only baseline 的 0.2671，相比 Official OR 获得 0.0206 的绝对 IoU 增益，并恢复约 37.4% 的 Patch-level Oracle headroom。进一步的机制分析表明，在保持近似 BA retention 的条件下，Learned Predictor 相比简单 confidence heuristic 将 HA admission 降低约 30%，说明显式建模局部协同价值能够更有效地区分有益与有害的路侧新增信息。

27. 论文中应避免的表述
当前不要写：
“统计显著提升”
原因：尚未进行多随机种子重复、bootstrap CI 或统计显著性检验。
当前不要写：
“恢复了 37.4% 的理论最优性能”
应写：
“恢复了 37.4% 的 Patch-level Oracle headroom”
当前不要写：
“远期预测一定更不可靠”
应写：
“Predictor 在当前 Test split 上随 horizon 增长表现出更保守的接受行为。”
当前不要将：
G4 offline / candidate-domain IoU
直接等同于：
UniV2X 官方完整 Occupancy benchmark IoU
28. 当前 G4 最终状态
G4-0 Offline Reconstruction        PASS
G4-1 Validation Threshold Search  PASS
G4-2 Held-out Test Evaluation     PASS
G4-3 Mechanism Analysis           PASS
G4-3 Heuristic Comparison         PASS

Selected threshold:
tau* = 0.70

Validation:
Ego      = 0.3018
Official = 0.3029
Learned  = 0.3235
Oracle   = 0.3650

Test:
Ego      = 0.2671
Official = 0.2526
Learned  = 0.2732
Oracle   = 0.3077

Learned vs Official:
+0.0206 absolute IoU
+8.2% relative

Learned vs Ego:
+0.0062 absolute IoU
+2.3% relative

Patch-level Oracle headroom recovery:
37.4%

Test mechanism:
Accept Rate  = 18.94%
BA Retention = 37.75%
HA Admission = 15.84%

Confidence comparison:
BA Retention difference = -0.43 pp
HA Admission reduction  = 6.88 pp
Relative HA reduction   ≈ 30%
Precision gain           = 6.50 pp

VERDICT:
G4 = GO
29. 第二项工作当前完整状态
G0  PASS
    Measurement / Protocol Calibration

G1  GO
    Negative Cooperation Exists

G2  GO
    Selective Fusion Has Patch-level Oracle Headroom

G3  GO
    Cooperation Utility Is Learnable

G4  GO
    Learned Utility Improves Actual Occupancy Fusion
完整证据链：
\[
\boxed{
\text{Negative Cooperation Diagnosis}
\rightarrow
\text{Patch-level Oracle Analysis}
\rightarrow
\text{Cooperation Utility Learning}
\rightarrow
\text{Utility-guided Selective Fusion}
}
\]
30. 下一阶段
G4 完成后，进入正式系统集成：
G5-0  STCV-Occ Online Integration
G5-1  Offline / Online Consistency Audit
G5-2  Frozen-backbone Training / Fine-tuning
G5-3  Optional Joint Fine-tuning
G5-4  Official UniV2X Evaluation
第一步不是从头重训 UniV2X，而是：
\[
\boxed{
\text{将当前已训练的 Cooperation Utility Predictor 接入 UniV2X OccFusion，
并验证在线 forward 是否能够复现 G4 offline Learned Fusion。}
}
\]
只有在线/离线一致性通过后，才进入正式训练或微调。
31. G4 最终结论
\[
\boxed{
\text{Patch-level Cooperation Utility 不仅可以被学习，
而且能够实际指导选择性融合并改善下游 Occupancy。}
}
\]
第二项工作已经从：
“可能存在负协同”
推进到：
“负协同可以被诊断”
→
“选择性融合存在可利用空间”
→
“协同价值可以被学习”
→
“学习到的协同价值可以实际改善融合”
最终：
\[
\boxed{\text{G4 COMPLETE — GO}}
\]