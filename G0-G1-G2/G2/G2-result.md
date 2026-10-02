G2 空间诊断与选择性融合潜力分析结论
为进一步分析路侧新增 Occupancy 的空间特征及选择性融合的潜在收益，本文在 G1 所定义的有效协同区域内开展空间诊断与 Oracle 实验。G2-A 结果表明，路侧新增信息的可靠性具有明显的空间异质性。错误新增比例 WAR 随 Ego 中心距离呈总体下降但局部波动的变化趋势，其中 0–5 m 区间 WAR 达到 96.71%，而 70–75 m 区间下降至 69.59%。因此，实验结果并不支持“距离越远，路侧新增越不可靠”的简单假设。与此同时，WAR 与 warp 有效区域边界距离之间表现出明显的非单调关系，表明仅依据 warp 边界距离同样难以判断路侧新增信息的可靠性。
进一步从样本级空间重叠关系分析发现，warp overlap ratio 与 WAR 的 Spearman 相关系数为
\[
\rho=-0.2598,\qquad p=1.2191\times10^{-9},
\]
说明空间重叠程度与错误新增比例之间存在显著但有限的负相关关系；然而，overlap ratio 与协同前后 IoU 变化 \(\Delta IoU\) 的相关系数仅为
\[
\rho=-0.0408,\qquad p=0.3414,
\]
未表现出显著的单调相关关系。上述结果说明，虽然几何空间条件会影响路侧新增信息的局部可靠性，但 Ego 距离、warp 边界距离以及整体空间重叠率等单一几何指标均不足以有效刻画最终协同价值。
在此基础上，G2-B 进一步利用 GT 构建 Cell Oracle 与不同尺度的 Patch Accept/Fallback Oracle，以评估选择性融合的理论收益空间。实验结果表明，Ego 与 Official OR 的 Candidate-domain IoU 分别为 0.2842 和 0.2710，说明无条件接受全部路侧新增会使整体 IoU 下降 0.0132。相比之下，Cell Oracle 的 IoU 达到 0.3726，相较 Ego 和 Official OR 分别获得 0.0884 和 0.1016 的绝对 IoU 提升，表明如果能够准确区分有益新增和有害新增，当前 Occupancy 协同仍存在较大的性能恢复空间。
区域级 Oracle 同样表现出明显收益。Patch-4、Patch-8 和 Patch-16 的 IoU 分别达到 0.3451、0.3329 和 0.3213，均高于 Ego 和 Official OR。其中，Patch-4 相较 Official OR 提升 0.0741 IoU，并恢复了约 72.9% 的 Cell Oracle 相对 Official OR 的潜在增益。随着 patch 尺度增大，Oracle 性能逐渐下降，说明路侧协同价值具有较明显的局部空间差异，更细粒度的区域决策能够更有效地区分有益与有害新增。
进一步分析 Patch-4 的 Accept/Fallback 决策发现，其接受区域中的 WAR 仅为 15.74%，而拒绝区域中的 WAR 达到 93.95%。Patch-4 共保留约 73.6% 的 Beneficial Addition，同时仅接受约 3.24% 的 Harmful Addition。这表明，路侧新增 Occupancy 中的有益信息与有害信息具有较强的局部可分结构，选择性融合可以在保留大量有效路侧信息的同时显著减少错误新增。
从 Precision–Recall 关系来看，Official OR 将 Recall 从 Ego 的 0.4067 提升至 0.5332，但 Precision 由 0.4856 降至 0.3554，说明直接 OR 融合虽然能够补充部分 Ego 漏检，却同时引入大量错误占用。Cell Oracle 在保持 Recall 为 0.5332 的同时，将 Precision 提升至 0.5531，F1 由 Official OR 的 0.4265 提升至 0.5430。由此可见，选择性融合的核心并非简单减少路侧信息，而是在尽可能保留路侧带来的 Recall 增益的同时抑制 Harmful Addition，从而改善 Precision、IoU 和 F1。
综合 G2-A 与 G2-B 的实验结果可以认为，路侧 Occupancy 的协同价值具有显著的空间异质性，且难以通过单一几何启发式规则进行可靠判断；与此同时，局部选择性 Accept/Fallback 融合具有明显的理论性能空间。 因此，后续有必要进一步利用车端与路侧 Occupancy 概率、局部空间结构以及不同预测时域的信息，学习推理阶段可获得的协同价值表示，从而在不依赖 GT 的情况下实现可靠的选择性 Occupancy 融合。
最后建议接你的方法章节：
基于上述分析，本文进一步设计 STCV-Occ，通过对不同空间区域及预测时域的协同价值进行建模，实现对路侧新增 Occupancy 的自适应选择与融合。