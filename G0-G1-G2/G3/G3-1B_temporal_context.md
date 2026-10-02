# G3-1B：Horizon Context 增量可学习性验证

## 1. 实验目的

G3-1A 已验证仅使用局部 Occupancy 信息时，Patch-level Cooperation Utility 存在基础可学习信号。

其中：

- Local Linear 在测试集上取得 AUROC = 0.6067、AUPRC = 0.1941；
- Local MLP 在测试集上取得 AUROC = 0.6066、AUPRC = 0.1873；
- Test 集正类比例为 14.12%，因此随机分类器的 AUPRC 参考水平约为 0.1412。

上述结果表明，局部 Occupancy evidence 中存在一定的 Cooperation Utility 判别信息，但单纯增加非线性模型容量并未明显改善跨场景泛化性能。

G3-1B 在此基础上进一步研究：

\[
\boxed{
\text{预测 Horizon 信息是否能够提供 Local Occupancy Evidence 之外的增量信息？}
}
\]

需要说明的是，本实验加入的是 Horizon Context，即显式标识当前预测时域 \(t=0,\ldots,4\)，并不等价于跨时域序列建模。

---

## 2. Horizon-dependent Heterogeneity 诊断

在进行模型训练前，首先统计不同预测 Horizon 下正协同效用 Patch 的比例。

### 2.1 Train

| Horizon | Patch 数 | Positive 数 | Positive Rate |
|---|---:|---:|---:|
| \(h=0\) | 18,022 | 3,042 | 16.88% |
| \(h=1\) | 18,446 | 3,102 | 16.82% |
| \(h=2\) | 18,448 | 2,638 | 14.30% |
| \(h=3\) | 17,008 | 2,160 | 12.70% |
| \(h=4\) | 14,219 | 1,778 | 12.50% |

### 2.2 Validation

| Horizon | Patch 数 | Positive 数 | Positive Rate |
|---|---:|---:|---:|
| \(h=0\) | 3,747 | 535 | 14.28% |
| \(h=1\) | 3,700 | 640 | 17.30% |
| \(h=2\) | 3,902 | 682 | 17.48% |
| \(h=3\) | 3,902 | 638 | 16.35% |
| \(h=4\) | 3,304 | 508 | 15.38% |

### 2.3 Test

| Horizon | Patch 数 | Positive 数 | Positive Rate |
|---|---:|---:|---:|
| \(h=0\) | 3,073 | 492 | 16.01% |
| \(h=1\) | 2,857 | 546 | 19.11% |
| \(h=2\) | 3,111 | 478 | 15.36% |
| \(h=3\) | 3,120 | 354 | 11.35% |
| \(h=4\) | 2,589 | 212 | 8.19% |

不同预测 Horizon 下正协同效用 Patch 的比例存在明显差异，而且这种差异在不同 Scene split 上并不表现为完全相同的单调规律。

因此，本实验不假设：

\[
\text{Horizon 越远，Cooperation Utility 必然越低}
\]

而采用更严格的表述：

\[
\boxed{
\text{Patch-level Cooperation Utility 存在明显的 Horizon-dependent Heterogeneity}
}
\]

这说明预测 Horizon 可能构成局部协同价值预测的重要上下文变量。

---

## 3. Horizon Context 表示

对于每个 Patch，将预测 Horizon：

\[
h\in\{0,1,2,3,4\}
\]

编码为 5 维 One-hot 向量：

\[
e_h\in\mathbb{R}^{5}
\]

例如：

\[
h=0
\rightarrow
[1,0,0,0,0]
\]

\[
h=4
\rightarrow
[0,0,0,0,1]
\]

需要强调：

\[
\boxed{
\text{Horizon One-hot 属于 Temporal/Horizon Context，而非完整 Temporal Modeling}
}
\]

真正的 Temporal Modeling 应进一步利用多个预测 Horizon 之间的状态演化、关联或一致性信息。

---

## 4. 实验模型

G3-1B 设置两个模型。

### 4.1 Horizon-only Baseline

仅输入 Horizon One-hot：

\[
X=e_h\in\mathbb{R}^{5}
\]

模型结构：

```text
5
↓
32
↓
1
```

其作用是测量：

\[
\boxed{
\text{仅知道当前预测 Horizon 本身可以提供多少判别信息}
}
\]

---

### 4.2 Local + Horizon MLP

G3-1A 的 Local Feature 为：

\[
X_{local}\in\mathbb{R}^{80}
\]

由：

\[
5\times4\times4
\]

局部特征展平得到。

加入 Horizon One-hot 后：

\[
X_{LH}
=
[X_{local},e_h]
\]

因此：

\[
X_{LH}\in\mathbb{R}^{85}
\]

模型结构保持与 G3-1A Local MLP 基本一致：

```text
85
↓
128
↓
64
↓
1
```

通过保持模型主体结构与训练协议一致，使性能变化主要反映 Horizon Context 所提供的增量信息。

---

## 5. 实验协议

G3-1B 与 G3-1A 使用完全相同的数据划分和训练协议。

### 数据划分

```text
Train : 13 scenes
Val   : 4 scenes
Test  : 4 scenes
```

对应：

| Split | Patches | Positive Rate |
|---|---:|---:|
| Train | 86,143 | 14.77% |
| Val | 18,555 | 16.18% |
| Test | 14,750 | 14.12% |

### 训练设置

```text
Seed          = 2026
Batch size    = 2048
Learning rate = 1e-3
Weight decay  = 1e-4
Optimizer     = AdamW
Loss          = BCEWithLogitsLoss(pos_weight)
Early stop    = 10 epochs
Best model    = selected by Validation AUPRC
Threshold     = selected on Validation set
Test          = evaluated with frozen threshold
```

Test 集正类率为：

\[
0.1412
\]

因此随机分类器的参考水平约为：

\[
AUROC_{random}=0.5
\]

\[
AUPRC_{random}\approx0.1412
\]

---

## 6. G3-1B 实验结果

整体结果如下：

| 模型 | 输入 | Test AUROC | Test AUPRC | ΔAUPRC vs Random |
|---|---|---:|---:|---:|
| Local Linear | Local | 0.6067 | 0.1941 | +0.0529 |
| Local MLP | Local | 0.6066 | 0.1873 | +0.0461 |
| Horizon-only | Horizon | 0.5845 | 0.1705 | +0.0293 |
| **Local + Horizon** | **Local + Horizon** | **0.6300** | **0.2089** | **+0.0677** |

---

## 7. Horizon-only 分析

Horizon-only Baseline 在测试集上取得：

\[
AUROC=0.5845
\]

\[
AUPRC=0.1705
\]

相对于随机 AUPRC：

\[
0.1705-0.1412
=
0.0293
\]

说明即使完全不使用局部 Occupancy Probability，仅知道预测 Horizon，也包含一定的 Patch-level Cooperation Utility 判别信息。

但是 Horizon-only 性能仍明显低于 Local + Horizon，因此 Horizon 本身不足以完成可靠的局部协同决策。

这一结果说明：

\[
\boxed{
\text{Horizon 是有效上下文变量，但不能替代 Local Occupancy Evidence}
}
\]

---

## 8. Local + Horizon 增量分析

Local MLP 的测试结果为：

\[
AUROC=0.6066
\]

\[
AUPRC=0.1873
\]

加入 Horizon Context 后：

\[
AUROC=0.6300
\]

\[
AUPRC=0.2089
\]

因此，相对于 Local MLP：

\[
\Delta AUROC
=
0.6300-0.6066
=
+0.0234
\]

\[
\Delta AUPRC
=
0.2089-0.1873
=
+0.0216
\]

AUPRC 相对提升约：

\[
\frac{0.2089-0.1873}{0.1873}
\times100\%
\approx
11.5\%
\]

同时，Local + Horizon 也超过 G3-1A 中表现最好的 Local Linear：

\[
0.2089-0.1941
=
+0.0148
\]

对应相对提升约：

\[
\frac{0.2089-0.1941}{0.1941}
\times100\%
\approx
7.6\%
\]

因此：

\[
\boxed{
\text{Horizon Context 提供了 Local Occupancy Evidence 之外的增量判别信息}
}
\]

---

## 9. G3-1B 结论

G3-1B 得到以下结论：

1. 不同预测 Horizon 下的正协同效用 Patch 比例存在明显差异，说明 Patch-level Cooperation Utility 具有 Horizon-dependent Heterogeneity；
2. Horizon-only Baseline 的 AUROC 和 AUPRC 均高于随机参考水平，说明预测 Horizon 本身包含一定的协同价值判别信息；
3. 将 Horizon Context 加入 Local Occupancy Feature 后，测试集 AUROC 从 0.6066 提升至 0.6300；
4. 测试集 AUPRC 从 0.1873 提升至 0.2089，相对提高约 11.5%；
5. Local + Horizon 同时超过此前表现最好的 Local Linear AUPRC 0.1941；
6. 因此，显式 Horizon Context 能够为 Patch-level Cooperation Utility Prediction 提供稳定的增量信息。

综合判断：

\[
\boxed{
\text{G3-1B: GO}
}
\]

需要注意，本实验当前仅使用单次固定 Scene split 和固定随机种子，因此上述性能差异应描述为“获得提升”“提供增量信息”或“稳定增量”，不宜在未进行多随机种子重复实验或统计检验的情况下使用“统计显著提升”等表述。

---

# 10. 论文中可直接使用的文字

## 10.1 时域异质性分析

进一步统计不同预测时域下局部候选区域的正协同效用比例可以发现，各 Horizon 的有效协同分布存在明显差异。例如，在测试场景中，\(h=1\) 时正协同效用区域占比为 19.11%，而在 \(h=4\) 时下降至 8.19%。与此同时，该变化趋势在训练集、验证集和测试集之间并非严格保持单调一致，说明预测时域与局部协同价值之间存在场景相关的非均匀关系，而不宜通过固定的时域规则直接进行融合决策。因此，本文将预测 Horizon 作为协同价值估计的上下文变量参与后续建模。

---

## 10.2 Horizon Context 增量实验

为验证预测时域信息能否为局部协同价值估计提供额外信息，本文在局部 Occupancy 特征基础上引入 5 维 Horizon One-hot 编码。在保持数据划分、网络主体结构及训练设置一致的条件下，Local MLP 在测试集上的 AUROC 和 AUPRC 分别为 0.6066 和 0.1873；加入 Horizon Context 后，两项指标分别提高至 0.6300 和 0.2089，其中 AUPRC 提高 0.0216，约为 11.5% 的相对增幅。

此外，仅使用 Horizon 信息的模型仍取得 0.5845 AUROC 和 0.1705 AUPRC，高于随机参考水平，表明预测时域本身包含一定的协同价值先验。然而，其性能明显低于融合局部 Occupancy Evidence 的模型，说明 Horizon Context 更适合作为局部协同价值估计的辅助条件，而非独立决策依据。

上述结果表明：

\[
\boxed{
\text{预测时域信息能够为局部协同价值估计提供 Local Feature 之外的增量信息}
}
\]

---

## 10.3 与 STCV-Occ 方法设计的衔接

G3-1A 表明，局部 Ego/Infrastructure Occupancy Evidence 中存在基础可学习的 Cooperation Utility Signal，但单纯增加非线性模型容量并未明显改善跨场景判别能力。

G3-1B 进一步发现，加入预测 Horizon Context 后，局部协同价值预测性能得到提升，说明不同预测时域下的协同信息具有不同的可靠性分布。

因此，后续 STCV-Occ 不应仅依据单一局部 Occupancy Probability 进行固定融合，而应进一步结合预测时域以及空间上下文，对不同区域和不同预测 Horizon 的协同价值进行条件化建模。

---

# 11. 当前 G3 实验链

```text
G3-0
│
├── Cooperation Utility Dataset
├── Scene-level Split
├── G2-B Sanity Check
│
▼
G3-1A
│
├── Local Linear
│   AUROC = 0.6067
│   AUPRC = 0.1941
│
├── Local MLP
│   AUROC = 0.6066
│   AUPRC = 0.1873
│
├── Conclusion:
│   Local Evidence 中存在基础可学习信号，
│   但仅增加模型非线性不足。
│
▼
G3-1B
│
├── Horizon-only
│   AUROC = 0.5845
│   AUPRC = 0.1705
│
├── Local + Horizon
│   AUROC = 0.6300
│   AUPRC = 0.2089
│
├── Conclusion:
│   Horizon Context 提供增量信息。
│
▼
G3-1C
│
└── Spatial Position Context
```

---

# 12. 当前状态

```text
G3-0   PASS
G3-1A  Weak GO
G3-1B  GO

Current Best:
Local + Horizon MLP

Test AUROC = 0.6300
Test AUPRC = 0.2089

Next:
G3-1C Spatial Position Context
```

\[
\boxed{
\text{G3-1B COMPLETE}
}
\]