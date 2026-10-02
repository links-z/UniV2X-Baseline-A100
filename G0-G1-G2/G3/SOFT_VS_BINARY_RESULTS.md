# G3-1A: Soft vs Binary Feature Comparison

## Critical Discovery

G3-0 原始设计错误使用了 **Binary Ov/Oi** (thresholded at 0.1) 作为预测器输入，而非设计规范中的 **Soft Pv/Pi** (probabilities)。

本实验验证修复此错误的效果。

## Experimental Setup

### Binary Branch (Baseline - 原始错误实现)
- **Input**: Ov/Oi (1[Pv>0.1], 1[Pi>0.1])
- **Feature channels**: [Ov, Oi, Oi-Ov, |Oi-Ov|, M^warp]
- **Dataset**: `G0-G1-G2/G3/binary/g3_dataset_{train,val,test}.npz`

### Soft Branch (Corrected - 符合设计规范)
- **Input**: Pv/Pi (soft probabilities)
- **Feature channels**: [Pv, Pi, Pi-Pv, |Pi-Pv|, M^warp]
- **Dataset**: `G0-G1-G2/G3/soft/g3_dataset_{train,val,test}.npz`
- **Verification**: `1[Pv>0.1] == Ov` cell-by-cell (max_diff=0.0, PASS)

### Key Points
- **Labels**: Both branches use IDENTICAL labels (Binary Ov/Oi for ADD/BA/HA/Utility) - CORRECT
- **Only difference**: Predictor input feature extraction (Binary vs Soft)
- **Metadata**: Completely preserved across branches

## Results

### Test AUPRC (Random baseline ≈ 0.1412)

| Model  | Binary (Ov/Oi) | Soft (Pv/Pi) | Delta      | Relative Gain |
|--------|----------------|--------------|------------|---------------|
| Linear | 0.194055       | 0.218133     | **+0.024078** | **+12.4%** |
| MLP    | 0.187340       | 0.231960     | **+0.044620** | **+23.8%** |

### Test AUROC

| Model  | Binary | Soft    | Delta      |
|--------|--------|---------|------------|
| Linear | 0.5761 | 0.6230  | **+0.0469** |
| MLP    | 0.6054 | 0.6810  | **+0.0756** |

## Key Findings

### 1. **Soft 输入显著优于 Binary** ✓
- Linear: +0.024 AUPRC (+12.4%)
- MLP: +0.045 AUPRC (+23.8%)
- 更复杂的模型（MLP）从 Soft 输入中获得更大提升

### 2. **Binary 输入信息损失严重**
- Binary thresholding 丢失了 Pv/Pi 的概率信息
- Soft 特征中 99%+ 的值是非二值的（non_binary_ratio > 0.99）
- 这些连续概率值包含预测所需的细粒度信息

### 3. **设计规范验证**
- 原始设计要求使用 Soft Pv/Pi 是正确的
- G3-0 的 Binary 实现是错误的

## Decision: 采用 Soft 作为主分支

### Rationale
1. **符合设计规范**: Pv/Pi 本身就是概率，不应该被二值化
2. **性能提升显著**: AUPRC +12.4% ~ +23.8%
3. **信息保留完整**: 保留了完整的概率分布信息

### Implementation
- **保留 Binary 结果**: 作为 ablation baseline（binary/ 目录）
- **继续使用 Soft**: 所有后续实验（G3-1B/C/D）使用 soft/ 数据集
- **标签不变**: 继续使用 Binary Ov/Oi 计算的 ADD/BA/HA/Utility 作为 GT

## Next Steps

继续 Soft 分支实验：
- [x] G3-1A Soft: Local features (DONE, AUPRC=0.232)
- [ ] G3-1B Soft: + Horizon encoding
- [ ] G3-1C Soft: + Spatial position
- [ ] G3-1D Soft: + Neighborhood context

## Files

### Binary Branch (backup)
```
G0-G1-G2/G3/binary/
├── g3_dataset_train.npz
├── g3_dataset_val.npz
├── g3_dataset_test.npz
└── scene_splits.txt

G0-G1-G2/G3/results/
├── g3_linear_learnability.json
└── g3_mlp_learnability.json
```

### Soft Branch (main)
```
G0-G1-G2/G3/soft/
├── g3_dataset_train.npz
├── g3_dataset_val.npz
└── g3_dataset_test.npz

G0-G1-G2/G3/results_soft/
├── g3_linear_learnability.json
└── g3_mlp_learnability.json
```

## Conclusion

修复 Binary → Soft 输入后，G3-1A 的可学习性从 **弱信号** 提升到 **中等信号**：
- Random baseline: 0.1412
- Binary MLP: 0.1873 (+0.0461)
- **Soft MLP: 0.2320 (+0.0908)** ← 采用此方案

**Verdict**: GO - Soft 输入显著改善可学习性，继续进行 G3-1B/C/D 实验。
