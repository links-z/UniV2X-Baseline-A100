# G3 Soft Context Ablation Summary

## Test AUPRC Results (Random baseline ≈ 0.1412)

| Model | AUPRC | AUROC | Incremental Gain | Notes |
|-------|-------|-------|------------------|-------|
| **Random baseline** | 0.1412 | 0.5000 | - | Class prior |
| Linear (Local only) | 0.2181 | 0.6230 | +0.0769 | G3-1A |
| **MLP (Local only)** | **0.2320** | 0.6810 | **+0.0908** | **G3-1A** |
| Horizon-only | 0.1705 | 0.5845 | - | G3-1B context ablation |
| MLP + Horizon | 0.2330 | 0.6850 | +0.0011 | G3-1B |
| Spatial-only | 0.1453 | 0.5051 | - | G3-1C context ablation |
| **MLP + H + S** | **0.2418** | 0.6882 | **+0.0088** | **G3-1C** |

## Context Contribution Analysis

### 1. Local Features (Soft Pv/Pi) - **PRIMARY**
- **Gain**: +0.0908 AUPRC (64.3% relative to random)
- **Verdict**: Strong signal - Soft probability features are highly predictive
- **Key insight**: 99%+ non-binary values contain critical information

### 2. Temporal Context (Horizon encoding) - **WEAK**
- **Gain**: +0.0011 AUPRC (0.5% relative gain)
- **Verdict**: Marginal improvement - near noise level
- **Explanation**: Horizon heterogeneity exists (h0=16%, h4=8%) but adds minimal predictive power

### 3. Spatial Context (Position u,v) - **MODERATE**
- **Gain**: +0.0088 AUPRC (3.8% relative gain)
- **Verdict**: Meaningful improvement - spatial bias exists
- **Note**: Spatial-only is near random (0.1453), but combined with Local+H shows value

## Total Performance

**Best model (MLP + H + S)**: 
- Test AUPRC = 0.2418
- Total gain = +0.1006 (71.3% improvement over random)
- AUROC = 0.6882

## Soft vs Binary Comparison

| Model | Binary AUPRC | Soft AUPRC | Improvement |
|-------|-------------|------------|-------------|
| Linear | 0.1941 | 0.2181 | +0.0240 (+12.4%) |
| MLP | 0.1873 | 0.2320 | +0.0447 (+23.9%) |
| MLP+H | 0.2089 | 0.2330 | +0.0241 (+11.5%) |
| MLP+H+S | 0.2118 | 0.2418 | +0.0300 (+14.2%) |

**Average improvement: +16.0%** by using Soft Pv/Pi instead of Binary Ov/Oi

## Next Step: G3-1D Neighborhood Context

已完成 Local, Temporal, Spatial 上下文实验，现在需要构建并测试 **Neighborhood Context**:

### G3-1D Design
- **Input**: 8-neighbor mean statistics (40-dim)
- **Architecture**: MLP + H + S + N (127-dim total)
- **Expectation**: Neighborhood averaging may smooth local noise and improve generalization

### Implementation Status
- [x] G3-1A Soft: Local features (AUPRC=0.232)
- [x] G3-1B Soft: + Horizon (AUPRC=0.233, +0.001)
- [x] G3-1C Soft: + Spatial (AUPRC=0.242, +0.009)
- [ ] **G3-1D Soft: + Neighborhood** (NEXT)

### Required Files
1. `build_g3_soft_neighborhood_dataset.py` - 从 Soft G3 + G0 cache 构建 127-dim 数据集
2. `train_learnability_neighborhood_soft.py` - 训练 MLP+H+S+N 模型

### Key Requirements
- Use **Soft Pv/Pi** from G0 cache (NOT Binary Ov/Oi)
- Build 50×50 complete patch grid per sample
- Extract 8-neighbor statistics for each candidate patch
- Verify center-patch consistency with existing Soft G3
- Expected sample counts: Train=86143, Val=18555, Test=14750

## Conclusion

**Verdict**: GO - Soft feature design shows strong learnability

1. **Primary signal**: Local Soft Pv/Pi features (+0.0908)
2. **Secondary signals**: Spatial position (+0.0088), Horizon (+0.0011)
3. **Soft superiority**: +16% average improvement over Binary
4. **Ready for**: G3-1D Neighborhood Context experiment
