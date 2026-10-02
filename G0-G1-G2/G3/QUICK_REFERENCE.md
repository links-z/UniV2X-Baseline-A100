# G3 Quick Reference Card

## 一句话总结
从 Soft Pv/Pi 特征预测 patch-level 合作效用，最佳模型 AUPRC=0.27（+92% vs 随机）

## 最佳模型
```
Model:      MLP+H+S+N (127→256→128→64→1)
AUPRC:      0.2712 (Test)
AUROC:      0.7035 (Test)
Checkpoint: checkpoints_soft/g3_neighborhood_best.pth
Dataset:    soft/g3_neighborhood_{train,val,test}.npz
```

## 关键数字
| 指标 | 值 |
|------|-----|
| 总样本数 | 119,448 patches |
| 正例率 | 14.9% |
| 相对随机提升 | +92.0% |
| 模型参数 | 73,985 |

## 信号贡献
```
Local (Pv/Pi)      +0.0908  (70%)  ⭐⭐⭐ PRIMARY
Neighborhood       +0.0293  (23%)  ⭐⭐   STRONG
Spatial (u,v)      +0.0088  ( 7%)  ⭐    MODERATE
Temporal (horizon) +0.0011  ( 1%)       WEAK
```

## 关键发现
1. ✅ **Soft > Binary**: Soft 特征比 Binary 平均提升 16%
2. ✅ **Local 主导**: 70% 的信号来自 4×4 local patch
3. ✅ **Neighborhood 重要**: 8 邻居统计贡献 23% 增益
4. ❌ **Temporal 微弱**: Horizon 编码几乎无效（+0.001）

## 快速使用

### 加载最佳模型
```python
import torch
from train_learnability_neighborhood_soft import NeighborhoodMLP

model = NeighborhoodMLP()
ckpt = torch.load('checkpoints_soft/g3_neighborhood_best.pth')
model.load_state_dict(ckpt['model'])
model.eval()
```

### 加载数据集
```python
import numpy as np

d = np.load('soft/g3_neighborhood_test.npz', allow_pickle=True)
features = d['features']  # [N, 127]
labels = d['labels']      # [N]
metadata = d['metadata']  # [N]
```

### 预测
```python
import torch

x = torch.from_numpy(features).float()
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)
    
# 使用验证集选定的 threshold=0.54
predictions = (probs >= 0.54).long()
```

## 特征结构 (127-dim)
```
[ 0: 79]  Local features (80-dim: 5ch × 4×4)
[80: 84]  Horizon encoding (5-dim: one-hot)
[85: 86]  Spatial position (2-dim: u,v)
[87:126]  Neighborhood (40-dim: 8nbr × 5stats)
```

## 文件位置
```
数据集:    soft/g3_neighborhood_{train,val,test}.npz
模型:      checkpoints_soft/g3_neighborhood_best.pth
结果:      results_soft/g3_neighborhood_learnability.json
完整报告:  README_FINAL.md
快速总结:  EXPERIMENT_SUMMARY.txt
```

## 性能指标 (Test, threshold=0.54)
| Metric | Value | 解释 |
|--------|-------|------|
| AUPRC | 0.2712 | 主要评估指标 |
| AUROC | 0.7035 | 排序能力 |
| Recall | 68.5% | 捕获有益 patch 的比例 |
| Precision | 22.7% | 接受 patch 的准确率 |
| F1 | 0.3405 | 平衡指标 |

## 判决
✅ **GO** - 强可学习性，准备进入 G4 端到端评估

## 下一步
进入 G4: 用学习预测器替换 Oracle，评估端到端融合性能
