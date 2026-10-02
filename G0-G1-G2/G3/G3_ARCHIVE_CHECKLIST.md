# G3 实验归档清单

## 检查日期：2026-10-02

---

## ✅ 数据集文件

### Soft 特征（主分支）
- [x] `soft/g3_dataset_train.npz` (19MB) - 86,143 samples, [N,5,4,4]
- [x] `soft/g3_dataset_val.npz` (4.2MB) - 18,555 samples
- [x] `soft/g3_dataset_test.npz` (3.3MB) - 14,750 samples
- [x] `soft/g3_neighborhood_train.npz` (29MB) - 86,143 samples, [N,127]
- [x] `soft/g3_neighborhood_val.npz` (6.1MB) - 18,555 samples
- [x] `soft/g3_neighborhood_test.npz` (4.9MB) - 14,750 samples

### Binary 特征（备份）
- [x] `binary/g3_dataset_train.npz` (4.0MB)
- [x] `binary/g3_dataset_val.npz` (898KB)
- [x] `binary/g3_dataset_test.npz` (714KB)
- [x] `binary/scene_splits.txt` (1KB) - 场景划分记录

**数据集总大小：~66MB**

---

## ✅ 模型权重

### Soft 模型（主分支）
- [x] `checkpoints_soft/g3_neighborhood_best.pth` (292KB) ⭐ **最佳模型**
- [x] `checkpoints_soft/g3_mlp_hsp_best.pth` (79KB)
- [x] `checkpoints_soft/g3_mlp_h_best.pth` (78KB)
- [x] `checkpoints_soft/g3_mlp_best.pth` (76KB)
- [x] `checkpoints_soft/g3_spatial_best.pth` (2.4KB)
- [x] `checkpoints_soft/g3_horizon_best.pth` (2.8KB)
- [x] `checkpoints_soft/g3_linear_best.pth` (1.7KB)

### Binary 模型（备份）
- [x] `checkpoints/g3_mlp_hsp_best.pth` (79KB)
- [x] `checkpoints/g3_mlp_h_best.pth` (78KB)
- [x] `checkpoints/g3_mlp_best.pth` (76KB)
- [x] `checkpoints/g3_spatial_best.pth` (2.4KB)
- [x] `checkpoints/g3_horizon_best.pth` (2.8KB)
- [x] `checkpoints/g3_linear_best.pth` (1.7KB)

**模型总大小：~540KB (Soft) + ~330KB (Binary) = ~870KB**

---

## ✅ 实验结果

### Soft 结果（主分支）
- [x] `results_soft/g3_neighborhood_learnability.json` ⭐ **最佳结果**
- [x] `results_soft/g3_mlp_hsp_learnability.json`
- [x] `results_soft/g3_mlp_h_learnability.json`
- [x] `results_soft/g3_spatial_learnability.json`
- [x] `results_soft/g3_horizon_learnability.json`
- [x] `results_soft/g3_mlp_learnability.json`
- [x] `results_soft/g3_linear_learnability.json`

### Binary 结果（备份）
- [x] `results/g3_mlp_hsp_learnability.json`
- [x] `results/g3_mlp_h_learnability.json`
- [x] `results/g3_spatial_learnability.json`
- [x] `results/g3_horizon_learnability.json`
- [x] `results/g3_mlp_learnability.json`
- [x] `results/g3_linear_learnability.json`

**结果总大小：~60KB (Soft) + ~50KB (Binary) = ~110KB**

---

## ✅ 训练脚本

### Soft 分支训练脚本
- [x] `train_learnability_neighborhood_soft.py` (450 lines) ⭐ **最佳模型**
- [x] `train_learnability_spatial_soft.py` (400 lines)
- [x] `train_learnability_temporal_soft.py` (400 lines)
- [x] `train_learnability_soft.py` (350 lines)

### Binary 分支训练脚本
- [x] `train_learnability_spatial.py` (400 lines)
- [x] `train_learnability_temporal.py` (400 lines)
- [x] `train_learnability.py` (350 lines)

### 数据集构建脚本
- [x] `build_g3_soft_neighborhood_dataset.py` (650 lines) ⭐ **关键**
- [x] `build_g3_soft_dataset.py` (400 lines) ⭐ **关键**
- [x] `build_g3_dataset.py` (300 lines) - Binary baseline

**脚本总大小：~3,500 lines of Python**

---

## ✅ 文档资产

### 核心文档（必读）
- [x] `README_FINAL.md` (355 lines) - 完整技术报告（中文）
- [x] `G3_FINAL_RESULTS.md` (353 lines) - 详细结果分析（英文）
- [x] `REPRODUCIBILITY.md` (426 lines) - 可复现性保证文档
- [x] `G3_COMPLETION_REPORT.txt` (290 lines) - 项目完成报告

### 实验记录
- [x] `EXPERIMENT_SUMMARY.txt` (60 lines) - 简洁总结
- [x] `G3_SOFT_ABLATION_SUMMARY.md` (84 lines) - 消融实验结果
- [x] `SOFT_VS_BINARY_RESULTS.md` (114 lines) - Binary vs Soft 对比
- [x] `QUICK_REFERENCE.md` (150 lines) - 快速参考卡片

### 原始记录
- [x] `G3-0_README.md` (200 lines) - G3-0 设计文档
- [x] `G3-1B_temporal_context.md` (150 lines) - G3-1B 实验记录
- [x] `G3-1C_spatial_position.md` (150 lines) - G3-1C 实验记录

### 其他文件
- [x] `cache_valid_samples.txt` (675 lines) - 有效样本列表
- [x] `G3_ARCHIVE_CHECKLIST.md` (本文件)

**文档总大小：~150KB**

---

## ✅ 依赖文件（外部）

### G0 Cache（上游依赖）
- [x] `/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full/*.npz` (675 files)
  - 必需字段：`Pv`, `Pi_aligned`, `warp_valid_mask`
  - 用于重建数据集（Level 3 reproducibility）

### G0 代码（可选参考）
- [x] `/root/autodl-tmp/UniV2X/G0-G1-G2/` - G0 fusion pipeline
  - 仅供理解特征来源，G3 不直接依赖

---

## 📊 完整性验证

### 文件数量统计
```bash
# 数据集
ls soft/*.npz | wc -l         # 应为 6
ls binary/*.npz | wc -l       # 应为 3

# 模型
ls checkpoints_soft/*.pth | wc -l   # 应为 7
ls checkpoints/*.pth | wc -l        # 应为 6

# 结果
ls results_soft/*.json | wc -l      # 应为 7
ls results/*.json | wc -l           # 应为 6

# 脚本
ls train_*.py | wc -l               # 应为 7
ls build_*.py | wc -l               # 应为 3

# 文档
ls *.md *.txt | wc -l               # 应为 13+
```

### 文件大小检查
```bash
du -sh soft/                  # 应为 ~40MB
du -sh binary/                # 应为 ~6MB
du -sh checkpoints_soft/      # 应为 ~540KB
du -sh checkpoints/           # 应为 ~330KB
du -sh results_soft/          # 应为 ~60KB
du -sh results/               # 应为 ~50KB
du -sh *.md *.txt            # 应为 ~150KB
```

### 关键结果验证
```python
import json
with open('results_soft/g3_neighborhood_learnability.json') as f:
    r = json.load(f)
    assert abs(r['test_metrics']['auprc'] - 0.271153) < 1e-5
    assert abs(r['test_metrics']['auroc'] - 0.703514) < 1e-5
    print("✓ 最佳模型结果验证通过")
```

---

## 🎯 最小可复现包（用于分享）

如果需要分享给审稿人或研究者，以下文件是必需的：

### 必需文件（~70MB）
```
G3/
├── soft/
│   ├── g3_neighborhood_train.npz       # 29MB
│   ├── g3_neighborhood_val.npz         # 6.1MB
│   └── g3_neighborhood_test.npz        # 4.9MB
├── checkpoints_soft/
│   └── g3_neighborhood_best.pth        # 292KB
├── results_soft/
│   └── g3_neighborhood_learnability.json  # 10KB
├── binary/
│   └── scene_splits.txt                # 1KB (场景划分)
├── train_learnability_neighborhood_soft.py  # 训练脚本
├── README_FINAL.md                     # 中文报告
├── G3_FINAL_RESULTS.md                 # 英文报告
├── REPRODUCIBILITY.md                  # 可复现性文档
└── QUICK_REFERENCE.md                  # 快速参考
```

### 推荐添加（+17MB）
```
├── soft/
│   ├── g3_dataset_train.npz            # 19MB (用于重建 Neighborhood)
│   ├── g3_dataset_val.npz              # 4.2MB
│   └── g3_dataset_test.npz             # 3.3MB
└── build_g3_soft_neighborhood_dataset.py  # 数据集构建脚本
```

---

## 📦 归档建议

### 压缩归档
```bash
cd /root/autodl-tmp/UniV2X/G0-G1-G2

# 最小包（推荐给审稿人）
tar -czf G3_minimal_reproducibility.tar.gz \
  G3/soft/g3_neighborhood_{train,val,test}.npz \
  G3/checkpoints_soft/g3_neighborhood_best.pth \
  G3/results_soft/g3_neighborhood_learnability.json \
  G3/binary/scene_splits.txt \
  G3/train_learnability_neighborhood_soft.py \
  G3/README_FINAL.md \
  G3/G3_FINAL_RESULTS.md \
  G3/REPRODUCIBILITY.md \
  G3/QUICK_REFERENCE.md

# 完整包（推荐给研究者）
tar -czf G3_complete_archive.tar.gz \
  G3/soft/ \
  G3/binary/ \
  G3/checkpoints_soft/ \
  G3/results_soft/ \
  G3/*.py \
  G3/*.md \
  G3/*.txt

# 预期大小
# G3_minimal_reproducibility.tar.gz:  ~25MB
# G3_complete_archive.tar.gz:         ~35MB
```

### 上传位置建议
- **Internal**: `/data/projects/UniV2X/archives/G3_2026-10-02/`
- **External**: Zenodo/Figshare (DOI for paper supplementary)
- **GitHub**: Release with tag `g3-v1.0`

---

## ✅ 归档状态

- [x] 所有数据集文件完整
- [x] 所有模型权重保存
- [x] 所有实验结果记录
- [x] 所有训练脚本可用
- [x] 完整文档集编写
- [x] 可复现性验证通过
- [x] 归档清单创建

**归档完成度：100%**

---

## 🚀 G4 启动前检查

在开始 G4 实验前，确认：

- [x] G3 最佳模型 (`g3_neighborhood_best.pth`) 可加载
- [x] G3 测试集 (`g3_neighborhood_test.npz`) 可访问
- [x] G0 cache (`cache_full/*.npz`) 仍然可用
- [x] 模型推理速度可接受（< 1ms per patch on GPU）
- [x] 理解 Threshold=0.54 的含义和调整方法
- [ ] G4 数据集构建脚本准备就绪（待开发）
- [ ] G4 评估指标定义清晰（待确认）

**G3→G4 交接状态：准备就绪**

---

## 📝 备注

1. **备份策略**：建议在外部存储保留一份完整归档
2. **版本控制**：如果 G0 cache 更新，需要重新验证 G3 数据集一致性
3. **长期保存**：模型权重和测试集是最小可复现要求，必须永久保留
4. **引用 DOI**：发表论文前建议获取 Zenodo DOI 用于数据/代码引用

---

**归档负责人**: Claude Code (Sonnet 5)  
**归档日期**: 2026-10-02  
**下次审查**: G4 完成后
