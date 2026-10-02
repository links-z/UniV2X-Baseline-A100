# G3 Complete File Index

Quick navigation guide for all G3 project files.

---

## 📚 Documentation (Start Here)

### Essential Reading
| File | Description | Lines | Audience |
|------|-------------|-------|----------|
| [README.md](README.md) | Top-level orientation, quick start | 200 | Everyone |
| [README_FINAL.md](README_FINAL.md) | Complete technical report | 355 | Researchers (中文) |
| [G3_FINAL_RESULTS.md](G3_FINAL_RESULTS.md) | Detailed analysis | 353 | Researchers (English) |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Step-by-step reproduction guide | 426 | Reviewers |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick usage examples | 150 | Developers |

### Project Records
| File | Description | Lines |
|------|-------------|-------|
| [G3_COMPLETION_REPORT.txt](G3_COMPLETION_REPORT.txt) | Project completion summary | 267 |
| [G3_DELIVERABLES_SUMMARY.txt](G3_DELIVERABLES_SUMMARY.txt) | Full deliverables list | 400+ |
| [G3_ARCHIVE_CHECKLIST.md](G3_ARCHIVE_CHECKLIST.md) | Archival checklist | 350 |
| [EXPERIMENT_SUMMARY.txt](EXPERIMENT_SUMMARY.txt) | Quick results overview | 60 |

### Experimental Details
| File | Description | Lines |
|------|-------------|-------|
| [G3_SOFT_ABLATION_SUMMARY.md](G3_SOFT_ABLATION_SUMMARY.md) | Ablation study results | 84 |
| [SOFT_VS_BINARY_RESULTS.md](SOFT_VS_BINARY_RESULTS.md) | Feature comparison | 114 |
| [G3-0_README.md](G3-0_README.md) | Original design (archived) | 200 |
| [G3-1B_temporal_context.md](G3-1B_temporal_context.md) | Temporal experiments | 150 |
| [G3-1C_spatial_position.md](G3-1C_spatial_position.md) | Spatial experiments | 150 |

---

## 🤖 Models

### Production Model
```
checkpoints_soft/g3_neighborhood_best.pth  (292KB) ⭐ BEST
```
- Architecture: MLP+H+S+N (127→256→128→64→1)
- Test AUPRC: 0.2712
- Test AUROC: 0.7035
- Parameters: 73,985
- Epoch: 7

### Ablation Models
```
checkpoints_soft/
├── g3_mlp_hsp_best.pth      (79KB)  - MLP+H+S (AUPRC 0.2418)
├── g3_mlp_h_best.pth        (78KB)  - MLP+H (AUPRC 0.2330)
├── g3_mlp_best.pth          (76KB)  - MLP (AUPRC 0.2320)
├── g3_spatial_best.pth      (2.4KB) - Spatial-only (AUPRC 0.1453)
├── g3_horizon_best.pth      (2.8KB) - Horizon-only (AUPRC 0.1705)
└── g3_linear_best.pth       (1.7KB) - Linear (AUPRC 0.2181)
```

### Baseline Models (Binary, Archived)
```
checkpoints/
├── g3_mlp_hsp_best.pth
├── g3_mlp_h_best.pth
├── g3_mlp_best.pth
├── g3_spatial_best.pth
├── g3_horizon_best.pth
└── g3_linear_best.pth
```

---

## 💾 Datasets

### Production (Soft Pv/Pi)
```
soft/
├── g3_neighborhood_train.npz  (29MB)  - 86,143 samples, 127-dim ⭐
├── g3_neighborhood_val.npz    (6.1MB) - 18,555 samples, 127-dim ⭐
├── g3_neighborhood_test.npz   (4.9MB) - 14,750 samples, 127-dim ⭐
├── g3_dataset_train.npz       (19MB)  - 86,143 samples, 5×4×4
├── g3_dataset_val.npz         (4.2MB) - 18,555 samples, 5×4×4
└── g3_dataset_test.npz        (3.3MB) - 14,750 samples, 5×4×4
```

**Features**:
- 127-dim: 80 (local) + 5 (horizon) + 2 (spatial) + 40 (neighborhood)
- 5 channels: Pv, Pi, Pi-Pv, |Pi-Pv|, warp_mask
- 4×4 patches on 200×200 grid

### Baseline (Binary Ov/Oi, Archived)
```
binary/
├── g3_dataset_train.npz  (4.0MB)
├── g3_dataset_val.npz    (898KB)
├── g3_dataset_test.npz   (714KB)
└── scene_splits.txt      (1KB) - Fixed scene split (SEED=2026)
```

---

## 📊 Results

### Production Results (Soft)
```
results_soft/
├── g3_neighborhood_learnability.json  ⭐ BEST (AUPRC 0.2712)
├── g3_mlp_hsp_learnability.json
├── g3_mlp_h_learnability.json
├── g3_spatial_learnability.json
├── g3_horizon_learnability.json
├── g3_mlp_learnability.json
└── g3_linear_learnability.json
```

### Baseline Results (Binary, Archived)
```
results/
├── g3_mlp_hsp_learnability.json
├── g3_mlp_h_learnability.json
├── g3_spatial_learnability.json
├── g3_horizon_learnability.json
├── g3_mlp_learnability.json
└── g3_linear_learnability.json
```

**Each JSON contains**:
- Training curves (loss, metrics per epoch)
- Best epoch selection
- Validation metrics
- Test metrics
- Hyperparameters
- Threshold selection

---

## 🔧 Source Code

### Dataset Construction
| File | Lines | Purpose |
|------|-------|---------|
| [build_g3_soft_neighborhood_dataset.py](build_g3_soft_neighborhood_dataset.py) | 650 | ⭐ Build 127-dim features (Local+H+S+N) |
| [build_g3_soft_dataset.py](build_g3_soft_dataset.py) | 400 | ⭐ Extract Soft Pv/Pi from G0 cache |
| [build_g3_dataset.py](build_g3_dataset.py) | 300 | Extract Binary Ov/Oi (archived) |

### Model Training (Soft)
| File | Lines | Model |
|------|-------|-------|
| [train_learnability_neighborhood_soft.py](train_learnability_neighborhood_soft.py) | 570 | ⭐ MLP+H+S+N (best) |
| [train_learnability_spatial_soft.py](train_learnability_spatial_soft.py) | 400 | MLP+H+S, Spatial-only |
| [train_learnability_temporal_soft.py](train_learnability_temporal_soft.py) | 400 | MLP+H, Horizon-only |
| [train_learnability_soft.py](train_learnability_soft.py) | 350 | MLP, Linear |

### Model Training (Binary, Archived)
| File | Lines | Model |
|------|-------|-------|
| [train_learnability_spatial.py](train_learnability_spatial.py) | 400 | Binary MLP+H+S |
| [train_learnability_temporal.py](train_learnability_temporal.py) | 400 | Binary MLP+H |
| [train_learnability.py](train_learnability.py) | 350 | Binary MLP |

---

## 📝 Metadata

```
cache_valid_samples.txt  - List of 675 valid G0 cache samples
FILE_INDEX.md            - This file
```

---

## 🗂️ Directory Structure

```
G3/
├── 📚 Documentation (14 .md/.txt files, ~150KB)
├── 🤖 Models (13 .pth files, ~1MB)
├── 💾 Datasets (9 .npz files, 66MB)
├── 📊 Results (13 .json files, ~110KB)
├── 🔧 Code (10 .py files, ~3,500 lines)
└── 📝 Metadata (2 files)
```

**Total Size**: ~87MB

---

## 🎯 Quick Access by Task

### I want to verify the results
1. Read: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) → Section 1 (Results Verification)
2. Run: Code snippet in Section 1
3. Expected: AUPRC 0.271153, AUROC 0.703514

### I want to retrain the model
1. Read: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) → Section 2 (Training)
2. Run: `python train_learnability_neighborhood_soft.py --seed 2026`
3. Expected: AUPRC 0.2712 ±0.001

### I want to understand the experiments
1. Read: [README_FINAL.md](README_FINAL.md) (中文) or [G3_FINAL_RESULTS.md](G3_FINAL_RESULTS.md) (English)
2. Check: [G3_SOFT_ABLATION_SUMMARY.md](G3_SOFT_ABLATION_SUMMARY.md) for ablation details

### I want to use the model in my code
1. Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Copy: Code snippets for loading and inference

### I want to rebuild the dataset
1. Read: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) → Section 3 (Data Rebuilding)
2. Run: `python build_g3_soft_neighborhood_dataset.py`
3. Requires: G0 cache at `/root/autodl-tmp/UniV2X/G0-G1-G2/cache_full/`

### I want to integrate into G4
1. Load model: `checkpoints_soft/g3_neighborhood_best.pth`
2. Load test data: `soft/g3_neighborhood_test.npz`
3. Reference: [README.md](README.md) → Section "Next Steps (G4)"

---

## 🔍 Search Tips

### Find by keyword
```bash
# Search in documentation
grep -r "AUPRC" *.md *.txt

# Search in code
grep -r "NeighborhoodMLP" *.py

# Search in results
grep "test_metrics" results_soft/*.json
```

### Find by file type
```bash
ls *.md        # Documentation (Markdown)
ls *.txt       # Plain text reports
ls *.py        # Python scripts
ls *.pth       # PyTorch models
ls *.npz       # NumPy datasets
ls *.json      # JSON results
```

### Find by size
```bash
du -h * | sort -h                # All files by size
du -h checkpoints_soft/*.pth    # Models by size
du -h soft/*.npz                # Datasets by size
```

---

## 📅 File Creation Timeline

1. **G3-0 (Binary, Archived)** - Week 1
   - build_g3_dataset.py
   - train_learnability*.py (binary versions)
   - binary/*.npz
   - results/*.json

2. **G3-1 (Soft, Correction)** - Week 2
   - build_g3_soft_dataset.py
   - build_g3_soft_neighborhood_dataset.py
   - train_learnability*_soft.py
   - soft/*.npz
   - results_soft/*.json
   - checkpoints_soft/*.pth

3. **Documentation** - Week 2
   - README.md, README_FINAL.md, G3_FINAL_RESULTS.md
   - REPRODUCIBILITY.md, QUICK_REFERENCE.md
   - EXPERIMENT_SUMMARY.txt
   - G3_SOFT_ABLATION_SUMMARY.md
   - SOFT_VS_BINARY_RESULTS.md

4. **Archival** - Week 2 End
   - G3_COMPLETION_REPORT.txt
   - G3_DELIVERABLES_SUMMARY.txt
   - G3_ARCHIVE_CHECKLIST.md
   - FILE_INDEX.md

---

## ✅ File Status

All files verified present and complete as of 2026-10-02.

---

**Last Updated**: 2026-10-02  
**Total Files**: 60+  
**Total Size**: 87MB  
**Status**: Complete
