# G4 Methodology Documentation

## Evaluation Protocol

### Dataset Splits
- **Validation Set**: 94 samples (4 scenes)
- **Test Set**: 86 samples (4 scenes, held-out)
- **Split Method**: Scene-level split from G3 (SEED=2026)
- **Note**: These are different from G2-B's 549 samples (full valid set)

### Threshold Selection (G4-1)
- **Method**: Grid search on Validation set
- **Range**: τ = 0.05, 0.10, 0.15, ..., 0.95 (19 thresholds)
- **Selection Criterion**: Maximize Val IoU (NOT classification F1)
- **Best Threshold**: τ* = 0.70
- **Accept Rate at τ***: 30.67% (conservative)
- **Frozen for Test**: τ* is selected on Val and NEVER tuned on Test

### Fusion Strategies (G4-2)

#### 1. Ego Baseline
```
O_ego = O_v
```
Only vehicle perception, no infrastructure.

#### 2. Official OR
```
O_official = O_v ∨ O_i
```
Naive union fusion (baseline to beat).

#### 3. Learned Selective Fusion
```
s_R = σ(f_θ(X_R))           # G3 predictor score
a_R = 1[s_R > τ*]            # Accept decision
O_learned = O_v ∨ (a_R ∧ O_i)  # Selective fusion
```
**Key property**: Only controls Infrastructure-added occupancy, preserves all Ego occupancy.

#### 4. Patch-level Oracle
```
a_R = 1[U_R > 0]             # GT utility-based accept
O_oracle = O_v ∨ (a_R ∧ O_i)
```
Upper bound for patch-level selective fusion.

---

## Metric Aggregation

### Critical: Macro-Averaging is Used

All metrics are computed **per sample** first, then averaged across samples:

```python
# For each sample i:
intersection_i = (pred_i ∧ gt_i).sum()
union_i = (pred_i ∨ gt_i).sum()
pred_pos_i = pred_i.sum()
gt_pos_i = gt_i.sum()

IoU_i = intersection_i / union_i
Precision_i = intersection_i / pred_pos_i
Recall_i = intersection_i / gt_pos_i
F1_i = 2 × Precision_i × Recall_i / (Precision_i + Recall_i)

# Then aggregate:
IoU = mean(IoU_1, IoU_2, ..., IoU_N)
Precision = mean(Precision_1, Precision_2, ..., Precision_N)
Recall = mean(Recall_1, Recall_2, ..., Recall_N)
F1 = mean(F1_1, F1_2, ..., F1_N)
```

### Important Consequence

**F1 ≠ 2 × Precision × Recall / (Precision + Recall)**

This is because:
- Reported **Precision** = macro-average of per-sample precisions
- Reported **Recall** = macro-average of per-sample recalls
- Reported **F1** = macro-average of per-sample F1s

This is **Macro F1**, not Micro F1 computed from aggregated TP/FP/FN.

### Aggregation Summary

| Metric | Aggregation Method |
|--------|-------------------|
| IoU | Macro-average (per-sample IoU → mean) |
| Precision | Macro-average (per-sample Precision → mean) |
| Recall | Macro-average (per-sample Recall → mean) |
| F1 | Macro-average (per-sample F1 → mean) |

**Rationale**: Macro-averaging gives equal weight to each sample/scene, avoiding dominance by high-occupancy samples.

---

## Key Results (Test Set)

### Performance Table

| Strategy | IoU | Precision | Recall | F1 | N |
|----------|-----|-----------|--------|-----|---|
| Ego | 0.2671 | 0.4339 | 0.3958 | 0.4029 | 86 |
| Official OR | 0.2526 | 0.3313 | 0.4811 | 0.3849 | 86 |
| **Learned** | **0.2732** | **0.4124** | **0.4241** | **0.4099** | 86 |
| Oracle | 0.3077 | 0.4582 | 0.4568 | 0.4480 | 86 |

### Key Comparisons

```
Learned vs Official: +0.0206 (+8.2% relative)
Learned vs Ego:      +0.0062 (+2.3% relative)

Oracle Gap = IoU_Oracle - IoU_Official = 0.0551
Learned Gain = IoU_Learned - IoU_Official = 0.0206
Recovery = Learned Gain / Oracle Gap = 37.4%
```

### Ordering

```
IoU_Learned > IoU_Ego > IoU_Official
0.2732     > 0.2671  > 0.2526
```

**Critical**: Learned Fusion not only fixes negative cooperation (beats Official), but also surpasses Ego baseline.

---

## Comparison with G2-B Oracle

### Different Evaluation Sets

**G2-B Oracle** (reported earlier):
- Evaluated on: 549 valid samples (full valid set)
- Results: IoU_Ego=0.2842, IoU_Official=0.2710, IoU_Oracle=0.3451

**G4 Learned Fusion** (current):
- Evaluated on: 86 test samples (4 held-out scenes)
- Results: IoU_Ego=0.2671, IoU_Official=0.2526, IoU_Learned=0.2732, IoU_Oracle=0.3077

### Why Different Numbers?

1. **Different sample sets**: G2-B used full valid set; G4 uses scene-level held-out test
2. **Scene variability**: Test scenes may have different cooperation characteristics
3. **Sample size**: 549 vs 86 samples

**Conclusion**: Cannot directly compare absolute IoU values between G2-B and G4. Each evaluation serves different purposes:
- G2-B: Diagnosed Oracle headroom on full valid set
- G4: Evaluated Learned Fusion generalization on held-out test scenes

---

## Oracle Definition

### Patch-level Oracle (Current)

```
Accept patch R if: U_R > 0
Where: U_R = BA_R - HA_R
```

This is **NOT** a global optimal occupancy oracle. It is the upper bound for the specific accept/fallback framework with 4×4 patches.

### Naming in Paper

Use: **"Patch-level Oracle headroom recovery"**

Do NOT write: "Recovered 37.4% of theoretical optimal performance"

The Oracle here is constrained to the patch-level accept/fallback decision framework.

---

## Statistical Significance

### Current Status: NOT TESTED

**What we can say**:
- "Learned Fusion achieved consistent improvement over Official OR on held-out test scenes"
- "Observed IoU gain of +0.0206 (8.2% relative improvement)"
- "Learned Fusion surpassed both Official OR and Ego baseline"

**What we CANNOT say yet**:
- "Statistically significant improvement"
- "Significant performance gain"

### What's Needed for Significance

To claim statistical significance, need:
1. Multiple random seeds (e.g., 5-10 runs)
2. Bootstrap confidence intervals
3. Paired statistical test (e.g., paired t-test, Wilcoxon)

**Current decision**: Report as "observed improvement" without significance claim.

---

## Go/No-Go Criterion

### Primary Criterion (MET ✅)

```
IoU_Learned > IoU_Official
0.2732 > 0.2526 ✅
```

**Result**: Learned Fusion fixes negative cooperation.

### Secondary Criterion (MET ✅)

```
IoU_Learned > IoU_Ego
0.2732 > 0.2671 ✅
```

**Result**: Learned Fusion not only fixes negative cooperation but provides net benefit over single-vehicle perception.

### Oracle Gap Recovery

```
Recovery = (IoU_Learned - IoU_Official) / (IoU_Oracle - IoU_Official)
         = 0.0206 / 0.0551
         = 37.4%
```

**Interpretation**: Learned predictor recovers 37.4% of the patch-level Oracle headroom.

### Final Verdict

**✅ GO - Learned Selective Fusion Validated**

The core closed loop is complete:
```
Negative Cooperation → Oracle Headroom → Utility Learnability → Learned Selective Fusion
```

---

## Reproducibility

### Random Seeds
- Scene split: SEED=2026 (fixed from G3)
- Model training: G3 checkpoint (epoch 7, Val AUPRC=0.2595)
- Threshold selection: Deterministic grid search on Val

### Key Files
- Model: `G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth`
- Mapping: `G0-G1-G2/G4/sample_idx_to_export_idx.json`
- Results: `G0-G1-G2/G4/results/g4_final_results.json`
- Script: `G0-G1-G2/G4/scripts/evaluate_learned_fusion.py`

### Execution
```bash
cd /root/autodl-tmp/UniV2X/G0-G1-G2/G4/scripts
python evaluate_learned_fusion.py
```

**Runtime**: ~25 seconds (19 threshold × 2 splits)

---

## Next Steps

### G4-3: Mechanism Analysis (Recommended)

1. **Accept Rate Analysis**
   - Per-horizon breakdown (t_0 to t_4)
   - BA retention rate
   - HA admission rate

2. **Simple Baseline Comparison**
   - Confidence heuristic: Accept if P_i - P_v > δ
   - Scan δ values on Val
   - Compare vs Learned predictor on Test
   - Answer: "Why Value Learning vs simple probability threshold?"

3. **Failure Mode Analysis**
   - Which patches are falsely accepted? (HA admission)
   - Which patches are falsely rejected? (BA loss)
   - Horizon-specific patterns

### Beyond G4

1. **Statistical Validation**
   - Multi-seed training (5-10 runs)
   - Bootstrap confidence intervals
   - Statistical significance testing

2. **Full Pipeline Integration**
   - Integrate STCV-Occ with UniV2X
   - Freeze backbone, train new modules
   - Joint fine-tuning if needed

---

**Last Updated**: 2026-10-02  
**Project**: UniV2X - Vehicle-Infrastructure Cooperative Perception
