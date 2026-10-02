# G4: Learned Selective Fusion - Complete Validation

## 🎯 Mission

Validate that **learned cooperation utility** can guide selective fusion to improve downstream occupancy prediction beyond both Official OR and Ego-only baselines.

## ✅ Status: COMPLETE

**Go/No-Go Verdict**: ✅ **GO**

All objectives validated:
- ✅ IoU_Learned (0.2732) > IoU_Official (0.2526) — Fixes negative cooperation (+8.2%)
- ✅ IoU_Learned (0.2732) > IoU_Ego (0.2671) — Provides net benefit (+2.3%)
- ✅ Recovers 37.4% of patch-level Oracle headroom
- ✅ Superior to confidence baseline (30% lower harmful patch admission)
- ✅ Generalizes to held-out test scenes

---

## 📊 Key Results (Test Set, 86 samples)

| Strategy | IoU | vs Official | vs Ego |
|----------|-----|-------------|--------|
| Official OR | 0.2526 | — | -5.4% |
| Ego | 0.2671 | +5.7% | — |
| **Learned** | **0.2732** | **+8.2%** | **+2.3%** |
| Oracle (P4) | 0.3077 | +21.8% | +15.2% |

**Oracle Gap Recovery**: 37.4% = (0.2732 - 0.2526) / (0.3077 - 0.2526)

---

## 🔬 Core Scientific Loop (Complete)

```
Negative Cooperation → Oracle Headroom → Utility Learnability → Learned Fusion
       (G1)                  (G2)              (G3)                  (G4)
```

**G1**: Identified negative cooperation (Official OR < Ego)  
**G2**: Quantified patch-level Oracle headroom (IoU 0.3077 vs 0.2526)  
**G3**: Proved cooperation utility is learnable (Val AUPRC 0.2595)  
**G4**: Validated learned utility improves downstream fusion (+8.2% IoU)

---

## 📁 Project Structure

```
G0-G1-G2/G4/
├── README.md                          # This file
├── G4_STATUS.sh                       # Quick status script
├── G4_COMPLETE.md                     # Detailed completion report
├── G4_RESULTS_SUMMARY.md              # Executive summary
├── G4_METHODOLOGY.md                  # Complete methodology
├── G4_MECHANISM_ANALYSIS.md           # Why Value Learning > confidence
├── FIGURES_INDEX.md                   # Figure documentation
│
├── results/
│   ├── g4_final_results.json          # Val/Test strategy comparison
│   ├── g4_threshold_scan_val.json     # Threshold selection (τ*=0.70)
│   └── g4_mechanism_analysis.json     # Per-horizon + baseline analysis
│
├── figures/
│   ├── fig1_stcv_occ_framework.*      # Overall method (PDF/PNG/SVG)
│   ├── fig2_g1_negative_cooperation.* # Problem motivation
│   ├── fig3_g2_oracle_headroom.*      # Oracle analysis
│   ├── fig4_g3_predictor_architecture.* # Method details
│   └── fig5_g4_learned_fusion.*       # Main results
│
├── scripts/
│   ├── evaluate_learned_fusion.py     # G4-0/1/2 main evaluation
│   ├── analyze_mechanism.py           # G4-3 mechanism analysis
│   ├── fair_comparison.py             # Baseline comparison
│   ├── generate_figures.py            # Fig.4 & Fig.5
│   ├── generate_fig2_fig3.py          # Fig.2 & Fig.3
│   └── generate_fig1.py               # Fig.1
│
└── sample_idx_to_export_idx.json      # Index mapping for cache loading
```

---

## 🚀 Quick Start

### View Status
```bash
bash G4_STATUS.sh
```

### Reproduce Evaluation
```bash
cd scripts/

# Main evaluation (G4-0/1/2)
python evaluate_learned_fusion.py    # ~25s

# Mechanism analysis (G4-3)
python analyze_mechanism.py           # ~5s

# Baseline comparison
python fair_comparison.py             # ~1s
```

### Regenerate Figures
```bash
cd scripts/

# All figures
python generate_figures.py            # Fig.4 & Fig.5
python generate_fig2_fig3.py          # Fig.2 & Fig.3
python generate_fig1.py               # Fig.1
```

---

## 📖 Documentation

### For Paper Writing
- **G4_RESULTS_SUMMARY.md** - Executive summary with all key numbers
- **G4_METHODOLOGY.md** - Complete methodology (threshold selection, metrics)
- **G4_MECHANISM_ANALYSIS.md** - Why learned > confidence baseline
- **FIGURES_INDEX.md** - Figure descriptions and captions

### For Development
- **G4_COMPLETE.md** - Full completion report with:
  - What we can/cannot say (statistical significance, terminology)
  - Known limitations
  - Comparison with G2-B Oracle
  - Review Q&A
  - Files for reproducibility

---

## 🎨 Core Figures (Publication-Ready)

All figures saved in `figures/` in PDF (vector), PNG (300 DPI), and SVG (editable) formats:

1. **Fig.1**: Overall STCV-Occ Framework ⭐
2. **Fig.2**: G1 Negative Cooperation Diagnosis
3. **Fig.3**: G2 Oracle Headroom Analysis
4. **Fig.4**: G3 Cooperation Utility Predictor Architecture ⭐
5. **Fig.5**: G4 Learned Fusion Results ⭐

See `FIGURES_INDEX.md` for detailed descriptions and suggested captions.

---

## 🔑 Key Contributions

1. **Beyond Prediction to Impact**: G3 showed utility is learnable; G4 proves it improves fusion
2. **Fixes Negative Cooperation**: Official OR (0.2526) → Learned (0.2732)
3. **Net Benefit Over Single-Vehicle**: Ego (0.2671) → Learned (0.2732)
4. **Generalizes to Held-Out Scenes**: Test scenes completely unseen during training
5. **Superior to Simple Heuristics**: 30% reduction in harmful patches vs confidence baseline

---

## 📐 Methodology Highlights

### Threshold Selection (G4-1)
- Scanned τ = 0.05 to 0.95 on **Validation set**
- Selection criterion: **Maximize Val IoU** (downstream metric, NOT classification F1)
- Optimal: **τ* = 0.70**
- **Frozen for Test evaluation** (no tuning on test)

### Evaluation Protocol (G4-2)
- Test set: 86 samples from 4 held-out scenes
- Four strategies compared: Official OR, Ego, Learned (τ*=0.70), Oracle
- Metrics: IoU, Precision, Recall, F1 (macro-averaged)

### Mechanism Analysis (G4-3)
- Per-horizon breakdown (t=0 to t=4)
- BA Retention (beneficial patch recall)
- HA Admission (harmful patch false positive rate)
- Fair comparison with confidence baseline at matched operating points

---

## 🎯 Main Claims (Safe for Paper)

✅ **Safe to say**:
- "Learned Fusion achieved consistent improvement over Official OR on held-out test scenes"
- "Observed IoU gain of +0.0206 (8.2% relative improvement)"
- "Recovered 37.4% of patch-level Oracle headroom"
- "Demonstrates superior precision over simple confidence heuristics"

❌ **Avoid without more work**:
- ~~"Statistically significant improvement"~~ (need multi-seed + tests)
- ~~"Theoretical optimal"~~ (Oracle is patch-level, not global)

See `G4_COMPLETE.md` for full guidance.

---

## 🔮 Future Work

### Before Submission (Recommended)
- [ ] Multi-seed training (5-10 runs) for significance testing
- [ ] Bootstrap confidence intervals
- [ ] Precision-recall curves (Learned vs Confidence)
- [ ] Qualitative examples (success/failure cases)

### Follow-up Research
- [ ] Better architecture: CNN encoder for patches
- [ ] Multi-scale patches (4×4, 8×8, 16×16)
- [ ] Temporal modeling: LSTM/Transformer across horizons
- [ ] Full STCV-Occ + UniV2X integration (G5)

---

## 📚 References

### Related Work
- **G0**: Cache preparation (G0_CACHE_EXPORT.md)
- **G1**: Negative cooperation diagnosis (G1_NEGATIVE_COOPERATION.md)
- **G2**: Oracle headroom analysis (G2_ORACLE_HEADROOM.md)
- **G3**: Utility learnability (G3_LEARNABILITY_RESULTS.md)

### Dependencies
- G3 best model: `../G3/checkpoints_soft/g3_neighborhood_best.pth`
- G3 datasets: `../G3/soft/g3_neighborhood_{train,val,test}.npz`
- G0 cache: `../G0/occupancy_cache/sample_{export_idx}.npz`

---

## 📝 Citation (Placeholder)

```bibtex
@article{stcv-occ-2026,
  title={STCV-Occ: Selective Temporal Cooperative V2X Occupancy Prediction via Learned Utility-Guided Fusion},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2026},
  note={Learned selective fusion achieves 8.2\% IoU improvement over baseline cooperation}
}
```

---

## 💡 Contact

For questions about:
- **Methodology**: See G4_METHODOLOGY.md
- **Results interpretation**: See G4_RESULTS_SUMMARY.md
- **Why learned > confidence**: See G4_MECHANISM_ANALYSIS.md
- **Figures**: See FIGURES_INDEX.md
- **Reproducibility**: See G4_COMPLETE.md

---

*Second Paper Core Validation: COMPLETE ✅*  
*Date: 2026-10-02*  
*Project: UniV2X - Vehicle-Infrastructure Cooperative Perception*
