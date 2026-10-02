# G4 Core Figures - Complete Index

## Overview

All 5 core figures for the second paper have been generated and saved in `/root/autodl-tmp/UniV2X/G0-G1-G2/G4/figures/`.

**Status**: ✅ COMPLETE  
**Date**: 2026-10-02  
**Formats**: PDF (publication), PNG (preview), SVG (editable, where applicable)

---

## Figure List

### Fig.1: STCV-Occ Overall Framework ⭐
**Files**: `fig1_stcv_occ_framework.{pdf,png,svg}`  
**Size**: 42KB (PDF), 465KB (PNG), 120KB (SVG)  
**Purpose**: Main method figure showing complete pipeline  

**Content**:
- Dual inputs (Ego Vehicle + Infrastructure)
- Occupancy prediction networks
- Spatial alignment
- **Cooperation Utility Predictor (core innovation)**
  - 4 feature groups: Local Soft Evidence, Horizon Encoding, Spatial Position, Neighborhood Context
  - MLP architecture (127→256→128→64→1)
  - Utility score sR ∈ [0,1]
- Accept/Reject decision (aR = 1[sR > τ*])
- Selective fusion branches
- Final fused occupancy

**Key message**: Complete utility-guided selective fusion framework from dual inputs to final output

---

### Fig.2: G1 Negative Cooperation Diagnosis
**Files**: `fig2_g1_negative_cooperation.{pdf,png}`  
**Size**: 32KB (PDF), 257KB (PNG)  
**Purpose**: Motivate the problem - why Official OR fails  

**Content**:
- **(a) Infrastructure Additions Composition**: Pie chart showing 14.9% BA (beneficial) vs 85.1% HA (harmful)
- **(b) Negative Cooperation Phenomenon**: Bar chart comparing Ego (0.2671), Official OR (0.2526), showing -5.4% drop
- **(c) Why Official OR Fails**: Conceptual diagram showing HA dominates (85%) when accepting all additions

**Key message**: Official OR hurts performance due to 85% harmful additions

---

### Fig.3: G2 Oracle Headroom Analysis
**Files**: `fig3_g2_oracle_headroom.{pdf,png}`  
**Size**: 32KB (PDF), 226KB (PNG)  
**Purpose**: Show potential of selective fusion  

**Content**:
- IoU progression: Ego (0.2671) → Official OR (0.2526) → Patch-16 (0.2650) → Patch-8 (0.2850) → Patch-4 (0.3077) → Cell Oracle (0.3300)
- Negative cooperation region highlighted
- Oracle headroom region (0.2526 → 0.3077) shaded in green
- Learned Fusion line at 0.2732 showing 37.4% recovery of P4 Oracle headroom

**Key message**: Selective fusion has significant potential (P4 Oracle: 0.3077), learned predictor recovers 37.4%

---

### Fig.4: G3 Cooperation Utility Predictor Architecture ⭐
**Files**: `fig4_g3_predictor_architecture.{pdf,png,svg}`  
**Size**: 35KB (PDF), 243KB (PNG), 81KB (SVG)  
**Purpose**: Core method - predictor design  

**Content**:
- **Feature Groups** (left):
  1. Local Soft Evidence (80-dim): Pv, Pi, ΔP, |ΔP|, Warp
  2. Horizon Encoding (5-dim): t=0,1,2,3,4
  3. Spatial Position (2-dim): u, v (normalized)
  4. Neighborhood Context (40-dim): 8 neighbors × 5 stats
- **Concatenation**: 127-dim feature vector
- **MLP Layers** (right):
  - 127 → 256 (ReLU + Dropout 0.1)
  - 256 → 128 (ReLU + Dropout 0.1)
  - 128 → 64 (ReLU + Dropout 0.1)
  - 64 → 1 (Sigmoid)
- **Output**: sR ∈ [0,1] (Cooperation Utility Score)
- **Total Parameters**: 73,985

**Key message**: Lightweight MLP with 4 diverse feature groups learns cooperation utility

---

### Fig.5: G4 Learned Fusion Results ⭐
**Files**: `fig5_g4_learned_fusion.{pdf,png}`  
**Size**: 26KB (PDF), 248KB (PNG)  
**Purpose**: Main results - learned fusion outperforms baselines  

**Content**:
- **(a) Downstream Fusion Performance**: Bar chart showing
  - Official OR: 0.2526
  - Ego: 0.2671
  - **Learned: 0.2732** (+8.2% vs Official)
  - Oracle: 0.3077
- **(b) Decision Quality Comparison**: Side-by-side comparison
  - BA Retention: Learned 37.75% vs Confidence 38.18% (similar)
  - HA Admission: **Learned 15.84%** vs Confidence 22.72% (-6.88%, **30% reduction**)

**Key message**: Learned > Ego > Official, with 30% lower harmful patch admission vs confidence baseline

---

## Usage in Paper

### Main Text Figures (Priority Order)
1. **Fig.1** - Introduction/Method overview
2. **Fig.5** - Main results (most important for convincing readers)
3. **Fig.4** - Method details (predictor architecture)
4. **Fig.2** - Problem motivation
5. **Fig.3** - Oracle analysis (can be supplementary if space limited)

### Figure Quality
- **PDF files**: Use for LaTeX submission (vector graphics, scalable)
- **PNG files**: Use for Word/PowerPoint (300 DPI, high resolution)
- **SVG files**: Use for further editing if needed (Fig.1, Fig.4 only)

### Caption Guidelines

**Fig.1 Caption (suggested)**:
> "Overview of STCV-Occ: Selective Temporal Cooperative V2X Occupancy Prediction. The framework takes dual inputs from ego vehicle and infrastructure, predicts occupancy independently, aligns infrastructure predictions spatially, then uses a learned Cooperation Utility Predictor to make patch-level accept/reject decisions. The predictor combines 4 feature groups (127-dim) through a lightweight MLP to output utility scores sR ∈ [0,1]. Patches with sR > τ* are accepted for fusion; otherwise ego-only prediction is kept."

**Fig.5 Caption (suggested)**:
> "Learned Fusion Results on Test Set (86 samples, 4 held-out scenes). (a) Downstream fusion performance: Learned fusion achieves IoU 0.2732, outperforming both Official OR (0.2526, +8.2%) and Ego baseline (0.2671, +2.3%), recovering 37.4% of patch-level Oracle headroom. (b) Decision quality: Compared to confidence baseline (Pi - Pv > δ) at matched selectivity, learned predictor maintains similar BA retention (37.75% vs 38.18%) but achieves 30% lower HA admission rate (15.84% vs 22.72%), demonstrating superior precision."

---

## Next Steps

### Before Submission
- [x] Generate all core figures
- [ ] Review figure quality with co-authors
- [ ] Finalize figure captions
- [ ] Check figure ordering in paper flow
- [ ] Generate supplementary figures if needed:
  - Precision-recall curves
  - Per-horizon performance breakdown
  - Qualitative examples (Accept vs Reject cases)
  - Failure case analysis

### Optional Enhancements
- [ ] Add error bars (requires multi-seed training)
- [ ] Generate higher resolution versions if journal requires
- [ ] Create animated version of Fig.1 for presentation
- [ ] Design graphical abstract based on Fig.1

---

## Files for Reproducibility

All figure generation scripts are saved in `scripts/`:

```bash
# Generate all figures at once
cd /root/autodl-tmp/UniV2X/G0-G1-G2/G4/scripts

# Fig.4 and Fig.5
python generate_figures.py

# Fig.2 and Fig.3
python generate_fig2_fig3.py

# Fig.1
python generate_fig1.py
```

**Runtime**: ~5 seconds total for all figures

---

## Design Notes

### Visual Style
- **Color palette**: 
  - Ego: Blue (#3b82f6, #1e40af)
  - Infrastructure: Green (#10b981, #059669)
  - Accept: Green (#10b981)
  - Reject: Red (#ef4444)
  - Oracle/Optimal: Dark green (#059669)
  - Learned: Blue (#3b82f6)
  - Warning/Attention: Orange/Yellow (#f59e0b, #fbbf24)
  
- **Typography**: Arial/Helvetica/DejaVu Sans, 7-13pt depending on element
- **Layout**: Clean, professional, publication-ready
- **Consistency**: Matched color scheme across all figures

### Technical Details
- DPI: 300 (publication quality)
- Font embedding: Type 42 (ensures compatibility)
- Border widths: 1.5-2.5pt for emphasis
- Alpha transparency: 0.15-0.85 for layering

---

*All figures complete and ready for paper writing!*  
*Date: 2026-10-02*  
*Project: STCV-Occ - UniV2X Selective Temporal Cooperative Perception*
