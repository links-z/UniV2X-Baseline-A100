# G5: STCV-Occ Online Integration

## 📊 Quick Status

**Phase**: G5-0 COMPLETE ✅  
**Next**: G5-1A Single Sample Exact Audit (READY)  
**Date**: 2026-10-02

```
✅ G5-0A: Candidate Audit          (86/86 exact match)
✅ G5-0B: Online Module             (478 lines, 6/6 tests)
✅ G5-0C: occ_head Integration      (4 modifications, 0 breaking)
⬜ G5-1A: Single Sample Audit       (READY)
⬜ G5-1B: Full Test Consistency     (PENDING)
⬜ G5-2:  Frozen Evaluation         (PENDING)
```

---

## 📁 File Structure

```
G0-G1-G2/G5/
├── README.md                    # This file - Quick guide
├── G5_STATUS.md                 # Detailed progress tracker
├── G5-0_COMPLETE_REPORT.md      # G5-0 final report
│
├── G5-0_online_integration/
│   ├── G5-0B_COMPLETE.md        # Module implementation details
│   └── test_stcv_occ.py         # Unit tests (6 tests)
│
├── G5-0C_occ_head_integration/
│   └── G5-0C_COMPLETE.md        # Integration guide
│
└── G5-1_consistency_check/
    └── G5-1_PLAN.md             # Next phase detailed plan
```

**Code Files**:
```
projects/mmdet3d_plugin/univ2x/dense_heads/
├── stcv_occ.py                  # STCV-Occ module (478 lines)
└── occ_head.py                  # Integration (4 modifications)
```

---

## 🎯 G5-0 Achievements

### G5-0A: Candidate Definition Audit ✅
- Verified candidate extraction matches G3 offline definition
- **Result**: 86/86 samples exact match (14,750 candidates)
- **Key finding**: Warp mask has no horizon dimension

### G5-0B: STCV Online Module ✅
- **Implementation**: `stcv_occ.py` (478 lines)
- **Architecture**: 73,985 parameters (exact G3 match)
- **Unit tests**: 6/6 passed
- **Components**:
  - NeighborhoodMLP (predictor)
  - STCVFeatureExtractor (127-dim)
  - STCVOcc (main module)
  - selective_fusion (patch-level)

### G5-0C: occ_head Integration ✅
- **Modifications**: 4 targeted changes to `occ_head.py`
- **Breaking changes**: 0
- **Safety checks**: 3 validation checks
- **Design**: Minimal invasiveness, toggle-able

---

## 🚀 Quick Start

### View Current Status
```bash
cat G0-G1-G2/G5/G5_STATUS.md
```

### View G5-0 Complete Report
```bash
cat G0-G1-G2/G5/G5-0_COMPLETE_REPORT.md
```

### View Implementation Details
```bash
# Module implementation
cat G0-G1-G2/G5/G5-0_online_integration/G5-0B_COMPLETE.md

# Integration guide
cat G0-G1-G2/G5/G5-0C_occ_head_integration/G5-0C_COMPLETE.md
```

### View Next Phase Plan
```bash
cat G0-G1-G2/G5/G5-1_consistency_check/G5-1_PLAN.md
```

### Run Unit Tests
```bash
cd G0-G1-G2/G5/G5-0_online_integration
python test_stcv_occ.py
```

---

## 🔑 Key Configuration

**Config file**: `projects/configs_e2e_univ2x/univ2x_coop_e2e.py`

**STCV parameters** (add to second OccHead only):
```python
use_stcv_occ=True,
stcv_checkpoint_path='/root/autodl-tmp/UniV2X/G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth',
stcv_threshold=0.70,
```

**Note**: Only enable on cooperative ego agent, NOT on infrastructure agent.

---

## 📊 What We Can Say (After G5-0)

### ✅ Safe Claims
- "Implemented online STCV-Occ module consistent with G3 method definition"
- "Passed functional unit tests with exact architecture match (73,985 params)"
- "Integration preserves all existing logic with zero breaking changes"
- "Candidate extraction verified across 86 samples with 100% exact match"

### ❌ Avoid (Before G5-1 PASS)
- "Online implementation numerically identical to offline evaluation"
- "Online forward pass reproduces G4 results exactly"
- "Offline +8.2% improvement validated in online benchmark"

### ✅ After G5-1 PASS (Expected)
- "Online forward pass precisely reproduced G4 offline results"
- "14,750 accept decisions matched 100% across 86 test samples"
- "Final IoU consistent with offline evaluation (0.2732)"

---

## 🎯 Next Phase: G5-1 Offline-Online Consistency

### G5-1A: Single Sample Exact Audit (READY)
**Objective**: Debug and validate one complete sample layer-by-layer

**Verification layers**:
1. Pv, Pi (input probabilities)
2. Ov, Oi (binary occupancy)
3. Candidate indices (h,r,c)
4. 127-dim features
5. Logits (zR)
6. Scores (sR)
7. Accept decisions (aR)
8. Learned occupancy (O_learned)

**Pass criteria**: All 8 layers pass their respective targets

---

### G5-1B: Full Test Set Consistency (PENDING G5-1A)
**Objective**: Validate consistency across 86 test samples

**Hard conditions** (must be 100% exact):
- 14750/14750 accept decisions exact match
- 86/86 learned occupancy exact match

**Soft conditions** (numerical tolerance):
- Feature/logit/score max_diff < 1e-5

**Final validation**:
- Online IoU ≈ 0.2732 (±0.0001)

---

### G5-2: Frozen Official Evaluation (CONDITIONAL)
**Condition**: G5-1 PASS

**Objective**: Test whether offline +8.2% transfers to online benchmark

**Key question**: Does G4's offline improvement translate to online?

**Note**: Cannot assume transfer; G5-2 is the actual validation

---

## 📈 Overall Progress

```
✅ G0: Cache Generation
✅ G1: Negative Cooperation Diagnosis
✅ G2: Oracle Headroom Analysis
✅ G3: Utility Learnability Training
    - Val AUPRC: 0.2595
    - Test AUPRC: 0.2712
    - Test AUROC: 0.7035
✅ G4: Learned Fusion Validation
    - Offline IoU: 0.2732 (+8.2% vs Official OR)
    - Oracle gap recovery: 37.4%
✅ G5-0A: Candidate Audit (86/86 exact)
✅ G5-0B: Online Module (6/6 tests)
✅ G5-0C: Integration (4 modifications)
⬜ G5-1A: Single Sample Audit (READY)
⬜ G5-1B: Full Test Consistency
⬜ G5-2: Frozen Evaluation
⬜ G5-3: Fine-tuning (conditional)
```

**Completion**: 8/12 major phases (67%)

---

## 🔧 Git Status

**Branch**: `agent-fusion-improved`

**Modified**:
- `projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py`

**New**:
- `projects/mmdet3d_plugin/univ2x/dense_heads/stcv_occ.py`
- `G0-G1-G2/G5/` (documentation and tests)

**Ready to commit**: Yes (all pre-commit checks passed)

---

## 📚 Key References

### Documentation
- `G5_STATUS.md` - Detailed progress tracker
- `G5-0_COMPLETE_REPORT.md` - G5-0 comprehensive report
- `G5-0B_COMPLETE.md` - Module implementation guide
- `G5-0C_COMPLETE.md` - Integration guide
- `G5-1_PLAN.md` - Next phase detailed plan

### G3/G4 Results
- G3 checkpoint: `G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth`
- G3 test data: `G0-G1-G2/G3/soft/g3_neighborhood_test.npz`
- G4 results: `G0-G1-G2/G4/results/g4_final_results.json`

### Cache
- G0 cache: `G0-G1-G2/cache_full/sample_*.npz`

---

## ✅ Design Principles

1. **Minimal Invasiveness** - Only 4 occ_head.py modifications
2. **Clear Separation** - stcv_occ.py standalone module
3. **Safety First** - 3 validation checks prevent misuse
4. **Baseline Preservation** - Official OR always computed
5. **Toggle-able** - Single config flag enables/disables
6. **Well-Tested** - 6/6 unit tests passed

---

*Last Updated: 2026-10-02*  
*G5-0 Status: COMPLETE ✅*  
*Next: G5-1A Single Sample Exact Audit*
