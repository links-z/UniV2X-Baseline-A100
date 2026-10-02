# G5-1: Offline-Online Consistency Verification

## 📋 Status

**Status**: ⬜ READY TO START
**Date**: 2026-10-02
**Phase**: G5-1 Numerical Consistency Check

---

## 🎯 Objective

Verify that online forward pass in UniV2X exactly reproduces G4 offline reconstruction results.

**Critical Question**: Does the online `STCVOcc` module produce identical numerical results to G3/G4 offline evaluation?

---

## 📊 Two-Level Verification Strategy

### Level 1: G5-1A Single Sample Exact Audit ⬜

**Purpose**: Debug and validate one complete sample from input to output

**Test Sample Selection**:
- Must be in `g3_neighborhood_test.npz`
- Preferably high candidate count (>100 patches)
- Example: Use first sample from G3 test split

**Layer-by-Layer Verification**:

```
Layer 0: Input Probabilities
├─ Pv (soft ego occupancy)
├─ Pi (soft aligned infra occupancy)
└─ Target: Already verified in G0 (should still check)

Layer 1: Binary Occupancy
├─ Ov = (Pv > 0.1)
├─ Oi = (Pi > 0.1)
└─ Target: Exact binary match

Layer 2: Candidate Extraction
├─ Candidate definition: ADD = (¬Ov ∧ Oi ∧ Mwarp)
├─ Traversal order: h → r → c
└─ Target: Exact (h,r,c) index match (G5-0A already verified on 86 samples)

Layer 3: 127-dim Features
├─ Local (80): 5-channel × 4×4 patch
├─ Horizon (5): One-hot encoding
├─ Spatial (2): Normalized (u,v)
├─ Neighborhood (40): 8 neighbors × 5 stats
└─ Target: max_abs_diff < 1e-5, mean_abs_diff < 1e-6

Layer 4: Logits
├─ zR = predictor(X_127)
└─ Target: max_abs_diff < 1e-5, mean_abs_diff < 1e-6

Layer 5: Scores
├─ sR = sigmoid(zR)
└─ Target: max_abs_diff < 1e-5, mean_abs_diff < 1e-6

Layer 6: Accept Decisions
├─ aR = 1[sR > 0.70]
└─ Target: EXACT MATCH (100%)

Layer 7: Learned Occupancy
├─ O_learned = selective_fusion(Ov, Oi, aR, candidates)
└─ Target: EXACT BINARY MATCH (all 5×200×200 voxels)
```

**G5-1A Pass Criteria**:
```
✅ All 7 layers pass their targets for the single test sample
```

---

### Level 2: G5-1B Full Test Set Consistency ⬜

**Purpose**: Validate consistency across all 86 held-out test samples

**Test Set**: G3 test split (86 samples, 4 held-out scenes)

**Aggregate Metrics**:

```
Sample-Level Metrics:
├─ Samples processed:              86 / 86
├─ Candidate index exact match:    86 / 86
├─ Accept decisions exact match:   86 / 86
└─ Learned occupancy exact match:  86 / 86

Patch-Level Metrics (14,750 total candidates):
├─ Feature max abs diff:           < 1e-5
├─ Feature mean abs diff:          < 1e-5
├─ Logit max abs diff:             < 1e-5
├─ Logit mean abs diff:            < 1e-5
├─ Score max abs diff:             < 1e-5
├─ Score mean abs diff:            < 1e-5
└─ Accept decisions exact:         14750 / 14750

Final IoU Metrics:
├─ G4 Offline Learned IoU:         0.2732
├─ G5 Online Learned IoU:          ?????
├─ Absolute IoU difference:        < 0.0001
└─ Relative IoU difference:        < 0.04%
```

**G5-1B Pass Criteria**:
```
✅ 14750/14750 accept decisions exact match
✅ 86/86 learned occupancy exact match
✅ Feature/logit/score within numerical tolerance
✅ Final IoU ≈ 0.2732 (±0.0001)
```

---

## 🔧 Implementation Approach

### Phase 1: G5-1A Single Sample (Expected: 2-4 hours)

**Step 1**: Select test sample from G3 test split
```python
# Load G3 test metadata
g3_test = np.load('G0-G1-G2/G3/soft/g3_neighborhood_test.npz')
sample_idx = g3_test['sample_idx'][0]  # First test sample
export_idx = SAMPLE_IDX_TO_EXPORT_IDX[sample_idx]
```

**Step 2**: Load G4 offline reference
```python
# G4 offline results for this sample
g4_accept = ...  # From G4 evaluation
g4_learned_occ = ...  # From G4 reconstruction
```

**Step 3**: Run online forward pass
```python
# UniV2X forward with STCV-Occ enabled
# Capture intermediate outputs at each layer
```

**Step 4**: Layer-by-layer comparison
```python
for layer in [candidates, features, logits, scores, accept, occupancy]:
    offline_result = g4_offline[layer]
    online_result = g5_online[layer]
    diff = compute_diff(offline_result, online_result)
    assert diff < tolerance, f"Layer {layer} mismatch"
```

**Step 5**: Debug any mismatches
- If candidates differ → Check traversal order, warp mask handling
- If features differ → Check patch extraction, neighborhood logic
- If logits differ → Check predictor weight loading, numerical precision
- If decisions differ → Check threshold application
- If occupancy differs → Check selective fusion logic

---

### Phase 2: G5-1B Full Test Set (Expected: 1-2 hours)

**Script**: `G5-1_consistency_sweep.py`

**Pseudocode**:
```python
results = {
    'samples_total': 0,
    'samples_passed': 0,
    'candidates_total': 0,
    'accept_exact_match': 0,
    'occupancy_exact_match': 0,
    'feature_diffs': [],
    'logit_diffs': [],
    'score_diffs': [],
    'iou_offline': [],
    'iou_online': [],
}

for sample_idx in g3_test_samples:
    # Load G4 offline reference
    offline_accept = ...
    offline_occ = ...

    # Run online forward
    online_accept = ...
    online_occ = ...

    # Compare
    results['candidates_total'] += len(offline_accept)
    results['accept_exact_match'] += (offline_accept == online_accept).sum()
    results['occupancy_exact_match'] += (offline_occ == online_occ).all()

    # Aggregate metrics
    ...

# Final report
print_consistency_report(results)
```

---

## 🎯 Success Criteria

### Hard Conditions (Must be 100% exact):
1. ✅ Candidate indices exact match
2. ✅ Accept decisions exact match (14750/14750)
3. ✅ Learned occupancy binary exact match (86/86)

### Soft Conditions (Numerical tolerance):
1. ✅ Feature max diff < 1e-5
2. ✅ Logit max diff < 1e-5
3. ✅ Score max diff < 1e-5

### Final Validation:
1. ✅ Online IoU ≈ Offline IoU (0.2732 ± 0.0001)

**Overall G5-1 PASS**:
```
IF all hard conditions exact match
AND all soft conditions within tolerance
AND final IoU matches
THEN G5-1 PASS → Proceed to G5-1B Frozen Evaluation
```

---

## 🚧 Expected Challenges

### Challenge 1: Floating Point Precision
**Issue**: GPU vs CPU, different PyTorch versions
**Mitigation**: Use relative tolerance, track max/mean diffs

### Challenge 2: Candidate Order
**Issue**: If traversal order differs, all downstream mismatches
**Mitigation**: G5-0A already verified, but double-check

### Challenge 3: Warp Mask Handling
**Issue**: Warp mask has no horizon dimension
**Mitigation**: Already handled in G5-0B, verified in G5-0A

### Challenge 4: Checkpoint Loading
**Issue**: State dict keys, device placement
**Mitigation**: G5-0B unit test already verified loading

### Challenge 5: Batch vs Single Processing
**Issue**: Different code paths for batch/single
**Mitigation**: G5-1A uses single sample, G5-1B processes as batch

---

## 📁 Required Files

### G4 Offline References
```
G0-G1-G2/G4/results/g4_final_results.json       # Final IoU metrics
G0-G1-G2/G3/soft/g3_neighborhood_test.npz       # Test sample metadata
G0-G1-G2/G3/results_soft/*.json                 # Per-sample offline results
```

### G5 Online Code
```
projects/mmdet3d_plugin/univ2x/dense_heads/stcv_occ.py   # STCV module
projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py   # Integration
projects/configs_e2e_univ2x/univ2x_coop_e2e.py           # Config
```

### G5-1 Scripts (to be created)
```
G0-G1-G2/G5/G5-1_consistency_check/
├── run_g5_1a_single_sample.py          # Single sample audit
├── run_g5_1b_full_sweep.py             # 86 samples sweep
└── consistency_utils.py                # Comparison utilities
```

---

## 📊 Expected Output

### G5-1A Output (Single Sample)
```
========================================
G5-1A: Single Sample Consistency Check
========================================

Sample: sample_idx=2889, export_idx=74
Candidates: 156 patches

Layer 0: Input Probabilities
  ✅ Pv shape: (5, 200, 200)
  ✅ Pi shape: (5, 200, 200)

Layer 1: Binary Occupancy
  ✅ Ov exact match
  ✅ Oi exact match

Layer 2: Candidate Extraction
  ✅ 156 candidates extracted
  ✅ (h,r,c) indices exact match

Layer 3: 127-dim Features
  ✅ Feature shape: (156, 127)
  ✅ Max abs diff: 3.45e-7
  ✅ Mean abs diff: 1.23e-8

Layer 4: Logits
  ✅ Max abs diff: 5.67e-7
  ✅ Mean abs diff: 2.34e-8

Layer 5: Scores
  ✅ Max abs diff: 4.89e-7
  ✅ Mean abs diff: 1.89e-8

Layer 6: Accept Decisions
  ✅ 156/156 exact match
  ✅ Accept count: 28 (offline), 28 (online)

Layer 7: Learned Occupancy
  ✅ Binary exact match: 5000/5000 voxels

========================================
G5-1A: PASS ✅
========================================
```

### G5-1B Output (Full Test Set)
```
========================================
G5-1B: Full Test Set Consistency Check
========================================

Samples: 86 / 86
Total candidates: 14750

Hard Conditions:
  ✅ Candidate indices:      14750 / 14750 exact
  ✅ Accept decisions:       14750 / 14750 exact
  ✅ Learned occupancy:      86 / 86 exact

Soft Conditions:
  ✅ Feature max diff:       8.92e-7
  ✅ Feature mean diff:      1.45e-8
  ✅ Logit max diff:         7.34e-7
  ✅ Logit mean diff:        2.01e-8
  ✅ Score max diff:         6.78e-7
  ✅ Score mean diff:        1.89e-8

Final IoU:
  Offline (G4):  0.2732
  Online (G5):   0.2732
  Absolute diff: 0.0000
  Relative diff: 0.00%

========================================
G5-1B: PASS ✅
========================================

✅ All hard conditions satisfied
✅ All soft conditions within tolerance
✅ Offline-online consistency verified

Ready for G5-1B 86-sample Sequential Consistency Sweep
```

---

## 🚀 Next Steps After G5-1 PASS

### G5-2: Frozen Official Evaluation
- Run frozen G3 predictor on full UniV2X benchmark
- Compare: Official OR vs STCV-Occ (frozen)
- Question: Does offline +8.2% transfer to online evaluation?

### G5-3: Fine-tuning (Conditional)
- If G5-2 shows modest improvement
- Freeze backbone, train predictor only
- Optimize for end-to-end IoU

---

## 📝 Implementation Checklist

### G5-1A Preparation
- [ ] Load G3 test metadata
- [ ] Select single test sample
- [ ] Load G4 offline reference for this sample
- [ ] Implement layer-by-layer comparison utilities
- [ ] Run single sample audit
- [ ] Debug any mismatches
- [ ] Document findings

### G5-1B Execution
- [ ] Implement full sweep script
- [ ] Process all 86 test samples
- [ ] Aggregate metrics
- [ ] Generate consistency report
- [ ] Verify all pass criteria
- [ ] Update G5_STATUS.md

---

*G5-1 Plan Complete - Ready to Execute*
*Date: 2026-10-02*
*Next: Run G5-1A Single Sample Audit*
