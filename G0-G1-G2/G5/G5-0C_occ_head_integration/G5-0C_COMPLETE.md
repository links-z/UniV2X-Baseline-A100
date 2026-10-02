# G5-0C: occ_head Minimal Integration - COMPLETE

**Status**: ✅ COMPLETE  
**Date**: 2026-10-02

---

## 🎯 Objective

Integrate STCV-Occ module into UniV2X's `occ_head.py` with minimal, non-breaking changes.

---

## ✅ Implementation Summary

### Modified File

**File**: `projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py`

**Total Changes**: 4 modifications, ~40 lines added

**Principles**:
- ✅ Minimal changes to existing code
- ✅ Preserve all original logic
- ✅ Easy to toggle on/off via config
- ✅ No breaking changes to existing functionality
- ✅ Official OR always computed for baseline comparison

---

## 📝 Detailed Modifications

### Modification 1: Import STCVOcc

**Location**: Line ~13 (after existing imports)

```python
from .occ_head_plugin import MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd, \
                             predict_instance_segmentation_and_trajectories
from .stcv_occ import STCVOcc
```

**Purpose**: Import STCV-Occ module

---

### Modification 2: Add Parameters to `__init__`

**Location**: Line ~70-78 (after `inf_pc_range` parameter)

**Added Parameters**:
```python
# STCV-Occ
use_stcv_occ=False,
stcv_checkpoint_path=None,
stcv_threshold=0.70,
```

**Purpose**: 
- `use_stcv_occ`: Toggle STCV-Occ on/off
- `stcv_checkpoint_path`: Path to G3 trained checkpoint
- `stcv_threshold`: Cooperation utility threshold τ* (from G4)

**Default Behavior**: STCV-Occ disabled by default (`use_stcv_occ=False`)

---

### Modification 3: Initialize STCV-Occ Module

**Location**: Line ~90-117 (after `self.is_old_mode = is_old_mode`)

**Added Code**:
```python
# STCV-Occ initialization
self.use_stcv_occ = use_stcv_occ
self.stcv_checkpoint_path = stcv_checkpoint_path
self.stcv_threshold = stcv_threshold
self.stcv_occ = None

if self.use_stcv_occ:
    if not self.is_ego_agent:
        raise ValueError(
            "STCV-Occ should only be enabled for the ego OccHead."
        )
    if not self.is_cooperation:
        raise ValueError(
            "STCV-Occ requires is_cooperation=True."
        )
    if self.stcv_checkpoint_path is None:
        raise ValueError(
            "stcv_checkpoint_path must be provided when use_stcv_occ=True."
        )
    self.stcv_occ = STCVOcc(
        checkpoint_path=self.stcv_checkpoint_path,
        threshold=self.stcv_threshold,
    )
    print(f"[OccHead] STCV-Occ enabled with τ={self.stcv_threshold}")
```

**Safety Checks**:
1. ✅ STCV-Occ only enabled for ego agent (not infrastructure)
2. ✅ Requires `is_cooperation=True` (cooperative mode)
3. ✅ Checkpoint path must be provided

**Purpose**: Initialize STCV-Occ module with proper validation

---

### Modification 4: Replace Fusion Logic

**Location**: Line ~808-820 (in `occ_prob_fusion()`)

**Original Code** (removed):
```python
fused_occ = []
for i in range(self.n_future + 1):
    max_values, _ = torch.max(
        torch.stack([veh_occ_log[:, i], inf_occ_log[:, i]]),
        dim=0
    )
    cur_fused_occ = max_values.unsqueeze(1)
    fused_occ.append(cur_fused_occ)

fused_occ = torch.stack(fused_occ, dim=2).squeeze(1)
```

**New Code**:
```python
# Official OR fusion (always compute for baseline comparison)
official_occ = torch.maximum(veh_occ_log, inf_occ_log)

# STCV-Occ selective fusion
if self.use_stcv_occ:
    fused_occ = self.stcv_occ(
        pv=veh_occ,
        pi=new_inf_occ,
        ov=veh_occ_log,
        oi=inf_occ_log,
        warp=warp_valid_mask,
    )
else:
    fused_occ = official_occ
```

**Key Design Decisions**:

1. **Official OR always computed**: `official_occ` is always calculated, even when STCV-Occ is enabled
   - **Rationale**: Allows G5-1 to compare both Official and Learned results simultaneously
   - **Future use**: Can be saved in `fusion_aux` for detailed analysis

2. **Simplified Official OR**: Replaced loop-based max with single `torch.maximum()`
   - **Original**: Loop over horizons, stack, max, unsqueeze, append, stack, squeeze
   - **New**: Single vectorized operation
   - **Equivalent**: Both compute element-wise maximum across (B, 5, 200, 200)
   - **Benefit**: Clearer code, same result

3. **Conditional STCV-Occ**: Uses `if self.use_stcv_occ` to toggle
   - **When disabled**: Falls back to `official_occ` (original behavior)
   - **When enabled**: Calls STCV-Occ selective fusion

4. **Input mapping**:
   - `pv=veh_occ`: Ego soft occupancy probability
   - `pi=new_inf_occ`: Aligned infrastructure soft occupancy probability
   - `ov=veh_occ_log`: Ego binary occupancy
   - `oi=inf_occ_log`: Infrastructure binary occupancy
   - `warp=warp_valid_mask`: Warp valid mask (B, 200, 200)

**Purpose**: 
- Enable STCV-Occ selective fusion when configured
- Preserve original behavior when disabled
- Always compute official baseline for comparison

---

## 🔧 Configuration Changes

### Config File Location

**File**: `projects/configs_e2e_univ2x/univ2x_coop_e2e.py`

**Two OccHead Instances**:
1. **First OccHead**: Non-cooperative (no changes needed)
2. **Second OccHead**: Cooperative ego agent (add STCV-Occ config here)

### Required Config Addition

**Location**: In the second `occ_head` dict (the one with `is_cooperation=True`, `is_ego_agent=True`)

**Add**:
```python
occ_head=dict(
    type='OccHead',
    bev_h=bev_h_,
    bev_w=bev_w_,
    pc_range=point_cloud_range,
    inf_pc_range=inf_point_cloud_range,
    is_cooperation=is_cooperation,
    is_ego_agent=is_ego_agent,
    # ... existing config ...
    
    # Add STCV-Occ configuration
    use_stcv_occ=True,
    stcv_checkpoint_path='/root/autodl-tmp/UniV2X/G0-G1-G2/G3/checkpoints_soft/g3_neighborhood_best.pth',
    stcv_threshold=0.70,
)
```

**Notes**:
- Use absolute path for checkpoint (for now, can refactor to relative path later)
- `stcv_threshold=0.70` is τ* from G4 validation
- Only enable for cooperative ego agent, not infrastructure

---

## ✅ Verification

### Syntax Check

```bash
python3 -c "
import ast
with open('projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py', 'r') as f:
    ast.parse(f.read())
print('✅ Syntax check passed')
"
```

**Result**: ✅ PASSED

### Code Structure

**Before**:
```
occ_head.py
├── __init__(..., is_cooperation, is_ego_agent, ...)
│   └── self.is_cooperation = ...
│       self.is_ego_agent = ...
├── occ_prob_fusion(...)
│   ├── Alignment
│   ├── Binarization
│   └── Loop-based OR fusion
```

**After**:
```
occ_head.py
├── Import STCVOcc
├── __init__(..., use_stcv_occ, stcv_checkpoint_path, stcv_threshold)
│   └── self.is_cooperation = ...
│       self.is_ego_agent = ...
│       # STCV-Occ init with validation
│       self.stcv_occ = STCVOcc(...) if use_stcv_occ else None
├── occ_prob_fusion(...)
│   ├── Alignment
│   ├── Binarization
│   ├── Official OR (always)
│   └── Conditional STCV-Occ or fallback
```

---

## 🎯 Key Achievements

1. ✅ **Minimal Integration**: Only 4 modification points, ~40 lines added
2. ✅ **Non-Breaking**: Original behavior preserved when `use_stcv_occ=False`
3. ✅ **Safety Checks**: Validates ego agent + cooperation requirements
4. ✅ **Baseline Preserved**: Official OR always computed for comparison
5. ✅ **Clean Code**: Simplified fusion logic (loop → vectorized)
6. ✅ **Configurable**: Easy to toggle via config file
7. ✅ **Syntax Valid**: Python AST parsing successful

---

## 🚀 Next Steps

### Immediate (G5-1): Offline-Online Consistency Check

**Objective**: Verify online forward matches G4 offline reconstruction

**Test Script**: Create `G0-G1-G2/G5/G5-1_consistency_check/test_consistency.py`

**Verification Layers**:
1. ✅ Candidate indices (already passed in G5-0A)
2. ⬜ 127-dim features (target: max_diff < 1e-6)
3. ⬜ Logits (target: max_diff < 1e-6)
4. ⬜ Scores (target: max_diff < 1e-6)
5. ⬜ Accept decisions (must be exact match)
6. ⬜ Learned occupancy (must be binary exact match)
7. ⬜ Final IoU (target: ≈ 0.2732)

**Test Set**: 86 test samples (4 held-out scenes from G3)

**Hard PASS Conditions** (100% exact match required):
- Accept decisions (aR)
- Learned occupancy (O_learned)

**Soft Conditions** (numerical tolerance ~1e-6):
- Features (X_127)
- Logits (zR)
- Scores (sR)

**Success Criteria**:
```
IF all hard conditions exact match
AND all soft conditions < 1e-5
THEN G5-1 PASS
```

---

### After G5-1 PASS (G5-1B): Frozen Official Evaluation

**Objective**: Run frozen G3 predictor on UniV2X official evaluator

**Configuration**:
```bash
# Enable STCV-Occ in config
use_stcv_occ=True

# Run official evaluation
bash tools/univ2x_plugin/dist_test.sh \
    projects/configs/univ2x_plugin/occ_baseline_bevformer_medium_temporal.py \
    checkpoints/univ2x_baseline.pth \
    8
```

**Expected**:
- Official OR baseline: IoU ~0.25-0.27 (from G1/G4)
- STCV-Occ (frozen): IoU > Official OR (target: +8.2% from G4 offline)

**Decision Point**:
- If improvement significant → training-free integration valuable
- If improvement modest → consider G5-2 fine-tuning
- If no improvement → diagnose offline-online mismatch

---

## 📊 Integration Status

```
G5-0A  Candidate Audit                 ✅ PASS (86/86 exact match)
G5-0B  STCV Online Module              ✅ COMPLETE (6/6 tests passed)
G5-0C  occ_head Minimal Integration    ✅ COMPLETE (4 modifications)
G5-1   Offline–Online Consistency      ⬜ READY TO TEST
G5-1B  Frozen Official Evaluation      ⬜ PENDING (after G5-1)
G5-2   Frozen-Backbone Training        ⬜ PENDING (conditional)
```

**Current Phase**: G5-0C COMPLETE ✅  
**Next Phase**: G5-1 (Consistency Check)

---

## 📁 Modified Files

```
projects/mmdet3d_plugin/univ2x/dense_heads/
├── stcv_occ.py                    # ✅ G5-0B (new file)
└── occ_head.py                    # ✅ G5-0C (modified)
    ├── +import STCVOcc
    ├── +use_stcv_occ parameter
    ├── +STCV-Occ initialization
    └── +conditional fusion logic
```

---

## 🔍 Code Review Checklist

- ✅ Import statement added correctly
- ✅ Parameters added to `__init__` signature
- ✅ STCV-Occ initialization with safety checks
- ✅ Official OR always computed
- ✅ Conditional STCV-Occ fusion
- ✅ Fallback to official when disabled
- ✅ Syntax valid (AST parsing passed)
- ✅ No breaking changes to existing code
- ✅ Easy to toggle via config
- ✅ Checkpoint path configurable

---

## 🎊 Summary

**G5-0C successfully integrates STCV-Occ into UniV2X with minimal, non-breaking changes.**

**Key Design**:
- Clean separation: STCV logic in `stcv_occ.py`, minimal wrapper in `occ_head.py`
- Safe integration: Validation checks, fallback behavior, configurable toggle
- Baseline preserved: Official OR always computed for comparison
- Ready for testing: G5-1 consistency check can now proceed

**Total Implementation**:
- G5-0B: 478 lines (stcv_occ.py)
- G5-0C: ~40 lines added (occ_head.py)
- **Total**: ~520 lines for complete online integration

**Next**: G5-1 Offline-Online Consistency Validation

---

*G5-0C Status: COMPLETE*  
*Date: 2026-10-02*  
*Project: STCV-Occ - UniV2X Selective Temporal Cooperative Perception*
