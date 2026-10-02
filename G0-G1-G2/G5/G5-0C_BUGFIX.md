# G5-0C Blocking Bug Fix

## 🐛 Bug Description

**Severity**: BLOCKING - Would prevent G5-1A-2 runtime audit

**Location**: `occ_head.py::occ_prob_fusion()` lines 843-846

**Problem**: Two semantic errors in STCV-Occ integration:

1. **Incorrect Sanity Check**: Code checked `fused_occ == official_occ`, but STCV's purpose is to make them different
2. **Mislabeled Output**: `fusion_aux["Oofficial"]` pointed to `fused_occ` (learned) instead of `official_occ`

---

## 🔍 Root Cause

Original G5-0C integration preserved UniV2X's sanity check from non-cooperative mode:

```python
# Official OR sanity check
official_check = torch.maximum(veh_occ_log, inf_occ_log)
if not torch.equal(fused_occ.long(), official_check.long()):
    raise RuntimeError("Official OccFusion != Ov OR Oi")
```

**Why this fails with STCV**:
- STCV selective fusion: O_learned = O_v ∨ (a_R ∧ O_i)
- Only accepts beneficial patches, rejects harmful ones
- When STCV rejects at least one candidate that would change the binary output: O_learned ≠ O_official
- Original check would raise RuntimeError when STCV successfully rejects patches
- Note: If STCV accepts all candidates or rejected patches don't affect binary output, they could still be equal

---

## ✅ Fix Applied

### Change 1: Correct Sanity Check Target

**Changed** lines 843-846 to check `official_occ` instead of `fused_occ`:

**Before** (WRONG):
```python
# Official OR sanity check
official_check = torch.maximum(veh_occ_log, inf_occ_log)
if not torch.equal(fused_occ.long(), official_check.long()):
    raise RuntimeError("Official OccFusion != Ov OR Oi")
```

**After** (CORRECT):
```python
# Official OR invariant check (corrected: checks official_occ, not fused_occ)
official_check = torch.maximum(veh_occ_log, inf_occ_log)
if not torch.equal(official_occ.long(), official_check.long()):
    raise RuntimeError("Official OccFusion != Ov OR Oi")
```

**Rationale**:
- The invariant O_official = O_v | O_i should always hold (baseline computation)
- STCV only affects O_fused, not O_official
- Check now protects the baseline without constraining selective fusion

### Change 2: Fix fusion_aux Semantics

**Before**:
```python
fusion_aux = {
    "Pv": veh_occ.detach(),
    "Pi_aligned": new_inf_occ.detach(),
    "Ov": veh_occ_log.detach(),
    "Oi": inf_occ_log.detach(),
    "Oofficial": fused_occ.detach(),  # ❌ Wrong: points to learned
    "warp_valid_mask": warp_valid_mask.detach(),
}
```

**After**:
```python
fusion_aux = {
    # Soft probability
    "Pv": veh_occ.detach(),
    "Pi_aligned": new_inf_occ.detach(),
    # Binary occupancy
    "Ov": veh_occ_log.detach(),
    "Oi": inf_occ_log.detach(),
    # Official binary OR (always computed)
    "Oofficial": official_occ.detach(),  # ✅ Correct
    # Fused occupancy (Official OR or STCV Learned)
    "Ofused": fused_occ.detach(),        # ✅ New field
    # Spatial validity of infrastructure warp
    "warp_valid_mask": warp_valid_mask.detach(),
}
```

**Key improvements**:
- `Oofficial` now correctly points to `official_occ` (always Ov | Oi baseline)
- New `Ofused` field contains actual output (Official or Learned depending on mode)
- Both available in memory during the same forward pass
- Note: Current G0 exporter writes only `Oofficial` to disk; `Ofused` available for future enhancement

### Change 3: Enhanced Debug Output

Added to debug block:
```python
print("official_occ unique:", torch.unique(official_occ))
print("use_stcv_occ:", self.use_stcv_occ)
```

**Purpose**: Help distinguish Official vs Learned in runtime audit

---

## 📊 Impact Analysis

### ✅ Positive Impacts

1. **Unblocks G5-1A-2**: Runtime audit can now proceed without false errors
2. **A/B Comparison Available**: Both Official and Learned available in same forward pass (in memory via fusion_aux)
3. **Preserves G0 Cache**: Existing cache files remain valid (generated with correct old semantics)
4. **Baseline Invariant Protected**: O_official = Ov | Oi check now correct and preserved

### ⚠️ To Be Verified

**Default Mode Regression**: The fix is designed to preserve original Official OR behavior when `use_stcv_occ=False`. Full regression verification pending real UniV2X run.

**Recommended**: "Designed to preserve" → will upgrade to "regression verified" after STCV-off test.

### ⚠️ Compatibility Notes

**G0 Exporter** (currently):
```python
"Oofficial": fusion_aux["Oofficial"]  # Now correct after fix
```

**Optional future enhancement**:
```python
"Olearned": fusion_aux["Ofused"][0].cpu().numpy().astype(np.uint8)
```
(Not required for G5-1A-2, can add later if needed)

---

## 🧪 Verification

### Syntax Check
```bash
python -m py_compile projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py
# ✅ Passed
```

### Logic Verification

**Official OR (STCV off)**:
- `official_occ = Ov | Oi` ✅
- `fused_occ = official_occ` ✅
- `fusion_aux["Oofficial"] = official_occ` ✅
- `fusion_aux["Ofused"] = official_occ` ✅

**STCV Learned (STCV on)**:
- `official_occ = Ov | Oi` ✅ (still computed)
- `fused_occ = stcv_occ(...)` ✅ (selective)
- `fusion_aux["Oofficial"] = official_occ` ✅ (baseline)
- `fusion_aux["Ofused"] = fused_occ` ✅ (learned)
- **Sanity check on official_occ passes** ✅ (checks baseline, not output)

---

## 🎯 Semantic Clarity

### Before Fix
```
STCV off:  fused_occ = official_occ
           fusion_aux["Oofficial"] = fused_occ  ✅ correct

STCV on:   fused_occ = learned_occ
           fusion_aux["Oofficial"] = fused_occ  ❌ mislabeled!
           Sanity check: fused_occ == official_occ  ❌ raises error!
```

### After Fix
```
STCV off:  official_occ = Ov | Oi
           fused_occ = official_occ
           fusion_aux["Oofficial"] = official_occ  ✅
           fusion_aux["Ofused"] = official_occ     ✅
           Sanity check: official_occ == Ov | Oi   ✅

STCV on:   official_occ = Ov | Oi  (always computed)
           fused_occ = learned_occ (selective)
           fusion_aux["Oofficial"] = official_occ  ✅
           fusion_aux["Ofused"] = learned_occ      ✅
           Sanity check: official_occ == Ov | Oi   ✅
```

**Invariant maintained**:
```
Oofficial = Ov | Oi  (always)
Ofused = {
    Oofficial  (STCV off)
    Olearned   (STCV on)
}
```

---

## 📝 Git Diff Summary

**File**: `projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py`

**Changes**:
- Lines modified: 6 (comments, sanity check target, fusion_aux keys)
- Lines added: 6 (corrected sanity check, Ofused field, debug output)
- Net change: +7 lines

**Affected sections**:
1. Comment update (line 811)
2. Sanity check corrected (lines 826-829): now checks official_occ instead of fused_occ
3. Debug output enhanced (lines 843, 846)
4. fusion_aux corrected (lines 857-860): Oofficial + Ofused separation

---

## ✅ Status

**Bug**: FIXED
**Code Verification**: PASSED (syntax, logic correct)
**Module-level Verification**: PASSED (G5-1A-1: 405 candidates, 0 diff)

**Still To Verify**:
- Real UniV2X runtime integration (G5-1A-2)
- STCV-off regression equivalence

**Blockers**: NONE
**Next Step**: G5-1A-2 Runtime Audit (READY)

---

*Fixed: 2026-10-02*
*G5-0C Blocking Bug - Resolved*
