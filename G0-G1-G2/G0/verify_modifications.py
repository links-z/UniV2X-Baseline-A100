#!/usr/bin/env python3
"""
Verify that occ_head.py modifications are correct before running evaluation.
"""
import ast
import sys

def check_imports():
    """Check that necessary imports are present."""
    with open('/root/autodl-tmp/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py', 'r') as f:
        content = f.read()

    required_imports = ['import os', 'import numpy as np']
    missing = [imp for imp in required_imports if imp not in content]

    if missing:
        print(f"❌ Missing imports: {missing}")
        return False
    print("✅ All required imports present")
    return True

def check_init_modifications():
    """Check that __init__ has G0 export initialization."""
    with open('/root/autodl-tmp/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py', 'r') as f:
        content = f.read()

    required_vars = [
        '_g012_export_dir',
        '_g012_export_limit',
        '_g012_export_idx'
    ]

    missing = [var for var in required_vars if var not in content]

    if missing:
        print(f"❌ Missing initialization variables: {missing}")
        return False
    print("✅ G0 export initialization present")
    return True

def check_fusion_aux_return():
    """Check that occ_prob_fusion returns fusion_aux."""
    with open('/root/autodl-tmp/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py', 'r') as f:
        content = f.read()

    # Check for fusion_aux dict creation
    if 'fusion_aux = {' not in content:
        print("❌ fusion_aux dict not created in occ_prob_fusion")
        return False

    # Check for required keys
    required_keys = ['Pv', 'Pi_aligned', 'Ov', 'Oi', 'Oofficial', 'warp_valid_mask']
    missing_keys = [key for key in required_keys if f'"{key}"' not in content]

    if missing_keys:
        print(f"❌ Missing fusion_aux keys: {missing_keys}")
        return False

    # Check return statement
    if 'return fused_occ, inf_occ_log, fusion_aux' not in content:
        print("❌ occ_prob_fusion does not return fusion_aux")
        return False

    print("✅ occ_prob_fusion returns fusion_aux correctly")
    return True

def check_forward_test_modifications():
    """Check that forward_test has G0 export logic."""
    with open('/root/autodl-tmp/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py', 'r') as f:
        content = f.read()

    # Check fusion_aux initialization
    if 'fusion_aux = None' not in content:
        print("❌ fusion_aux not initialized in forward_test")
        return False

    # Check fusion call update
    if 'new_pred_seg_scores, new_inf_occ, fusion_aux = self.occ_prob_fusion' not in content:
        print("❌ occ_prob_fusion call not updated to receive fusion_aux")
        return False

    # Check G0 export logic
    if 'G0/G1/G2 offline export' not in content:
        print("❌ G0 export logic not present")
        return False

    # Check export data dictionary
    export_keys = ['Pv', 'Pi_aligned', 'Ov', 'Oi', 'Oofficial', 'GT',
                   'gt_cell_valid_mask', 'future_valid_mask', 'warp_valid_mask',
                   'test_seg_thresh', 'export_idx']

    missing_export = [key for key in export_keys if f'"{key}"' not in content]
    if missing_export:
        print(f"❌ Missing export data keys: {missing_export}")
        return False

    # Check npz save
    if 'np.savez_compressed' not in content:
        print("❌ np.savez_compressed not found")
        return False

    print("✅ forward_test G0 export logic complete")
    return True

def check_warp_valid_mask():
    """Check that warp_valid_mask is computed once and reused."""
    with open('/root/autodl-tmp/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py', 'r') as f:
        content = f.read()

    # Should compute once after grid normalization
    if 'warp_valid_mask = (' not in content:
        print("❌ warp_valid_mask not computed")
        return False

    print("✅ warp_valid_mask computed correctly")
    return True

def main():
    print("=" * 60)
    print("Verifying occ_head.py modifications for G0 export")
    print("=" * 60)

    checks = [
        check_imports,
        check_init_modifications,
        check_warp_valid_mask,
        check_fusion_aux_return,
        check_forward_test_modifications,
    ]

    results = [check() for check in checks]

    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL CHECKS PASSED - Ready for smoke test")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix before running")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
