#!/usr/bin/env python3
"""
G3: Identity Consistency Audit

Compare cache valid samples with evaluator valid samples to verify
that future_valid_mask.all() ⟺ occ_to_eval for all 675 samples.
"""

import os

CACHE_FILE = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3/cache_valid_samples.txt"
EVAL_FILE = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3/evaluator_valid_samples.txt"
OUT_FILE = "/root/autodl-tmp/UniV2X/G0-G1-G2/G3/identity_audit_report.txt"

print("=" * 70)
print("G3: IDENTITY CONSISTENCY AUDIT")
print("=" * 70)
print()

# Load cache valid samples
cache_samples = set()
with open(CACHE_FILE, 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            sample_idx = parts[0]
            cache_samples.add(sample_idx)

# Load evaluator valid samples
eval_samples = set()
with open(EVAL_FILE, 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            sample_idx = parts[0]
            eval_samples.add(sample_idx)

print(f"Cache valid samples     : {len(cache_samples)}")
print(f"Evaluator valid samples : {len(eval_samples)}")
print()

# Compute set differences
cache_only = cache_samples - eval_samples
eval_only = eval_samples - cache_samples
intersection = cache_samples & eval_samples

print(f"Intersection            : {len(intersection)}")
print(f"Cache only              : {len(cache_only)}")
print(f"Evaluator only          : {len(eval_only)}")
print()

# Identity check
if cache_samples == eval_samples:
    result = "PASS"
    status = "✓"
else:
    result = "FAIL"
    status = "✗"

print("=" * 70)
print(f"IDENTITY CHECK: {status} {result}")
print("=" * 70)
print()

# Save report
with open(OUT_FILE, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("G3: IDENTITY CONSISTENCY AUDIT REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("Comparison:\n")
    f.write(f"  Cache valid samples     : {len(cache_samples)}\n")
    f.write(f"  Evaluator valid samples : {len(eval_samples)}\n")
    f.write(f"  Intersection            : {len(intersection)}\n")
    f.write(f"  Cache only              : {len(cache_only)}\n")
    f.write(f"  Evaluator only          : {len(eval_only)}\n\n")

    f.write(f"Identity Check: {status} {result}\n\n")

    if cache_samples == eval_samples:
        f.write("Conclusion:\n")
        f.write("The 549 valid samples identified by cache (future_valid_mask.all())\n")
        f.write("are IDENTICAL to the 549 valid samples identified by evaluator\n")
        f.write("(not gt_occ_has_invalid_frame).\n\n")
        f.write("This confirms:\n")
        f.write("  future_valid_mask.all() ⟺ occ_to_eval\n\n")
        f.write("Therefore, G1/G2 analyses are using the exact same 549 samples\n")
        f.write("that the official UniV2X baseline evaluator uses for Occupancy IoU.\n")
    else:
        f.write("Discrepancies:\n\n")
        if cache_only:
            f.write(f"Samples in cache but NOT in evaluator ({len(cache_only)}):\n")
            for sample_idx in sorted(cache_only):
                f.write(f"  {sample_idx}\n")
            f.write("\n")
        if eval_only:
            f.write(f"Samples in evaluator but NOT in cache ({len(eval_only)}):\n")
            for sample_idx in sorted(eval_only):
                f.write(f"  {sample_idx}\n")
            f.write("\n")

print(f"Report saved to: {OUT_FILE}")
print()

if cache_samples == eval_samples:
    print("✓ Identity verified: cache and evaluator use the same 549 samples")
else:
    print("✗ Identity mismatch detected - see report for details")
