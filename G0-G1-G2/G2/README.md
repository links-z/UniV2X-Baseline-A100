# G2: Spatial Diagnosis and Oracle Analysis

G2 investigates whether harmful infrastructure-added occupancy exhibits structured spatial characteristics and whether selective acceptance of infrastructure additions provides meaningful performance headroom.

---

## Structure

G2 consists of two complementary analyses.

### G2-A: Spatial Diagnosis

- Analyze the spatial distribution of Harmful Additions (HA) and Beneficial Additions (BA).
- Study WAR with respect to ego-centric distance and warp-boundary distance.
- Examine the relationship between warp overlap ratio and addition reliability / cooperation gain.
- Output:
  - `g2_spatial_stats.npz`
  - `G2-A-readme.md`

### G2-B: Patch Accept/Fallback Oracle Analysis

- Quantify the theoretical performance headroom of selective fusion.
- Compare Cell Oracle with Patch Oracles at different spatial granularities.
- Use the patch utility:

\[
U_R = BA_R - HA_R
\]

Decision rule:

\[
U_R > 0
\Rightarrow \text{Accept infrastructure additions}
\]

\[
U_R \le 0
\Rightarrow \text{Fallback to Ego}
\]

- Output:
  - `g2_oracle_stats.npz`
  - `G2-B-readme.md`

---

## G1 Consistency Verification

Both G2-A and G2-B exactly reproduce the G1 candidate-domain statistics:

```text
Total ADD : 799,179
Total HA  : 646,667
Total BA  : 152,512
WAR       : 80.92%
```

Status:

```text
PASS
```

This confirms that G2 uses the same candidate-domain definition and sample population as G1.

---

# G2-A: Spatial Diagnosis

## Spatial Findings

### Ego-centric Distance

WAR exhibits a clear distance-dependent pattern with local fluctuations.

Representative results include:

```text
0-5 m   : WAR = 96.71%
5-10 m  : WAR = 89.21%
20-25 m : WAR = 80.07%
45-50 m : WAR = 78.44%
50-55 m : WAR = 76.93%
55-60 m : WAR = 76.20%
65-70 m : WAR = 71.87%
70-75 m : WAR = 69.59%
```

Overall, WAR tends to decrease with ego-centric distance rather than increase.

Therefore, the results do not support the simple hypothesis that farther infrastructure additions are necessarily less reliable.

The farthest distance bins contain fewer additions and should therefore be interpreted cautiously.

---

### Warp-Boundary Distance

WAR varies non-monotonically with distance to the warp-valid boundary.

Examples:

```text
0-2 cells   : WAR = 77.61%
8-10 cells  : WAR = 76.12%
14-16 cells : WAR = 81.34%
30-32 cells : WAR = 71.35%
38-40 cells : WAR = 84.31%
48-50 cells : WAR = 84.77%
```

No simple monotonic relationship of

```text
closer to warp boundary -> higher WAR
```

is observed.

Thus, warp-boundary distance alone is insufficient to characterize addition reliability.

---

### Warp Overlap Ratio

Sample-level Spearman correlation between overlap ratio and WAR:

\[
\rho=-0.2598,\qquad
p=1.2191\times10^{-9}
\]

This indicates a statistically significant but moderate negative monotonic association between overlap ratio and addition error rate.

Sample-level Spearman correlation between overlap ratio and cooperation gain:

\[
\rho=-0.0408,\qquad
p=0.3414
\]

No statistically significant monotonic association between overlap ratio and \(\Delta IoU\) is observed.

---

## G2-A Conclusion

G2-A demonstrates that infrastructure-added occupancy exhibits clear spatial heterogeneity.

However, the tested individual geometric indicators, including:

```text
ego-centric distance
warp-boundary distance
overall warp overlap ratio
```

cannot individually provide a sufficient description of cooperation value.

In particular, overlap ratio is associated with addition reliability but shows no significant monotonic association with the final candidate-domain IoU change.

Therefore, simple single-variable geometric heuristics are insufficient to determine whether infrastructure information should be accepted.

---

# G2-B: Patch Accept/Fallback Oracle Analysis

## Strategies

The following strategies are compared:

```text
Ego
Official OR
Cell Oracle
Patch-4 Oracle
Patch-8 Oracle
Patch-16 Oracle
```

Cell Oracle accepts only GT-confirmed beneficial additions:

\[
O_{\mathrm{cell}} = O_v \lor BA
\]

Patch Oracles make one Accept/Fallback decision for each spatial patch according to:

\[
U_R=BA_R-HA_R
\]

These Oracle strategies use ground truth and therefore represent diagnostic upper bounds rather than deployable methods.

---

## Overall Performance

| Strategy | IoU | Precision | Recall | F1 | Δ vs Ego | Δ vs Official |
|---|---:|---:|---:|---:|---:|---:|
| Ego | 0.2842 | 0.4856 | 0.4067 | 0.4427 | - | - |
| Official OR | 0.2710 | 0.3554 | 0.5332 | 0.4265 | -0.0132 | - |
| Cell Oracle | 0.3726 | 0.5531 | 0.5332 | 0.5430 | +0.0884 | +0.1016 |
| Patch-4 | 0.3451 | 0.5272 | 0.4998 | 0.5131 | +0.0609 | +0.0741 |
| Patch-8 | 0.3329 | 0.5144 | 0.4854 | 0.4995 | +0.0486 | +0.0618 |
| Patch-16 | 0.3213 | 0.5040 | 0.4700 | 0.4864 | +0.0371 | +0.0503 |

Official OR decreases candidate-domain IoU relative to Ego:

\[
0.2842 \rightarrow 0.2710
\]

whereas all Oracle strategies outperform both Official OR and Ego.

The Cell Oracle provides an absolute IoU gain of:

\[
+0.0884
\]

over Ego and:

\[
+0.1016
\]

over Official OR.

These values characterize the upper-bound headroom available when beneficial and harmful additions can be perfectly distinguished at cell level.

---

## Patch Granularity

Patch Oracle performance decreases as the spatial decision granularity becomes coarser:

```text
Patch-4  : IoU = 0.3451
Patch-8  : IoU = 0.3329
Patch-16 : IoU = 0.3213
```

Relative to Official OR, Patch-4 recovers:

\[
\frac{0.3451-0.2710}
     {0.3726-0.2710}
\approx72.9\%
\]

of the Cell-Oracle improvement.

Thus, finer spatial decision granularity provides substantially greater opportunity to distinguish locally beneficial and harmful cooperation.

---

## Patch-4 Decision Analysis

Patch-4 is the best-performing patch-level Oracle.

Acceptance statistics:

```text
Accepted patches : 17,805 / 119,448
Acceptance rate  : 14.9%
```

Accepted additions:

```text
ADD = 133,206
BA  = 112,239
HA  = 20,967
WAR = 15.74%
```

Rejected additions:

```text
ADD = 665,973
BA  = 40,273
HA  = 625,700
WAR = 93.95%
```

Compared with the overall addition WAR:

```text
Overall WAR = 80.92%
```

the accepted Patch-4 regions contain substantially more reliable infrastructure additions.

Patch-4 retains approximately:

\[
\frac{112239}{152512}
\approx73.6\%
\]

of all Beneficial Additions while admitting only:

\[
\frac{20967}{646667}
\approx3.24\%
\]

of all Harmful Additions.

This demonstrates substantial local separability between beneficial and harmful infrastructure additions under the GT-defined Oracle rule.

---

## Other Patch Oracles

### Patch-8

```text
Accepted patches : 7,777 / 58,829
Acceptance rate  : 13.2%

Accepted:
ADD = 127,761
BA  = 94,866
HA  = 32,895
WAR = 25.75%

Rejected:
ADD = 671,418
BA  = 57,646
HA  = 613,772
WAR = 91.41%
```

BA retention is approximately 62.2%, while approximately 5.09% of all HA is admitted.

### Patch-16

```text
Accepted patches : 3,674 / 33,490
Acceptance rate  : 11.0%

Accepted:
ADD = 114,503
BA  = 76,291
HA  = 38,212
WAR = 33.37%

Rejected:
ADD = 684,676
BA  = 76,221
HA  = 608,455
WAR = 88.87%
```

BA retention is approximately 50.0%, while approximately 5.91% of all HA is admitted.

---

## Horizon-wise Results

| Horizon | Ego | Official | Cell | Patch-4 | Patch-8 | Patch-16 |
|---|---:|---:|---:|---:|---:|---:|
| t=0 | 0.3756 | 0.3525 | 0.4615 | 0.4334 | 0.4219 | 0.4138 |
| t=1 | 0.2932 | 0.2919 | 0.3905 | 0.3615 | 0.3482 | 0.3366 |
| t=2 | 0.2536 | 0.2492 | 0.3513 | 0.3215 | 0.3075 | 0.2924 |
| t=3 | 0.2419 | 0.2278 | 0.3318 | 0.3031 | 0.2902 | 0.2762 |
| t=4 | 0.2387 | 0.2232 | 0.3075 | 0.2860 | 0.2770 | 0.2675 |

All Oracle strategies outperform Official OR across all five prediction horizons.

---

# G2 Overall Conclusion

G2 provides two complementary findings.

First, G2-A shows that the reliability of infrastructure-added occupancy is spatially heterogeneous, but the tested individual geometric indicators are insufficient to characterize final cooperation utility.

Second, G2-B establishes substantial Oracle headroom for local selective fusion.

The Cell Oracle improves candidate-domain IoU from:

```text
Official OR : 0.2710
```

to:

```text
Cell Oracle : 0.3726
```

corresponding to an absolute gain of:

```text
+0.1016 IoU
```

over Official OR.

Even the structured Patch-4 Oracle reaches:

```text
IoU = 0.3451
```

and recovers approximately 72.9% of the Cell-Oracle gain over Official OR.

These results demonstrate that beneficial and harmful infrastructure additions exhibit substantial local decision structure.

However, because all Oracle decisions rely on ground truth, these experiments do not demonstrate deployable performance or prove that the Oracle decisions can already be predicted from available inference-time features.

Instead, they provide strong motivation for learning an inference-time cooperation-value estimator using occupancy probabilities, spatial structure, and temporal information.

---

## Status

```text
G0   : PASS
G1   : GO
G2-A : PASS
G2-B : GO
```

---

## Files

### Scripts

```text
compute_g2_spatial.py
compute_g2_oracle.py
```

### Outputs

```text
g2_spatial_stats.npz
g2_oracle_stats.npz
G2-A-readme.md
G2-B-readme.md
```

---

## Running G2

```bash
cd /root/autodl-tmp/UniV2X

python G0-G1-G2/G2/compute_g2_spatial.py

python G0-G1-G2/G2/compute_g2_oracle.py
```

Both analyses are performed offline using the cached G0 outputs.

No model retraining or repeated neural-network inference is required.

---

## Next Step

The next stage is to investigate whether the GT-defined cooperation value demonstrated by G2 can be predicted from inference-time information.

The planned STCV-Occ model will use available occupancy probabilities, spatial structure, and temporal evolution to estimate local cooperation value and perform selective Accept/Fallback fusion without access to ground truth.