G2-A Spatial Diagnosis

G1 consistency:
ADD = 799,179
HA  = 646,667
BA  = 152,512
WAR = 80.92%
PASS

Distance:
WAR shows an overall decreasing tendency with ego-centric distance.
0-5 m   : 96.71%
5-10 m  : 89.21%
...
55-60 m : 76.20%
70-75 m : 69.59%

Conclusion:
No evidence that farther ego distance leads to higher WAR.
Instead, near-range additions exhibit substantially higher WAR.

Warp boundary:
WAR varies non-monotonically with boundary distance.
No simple "closer to warp boundary -> higher WAR" relationship is observed.

Overlap:
Spearman(overlap, WAR)
rho = -0.2598
p = 1.2191e-09

Spearman(overlap, Delta_IoU)
rho = -0.0408
p = 0.3414

Conclusion:
Spatial/geometric factors affect addition reliability, but simple
geometric indicators cannot adequately characterize cooperation utility.



