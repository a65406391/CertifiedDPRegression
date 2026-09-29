# Certified native-PLD all-allocation Blobs summary

All intervals are descriptive and unadjusted for multiplicity. Every retained rule is included; no test-best rule is selected.

| epsilon | rules | positive vs matched DP | positive vs Poisson DP |
|---:|---:|---:|---:|
| 15 | 9 | 1 | 0 |
| 20 | 9 | 2 | 0 |
| 25 | 9 | 5 | 0 |
| 30 | 9 | 5 | 2 |

## Epsilon 15 details

| rule | AGS | matched DP | difference | 95% CI |
|:---|---:|---:|---:|---:|
| uniform | 0.8793 | 0.8763 | -0.0061 | [-0.0192, +0.0065] |
| public_rms_power_0.25 | 0.8857 | 0.8763 | +0.0100 | [+0.0005, +0.0197] |
| public_rms_power_0.5 | 0.8510 | 0.8763 | -0.0400 | [-0.0618, -0.0198] |
| public_rms_power_0.75 | 0.6973 | 0.8763 | -0.2075 | [-0.2521, -0.1642] |
| weight_bias_ratio_0.25 | 0.5110 | 0.8763 | -0.3418 | [-0.3589, -0.3227] |
| weight_bias_ratio_0.5 | 0.5240 | 0.8763 | -0.2629 | [-0.2910, -0.2342] |
| weight_bias_ratio_1.0 | 0.8793 | 0.8763 | -0.0061 | [-0.0196, +0.0063] |
| weight_bias_ratio_2.0 | 0.8753 | 0.8763 | -0.0013 | [-0.0113, +0.0087] |
| weight_bias_ratio_4.0 | 0.6623 | 0.8763 | -0.1990 | [-0.2177, -0.1802] |
