# Overnight AGS versus DP-SGD summary

Every Blobs allocation/beta cell is individually private and retained. The displayed cell is an exploratory research selection using a disjoint public synthetic validation set, but selection evaluates private-trained candidate releases and is not itself privacy-accounted; it is therefore not a deployable selected mechanism. The fixed reference cell was chosen during exploratory development, so its bootstrap interval is descriptive, not confirmatory. PCA has no test-driven selection.

Table confidence intervals independently resample the AGS and DP-SGD trials; paired-seed intervals are also retained in JSON.

## Audit corrections applied

- Corrected cross-radius Abar uses T_k and T_(k+1); widths remain diagnostics.
- Sparse adjacent pairs after a missing radius are retained.
- The final Blobs dataset diameter is N=200; F is only a concretization frequency.
- Fixed cyclic DP-SGD composes 20 record-touching releases via adaptive parallel composition, not all 200 updates.
- Coordinate smooth-Laplace RDP maxima are enclosed continuously by interval branch-and-bound.

## Blobs — matched conservative generic PLD (certified)

| epsilon | AGS median | DP-SGD median | paired mean difference | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 0.4980 | 0.5048 | -0.0071 | [-0.0191, +0.0046] |
| 2 | 0.4988 | 0.5080 | -0.0168 | [-0.0291, -0.0047] |
| 5 | 0.5032 | 0.5475 | -0.0603 | [-0.0756, -0.0456] |
| 10 | 0.5080 | 0.6432 | -0.1352 | [-0.1541, -0.1163] |
| 15 | 0.5155 | 0.7140 | -0.1821 | [-0.2024, -0.1614] |
| 20 | 0.5405 | 0.7548 | -0.1992 | [-0.2205, -0.1776] |
| 25 | 0.5740 | 0.7800 | -0.1968 | [-0.2188, -0.1743] |
| 30 | 0.6162 | 0.8003 | -0.1811 | [-0.2041, -0.1574] |
| 40 | 0.7382 | 0.8327 | -0.1181 | [-0.1439, -0.0927] |
| 50 | 0.8355 | 0.8510 | -0.0434 | [-0.0650, -0.0225] |
| 100 | 0.9005 | 0.8868 | +0.0152 | [+0.0107, +0.0197] |

## Blobs — matched mechanism-specific RDP (fixed reference cell)

Exploratory-development reference: beta fraction 4, public bias-to-weight epsilon ratio 2.

| epsilon | raw AGS | raw DP-SGD | public-bias AGS | public-bias DP-SGD | raw paired difference | 95% CI | certified epsilon upper |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4957 | 0.5673 | 0.5063 | 0.6053 | -0.0840 | [-0.1024, -0.0657] | 0.999997 |
| 2 | 0.4947 | 0.6667 | 0.5023 | 0.7230 | -0.1672 | [-0.1889, -0.1456] | 1.999995 |
| 5 | 0.5370 | 0.7833 | 0.5337 | 0.8503 | -0.2449 | [-0.2678, -0.2217] | 4.999989 |
| 10 | 0.6803 | 0.8477 | 0.6803 | 0.8910 | -0.1561 | [-0.1760, -0.1358] | 9.999978 |
| 15 | 0.8503 | 0.8727 | 0.8540 | 0.8963 | -0.0195 | [-0.0314, -0.0077] | 14.999942 |
| 20 | 0.8950 | 0.8837 | 0.9007 | 0.8973 | +0.0142 | [+0.0072, +0.0213] | 19.999954 |
| 25 | 0.9000 | 0.8880 | 0.9030 | 0.9017 | +0.0133 | [+0.0077, +0.0189] | 24.999936 |
| 30 | 0.9017 | 0.8910 | 0.9043 | 0.9013 | +0.0109 | [+0.0063, +0.0154] | 29.999933 |
| 40 | 0.9020 | 0.8940 | 0.9047 | 0.9023 | +0.0078 | [+0.0043, +0.0113] | 39.999864 |
| 50 | 0.9020 | 0.8963 | 0.9043 | 0.9037 | +0.0058 | [+0.0029, +0.0087] | 49.999738 |
| 100 | 0.9023 | 0.8997 | 0.9047 | 0.9047 | +0.0025 | [+0.0007, +0.0042] | 99.998175 |

## Blobs — exploratory public-validation selection

| epsilon | beta fraction | public allocation | AGS median | DP-SGD median | paired mean difference | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.25 | bias_ratio_0.25 | 0.5003 | 0.5673 | -0.0828 | [-0.0981, -0.0682] |
| 2 | 0.25 | bias_ratio_0.25 | 0.5010 | 0.6667 | -0.1708 | [-0.1889, -0.1531] |
| 5 | 8.0 | public_rms_power_0.25 | 0.5413 | 0.7833 | -0.2306 | [-0.2592, -0.2017] |
| 10 | 8.0 | bias_ratio_1.0 | 0.8130 | 0.8477 | -0.0661 | [-0.0903, -0.0424] |
| 15 | 4.0 | public_rms_power_0.25 | 0.8800 | 0.8727 | +0.0067 | [-0.0041, +0.0173] |
| 20 | 4.0 | bias_ratio_2.0 | 0.8950 | 0.8837 | +0.0142 | [+0.0072, +0.0213] |
| 25 | 4.0 | bias_ratio_2.0 | 0.9000 | 0.8880 | +0.0133 | [+0.0077, +0.0189] |
| 30 | 4.0 | bias_ratio_1.0 | 0.9003 | 0.8910 | +0.0095 | [+0.0048, +0.0142] |
| 40 | 2.0 | bias_ratio_2.0 | 0.9017 | 0.8940 | +0.0079 | [+0.0046, +0.0114] |
| 50 | 2.0 | bias_ratio_2.0 | 0.9020 | 0.8963 | +0.0059 | [+0.0032, +0.0088] |
| 100 | 0.5 | bias_ratio_2.0 | 0.9023 | 0.8997 | +0.0025 | [+0.0007, +0.0042] |

## PCA-MNIST — common analytic RDP

| epsilon | AGS median | DP-SGD median | paired mean difference | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 0.2300 | 0.1380 | +0.0871 | [+0.0737, +0.1007] |
| 2 | 0.3740 | 0.1980 | +0.1665 | [+0.1518, +0.1813] |
| 4 | 0.5720 | 0.3810 | +0.1934 | [+0.1822, +0.2046] |
| 5 | 0.6280 | 0.4620 | +0.1706 | [+0.1595, +0.1819] |
| 8 | 0.7240 | 0.6260 | +0.1055 | [+0.0972, +0.1141] |
| 10 | 0.7520 | 0.6780 | +0.0763 | [+0.0686, +0.0840] |
| 20 | 0.8010 | 0.7750 | +0.0262 | [+0.0218, +0.0305] |
| 50 | 0.8200 | 0.8130 | +0.0074 | [+0.0051, +0.0097] |
| 100 | 0.8240 | 0.8230 | +0.0028 | [+0.0006, +0.0049] |

> PCA sensitivity status: analytic_rdp_conditional_sensitivity_certificate

The closed-form final-coordinate sensitivity assumes the clipped update map is non-expansive. The repository checks the unclipped softmax smoothness condition, but a proof covering elementwise per-example gradient clamping is still required before treating this PCA panel as fully certified.
