# New evidence computed by Claude after round 1 (RT held-out, same predictions as rt_s6enc_report.json)

## A. FILE rank check (replaces the retracted mean comparison)
AUC of each mixture slice (positives) vs ALL real test files (negatives):
| config | slice | n | FILE AUC | component AUC |
| S6 | realV+fakeM | 86 | 0.922 | MUSIC 0.904 |
| S6 | fakeV+fakeM | 398 | 0.987 | MUSIC 0.966 |
| S6 | fakeV+realM | 214 | 0.956 | VOICE 0.960 |
| S6+both | realV+fakeM | 86 | 0.910 | MUSIC 0.904 |
-> FILE preserves/raises the component ranking; the "FILE defect" claim is retracted in the report.

## B. Leave-one-unseen-generator-out (held-out RT ADS deltas)
| excluded generator | rows | S6-S5 | S6+both - S6 |
| none | 0 | +0.00204 | +0.01044 |
| fx_encoder_style_transfer | 227 | +0.00003 | +0.00743 |
| lyria-3 pro | 227 | -0.00081 | +0.00834 |
| openvoice | 427 | +0.00408 | +0.01207 |
| voxcpm-0.5b | 15 | +0.00205 | +0.01051 |
| voxcpm-1.5 | 17 | +0.00231 | +0.01044 |
| voxcpm2 | 222 | +0.00300 | +0.01022 |

## C. Per-seed held-out music EER, paired seeds (from run files test_preds, same ids/labels)
control mert6_12+eat16 vs +sslampt5: seed41 0.1545 -> 0.1401 (-0.0144); seed42 0.1452 -> 0.1531 (+0.0079); seed43 0.1573 -> 0.1336 (-0.0237)

## D. Static: component_fusion.fuse line 18 `w = {n: x for n, x in spec['weights'].items() if x and n in values}`
-> a weighted member missing at runtime is silently dropped (confirmed).
