# V4 retry: initial live metrics

Status: development evidence through six probes and P1 only. P2-P4, final development selection, refit, weight lock, and sealed evaluation remain pending. Frozen source commit: `a01a50e0c774f181c20238109127c5b81cc95dc1`. No sealed data was accessed.

## G0 integrity and label coverage

The existing evaluator accepted `.omo/evidence/v4-retry-G0_predictions.csv` against `v4_validation/dev.csv`. Both have 210 unique IDs in the same order and the exact six prediction columns. Panel counts are `music_real=10`, `music_seen=30`, `music_ood=10`, `speech=40`, `mixture=80`, and `splice=40`.

Known-label coverage is complete for file EER (210/210), music EER (170/170), and music-presence AUC (210/210). Voice EER has 100 known and 110 unknown relevant labels. Voice-presence AUC has 120 known and 90 unknown relevant labels, and the known subset has only one class. The five-head absolute score is therefore `null`; no overall score is reported for G0, any probe, or P1.

| Scope | File EER | Music EER | Music AUC |
| --- | ---: | ---: | ---: |
| All known dev labels | 0.414286 | 0.541429 | n/a |
| Seen music | 0.500000 | 0.400000 | 0.566667 |
| OOD music | 0.400000 | 0.400000 | 0.580000 |
| Speech | 0.000000 | n/a | n/a |
| Mixture | 0.558333 | 0.600000 | 0.404375 |
| Splice | 0.400000 | 0.400000 | 0.650000 |

G0 music-presence AUC is 0.995441 and known-subset voice EER is 0.150000. G0 slice values embedded in the first probe receipt match the independent evaluator result within `1e-12` for every available file/music EER and AUC.

## Six-probe snapshot

All probes completed the prescribed 60 updates. None passes the complete quality gate because every probe has negative OOD D and fails the OOD bootstrap win-rate and p10 requirements.

| Probe | Seconds | File EER | Music EER | Seen D | OOD D | Rank=min(D) | Seen win / p10 | OOD win / p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `probe_sonics_1e-05` | 48.496 | 0.371429 | 0.412143 | -0.022500 | -0.189000 | -0.189000 | 0.3590 / -0.1365 | 0.0200 / -0.2025 |
| `probe_sonics_3e-05` | 17.345 | 0.342857 | 0.358571 | 0.045000 | -0.117000 | -0.117000 | 0.6370 / -0.0810 | 0.0985 / -0.1665 |
| `probe_sonics_1e-04` | 16.480 | 0.332143 | 0.312143 | 0.037500 | -0.144000 | -0.144000 | 0.7990 / -0.0195 | 0.2495 / -0.1710 |
| `probe_ast_1e-06` | 58.338 | 0.310714 | 0.300000 | 0.045000 | -0.072000 | -0.072000 | 0.5575 / -0.1200 | 0.4760 / -0.1215 |
| `probe_ast_3e-06` | 70.924 | 0.342857 | 0.417143 | 0.117000 | -0.045000 | -0.045000 | 0.9725 / 0.0390 | 0.3855 / -0.1305 |
| `probe_ast_1e-05` | 75.655 | 0.310714 | 0.387857 | 0.216000 | -0.144000 | -0.144000 | 0.9975 / 0.1215 | 0.1400 / -0.1665 |

`probe_ast_3e-06` has the highest observed probe rank (-0.045000), but it still fails OOD gain, OOD file-EER, OOD music-AUC, and both OOD bootstrap gates. This is a probe observation, not a final quality or selection result.

## Initial P1 main run

P1 completed five epochs and 175 updates in 67.789 seconds, with 5,580 window exposures and 17,172,493 trainable parameters. The terminal result has no top-level `evaluation`, so the reported values come from `result.reports[result.best_epoch-1].evaluation`, which is epoch 5. Epoch rank scores were `[-0.189, -0.144, -0.144, -0.144, -0.045]`.

| File EER | Music EER | Seen D | OOD D | Rank | Seen win / p10 | OOD win / p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.342857 | 0.341429 | 0.109500 | -0.045000 | -0.045000 | 0.9075 / 0.0030 | 0.4125 / -0.0900 |

P1 passes the seen-panel statistical gates and the controlled speech/mixture file-EER gates. It fails OOD gain, OOD file-EER, OOD music-AUC, OOD bootstrap win rate, and OOD bootstrap p10, so it is not currently eligible under the frozen selection contract.

The initial cold forecast capped P1 at five epochs before empirical main-run throughput existed. The completed 67.789-second run is post-run evidence that may support a prospective extension decision, but such a decision can only occur before sealed access. This report neither authorizes an extension nor selects P1.

The candidate prediction CSVs were not supplied with these terminal receipts, so row-level protected-column equality was not independently rechecked. Full input hashes, gate booleans, and numeric values are recorded in `.omo/evidence/v4-retry-initial-metrics.json`.
