# Cost ledger (option B, cap $12.00, hard stop cumulative $15.00)
| time | event | workspace | balance |
|---|---|---|---|
| 2026-09-28 22:03 KST | start (approved B) | CPU wsp-56syy2qpy2oe start | $18.75 |
| 2026-09-28 22:19 KST | x1 GPU start (feature dump + DB jobs) | H100 wsp-9683432r72m7 | $18.70 |
| 2026-09-28 23:20 KST | DB jobs done; P1 s7_bc started (overlaps gate evaluation; killed if gates fail) | x1 | $15.39
Balance |
| 2026-09-28 23:29 KST | X2 rejected by pre-registered gates; P1 killed; x1 pause requested | x1 | $14.90 |
