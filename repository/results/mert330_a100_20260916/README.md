# C08 MERT330 MLP comparison on A100

Launch on existing workspace `wsp-5i3knd3gxeox` (`jolly-routine`), A100 SXM 80GB, 1.48 credits/hour. The initial compute ceiling is two hours; credit balance before launch was 7.028079.

Model: frozen Demucs and MERT; 64-channel three-convolution residual trunk, max pooling, Linear(64,32), GELU, Dropout(0.1), Linear(32,2), fake-class softmax. Training uses class-balanced cross-entropy and AdamW weight decay 0.01. MERT95/768 and MERT330/1024 each train three fresh heads, seeds 41/42/43, against the same source-separated split and tune-only selection.

The MERT95 features were previously extracted on H100 and are reused with sealed provenance. New MERT330 features and all six heads run on A100. This hardware difference is recorded; local metrics do not establish the official total score or L4 deployment time.

- Input: `C08_MERT330_A100_Input.zip`
- Bootstrap: `a100_bootstrap.sh`
- Remote root: `/root/C08_MERT330_A100`
- Object prefix: `/root-1/temporal/mert330-a100-20260916/`
- Notebook: `C08_MERT330_A100.ipynb`
- Monitor: existing three-minute heartbeat; persist and verify results before pausing the workspace.

The original C08 submission ZIP and earlier experiment packages remain intact. An executed notebook and real run state, rather than this preparation record, establish execution progress.

## Completed result

All 1,446 feature records, six fresh head fits and eight historical reports completed. The winner selected by tune music EER was `M330_mlp` (mean 0.2555798451 versus M95's 0.2712061102). Historical test data did not enter that selection.

| Model | Historical music EER | Historical file EER |
|---|---:|---:|
| Official baseline | 0.5527777778 | 0.4320512821 |
| Original C08, prior three-seed mean | 0.2953703704 | 0.3213675214 |
| MERT95 + regularized MLP, three-seed mean | 0.2972222222 | 0.3213675214 |
| MERT330 + regularized MLP, three-seed mean | 0.2203703704 | 0.2666666667 |

These are means of independently evaluated seeds, not ensemble scores. MERT330 music EER by seed41/42/43 was 0.2138888889 / 0.2333333333 / 0.2138888889 (population standard deviation 0.0091661990). Its selected epochs were 14 / 19 / 17. MERT95 selected epochs were 12 / 15 / 14. The historical panel contains 190 files, including 150 with known present-music labels. It is a reused development panel, not a fresh blind leaderboard test. Incomplete voice/presence labels prevent reporting a full official score.

Extraction took 4,973.93 seconds (82m54s). All six classifier fits took 83.16 seconds in total. Results ZIP SHA256: `bfe80668bd2112cdcf0373cb39ca80ee24f9b7e0611c2597a93baa66d1ed1d6c` (4,998,622 bytes). The separately persisted feature ZIP is 955,063,193 bytes, SHA256 `7501f3c0f90fb1579634aafb200088e31861fec3c835f6d0b84acbea8675d4e3`.

`verify_results.py` recomputed all six tune predictions' metrics and eight historical metric reports, checked split isolation, frozen selection hash, checkpoint hashes and strict loading, and verified 1,446 audio/cache receipts. `verify_remote_archive.py` checked the persisted feature archive SHA, CRC and every cache member against those receipts. The executed notebook has no cell errors and its plotted values were visually checked.

Workspace state is **stopped / standby**, current burn is **0**, and the heartbeat is **PAUSED**. Credit balance changed from 7.028079 to 4.691360: **2.336719 credits** used. This includes preparation and artifact verification time. The original C08 submission package was not replaced; no competition submission was performed.
