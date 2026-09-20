# DACON 236749: five-graph Colab pilot

Latest artifact (2026-09-13): [AST full-data final refit](submissions/AST_full1029_epoch17/README.md).
All 1,029 public-corpus files participated in head warmup (8 epochs) and last-block
fine-tuning (17 epochs), executed locally on MPS. The final ZIP passed offline replay;
submission 88673 scored 0.6174819841 in 82 seconds, below baseline 88418 (0.6909110688).
The renamed uploaded ZIP matches the full-refit hash. There is no remaining independent holdout.
[Executed audit notebook](DACON_236749_full_refit_run_record.ipynb) ·
[Colab reproduction notebook](DACON_236749_full_refit.ipynb).

- [Colab notebook](https://colab.research.google.com/drive/18oA4W3x-FaqJky9iMa50eANknK229pQa)
- [Drive data, sources, and results](https://drive.google.com/drive/folders/1jKgz89YyiNRFIAsmM7canCtm0QODanNT)
- [Pipeline graph and hypotheses](PIPELINE_GRAPH.md)
- [Official competition contract](COMPETITION_CONTRACT.md)
- [Validation data provenance and limitations](DATA_SOURCES.md)

The notebook shares PANNs, HTDemucs, and DF-Arena window evidence across G0–G4.
Only G4 fits small logistic fusion heads, using dev groups. Frozen inference
never uses another evaluation file's predictions or statistics.

The public validation bank contains 225 clips (160 dev / 65 holdout), with
original speech, synthesized speech, music, mixtures, and telephone transforms.
Holdout contains 21 atomic source groups; transforms and mixtures are correlated.
There are no confirmed voice-absent negatives, so voice-presence AUC and the
complete DACON score cannot be estimated. Partial metrics and a labeled pilot
selection proxy are reported separately. The bank is not generator-disjoint
and can overlap upstream detector training (MLAAD).

The experiment has a 120-minute GPU ceiling, per-file Drive checkpoints, an
official-reference parity check, a 60-second input stress check, and a fresh
offline execution of the selected submission ZIP. The Colab runtime differs
from the competition server; A100 results do not establish L4 time compliance.

Rebuild the notebook with `.venv/bin/python build_notebook.py`. Prepare the
auditable dataset with `uv run prepare_validation.py`; use `--check` to run its
source-lineage, paired-label, decoding, and telephone-filter checks. Numerical
and independence checks run with:

```sh
.venv/bin/python -m pytest -q test_graph_scoring.py test_graph_fusion.py
```

Local runtime evidence is stored under `results/pilot_20260912_160943/`.
The actual Colab session began at 2026-09-13 01:09 (Asia/Seoul).
All 225 clips completed; the fresh G0 archive reproduced notebook predictions
with maximum probability difference 0.0. Total notebook runtime was 614.35 seconds;
the A100 was released and Colab showed zero active sessions. G0 tied G2 and
outperformed the other variants on the predeclared pilot selection proxy, so the
existing official submission was retained without spending another daily attempt.
See [RESULTS.md](RESULTS.md) for metrics, limits, and Drive links.

## Official baseline submission record

Official, unmodified PANNs Cnn14 / HTDemucs / DF-Arena 1B inference baseline.

- Competition: https://dacon.io/competitions/official/236749/overview/description
- Shared notebook: https://dacon.io/competitions/official/236749/codeshare/14153
- Official archive source: https://dacon.io/competitions/official/236749/data
- Submission page: https://dacon.io/competitions/official/236749/mysubmission
- Downloaded via Aside Computer Use on 2026-09-13 (Asia/Seoul).

`baseline_original/baseline_submit.zip` was extracted unchanged from the official
`open.zip`. The submission archive and dummy audio are retained locally and
excluded from Git. The extracted script, requirements, notebook, model source
record, and checksums preserve the official files without modification.

`baseline_original/verification.json` records archive size, SHA-256, successful
CRC checks for every archive entry, and matches against all three published
model checksums. The archive expands to 5,003,422,551 bytes; its root entries are
`model/`, `script.py`, and `requirements.txt`.

The notebook differs from the packaged script only in notebook-compatible base
directory resolution and command-line argument handling. Prediction logic is
unchanged. At initial inspection, local inference was not run because the unmodified default
requires CUDA. The official evaluator subsequently returned the score below.

The official `requirements.txt` pins `matplotlib==3.9.2`; it is intentionally
preserved. The initial inspection did not verify evaluator installation; the later scored
submission confirms that the official evaluator completed the baseline run.

Submission completed once through Aside Computer Use. DACON displayed its
submission-complete dialog and then listed `baseline_submit.zip` with the memo
`Official baseline: PANNs Cnn14 + HTDemucs + DF-Arena 1B (unmodified baseline_submit.zip)`.

- Submission time: `2026-09-13 00:35:37` (Asia/Seoul).
- Updated observed result: score `0.6909110688`, ADS `0.6577539683`, CPS `0.9893249735`; evaluator inference time `21분 59초`.
- Current evidence: `results/v2_evidence/baseline_88418_scored.json` and `.jpeg`.
- Local UI evidence: `_workspace/2026-09-13-001/submitted.jpg` (excluded from Git).

## V2 AST diagnosis and actual Colab experiment

The original three-candidate development selection remains **G0**. Frozen AST heads
and last-block fine-tuning were trained on 633 files and selected on 122 files,
then evaluated on 176 generator-held and 98 source-held files. A later fixed
G0/AST routing candidate is preserved as **exploratory**, with its separate
protocol and timestamps; it is not a replacement for the original locked selection.
It improves the combined partial-label proxy but regresses on held-out generators.
No new DACON submission was made.

- [Full V2 report](RESULTS_V2.md)
- [V2 pipeline graph](PIPELINE_GRAPH_V2.md)
- [V2 data provenance](DATA_SOURCES_V2.md)
- [Executed training Colab](https://colab.research.google.com/drive/1cesnr-fH1lQm5-fFraiw4D2n-NfujeKa)
- [Executed precision and offline replay Colab](https://colab.research.google.com/drive/19_Uvy8xgwlED-akOyBQRIk0bWAJs8CKz)
- [V2 Drive results](https://drive.google.com/drive/folders/1IqE0FBeDhoSrxtrwZQOZeJoK93faWosO)

The primary package replay exposed a cuDNN TF32 mismatch. The separate GPU
diagnostic reproduced it, disabled TF32 at the shared AST load boundary, and
verified fresh AST and exploratory hybrid ZIP execution with sockets blocked:
maximum prediction difference 0.0 for four inputs including a 60-second file.
The executed notebooks preserve the original error and successful recovery.
The local `DACON_236749_v2_ast.ipynb` is rebuilt from corrected runtime sources;
its rerun is a reproduction, not a new untouched evaluation. Rebuild it with
`.venv/bin/python build_v2_notebook.py`. Run meaningful local checks using
`.venv/bin/python -m pytest -q`. Both active GPU sessions were released; final
Colab UI showed zero active sessions. L4 throughput remains unverified.
