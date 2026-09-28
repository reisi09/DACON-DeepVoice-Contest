# V6 H100 execution handoff

## User authorization

User requested the whole final pipeline, then explicitly requested Computer Use
on VESSL, their H100 workspace currently paused, notebook/data upload, training,
and long intervals between progress checks. Earlier total support ceiling $50,
first trial $20. Do not top up or purchase additional credits. Inspect live
balance/rate, keep compute budget <= $15 and preserve reserve. GPU start and
training are already authorized; don't ask again solely because of a skill gate.

## Prepared

- `V6_H100_Experiment.ipynb`: validated notebook and compile-checked code cells.
- `V6_FINAL_README.md`: final single-pipeline behavior and run instructions.
- `_workspace/V6_H100_RunBundle.zip`: complete upload bundle (see JSON sidecar
  for successful package verification and digest when present).
- `_workspace/v6_assets`: sealed pretrained assets, downloaded and tested locally.
- `_workspace/v6_local_check`: full split audit and genuine-audio integration smoke.
- `python -m unittest test_v6_final`: meaningful numerical/integration checks.

Large backbones are frozen. Voice uses the published XLS-R+AASIST **LA ONNX**.
Music uses frozen MERT95 with a newly fitted linear head, file-level max/MIL BCE.
File fusion is logistic and fit on separate source groups. Original PANNs,
HTDemucs and raw DF-Arena provide presence/separation/raw auxiliary evidence.
The full split has expert=1137, fusion=459 (including 80 mixtures), dev=210.
Eligible music training files=823. Existing dev is not a fresh holdout.

## Blocker

`@oai/sky` `get_app_state` for Aside / `at.studio.AsideBrowser` failed repeatedly:
`Computer Use server error -10005: failedToCreateImageDestination`.
After `mcp__node_repl__js_reset`, bootstrap failed with
`failed to write kernel assets: No such file or directory (os error 2)`.
The requested screenshot attachment also referenced a missing temporary file.
Do not infer actual VESSL current state from prior observations. The task did not
resume the GPU, upload to VESSL, or start remote training. A fresh Codex runtime
is needed to retry; no browser or account permissions were changed.

## Resume steps

1. Bootstrap Computer Use and read Aside; use fresh element indices after actions.
2. Inspect actual paused H100 workspace, hourly price, current credits, storage
   and any existing data. Keep prepaid trial bounds and record resume UTC time.
3. Resume authorized workspace and connect to its Jupyter service. Reuse matching
   existing assets if available; otherwise upload the notebook and full ZIP.
4. Fill notebook live-budget parameters, execute top-to-bottom, save outputs.
5. After a successful launch, use a thread heartbeat every 15 minutes to inspect
   progress/errors/budget, staying quiet when nothing actionable changes. Avoid
   repeated status reads. Notebook has a process deadline; that does NOT pause
   VESSL billing. Pause workspace on completion/error/near budget deadline.
6. Download trained files, metrics, executed notebook, final submission ZIP;
   validate archive/runtime. Do not submit to the leaderboard without a request.

## Scope of completed validation

Actual CPU inference traversed all five pretrained components on real audio.
A separate 10-file integration smoke exercised real cached features, 30-epoch
music head fitting, fusion fitting, evaluation and safe artifact save. That run
is explicitly `purpose=smoke`, and the exporter rejects it. It is not the H100
trial, full-dataset training, a quality estimate, or L4 timing evidence.

Unrelated V4 evidence files were already dirty when work started; preserve them.
