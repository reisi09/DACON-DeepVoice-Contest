# Full-bank pre-H100 code review

Status: **WATCH**. Review is read-only; no GPU or cloud operation was run.

## Scope and evidence

- `full_bank_features.py`, `full_bank_train.py`, `full_bank_evaluate.py`, `full_bank_split.py`, `full_bank_upload.py`
- Current split artifacts: `results/full_bank_mil_8020_20260918/{all_sources,train,val,admission_ledger}.csv`
- `test_full_bank_training.py` passed with `.venv/bin/python test_full_bank_training.py`.
- Split audit: 16,042 admitted rows, 12,826 train / 3,216 validation (20.047%), with no overlap by `component_id`, source bytes, native PCM, or PCM.
- Final deployment evidence now records all 16,042 items in 56 shards (15,054,431,524 bytes), `all_sources_uploaded: true`, remote object metadata checks, and a deployment manifest that maps every row to `audio/<sha256>.<ext>`.

## Findings

### CRITICAL

None.

### HIGH

None in the finalized deployment path. The former empty-`audio_sha256` blocker is resolved by the finalized deployment manifest, which fills it with the verified source hash. The user explicitly superseded the former local holdouts with the fresh all-data 80:20 split; the split artifact discloses that this validation is not historically independent and must not be presented as an independent test.

### MEDIUM

1. `full_bank_submission.py:69-75` peak-normalizes every input above 0.6, while `full_bank_features.py:171-173` intentionally preserves gain for the 875 legacy/constructed training rows. This does not affect the full-bank H100 training launch, but it prevents claiming exact cached-validation inference parity until that boundary is tested or aligned. The package report correctly currently declares parity and L4 validation false.

### LOW

1. `full_bank_features.py:71-85` accepts an empty `mert_windows` array during cache resume; training later rejects it at `full_bank_train.py:50-51`. Rejecting it during cache validation would fail closer to the corrupted artifact. This is fail-fast quality only, not a silent training error.

## Verified non-findings

- Full-duration PANNs are evaluated one baseline segment at a time (`full_bank_features.py:186-192`); they do not construct a whole-file PANN batch.
- HTDemucs uses the baseline's `apply_model(..., split=True)` path, so long files are chunked for GPU separation. The longest admitted music file is 784.54 seconds; raw/stem timeline retention is a host-RAM cost, not an unbounded GPU tensor.
- MERT uses fixed 64,600-sample windows and an empirical encoder-batch probe; head training groups bags by a measured window limit and rejects a file above that limit. The full bank is 250.27 hours / 437,663 MERT windows, approximately 1.67 GiB for float32 MERT embeddings alone.
- Music train/evaluation use the same WindowMIL log-mean-exp aggregation. Voice training reloads the existing DF-Arena `fc5` weights into a 1280-to-2 linear layer; its logit-difference max is algebraically equivalent to the inference max softmax fake probability. The meaningful contract test covers ragged MIL equivalence, gradients, full epoch coverage, validation exclusion, and the real `fc5` architecture.
- The 86 non-admitted entries are 84 full-original replacements for legacy crops, one exact duplicate, and one under-0.25-second file. This preserves the user's all-distinct-source objective rather than silently capping the data.

## Skill-perspective check

Consulted `omo:programming` and `omo:remove-ai-slops`. The relevant tests are meaningful rather than deletion-only or implementation-constant tests. The production code has no needless new parsing/normalization for this goal. The supplied full-bank scripts exceed the programming skill's strict script/style conventions and `full_bank_train.py` is over 250 pure LOC, but those are preflight maintainability observations, not the minimal correctness blockers above; splitting them now would add risk before the authorized run.

## Recommendation

`APPROVE` for the verified H100 training launch. Keep the package's inference-parity and L4-runtime claims unverified until a real package run measures them.
