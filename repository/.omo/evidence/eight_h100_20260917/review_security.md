# Eight H100 security review

- Verdict: **PASS**
- Reviewed commit: `2a0697edef92903f604cdb50f77ef807a940c5d0`
- Review mode: independent, read-only review of the scoped tracked code and operation artifacts; no GPU, API, UI, or external security scan was used.

## Scope

Reviewed `eight_h100_launch.py`, `eight_h100_watch.py`, `eight_h100_run.sh`, the reused archive/stop helpers in `redesign_h100_watch.py`, the three preserved files and receipt under `results/eight_h100_20260917/runtime_sources/`, and the tracked receipts under `results/eight_h100_20260917/` relevant to export verification, the hard deadline, compute stop, and bootstrap restoration.

## Findings

No blocking security finding was reproduced in scope.

### Secrets and private payloads

PASS. A pattern scan of the scoped tracked text artifacts found no private-key block, AWS access-key-shaped value, GitHub token-shaped value, OpenAI key-shaped value, or bearer token. The code uses fixed workspace, organization, team, volume, and object-prefix identifiers but contains no authentication secret. The only generic secret-related hit was prose in `preflight_independent.json` referring to a prior secret-output audit; it does not contain a secret value.

### Export archive integrity and path safety

PASS. I independently opened `results/eight_h100_20260917/downloaded/results.zip` and reproduced:

- archive SHA-256 `681d41de10397d31a371daaa276a4a90d19f04789b85e01fdd6f936399dd23d2`;
- 7,315 ZIP entries: 7,314 payload members plus `EXPORT_SHA256.json`;
- no CRC failure, duplicate member, absolute member, parent traversal component, or backslash path;
- exact manifest coverage of all 7,314 payload members;
- zero payload SHA-256 mismatches.

This agrees with `download_verification.json`, `downloaded/export_receipt.json`, and `downloaded/run_state.json`. `redesign_h100_watch.verify_archive` additionally rejects non-canonical paths, directory entries, links/special files, duplicates, manifest gaps, digest mismatches, and resolved destinations outside the extraction root before extraction. Its local self-check reproduced acceptance of a valid archive and rejection of corrupt, traversal, link, and duplicate archives.

The runtime source and fixed wheel extraction paths also check CRC, duplicates where applicable, absolute/parent paths, symlinks, manifest coverage, and member hashes before `extractall`. The wheel is pinned by a fixed SHA-256. The source bundle is produced locally with member hashes and is verified again before use.

### Deadline and stop protection

PASS. Launch metadata fixes the deadline to four hours from the original start and refuses an existing run metadata file, preventing a normal relaunch from resetting it. The preserved attempt-6 config retains `started_epoch=1789648960` and `pause_deadline_epoch=1789663360`, exactly 14,400 seconds apart; `runtime_sources/receipt.json` records the same immutable values. Runtime execution rejects invalid or over-four-hour spans, requires more than the 600-second export reserve, and wraps execution in a deadline-bounded `timeout`.

The watcher validates finite start/deadline values and the four-hour ceiling, bounds remote calls by remaining time, retries a failed pause request after deadline, and pauses after a verified terminal archive or terminal failure. `resource_stop.json` records `pause_verified=true`, phase `standby`, workspace `wsp-9683432r72m7`, with reason `archive_verified_exported`. The local watcher self-check reproduced retry after a transient pause failure.

### Bootstrap restoration and frozen runtime sources

PASS. `eight_h100_watch.restore_bootstrap` refuses restoration until `resource_stop.json` proves the compute pause, uploads the saved original bootstrap, downloads only the intended bootstrap object, and compares SHA-256 before writing its receipt. `bootstrap_restored.json` records `restored=true` and SHA-256 `d027558a7774b4581c6ed1485ed9e94f8bec5b0b8fc88687aa5366e520b07a94`.

The three preserved attempt-6 runtime files reproduce every SHA-256 listed in `runtime_sources/receipt.json`:

- `config.attempt6.json`: `deae392fdaf8597eb2a66a22a7f2158876a714b99005441280e258e545b49baa`
- `eight_h100_run_retry.attempt6.sh`: `5e26fb775d04f0f6a1986c0fad551eea4af867a180ed9ba96b0bd0f3a164bca6`
- `full_overlay_then_main.attempt6.sh`: `be971b37ae1f836253bff662efc0d20d07c40918231934d8315e33e905966ece`

## Slop and maintenance review

The scoped diff does not add a broad framework, speculative abstraction, dependency, or low-value test suite. The archive self-check covers concrete failure modes: corruption, traversal, links, duplicates, and stop retry. The preserved runtime scripts are operational evidence for the exact successful attempt, so their duplication of deployed shell content has direct audit value. No deletion-only, tautological, shape-only, or implementation-mirroring test was found in scope.

## Evidence pointers

- `redesign_h100_watch.py`: `_safe_member_name`, `verify_archive`, `_load_run`, `_download`, `_verify_export`, `_pause_and_verify`, `monitor`, and `self_check`
- `eight_h100_launch.py`: `base_config`, `upload_and_readback`, `start`
- `eight_h100_watch.py`: `restore_bootstrap`, `main`
- `eight_h100_run.sh`: fixed wheel verification, source archive verification, deadline validation, bounded runtime
- `results/eight_h100_20260917/download_verification.json`
- `results/eight_h100_20260917/resource_stop.json`
- `results/eight_h100_20260917/bootstrap_restored.json`
- `results/eight_h100_20260917/downloaded/results.zip`
- `results/eight_h100_20260917/downloaded/export_receipt.json`
- `results/eight_h100_20260917/downloaded/run_state.json`
- `results/eight_h100_20260917/runtime_sources/receipt.json`

## Residual notes

The review establishes integrity and safe handling of the tracked/downloaded artifacts at the reviewed commit. It does not independently query the live VESSL control plane; the stopped and restored states are supported by the recorded, hash-linked local receipts as required by this scoped review.
