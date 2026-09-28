# H100 recovery integration audit

## Scope and basis

Read-only review of the recovery path for the existing `wsp-9683432r72m7` workspace. No remote command, workspace operation, code change, or test-data mutation was performed.

Reviewed file SHA-256 values:

- `eight_h100_run.sh`: `9bd84b39249800f310a2f14dc88406a941df912810e39f1df37a0ba308f7e2aa`
- `eight_h100_launch.py`: `bc73a68cf7ad46a7d68acd49e0a35070df823315091f25381544939a80d5931c`
- `eight_h100_watch.py`: `dedd5f5a8c6bed135b707d9e188d67186c565013ac6123d3ca37a193adb75764`
- `redesign_h100_watch.py`: `ac61434e2a008c408f008cb895eb7a964cdb41e19966d0ac938040f3decc974e`
- `_workspace/eight_h100_20260917/config.json`: `ea2621a17f8c8c88f2d55d964012f62341240fa298d518c96fd808682bfff577`
- `_workspace/eight_h100_20260917/readiness_audit.sh`: `4aab003e22909ceb2e8043e486de42c7c83d4f1b292d67d499357e24e0d641d3`

The corrected config preserves `started_epoch=1789648960` and `pause_deadline_epoch=1789663360`, uses attempt 4, and points to `/root/redesign_h100_20260917/assets`. The remote readiness receipt is nevertheless currently `blocked`.

## Skill-perspective check

`omo:programming` and `omo:remove-ai-slops` were loaded before reviewing maintainability and test relevance. This pass found no deletion-only test, requested-removal test, tautological test, implementation-constant-only test, brittle prompt test, untyped escape hatch, or unnecessary production parsing/normalization relevant to this recovery diff. The recovery code does not violate either skill perspective in a release-blocking way. The missing checks below are integration contracts with real remote state; shallow unit tests would not resolve them.

## Findings

### CRITICAL

None.

### HIGH / P1

1. **The authoritative readiness gate failed and its bootstrap cannot repair the failed prerequisite.** The latest receipt, `results/eight_h100_20260917/readiness_audit.json`, reports `status: blocked`, 9/10 checks passing, and `kernel_registration` failed with `kernels=['python3']`. The audit only checks for `eight-h100` at [`readiness_audit.sh:79`](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/eight_h100_20260917/readiness_audit.sh:79) and writes the blocked result at [`readiness_audit.sh:106`](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/eight_h100_20260917/readiness_audit.sh:106); it never registers the kernel. The real execution bootstrap registers it later, but that means this audit-only run cannot establish the claimed complete readiness inventory. **Fix:** register `eight-h100` with the intended virtualenv before the audit check, then rerun and read back a `ready` receipt; do not begin the experiment path from the present blocked receipt.

2. **The current readiness run has no fresh, attempt-bound shared state for the watcher.** The current config is attempt 4, but the newest checked-in/readback shared `run_state.json` is attempt 3, `status: running`, `stage: preparing`. The readiness bootstrap ends after publishing `readiness_audit.json` at [`readiness_audit.sh:109`](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/eight_h100_20260917/readiness_audit.sh:109), so it never replaces that stale state. The full retry bootstrap only isolates the stale state when that separate script runs ([`eight_h100_run.sh:57`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_run.sh:57)-[`eight_h100_run.sh:77`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_run.sh:77)). Meanwhile the watcher accepts any `status: running` state without validating attempt, stage, start, or deadline ([`redesign_h100_watch.py:200`](/Users/choiwuseck/Documents/ChatGPT/데이콘/redesign_h100_watch.py:200)). This can make a new watcher observe attempt 3 or act on a stale terminal state, violating retry isolation. **Fix:** after old watcher termination and before a new watcher starts, atomically publish/read back `run_state.json` with `attempt: 4`, `status: running`, `stage: preparing`, original start/deadline, and `archive_ready: false`; make the watcher reject an attempt/boundary mismatch.

3. **The normal launcher is unsafe to use for this bounded recovery.** [`eight_h100_launch.py:152`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_launch.py:152)-[`eight_h100_launch.py:159`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_launch.py:159) generates the old remote asset path `/root/redesign_full_20260917/assets` and derives a new deadline. Its `start()` path takes a fresh current timestamp and overwrites `config.json` ([`eight_h100_launch.py:314`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_launch.py:314)-[`eight_h100_launch.py:317`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_launch.py:317)), then writes new run metadata with `started + 14_400` ([`eight_h100_launch.py:338`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_launch.py:338)-[`eight_h100_launch.py:352`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_launch.py:352)). Invoking it would reintroduce the failed asset route and extend the immutable run boundary. **Fix:** do not use `start()` for recovery; either remove/guard that recovery-incompatible path or add a narrowly validated recovery route that consumes the already-fixed config and refuses changed start/deadline/assets.

### MEDIUM / P2

None.

### LOW

None.

## Positive verification

- The recovery shell diff sets executable permission before `imageio_ffmpeg.get_ffmpeg_exe()` and verifies the selected binary with `-version` ([`eight_h100_run.sh:96`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_run.sh:96)-[`eight_h100_run.sh:107]).
- Extracted-source validation deliberately excludes only `__pycache__/*.pyc` ([`eight_h100_run.sh:122`](/Users/choiwuseck/Documents/ChatGPT/데이콘/eight_h100_run.sh:122)-[`eight_h100_run.sh:145]).
- The executed readiness receipt did verify FFmpeg execution, all 27 remote asset hashes, all 7,133 source members, AntiDeepfake hashes, imports, one H100, and the corrected remote asset path; kernel registration is the only reported failed check.
- `python3 -m py_compile` succeeded for the three watcher/launcher modules and `eight_h100_run.py`; `bash -n` succeeded for the recovery and readiness scripts. `python3 redesign_h100_watch.py --self-check` passed its archive-safety and deadline-pause-retry checks. These local checks do not supersede the blocked remote readiness receipt.

## Decision

- `codeQualityStatus`: **BLOCK**
- `recommendation`: **REQUEST_CHANGES**
- `blockers`:
  1. Register `eight-h100`, rerun readiness, and obtain/read back `status: ready`.
  2. Publish/read back a fresh attempt-4 shared `run_state.json` with the fixed original boundary before watcher launch.
  3. Keep `eight_h100_launch.py start()` out of this recovery until it cannot regenerate the wrong asset path or a new deadline.
