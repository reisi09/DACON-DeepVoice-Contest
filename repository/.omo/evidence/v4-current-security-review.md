# V4 current security review

**Verdict: PASS**

Reviewed immutable source commit `d2397ad8cc96667e5bc2e4132d86405afe093f4e` in `/tmp/dacon-v4-review-d2397ad`, with a clean detached worktree, against delta `a01a50e0c774f181c20238109127c5b81cc95dc1..d2397ad8cc96667e5bc2e4132d86405afe093f4e`. Context: `/Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-final-current-context.md`.

## Security boundary results

- **Dependency mutability: PASS.** Both execution paths pin `imageio-ffmpeg==0.6.0` and `safetensors==0.8.0`; requested and resolved versions are persisted in `environment.json` (`build_v4_notebook.py:77-83`, `:316-325`). This resolves the two prior mutable direct-dependency blockers.
- **Outer baseline ZIP: PASS.** The full notebook no longer copies or extracts the outer `open.zip`. It accepts exactly one safe basename `baseline_submit.zip`, requires the exact declared uncompressed size `4562809110`, bounds compressed size, streams only that member, then requires exact output bytes and SHA-256 `cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70` before nested extraction (`build_v4_notebook.py:233-241`, `:342-343`). Other outer members are never extracted.
- **Archive paths, hashes, and sizes: PASS.** All staged input archives are reconstructed only from receipt-listed Drive parts after per-part byte-count and SHA-256 checks; assembled archives are checked again before extraction. ZIP members with absolute paths or `..` components are rejected (`build_v4_notebook.py:226-240`, `:326-340`; matched path `:62-64`, `:83-91`). These archives are fixed by their full SHA-256 receipts, so subsequent extraction is bound to known bytes.
- **Matched recheck data scope: PASS.** Its builder filters inputs to exactly train, dev, metadata, and SONICS receipts and requires exactly four. The generated notebook does not stage AST, sealed data, the official baseline, or `open.zip` (`build_v4_notebook.py:34-36`, `:93-99`). The raw G0 file, train manifest, and dev manifest are SHA-bound; the wrapper additionally requires exact ordered dev IDs and counts of 1,116 eligible training rows and 210 development rows (`recheck_v4_matched.py:53-68`).
- **Deadline/resource boundary: PASS.** The matched wrapper computes `min(user_deadline, allocation_start + 7080)` and rejects expired execution. The generated command pins the original start `1789283305` and deadline `1789290385`; the notebook watchdog and polling loop enforce release with a 60-second shutdown margin (`recheck_v4_matched.py:53-55`; `build_v4_notebook.py:42-48`, `:99-110`). The full path likewise derives `SAFE_DEADLINE` from the allocation start and clamps the runner deadline.
- **Network boundary: PASS.** Network use is confined to the explicit pinned package installation and Colab Drive mount. Training subprocesses set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; no new HTTP client, URL fetch, upload, submission, or external service call was introduced (`build_v4_notebook.py:77-101`, `:316-352`).
- **Secrets: PASS.** Review of the changed production sources and generated matched notebook found no embedded API key, access token, password, private key, bearer credential, or submission credential. Environment inheritance is used only to start the local runner; the recorded command contains paths, hashes, and deadlines, not credentials.
- **Source identity: PASS.** Current hashes are: `build_v4_notebook.py` `1fd3f190b0db7730331036798e34468182e7f93ad9f552685bd7ba75cffd4982`; `recheck_v4_matched.py` `130e078b393f627b871e549e695e9fe8f4061cfe10a1a16494b60d528aab825c`; `DACON_236749_v4_matched_recheck.ipynb` `3d465f9285c887dbe16db10deaaef7d70e721a301380fab98cfd201a1cc661c1`; recheck receipt `a23b52a27afca9494f0d249ebfaeb66d3e1c177dc660d7de5e0530674d2d19f3`. The receipt lists all 14 canonical embedded/runtime sources, including the current wrapper hash.

## Findings

No security blocker was found in the requested delta. No GPU, browser, network probe, commit, or source mutation was performed. Review applied the `omo:programming` boundary/error-handling criteria and the scoped security-diff criteria; unrelated architecture and service exposure were not audited.

**Recommendation: PASS at exact full SHA `d2397ad8cc96667e5bc2e4132d86405afe093f4e`.**
