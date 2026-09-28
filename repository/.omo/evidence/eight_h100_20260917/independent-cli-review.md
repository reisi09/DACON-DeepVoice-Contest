# Independent Codex CLI plan review receipt

## Machine receipt

```json
{
  "reviewer": "codex-cli:configured-default",
  "target": ".omo/plans/eight-h100-20260917.md",
  "plan_sha256": "c68cce193ed3c5428db36fb2bc2375ee1f332b39d2acaf3e2f99502b7d12cae0",
  "source_size_bytes": 15960,
  "review_workspace": "/tmp/eight-h100-cli-review.rhJ61U",
  "runtime_home": "/tmp/eight-h100-cli-review.rhJ61U/codex-home",
  "verdict": "INCONCLUSIVE",
  "reason": "The isolated CODEX_HOME had no authentication. codex login status returned Not logged in; codex exec reached the API but failed with HTTP 401 Unauthorized / Missing bearer or basic authentication.",
  "review_started": "2026-09-17T10:42:55Z",
  "review_exit_code": 1,
  "cleanup": {
    "process_check": "No codex exec process remained after exit (the only pgrep match was the checking shell/rg command).",
    "temporary_workspace_removed": true,
    "removed_path": "/tmp/eight-h100-cli-review.rhJ61U"
  }
}
```

## Binding and invocation

The source plan and disposable copy were both read through an `openat`/`O_NOFOLLOW` descriptor chain (`workspace/.omo/plans/eight-h100-20260917.md`), and both produced SHA-256 `c68cce193ed3c5428db36fb2bc2375ee1f332b39d2acaf3e2f99502b7d12cae0` over 15,960 bytes.

Exact bounded invocation (configured default model; no model override):

```sh
cd /tmp/eight-h100-cli-review.rhJ61U && CODEX_HOME=/tmp/eight-h100-cli-review.rhJ61U/codex-home perl -e '$SIG{ALRM}=sub { exit 124 }; alarm 270; exec @ARGV' codex exec --ephemeral --skip-git-repo-check -s read-only --json -o /tmp/eight-h100-cli-review.rhJ61U/last-message.txt "$prompt" > /tmp/eight-h100-cli-review.rhJ61U/events.jsonl 2> /tmp/eight-h100-cli-review.rhJ61U/stderr.log
```

The prompt required the first action to read the exact target path and restricted review to the plan and eight copied references. No product files, GPU jobs, network resources, or external state were mutated.

## Binary observable

- `codex login status` under the isolated runtime home: `Not logged in`.
- CLI process exit: `1`.
- Captured event: `turn.failed` with `unexpected status 401 Unauthorized: Missing bearer or basic authentication`.
- Captured stderr also recorded the same 401 on WebSocket and HTTPS fallback attempts.
- No `last-message.txt` was produced, so no review verdict was claimed. The only valid lane verdict is `INCONCLUSIVE`.

Captured artifact: `.omo/evidence/eight_h100_20260917/independent-cli-review.md`.
