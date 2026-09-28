# V5 temporal recovery evidence

Scenario: a deterministic 20-second training context has a complete aligned 16 kHz raw/voice/music trio, then one stem becomes one frame short.

Invocation: `.venv/bin/pytest -vv test_v5_resume_temporal.py`

Binary observable: `test_cached_trio_is_reused_only_when_context_rate_and_frames_align PASSED`; the valid trio is reused with the exact deterministic offset and the damaged trio is rejected for regeneration.

Captured artifact: `.omo/evidence/v5-resume-temporal-scenarios.log`.

Static checks:

- `uvx ruff check v5_resume_temporal.py test_v5_resume_temporal.py` -> `All checks passed!` in `.omo/evidence/v5-resume-temporal-ruff.log`.
- `uvx basedpyright --level error v5_resume_temporal.py test_v5_resume_temporal.py` -> `0 errors, 0 warnings` in `.omo/evidence/v5-resume-temporal-basedpyright.log`.
- Python no-excuse checker -> `no violations in 2 file(s)` in `.omo/evidence/v5-resume-temporal-no-excuse.log`.
