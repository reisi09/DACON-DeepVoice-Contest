# Existing9504 I/O recovery — 2026-09-19

Not resolved as of14:07UTC: pause accepted, workspace remains stopping/standby; restart not issued. Existing recurring heartbeat remains PAUSED by user request.

See `io_diagnosis.json` for actual blocked syscalls and storage/memory evidence, `current_state.json` for the latest control-plane observation, `cache_index_before.json` for the681-entry recoverable index, and `object_volume_hotfix_verification.json` for preserved notebook/archive/deployment hashes. A local full-cache backup was attempted but failed HTTP503; no claim that all681 cache files were backed up or independently revalidated.

`SUPPORT_REQUEST.md` is a concrete unsent support draft. No data, volumes, or workspace were deleted; no model, split, or training logic was changed. The underlying Ceph/backend cause is not visible to the notebook user.

Official workspace lifecycle reference: https://docs.cloud.vessl.ai/cli/commands/workspace . Supported CLI commands provide pause/start; terminate is permanent deletion and was not used. The current blocker is the platform not completing its stop transition.

User authorized workspace replacement afterward. See `../workspace_replacement_20260919/README.md` for subsequent recovery; the status above is an earlier snapshot.
