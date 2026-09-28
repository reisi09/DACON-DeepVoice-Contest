# VESSL Cloud support request — stalled Ceph I/O and stuck workspace pause

Draft only; not sent. Contact: support@vessl.ai (https://docs.cloud.vessl.ai/guides/get-started/support).

Please recover the existing workspace `wsp-5i3knd3gxeox` (jolly-routine, cluster-betelgeuse, A100 x1), preserving its identity and all attached volumes. Do not terminate/delete it or its data. Its persistent `/root` is Cluster volume `clustervol-4jnjmzdw5rik`; the deployment is on Object volume `objvol-a5gxdg7er70i`.

On 2026-09-19, feature-extraction progress stopped at 13:39:03 UTC. GPU worker PID3469 was blocked in write(fd1), and notebook controller PID231 was blocked in fsync(fd78). Both were D-state with `folio_wait_bit_common`; both file descriptors belonged to the Ceph `/root` mount. Memory PSI was zero and cgroup OOM/OOM-kill counters were zero; I/O PSI full avg10 was84.16. The specific backend cause remains unknown.

Pause was accepted at approximately13:58 UTC. At14:07:28 UTC the workspace still showed `state=stopping`, `phase=standby`. Jupyter subsequently returned HTTP503; new kernel/terminal access could not be established. No restart or termination has been issued. Please inspect stuck container/storage operations, complete a safe stop, and confirm GPU resource release and billing treatment since the accepted pause.

The feature-cache index records681 entries. Its index and expected hashes were saved locally; individual cache-file backup could not complete because HTTP503 prevented reads. Original files remain on the persistent volume but require re-verification after recovery. The hotfixed notebook, author archive and deployment manifest on Object storage were independently SHA256-verified against local copies.

After the platform returns the existing workspace to a usable stopped state, it can be restarted with the same A100 resource. There is no startup script. The notebook at `/root-1/existing-only-9504-20260919/DACON_Existing9504_A100.ipynb` should resume `/root/dacon_existing9504_a100_20260919`, verify cached entries, and avoid duplicate feature/training processes.
