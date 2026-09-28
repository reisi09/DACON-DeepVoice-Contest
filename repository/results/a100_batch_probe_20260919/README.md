# A100 batching probe — not executed

A bounded train-only Demucs/MERT batching benchmark was prepared in `benchmark.py` but was not uploaded or executed. No production pause signal or runtime modification was made.

Live inspection found the production feature worker and notebook controller blocked again on fsync; the worker descriptor resolves to the Ceph mount at `/root`. Progress stayed at 2,978/9,504 from 14:48:41 through 14:55:41 UTC. See `blocked_by_ceph.json`. The previous replacement-workspace health checks were valid at their observation time but did not establish that the storage problem was permanently resolved.

Do not use the unexecuted benchmark as speedup/parity evidence. Resolve storage write availability before attempting concurrent or paused-worker GPU benchmarking. In particular, the existing worker retains GPU allocations while blocked.
