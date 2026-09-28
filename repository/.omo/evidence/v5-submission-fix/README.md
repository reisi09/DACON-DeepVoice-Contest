# B1/B2 submission dependency repair

## Observed failure

DACON submission 89372, V5_B1.zip, 2026-09-14 11:17:33 KST, failed after three seconds:
`OSError: Could not load this library: /usr/local/lib/python3.11/dist-packages/torchaudio/lib/libtorchaudio.so` at `/app/script.py:13`.
B2 was not visible in the latest submission table; its exact server error was not independently observed.

The downloaded B1 SHA-256 matches the previously delivered archive:
`77f949d126ab7f22f3ebbaa2839febcc30b5c2bc22761b143e4e1b28beef8e9d`.

## Mechanism and change

`timm==1.0.15` requires unpinned torchvision. Real pip resolution with simulated official installed metadata chose torchvision 0.29.0, which requires torch 2.14.0, leaving torchaudio 2.7.1+cu128 unchanged. Matching torchvision 0.22.1+cu128 and exact official torch/torchaudio pins prevent that upgrade. See `dependencies/README.md` for evidence and simulation limits.

Only requirements change in the ZIPs. No retraining, architecture changes, label changes, or weight modification. The local notebook's embedded packager is synchronized with current source, including earlier selected-arm/compression fixes.

## Verification

- Original B1 executes successfully on macOS Python 3.11.15 with torch/torchaudio 2.7.1 and torchvision 0.22.1. Two real complete audio files produce the required ordered CSV in 85.52 seconds, maximum resident set 9,292,890,112 bytes. This independently excludes corrupt B1 weights or an unconditional entrypoint failure.
- The existing behavioral checks pass for tail coverage, aggregation, file ordering, decoder agreement, archive content, and selected-arm packaging (`test_v5_submission.py`).
- Linux execution and CUDA library checks are recorded separately under `runtime/`; do not treat the macOS run or resolver metadata as L4 evaluation.
- Actual clean Linux installation of the fixed requirements successfully imports `torch 2.7.1+cu128`, `torchaudio 2.7.1+cu128`, and `torchvision 0.22.1+cu128`. See `runtime/cuda128-install.log`. This check used a CPU Colab VM and did not exercise CUDA kernels or L4 throughput.
- No DACON submission is performed by this repair.
- A separate CPU Linux B1 inference attempt exceeded its 1,200-second QA timeout before producing a CSV. B2 inference was not reached. This is not an L4 runtime measurement or a successful end-to-end Linux validation. At the user's direction, further inference was stopped in favor of original-versus-fixed archive comparison; no GPU was allocated.
- `compare_repair_v5_pair.py` validates each original archive against its previously delivered SHA-256, checks identical member sets without duplicates, and compares every unchanged member by SHA-256. It replaces the two Drive archives only after both comparisons pass. The resulting receipt records the actual final archive hashes and sizes.

## Drive cleanup

Eight old V1/V2/V4/recovery/AST experiment folders were moved to Trash through Aside. The UI confirmed eight items moved; a fresh connector listing returned only `runs_final_v5` and `runs_compare_v5` as direct experiment folders. Both contain B1/B2 checkpoints and records and are retained, together with source datasets and reproducibility inputs. Five unused AST asset parts and the obsolete V2 upload manifest were also moved to Trash. See `drive-cleanup-plan.json`. Trash was not emptied and no storage reclamation is claimed.

## Delivered comparison result

Both archives passed member-by-member SHA-256 comparison: 54 unchanged members each, requirements.txt as the only changed member. `dependency_fix_comparison.json` was fetched back from Drive, and Drive metadata confirms final byte sizes. The existing Drive files were renamed V5_B1_fixed.zip and V5_B2_fixed.zip after replacement.

- B1: 4,546,433,930 bytes; SHA-256 `0f4c9e705ee6162a3a7267810cd6a1993876e81315dba3b13126c7f132ad2e5c`.
- B2: 4,547,606,458 bytes; SHA-256 `b5b79050141df0a33e52327b15b2caa105d226f72b4da624ee90a96f63e1d575`.
