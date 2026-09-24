# PANNs v1 execution record

The finalized workflow is being executed. This milestone validates the trainable architecture and file-level loss; it does not establish improved ADS/CPS, select a candidate, or finish W4-W7.

- W0: DataV4 archive SHA-256 `05ae786864dc282de6ae750a451bc297a870be34f2d1e971b14429eca0e47419`, every archive member, and the existing extracted package reverified against the receipt. Equivalence to the user's leaderboard-best submission remains unconfirmed.
- W1: all 43,516 available audio payloads hashed and decoded. Of 44,424 manifest rows, 900 MUSAN noise and eight HAIM files are absent. D0 excludes 1,477 supplemental rows, leaving 42,039 verified rows. Historical roles are preserved; parent closure is checked before admission. Both-absent quality coverage is missing.
- W2: implemented identity-initialized SE, optional temporal pooling, frozen Cnn14 prefix/BN/FC, differentiable 24-feature logistic readout and masked file loss. A 32-file train-only engineering panel (0.448125–69.3125 seconds, 105 segments) uses freshly computed CPU Demucs stems and FP32 block5 caches. Base tags match exactly; maximum presence error is `1.119346799971499e-08`.
- W3: R1/R2/R3/R5 received five real optimizer steps on CPU and MPS; R3 also tested microbatch 1, 2 and 4 on both devices. All intended parameters changed, frozen tensors and BN buffers stayed unchanged, and an in-process checkpoint reload reproduced weights and probabilities exactly. That measurement did not exercise an interrupted CLI process. R3 CPU/MPS first-step gradient relative error was at most `1.3466006794746556e-06`. These are engineering pilots, not quality comparisons or full training runs.

R3 MPS microbatch 4 averaged 0.303 seconds per effective four-file step on this small panel, with maximum conservative RSS-plus-driver sum 3.818 GiB. The panel is not a production throughput guarantee. Whole-dataset preparation, especially fixed Demucs stems, must be accounted for separately. The verified D0 contains 110.41 hours of audio and approximately 119,315 baseline segments; FP32 block5 values alone would occupy about 10.92 GiB, before index/filesystem overhead.

Executable commands are `audit`, `cache` and `pilot` in `repository/panns_adaptation_run.py`. `--run` binds inputs and architecture specification hashes; pilot accepts only fixed v1 arms and microbatch sizes. Checkpoint loading uses `weights_only=True`. The local runs denied network and private-home reads in an OS sandbox. No new paid CPU/GPU instance was created.

`train`/`evaluate`, full D0 caching, final training budget, independent quality evaluation and deployment integration remain subsequent gates. No placeholder implementations or replacement submission were created. Historical design lock files remain unchanged.

Evidence: [engineering validation](evidence/panns_cnn_execution_20260925/engineering_validation.json). Bulk audio, activation and checkpoint artifacts are outside Git under the original project's `_workspace/panns_cnn_v1`.

## Continued execution and review fixes

The initial evidence above is historical. The four Important findings against commit `44eae7b` were addressed in one fix pass. Cache extraction and data admission now live in focused modules; the CLI retains the audit/cache/pilot entry points.

- Versioned noise admission recovered 900 raw sources and regenerated their processed audio. All 900 have different bytes from the historical processed hashes; original hashes, labels, roles, group keys and exposure remain recorded. Repeating regeneration reproduced all 900 new hashes. The full bank audit verified 44,416 files, leaving eight missing HAIM rows. D0 excludes the 1,477 supplemental rows and now contains 42,939 files, including 527 both-absent training files.
- A new 43-file train-only cache includes all 11 recovered codec types. CPU stems and FP32 activations were recomputed with source/runtime provenance. Changed preprocessing/runtime and incomplete-pilot overwrites are rejected.
- R3 CPU and MPS now explicitly train on the longest 69.3125-second file. MPS microbatch 4 measured 4.738 GiB conservative RSS-plus-driver peak. CPU/MPS first-gradient relative L2 difference was at most 1.853e-6. Frozen tensors remained unchanged.
- An actual MPS CLI process was terminated after its step-two checkpoint. An ordinary restart was rejected without changing the checkpoint. Explicit resume completed, and a separate uninterrupted run matched all final trainable weights, recorded probabilities and sampler order exactly. Replay verification shares the resource budget; total continuous-plus-replay updates cannot exceed 200.
- The 43-file Demucs device probe measured CPU 137.096 s versus MPS 23.164 s (5.92x). MPS presence error reached 3.75545e-5, exceeding the predeclared 1e-5 tolerance. MPS stems were **not adopted** and the CPU cache remains unchanged. This is an engineering discrepancy, not a measured ADS/CPS regression.

W1 admission and the scoped engineering corrections are complete. W3 still requires a locked production budget; W4-W7 remain unexecuted. Full D0 preparation, cost reduction for fixed stems, full training/evaluation and independent candidate validation remain outstanding. No paid instance or official submission was started.

Evidence: [review-fix validation](evidence/panns_cnn_execution_20260925/review_fix_validation.json). This receipt records executor verification and the original review findings; it does not claim another reviewer approved the new commit.
