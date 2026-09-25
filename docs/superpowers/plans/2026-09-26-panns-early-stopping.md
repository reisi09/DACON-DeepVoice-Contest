# PANNs high-epoch training with early stopping

The user requested a high epoch ceiling and early stopping, explicitly allowing longer training. This supersedes the equal-step, final-checkpoint, four-hour training-phase policy. It does not authorize an unlimited cloud allocation or reset an existing allocation's deadline.

## Training policy

- Pin `early_stopping = {max_epochs: 200, patience: 15, min_delta: 0.0001, monitor: selection_presence_bce}` in the prepared run identity before training. These are starting settings, not empirically optimal hyperparameters.
- Preserve the four arms, three seeds, training data, optimizer, frozen parameters and loss. Evaluate the untrained arm at epoch 0, then validate only after each complete shuffled training pass, including its final partial batch.
- Monitor selection-only file-level presence BCE: compute the mean over known labels separately for voice and music, then average the two target means. Reject missing target coverage or nonfinite values. Development, calibration and test rows never enter this monitor.
- Save weights for every strict improvement of the true minimum validation loss. Reset the patience counter only after a decrease of at least `min_delta` from the previous patience anchor. Stop after 15 consecutive validation epochs without that improvement, or after 200 complete epochs. All conditions use the same rule; their actual training lengths may differ.
- This is a presence-loss stopping rule. It does not establish an ADS improvement; frozen voice/music fake heads remain unchanged. Selection is now reused adaptively for stopping and arm comparison, so its results remain development diagnostics rather than independent evidence.

## Persistence and execution

- Keep the current optimizer/model, sampler, RNG, last validated epoch, patience state and best model together in one atomically replaced checkpoint. Resume uses current training state; evaluation restores the best weights. Do not create an unbounded series of large snapshots.
- An interruption at an epoch boundary must validate that epoch after resume before taking another update or declaring completion. A time limit, signal, partial checkpoint or failed validation is not an early-stopping success.
- Throughput and checkpoint measurements estimate time and storage. They no longer shrink the epoch ceiling to fit a time allowance, and the former 100,000-step cap is removed.
- The external allocation controller and absolute session deadline still bound paid execution. The separate four-hour training-phase limit is removed. Existing releases, deadlines and ownership records remain immutable; any new allocation limit must be established before launching it.
- Report each condition's actual completed epochs, best epoch, validation loss and stop reason. Completion requires a verified `early_stopping` or `max_epochs` receipt bound to the combined checkpoint. Interrupted conditions are excluded from completed quality comparisons.

## Work and verification

1. Trainer/state: implement full-epoch selection validation, best/current checkpoint state and terminal validation. Verify resumed and continuous optimization agree, including the pending-validation boundary; verify wrong-role access and nonfinite scores cannot pass.
2. Profile/pipeline/report: replace equal-step assumptions with the pinned epoch policy and shared completion validation. Keep CUDA parity, real Engine preflight, child cancellation and session-cost controls intact.
3. Preparation/notebook/docs: expose the new policy, remove the old training-hours argument, and describe the metric and evidence limits consistently.
4. Execute a bounded real CPU Cnn14 training/replay using actual cached audio features. Separate fault-injection checks from real model evidence. Review staged source and evidence, commit and push only task changes.

The independent A100 numerical discrepancy remains unresolved. High-epoch training must not bypass the unchanged numerical gate. GPU training and leaderboard improvement are not claimed by implementing this policy.
