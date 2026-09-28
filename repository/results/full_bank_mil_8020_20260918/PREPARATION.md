# Full-bank H100 run

- Admitted: 16,042 = 15,167 bank originals + 875 legacy/constructed recordings.
- Train 12,826; validation 3,216. Connected source/artist/prompt/speaker identities do not cross the split.
- One exact duplicate and one 27 ms unusable recording excluded; 84 superseded legacy crops replaced by full originals. Soft warnings retained.
- Every admitted original uploaded without transcoding: 15,054,431,524 bytes in 56 ZIP shards. Deployment maps each ID to its verified original SHA256 and relative archive path.
- Same base HTDemucs, frozen MERT330M layer12, WindowMIL (1024→128→1), frozen PANNs, DF-Arena original fc5 (1280→2). Three seeds 41/42/43.
- All-duration windows, no source quota. Every training row once per epoch. Unknown component/local labels remain unknown.
- Training uses validation EER then BCE; 50 epochs maximum and 5 stale epochs for convergence, no wall-clock limit. No independent-test or official-score claim.
- Old local holdouts and Mustango participate in the new split as explicitly requested. The new validation set is development data and is not historically independent.
- Peak ceiling .6 applies to new originals. Legacy/constructed audio retains its prior gain; 38 entries in the earlier QC catalogue exceed .6. Submission inputs use the .6 ceiling, so exact validation-cache parity must not be claimed for those legacy entries.
- H100 feature extraction is running in existing workspace `wsp-9683432r72m7`; completion is pending. First runtime probe verified 41 cached files, GPU utilization 94%, and MERT batch 256 with measured peak allocation 26,929,689,088 bytes. `full_bank_ops.py watch` polls every 5 minutes, verifies the downloaded result archive, then stops GPU compute.

Meaningful CPU verification: complete epoch coverage, validation optimizer rejection, MIL forward and gradient parity, original fc5 gradient and fake-index parity, unknown-label masks, and truncated/tampered cache rejection. Deployment manifest verifies all 16,042 original hashes and paths; remote shards verified against multipart ETags. Bootstrap re-hashes every shard and audio file.
