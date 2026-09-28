# C3 final submission — complete

Submit `C3_full_clean_submission.zip` directly (6,005,189,559 bytes). Do not upload the smaller `C3_final_results.zip`, which is the experiment record.

The submission has root `script.py` and `requirements.txt`, offline model assets, runtime code, and trained C3 heads. No training audio is included. SHA256 matches the server artifact; all ZIP member CRCs and paths passed. Downloaded runtime/head members match the actual packaged CLI used in the A100 smoke check.

Full-data training: all eligible public train/dev/sealed data, 1,206 music-head files and 1,886 calibrator files; 300 head epochs, frozen encoders, five source-component-disjoint OOF folds for calibration. Exclusions are recorded in `unpacked/exclusions.csv`.

Actual packaged CLI and reference predictions on four real inputs matched exactly. All five notebook code cells completed without errors. This is not a full L4 runtime test. Since all public splits were used for final fitting, no independent final holdout score is available. The user subsequently supplied a scored DACON submission receipt; see the result below.

The A100 workspace is confirmed stopped; current billing burn is zero. This run used 1.183908 credits based on the observed balance change, leaving 12.793524 credits. See `download_and_pause.json` for verification evidence and `unpacked/C3_FullData_Executed.ipynb` for the executed notebook.

## User-supplied leaderboard result

Submission 90217, displayed submission time 2026-09-15 14:06:12: score 0.6843536931, ADS 0.6504603175, CPS 0.9893940741, server duration 21m24s. The screenshot omits headers; metric identities follow the previously verified column order and reproduce the displayed total via 0.9*ADS + 0.1*CPS. Original official baseline submission 88418 scored 0.6909110688, ADS 0.6577539683, CPS 0.9893249735.

C3 is worse by 0.0065573757 in total score. The ADS decline (-0.0072936508) accounts for the regression; CPS is nearly unchanged (+0.0000691006). Aggregate scores do not identify which of file, voice or music EER regressed. Exploratory dev improvements did not translate to a baseline-beating full-data submission. This evidence does not isolate the effect of data filtering, the final OOF calibrator, the music head or the replaced voice path. The user submitted the ZIP; the assistant performed no submission and has not restarted a GPU.
