# V6 submission 89989: NumPy ABI failure

Observed in the user's DACON submission page for competition 236749:

- File: `V6_final_submission.zip`, submitted 2026-09-15 07:22:59.
- Status: submission error; runtime 2 seconds.
- Error: `ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject (File: /app/script.py, Line: 5)`.
- The referenced line imports `run_v6_final`. Its top-level `v6_train` import loads
  `sklearn.linear_model`; scikit-learn's compatibility module imports pandas if
  installed. The ZIP pins NumPy 2.2.6 but did not constrain pandas.

Hypotheses considered: incompatible preinstalled pandas, another NumPy-dependent
binary extension, or model/GPU failure. The reported dtype mismatch establishes
an ABI failure; the 2-second module-import location precedes model loading.
An isolated Python3.11 reproduction with NumPy2.2.6, pandas1.5.3, SciPy1.15.3,
scikit-learn1.8.0 produces the exact error through sklearn -> pandas._libs.interval.
Changing only pandas to 2.2.3 makes the same import succeed. The grader's exact
pandas version and full internal traceback were not exposed, so pandas is the
reproduced likely offender, not a directly observed grader package version.

Fix: add `pandas==2.2.3` to the package requirements. Keep NumPy, all model weights,
all runtime source, and trained heads unchanged. The corrected ZIP is a separate
file, `results/v6_l40s_20260915/V6_final_submission_fixed.zip`.

Reproducible commands (before is expected to fail, after must pass):

```sh
uv run --python 3.11 --with numpy==2.2.6 --with pandas==1.5.3 --with scipy==1.15.3 --with scikit-learn==1.8.0 python -c 'from sklearn.linear_model import LogisticRegression'
uv run --python 3.11 --with numpy==2.2.6 --with pandas==2.2.3 --with scipy==1.15.3 --with scikit-learn==1.8.0 python -c 'from sklearn.linear_model import LogisticRegression'
```

See `before.log`, `after.log`, `reproduction.json`, and `archive_comparison.json`
for the actual outcomes and archive integrity comparison. This local reproduction
does not establish successful full inference on the DACON Linux/L4 environment.
No GPU workspace was started, no retraining performed, and no resubmission made.

Primary reference: https://pandas.pydata.org/pandas-docs/version/2.2/whatsnew/v2.2.2.html
(pandas2.2.2 introduced general NumPy2 compatibility).
