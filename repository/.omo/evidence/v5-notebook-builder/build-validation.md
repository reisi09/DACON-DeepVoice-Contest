# V5 notebook builder validation

- Scenario: build the source-embedded V5 comparison notebook from the current 20-file Python closure.
- Invocation: `.venv/bin/python build_v5_notebook.py`.
- Observable: `DACON_236749_compare_v5.ipynb` was written with 29 cells and 20 `%%writefile /content/dacon_v5/` source cells.
- Invocation: `.venv/bin/python -` with `nbformat.read`, `nbformat.validate`, bootstrap/deadline assertions, exact source-closure assertions, and sealed/AST staging absence assertions.
- Observable: `notebook=DACON_236749_compare_v5.ipynb cells=29 embedded_python_sources=20 closure=ok sealed_ast_staging=absent`.
- Artifact: `DACON_236749_compare_v5.ipynb` SHA-256 `b07959eef66801e866969b106c4812b60f9251ba847e62c53ecbcf9ea9d1bd2c`.
- Builder: `build_v5_notebook.py` SHA-256 `2edfbeb4015f24708e4abf72c0fd8e6ecc99752ccfd5494825c51738b0bb78de`.
