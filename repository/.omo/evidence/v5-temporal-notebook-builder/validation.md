# V5 temporal-only notebook validation

- Scenario: generate B3/B4-only Colab source labelled as requiring additional budget authorization, without allocating a runtime.
- Invocation: `.venv/bin/python build_v5_temporal_notebook.py`.
- Observable: `DACON_236749_temporal_only_v5.ipynb` contains 30 cells and 22 embedded Python sources.
- Invocation: `.venv/bin/python -` with `nbformat.validate`, byte-exact embedded-source checks, temporal CLI checks, origin-weight-copy exclusion, authorization metadata checks, deadline checks, and sealed/AST staging exclusion.
- Observable: `notebook=DACON_236749_temporal_only_v5.ipynb cells=30 embedded_sources=22 source_content_match=ok temporal_cli_only=ok origin_weight_copy=absent`.
- Artifact: notebook SHA-256 `ccc56330f04fc33e174f0ec0d7893e024bc044bd1cb57d6d0d36c01303640ae3`.
- Preserved artifact: `.omo/evidence/v5-colab-runtime/results/executed.ipynb` remains SHA-256 `4993fb5b14b65d65f25e4ac968202a2b5f8bdd51d6fac48aaf24d40da876d457`.
