# Corrected V5 rerunnable notebook validation

- Scenario: regenerate the corrected source-only notebook after the flexible temporal-phase and reporting fixes, without allocating or running a GPU session.
- Invocation: `.venv/bin/python build_v5_notebook.py`.
- Observable: `DACON_236749_compare_v5.ipynb` was regenerated with 31 cells and 22 embedded Python sources.
- Invocation: `.venv/bin/python -` with `nbformat.validate`, exact embedded-source content checks, corrected-metadata checks, analyzer-finally release checks, and forbidden sealed/AST staging checks.
- Observable: `notebook=DACON_236749_compare_v5.ipynb cells=31 embedded_sources=22 source_content_match=ok corrected_metadata=ok analyzer_finally_release=ok`.
- Artifact: regenerated notebook SHA-256 `9b995465ca6333ca1b228aecc118309aa9b616805a9e2b6cc24a6c47a521609f`.
- Preserved execution artifact: `.omo/evidence/v5-colab-runtime/results/executed.ipynb` SHA-256 remains `4993fb5b14b65d65f25e4ac968202a2b5f8bdd51d6fac48aaf24d40da876d457`.
