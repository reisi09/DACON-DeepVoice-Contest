# V5 final notebook builder validation

Scenario: build the source-embedded B1--B4 Colab artifact only after its complete local source closure is present.

Invocation:

```sh
uv run build_v5_final_notebook.py
uv run --no-project python - <<'PY'
import nbformat
n = nbformat.read('DACON_236749_final_B1_B4.ipynb', as_version=4)
nbformat.validate(n)
for index, cell in enumerate(cell for cell in n.cells if cell.cell_type == 'code'):
    source = cell.source.split('\n', 1)[1] if cell.source.startswith('%%writefile ') else cell.source
    compile(source, f'cell_{index}', 'exec')
print(len(n.cells))
PY
git diff --check -- build_v5_final_notebook.py DACON_236749_final_B1_B4.ipynb
```

Binary observables: `nbformat_valid=true`; 28 code cells compiled; 24 current source files embedded; all code-cell outputs empty; whitespace check passed. A separate SHA-256 comparison of every embedded `%%writefile` payload against `SOURCES` reported `source_closure_hash_match=true count=24`.

Artifacts:

- `build_v5_final_notebook.py` SHA-256: `b3e87a2417043c15104d5263d2fe262ce0221aeb8cde6e9dd18715b5f2db76e4`
- `DACON_236749_final_B1_B4.ipynb` SHA-256: `e5102329e4808de06d307fd19a1b0f7aa6724812fae494f333ad8838302d5614`

Scope boundary: this is static builder validation only. The notebook has not mounted Drive, allocated a GPU, run the refit, created package ZIPs, or submitted to DACON.
