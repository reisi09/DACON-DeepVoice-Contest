"""Seal the existing MLP experiment for A100 without changing model settings."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import nbformat

root=Path.cwd()
output=root/'results/mert330_a100_20260916'
stage=root/'_workspace/mert330_a100_input'
stage.mkdir(exist_ok=True)
source=root/'results/mert330_h100_20260916/C08_MERT330_MLP_Input.zip'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(source)=='2147a0169ec1ff1a8b098e2033d6c7de91d15637e5023b8a71225ea12424d138'
with zipfile.ZipFile(source) as z:
    assert all((stage/name).resolve().is_relative_to(stage) for name in z.namelist())
    for name,digest in json.loads(z.read('input_manifest.json'))['sha256'].items():
        assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    z.extractall(stage)
shutil.copy2(root/'temporal_backbone.py',stage/'temporal_backbone.py')
notebook=nbformat.read(stage/'C08_MERT330_H100.ipynb',as_version=4)
notebook.cells[0].source+='\n\nA100 run: two-hour compute limit; same six MLP head fits. The frozen MERT95 feature cache was extracted on H100 and is reused with its original provenance; new MERT330 extraction and both classifier training arms run on A100.'
notebook.metadata.kernelspec=dict(display_name='Python (Temporal A100)',language='python',name='temporal-a100')
nbformat.write(notebook,stage/'C08_MERT330_A100.ipynb')
(stage/'C08_MERT330_H100.ipynb').unlink()
shutil.copy2(stage/'C08_MERT330_A100.ipynb',root/'C08_MERT330_A100.ipynb')
manifest={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='input_manifest.json'}
(stage/'input_manifest.json').write_text(json.dumps(dict(sha256=manifest),indent=2))
archive=output/'C08_MERT330_A100_Input.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in stage.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(stage))
r=dict(file=archive.name,sha256=sha(archive),bytes=archive.stat().st_size)
archive.with_suffix('.json').write_text(json.dumps(r,indent=2));print(r)
