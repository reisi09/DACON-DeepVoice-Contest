"""Seal the experiment code, exact prior controls and pinned encoder metadata."""
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat

root=Path(__file__).resolve().parents[2]
p=Path(__file__).resolve().parent;base=root/'results/methods_a100_20260916/unpacked'
with zipfile.ZipFile(root/'results/methods_a100_20260916/C08_Methods_A100_Input.zip') as z:
    files={n:z.read(n) for n in z.namelist() if not n.startswith(('results/','assets/')) and n!='input_manifest.json'}
for name in ['music_temporal.py','temporal_experiment.py','afwhisper_features.py','temporal_afwhisper.py','test_afwhisper.py']:
    files[name]=(root/name).read_bytes()
files['afwhisper_metadata.json']=(p/'afwhisper_metadata.json').read_bytes()
files['results/afwhisper_a100_20260916/source/modeling_whisper.py']=(p/'source/modeling_whisper.py').read_bytes()
for name in ['config.json','NVIDIA_OneWay_Noncommercial_License.docx','README.md']:
    src=p/'source'/name
    if src.exists():files['assets/afwhisper/'+name]=src.read_bytes()
for smoothing in (0,5):
    for seed in (41,42,43):
        for name in ('training.json','temporal_music.pt','tune_predictions.npz'):
            files[f'results/M330_mlp_ls{smoothing}_seed{seed}/{name}']=(base/f'results/mlp_ls{smoothing}_lr0_seed{seed}'/name).read_bytes()
for name in ('linear_control.npz','linear_control.json'):
    files['results/'+name]=(base/'results'/name).read_bytes()
notebook_path=root/'C08_AFWhisper_A100.ipynb'
if not notebook_path.exists():
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# AF-Whisper 637M vs MERT330\n\nFrozen audio encoders; original compact MLP with label smoothing0/0.05, seeds41/42/43. Source-group-disjoint locked split; early stopping max300/patience20/min_delta1e-4. Six new AF637 fits and six sealed MERT controls. No submission replacement.'),nbformat.v4.new_code_cell('from temporal_afwhisper import main\nmain()')],metadata={'kernelspec':{'name':'temporal-a100','display_name':'Python (Temporal A100)','language':'python'}})
    nbformat.write(nb,notebook_path)
files[notebook_path.name]=notebook_path.read_bytes()
files['input_manifest.json']=json.dumps({'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}},indent=2).encode()
out=p/'C08_AFWhisper_A100_Input.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in files.items():z.writestr(name,data)
receipt=dict(file=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(p/'input_verification.json').write_text(json.dumps(receipt,indent=2));print(receipt)
