from pathlib import Path
import shutil,json,hashlib,zipfile,nbformat as n
root=Path.cwd();stage=root/'_workspace/mert330_mlp_input';stage.mkdir(exist_ok=True)
old=root/'results/temporal_h100_resolution_20260915/unpacked'
for name in ['music_temporal.py','temporal_augmentation.py','temporal_experiment.py','temporal_resolution.py','temporal_tuning.py','temporal_backbone.py','v6_core.py','v6_data.py','v6_train.py','v6_runtime.py','v7_ablation.py','c3_data.py','graph_scoring.py','v7_notebook_runner.py']:
 shutil.copy2(root/name,stage/name)
for name in ['locked_split.json','full_data_plan.json','baseline_predictions.json']:
 shutil.copy2(old/name,stage/name)
shutil.copy2(old/'results/features_manifest.json',stage/'reference95_features_manifest.json')
shutil.copy2(root/'results/temporal_h100_resolution_20260915/existing_audio_map.json',stage/'existing_audio_map.json')
shutil.copy2(root/'_workspace/mert330_metadata.json',stage/'mert330_metadata.json')
shutil.copytree(root/'_workspace/mert330_source',stage/'assets/mert',dirs_exist_ok=True)
for seed in [41,42,43]:
 target=stage/f'results/M95_seed{seed}';target.mkdir(parents=True,exist_ok=True)
 for name in ['temporal_music.pt','training.json','tune_predictions.npz']:
  shutil.copy2(old/f'results/R1_sigmoid_seed{seed}'/name,target/name)
shutil.copy2(old/'results/linear_control.npz',stage/'results/linear_control.npz')
shutil.copy2(old/'results/linear_control.json',stage/'results/linear_control.json')
req=(root/'_workspace/temporal_resolution_input/requirements-resolution.txt').read_text()+'torchcodec==0.8.1\n'
(stage/'requirements-backbone.txt').write_text(req)
nb=n.v4.new_notebook(cells=[n.v4.new_markdown_cell('''# C08: MERT-95M versus MERT-330M

Frozen pretrained encoder changes from 768 to 1024 output features. The classifier remains a 64-channel three-layer residual Conv1D with GELU, max pooling, and a 64-to-32-to-2 MLP (GELU, dropout 0.1) with two-class softmax. Training uses class-balanced cross-entropy on raw logits; fake is class 1. AdamW weight decay is 0.01; other training settings are unchanged. Both backbones train fresh heads with seeds41–43 (six fits). MERT95 reuses verified cached features; prior unregularized checkpoints are retained only as historical inputs. MERT330 uses the same raw audio and separation, newly extracted once. Tune-only selection precedes historical evaluation. Original submission ZIP is preserved.'''),n.v4.new_code_cell('from temporal_backbone import main\nmain()'),n.v4.new_code_cell('''import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
out=Path('results')
selection=json.loads((out/'selection.json').read_text())
lookup={r['name']:r for r in json.loads((out/'test_report.json').read_text())['results']}
summary=[]
for group in selection['summaries']:
    rows=[lookup[k] for k in group['candidates']]
    summary.append(dict(name=group['name'],tune_eer=group['music_eer_mean'],tune_bce=group['balanced_bce_mean'],
      test_music_eer=float(np.mean([r['music_eer'] for r in rows])),
      test_file_eer=float(np.mean([r['metrics']['metrics']['file_eer'] for r in rows]))))
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,key,title in zip(axes,['tune_eer','test_music_eer','test_file_eer'],['Tune music EER','Historical music EER','Historical file EER']):
    bars=ax.bar(['MERT95 / 768','MERT330 / 1024'],[r[key] for r in summary],color=['#6c8eb4','#ee9b49'])
    ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,.65);ax.set_title(title+' (lower better)');ax.grid(axis='y',alpha=.2)
fig.savefig(out/'comparison.png',dpi=170);plt.show()
(out/'summary.json').write_text(json.dumps(dict(groups=summary,winner=selection['winner']['name']),indent=2))
print(json.dumps(summary,indent=2))''')])
nb.metadata={'kernelspec':{'display_name':'Python (Temporal H100)','language':'python','name':'temporal-h100'}}
n.write(nb,stage/'C08_MERT330_H100.ipynb');shutil.copy2(stage/'C08_MERT330_H100.ipynb',root/'C08_MERT330_H100.ipynb')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='input_manifest.json'}
(stage/'input_manifest.json').write_text(json.dumps({'sha256':manifest},indent=2))
archive=root/'results/mert330_h100_20260916/C08_MERT330_MLP_Input.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in stage.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(stage))
r=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive));archive.with_suffix('.json').write_text(json.dumps(r,indent=2));print(r)
