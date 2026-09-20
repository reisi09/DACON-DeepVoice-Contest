import csv,json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).absolute().parents[2]))
import numpy as np
import nbformat
from eight_metrics import eer,grouped_bootstrap_delta
from localization_export import completion
root=Path('results/expanded_h100_20260918');out=root/'downloaded/unpacked/output';train=out/'train'
r=json.loads((train/'results.json').read_text());assert completion(out,0)[0]
assert r==json.loads((root/'live/train/results.json').read_text())
assert not any(o.output_type=='error' for c in nbformat.read(out/'DACON_Expanded_H100_Executed.ipynb',4).cells if c.cell_type=='code' for o in c.outputs)
pre=json.loads((root/'sampling_preflight.json').read_text())
for cohort in ['expanded','original831']:
 e=json.loads((train/f'{cohort}_exposure.json').read_text())
 for seed in ['41','42','43']:
  assert e[seed]['draw_sha256']==pre[cohort][seed]['schedule_sha256']
  assert not e[seed]['unseen_ids'] and not e[seed]['unseen_parents']
ensembles={};seed_results={};verifications=0
for arm in r['metrics']:
 rows=list(csv.DictReader((train/f'{arm}_ensemble.csv').open()));ensembles[arm]=rows
 saved=[np.load(train/f'{arm}_{seed}_predictions.npz') for seed in [41,42,43]]
 assert all(s['ids'].tolist()==[x['ID'] for x in rows] for s in saved)
 p=np.stack([s['music_probability'] for s in saved]).mean(0)
 np.testing.assert_array_equal(p,np.array([float(x['MUSIC_FAKE_PROB']) for x in rows],dtype=np.float32))
 seed_results[arm]={}
 for split in ['dev','calibration','test','confirmation_external']:
  indices=[i for i,x in enumerate(rows) if x['split']==split]
  y=[int(rows[i]['MUSIC_FAKE']) for i in indices]
  value=eer(y,p[indices])['eer'];assert abs(value-r['metrics'][arm][split]['music_eer'])<1e-12
  seed_results[arm][split]=[eer(y,s['music_probability'][indices])['eer'] for s in saved]
  verifications+=1
all_rows=ensembles['mean'];parents={x['ID']:x['ID'] for x in all_rows};atoms={}
def find(x):
 while parents[x]!=x:
  parents[x]=parents[parents[x]];x=parents[x]
 return x
for row in all_rows:
 for atom in row['group_id'].split('|')+['parent:'+v for v in json.loads(row['parent_ids'])]:
  if atom in atoms: parents[find(row['ID'])]=find(atoms[atom])
  else:atoms[atom]=row['ID']
lookup={arm:{x['ID']:float(x['MUSIC_FAKE_PROB']) for x in rows} for arm,rows in ensembles.items()}
comparisons={}
for split in ['test','confirmation_external']:
 data=[{'label':int(x['MUSIC_FAKE']),'component_id':find(x['ID']),**{a:v[x['ID']] for a,v in lookup.items()}} for x in all_rows if x['split']==split]
 comparisons[split]={}
 for control,candidate in [('original831_mean','mean'),('original831_mean','mil'),('mean','mil'),('mean','ast')]:
  comparisons[split][control+' -> '+candidate]=grouped_bootstrap_delta(data,control,candidate,n_boot=1000,seed=41)
result={'verified_ensemble_metrics':verifications,'ensemble_matches_saved_seed_mean':True,'all_12_locked_checkpoints_match':True,'gpu_sampling_matches_local_schedules':True,'notebook_error_outputs':0,'seed_music_eer':seed_results,'paired_group_bootstrap':comparisons,'bootstrap_seed':41,'bootstrap_samples':1000}
(root/'postrun_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
