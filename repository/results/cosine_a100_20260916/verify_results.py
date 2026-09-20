"""Recompute measured scores and check the locked experiment's execution evidence."""
import hashlib,json,sys
from pathlib import Path
import nbformat
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from music_temporal import load_temporal_head
from temporal_resolution import select_resolution
from temporal_experiment import music_metrics
from graph_scoring import official_metrics
from v6_data import assert_disjoint

def read(p):return json.loads(p.read_text())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 unpacked=ROOT/'unpacked';out=unpacked/'results'
 assert read(unpacked/'run_state.json')['stage']=='complete'
 nb=nbformat.read(unpacked/'C08_Cosine_A100_Executed.ipynb',as_version=4)
 assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
 split=read(out/'split.json');assert split==read(ROOT.parent/'temporal_h100_compact_20260915/unpacked/results/split.json')
 for a,b in [('train','tune'),('train','test'),('tune','test')]:assert_disjoint(split[a],split[b])
 protocol=read(out/'search_protocol.json');trials=read(out/'tuning.json');selection=read(out/'selection.json')
 assert len(trials)==6 and len({r['name'] for r in trials})==6 and not selection['test_used']
 for trial in trials:
  folder=out/trial['name'];trace=read(folder/'training.json')
  recipe=dict(protocol['recipe'],seed=trial['seed'],input_dim=768 if trial['setting']=='M95_cosine' else 1024)
  assert trace['recipe']==trial['recipe']==recipe
  assert trace['train_ids']==[r['ID'] for r in split['train']]
  assert trace['validation_ids']==[r['ID'] for r in split['tune']]
  assert trace['checkpoint_sha256']==digest(folder/'temporal_music.pt')
  assert recipe['cosine_output'] and not recipe['mlp_output'] and recipe['label_smoothing']==.05
  assert recipe['lr_plateau'] and len(trace['learning_rates'])==trace['epochs_completed']
  rate=recipe['learning_rate'];best=float('inf');stale=0
  for actual,loss in zip(trace['learning_rates'],trace['validation_losses'],strict=True):
   np.testing.assert_allclose(actual,rate,atol=1e-15,rtol=0)
   if loss<best-recipe['min_delta']:best=loss;stale=0
   else:stale+=1
   if stale>=5:rate=max(rate*.5,1e-6);stale=0
  model=load_temporal_head(folder/'temporal_music.pt')
  assert model.temporal_bins==recipe['temporal_bins'] and model.softmax_output==recipe['softmax_output']
  assert sum(p.numel() for p in model.parameters())==trace['parameters']==trial['parameters']
  assert trace['selected_epoch']==int(np.argmin(trace['validation_losses']))+1
  with np.load(folder/'tune_predictions.npz') as saved:
   assert saved['ids'].tolist()==trace['validation_ids']
   metrics=music_metrics(split['tune'],saved['probability'])
  for key,value in metrics.items():np.testing.assert_allclose(value,trial[key],atol=1e-12,rtol=0)
  np.testing.assert_allclose(metrics['balanced_bce'],trace['best_validation_loss'],atol=2e-6,rtol=0)
 for group in selection['summaries']:
  rows=[r for r in trials if r['setting']==group['name']]
  assert {r['seed'] for r in rows}=={41,42,43}
  for key in ['music_eer','balanced_bce']:
   np.testing.assert_allclose(group[key+'_mean'],np.mean([r[key] for r in rows]),atol=1e-12)
  np.testing.assert_allclose(group['music_eer_std'],np.std([r['music_eer'] for r in rows]),atol=1e-12)
 winner=select_resolution(selection['summaries'])
 assert selection['winner']==winner
 report=read(out/'test_report.json');assert report['selection_sha256']==digest(out/'selection.json')
 baseline=read(out/'Official_baseline_test_predictions.json')
 assert len(report['results'])==8
 for result in report['results']:
  predictions=read(out/(result['name']+'_test_predictions.json'))
  metrics=official_metrics(split['test'],predictions)
  for key,value in metrics['metrics'].items():
   if value is None:assert result['metrics']['metrics'][key] is None
   else:np.testing.assert_allclose(value,result['metrics']['metrics'][key],atol=1e-12,rtol=0)
  assert {k:v for k,v in metrics.items() if k!='metrics'}=={k:v for k,v in result['metrics'].items() if k!='metrics'}
  for pred,old in zip(predictions,baseline,strict=True):
   for key in ['ID','VOICE_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB']:assert pred[key]==old[key]
   if result['name']!='Official_baseline':assert pred['FILE_FAKE_PROB']==max(pred['VOICE_FAKE_PROB']*pred['VOICE_PRESENT_PROB'],pred['MUSIC_FAKE_PROB']*pred['MUSIC_PRESENT_PROB'])
 features=read(out/'features_manifest.json')
 rows=[r for part in split.values() for r in part]
 assert features['records']==read(unpacked/'reference330_features_manifest.json')['records']
 assert set(features['records'])=={r['ID'] for r in rows}
 for row in rows:
  item=features['records'][row['ID']]
  assert item['identity']==row['audio_sha256']+':'+features['cache_id']
  assert item['audio_sha256']==row['audio_sha256']
  assert item['music_1_sha256']!=item['music_8_sha256']
 result=dict(checkpoints=6,tune_predictions_recomputed=6,historical_reports_recomputed=8,
             audio_cache_receipts=len(rows),winner=winner['name'],selection_sha256=report['selection_sha256'])
 (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
