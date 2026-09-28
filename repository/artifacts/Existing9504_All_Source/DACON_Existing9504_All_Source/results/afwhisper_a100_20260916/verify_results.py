"""Recompute larger-backbone metrics and audit actual early stopping and provenance."""
import json
from pathlib import Path
import sys

import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from graph_scoring import official_metrics
from music_temporal import load_temporal_head
from temporal_experiment import music_metrics
from temporal_resolution import select_resolution
from v6_data import assert_disjoint,sha256


def main():
    run=ROOT/'unpacked';out=run/'results'
    assert json.loads((run/'run_state.json').read_text())['stage']=='complete'
    nb=nbformat.read(run/'C08_AFWhisper_A100_Executed.ipynb',as_version=4)
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
               for c in nb.cells if c.cell_type=='code')
    split=json.loads((run/'locked_split.json').read_text())
    for a,b in [('train','tune'),('train','test'),('tune','test')]:assert_disjoint(split[a],split[b])
    parity=json.loads((out/'encoder_parity.json').read_text())
    assert parity['audio_parameters']==636968960 and parity['trainable_audio_parameters']==0
    assert parity['full_masked_vs_trimmed_max_abs']<=1e-4
    manifest=json.loads((out/'afwhisper_features_manifest.json').read_text())
    rows=[r for part in ('train','tune','test') for r in split[part]]
    assert set(manifest['records'])=={r['ID'] for r in rows}
    for row in rows:
        item=manifest['records'][row['ID']]
        assert item['identity']==row['audio_sha256']+':'+manifest['cache_id']
        assert item['audio_sha256']==row['audio_sha256']
    selection=json.loads((out/'selection.json').read_text())
    report=json.loads((out/'test_report.json').read_text())
    assert report['selection_sha256']==sha256(out/'selection.json')
    assert selection['test_used'] is False
    assert selection['winner']['name']==select_resolution(selection['summaries'])['name']
    assert len(selection['evaluation_candidates'])==12
    groups=[]
    for setting in selection['comparison_groups']:
        seeds=[]
        for seed in (41,42,43):
            name=f'{setting}_seed{seed}';folder=out/name
            trace=json.loads((folder/'training.json').read_text());recipe=trace['recipe']
            assert trace['status']=='complete'
            assert trace['train_ids']==[r['ID'] for r in split['train']]
            assert trace['validation_ids']==[r['ID'] for r in split['tune']]
            assert recipe['epochs']==300 and recipe['patience']==20 and recipe['min_delta']==1e-4
            assert recipe['learning_rate']==1e-4 and not recipe['lr_plateau']
            assert recipe['nsynth_mass']==0. and recipe['dropout']==.1 and recipe['weight_decay']==.01
            assert recipe['mlp_output'] and not recipe['cosine_output'] and not recipe.get('alternative_output','')
            assert recipe['label_smoothing']==(.05 if setting.endswith('ls5') else 0.)
            assert recipe['input_dim']==(1280 if setting.startswith('AF637') else 1024)
            assert sha256(folder/'temporal_music.pt')==trace['checkpoint_sha256']
            model=load_temporal_head(folder/'temporal_music.pt')
            assert sum(p.numel() for p in model.parameters())==trace['parameters']
            losses=trace['validation_losses'];assert np.isfinite(losses).all()
            assert len(losses)==trace['epochs_completed'] and trace['selected_epoch']==int(np.argmin(losses))+1
            reference=float('inf');stale=0
            for loss in losses:
                assert stale<20
                if loss<reference-1e-4:reference=loss;stale=0
                else:stale+=1
            assert stale==20 or len(losses)==300
            with np.load(folder/'tune_predictions.npz') as z:
                assert z['ids'].tolist()==[r['ID'] for r in split['tune']]
                tune=music_metrics(split['tune'],z['probability'])
            predictions=json.loads((out/f'{name}_test_predictions.json').read_text())
            assert [p['ID'] for p in predictions]==[r['ID'] for r in split['test']]
            metrics=official_metrics(split['test'],predictions)['metrics']
            saved=next(r for r in report['results'] if r['name']==name)['metrics']['metrics']
            for k,v in saved.items():
                if v is None:assert metrics[k] is None
                else:np.testing.assert_allclose(metrics[k],v,atol=1e-12,rtol=0)
            seeds.append(dict(seed=seed,tune_eer=tune['music_eer'],tune_bce=tune['balanced_bce'],
                music_eer=metrics['music_eer'],file_eer=metrics['file_eer'],
                selected_epoch=trace['selected_epoch'],epochs_completed=len(losses),parameters=trace['parameters']))
        groups.append(dict(name=setting,seeds=seeds,**{
            k+suffix:float(fn([s[k] for s in seeds])) for k in ('tune_eer','tune_bce','music_eer','file_eer')
            for suffix,fn in (('_mean',np.mean),('_std',np.std))}))
    result=dict(status='verified',new_fits=6,reused_controls=6,winner=selection['winner']['name'],
        encoder_parity=parity,features_verified=len(rows),groups=groups,early_stopping_verified=True,
        official_score=None,independent_test=False,submission_changed=False)
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
