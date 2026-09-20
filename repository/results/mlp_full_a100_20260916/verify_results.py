"""Audit five-fold early stopping and complete, positive-weight data coverage."""
import json
from pathlib import Path
import sys

import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from music_temporal import load_temporal_head
from v6_data import assert_disjoint,sha256
from v7_ablation import source_weights


def main():
    run=ROOT/'unpacked';out=run/'final_run'
    assert json.loads((run/'run_state.json').read_text())['stage']=='complete'
    nb=nbformat.read(run/'C08_MLP_Full_A100_Executed.ipynb',as_version=4)
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
               for c in nb.cells if c.cell_type=='code')
    plan=json.loads((out/'full_data_plan.json').read_text())
    rows=[r for r in plan['rows'] if r['music_training']=='1'];by_id={r['ID']:r for r in rows}
    metadata=json.loads((out/'training.json').read_text())
    assert len(rows)==len(by_id)==1206 and len(metadata['folds'])==5
    assert metadata['train_ids']==list(by_id)
    epochs=[]
    for f in metadata['folds']:
        p=out/f"fold{f['fold']}";t=json.loads((p/'training.json').read_text());recipe=t['recipe']
        assert sha256(p/'temporal_music.pt')==t['checkpoint_sha256']==f['checkpoint_sha256']
        model=load_temporal_head(p/'temporal_music.pt')
        assert model.architecture=='mert-residual-64x3-max-d1024-mlp32-softmax-v1'
        assert sum(p.numel() for p in model.parameters())==107234
        assert recipe['mlp_output'] and not recipe['cosine_output']
        assert recipe['label_smoothing']==0 and not recipe['lr_plateau']
        assert recipe['nsynth_mass']==.05 and recipe['epochs']==300 and recipe['patience']==20
        assert recipe['dropout']==.1 and recipe['learning_rate']==1e-4 and recipe['weight_decay']==.01
        assert t['train_ids']==f['train_ids'] and t['validation_ids']==f['validation_ids']
        train=[by_id[k] for k in t['train_ids']];valid=[by_id[k] for k in t['validation_ids']]
        assert_disjoint(train,valid)
        assert set(t['train_ids'])|set(t['validation_ids'])==set(by_id)
        assert np.all(source_weights(train,.05)>0)
        assert t['selected_epoch']==int(np.argmin(t['validation_losses']))+1
        best=float('inf');stale=0
        for loss in t['validation_losses']:
            if loss<best-recipe['min_delta']:best=loss;stale=0
            else:stale+=1
        assert stale>=20 or t['epochs_completed']==300
        epochs.append(t['selected_epoch'])
    for identifier in by_id:
        assert sum(identifier in f['train_ids'] for f in metadata['folds'])==4
        assert sum(identifier in f['validation_ids'] for f in metadata['folds'])==1
    with np.load(out/'oof_predictions.npz') as z:
        assert z['ids'].tolist()==list(by_id) and np.isfinite(z['probability']).all()
    result=dict(files=1206,folds=5,positive_weight_coverage=True,group_disjoint=True,
                selected_epochs=epochs,early_stopping_verified=True,independent_final_score=None)
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
