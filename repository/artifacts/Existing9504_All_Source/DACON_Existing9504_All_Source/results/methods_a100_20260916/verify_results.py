"""Audit real early-stopped runs and recompute comparison/calibration metrics."""
import json
from pathlib import Path
import sys

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from graph_scoring import official_metrics
from music_temporal import load_temporal_head
from temporal_experiment import music_metrics
from temporal_resolution import select_resolution
from v6_data import assert_disjoint, sha256


def main():
    run = ROOT/'unpacked'; out = run/'results'
    assert json.loads((run/'run_state.json').read_text())['stage'] == 'complete'
    nb = nbformat.read(run/'C08_Methods_A100_Executed.ipynb', as_version=4)
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.outputs)
               for c in nb.cells if c.cell_type == 'code')
    split = json.loads((run/'locked_split.json').read_text())
    for a,b in (('train','tune'),('train','test'),('tune','test')):
        assert_disjoint(split[a], split[b])
    protocol = json.loads((out/'search_protocol.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    report = json.loads((out/'test_report.json').read_text())
    assert report['selection_sha256'] == sha256(out/'selection.json')
    assert selection['test_used'] is False
    assert select_resolution(selection['summaries'])['name'] == selection['winner']['name']
    trials = json.loads((out/'tuning.json').read_text())
    assert len(trials) == 48 and len({t['name'] for t in trials}) == 48
    summaries = []
    for setting in protocol['settings']:
        items = []
        for seed in (41,42,43):
            name = f'{setting}_seed{seed}'
            folder = out/name
            trace = json.loads((folder/'training.json').read_text()); recipe = trace['recipe']
            assert trace['status'] == 'complete' and recipe['patience'] == 20 and recipe['epochs'] == 300
            assert recipe['nsynth_mass'] == 0. and recipe['seed'] == seed
            assert trace['train_ids'] == [r['ID'] for r in split['train']]
            assert trace['validation_ids'] == [r['ID'] for r in split['tune']]
            assert sha256(folder/'temporal_music.pt') == trace['checkpoint_sha256']
            model = load_temporal_head(folder/'temporal_music.pt')
            assert sum(p.numel() for p in model.parameters()) == trace['parameters']
            losses = trace['validation_losses']
            assert len(losses) == trace['epochs_completed']
            assert trace['selected_epoch'] == int(np.argmin(losses))+1
            reference = float('inf'); stale = 0
            for epoch, loss in enumerate(losses, 1):
                assert stale < 20, 'Training continued after early stopping'
                if loss < reference-recipe['min_delta']:reference=loss;stale=0
                else:stale+=1
            assert stale == 20 or len(losses) == 300
            lr = trace['learning_rates']
            assert len(lr) == len(losses) and lr[0] == recipe['learning_rate']
            assert np.all(np.diff(lr) <= 0)
            if not recipe['lr_plateau']:assert len(set(lr)) == 1
            with np.load(folder/'tune_predictions.npz') as z:
                assert z['ids'].tolist() == [r['ID'] for r in split['tune']]
                tune = music_metrics(split['tune'], z['probability'])
            predictions = json.loads((out/f'{name}_test_predictions.json').read_text())
            assert [p['ID'] for p in predictions] == [r['ID'] for r in split['test']]
            test = official_metrics(split['test'], predictions)['metrics']
            saved = next(r for r in report['results'] if r['name'] == name)['metrics']['metrics']
            for key, value in saved.items():
                if value is None:assert test[key] is None
                else:np.testing.assert_allclose(test[key], value, atol=1e-12, rtol=0)
            items.append(dict(seed=seed, tune_eer=tune['music_eer'], tune_bce=tune['balanced_bce'],
                music_eer=test['music_eer'], file_eer=test['file_eer'],
                selected_epoch=trace['selected_epoch'], epochs_completed=len(losses),
                lr_reductions=int(np.count_nonzero(np.diff(lr))), parameters=trace['parameters']))
        summaries.append(dict(name=setting, seeds=items, **{
            key+suffix:float(fn([r[key] for r in items]))
            for key in ('tune_eer','tune_bce','music_eer','file_eer')
            for suffix,fn in (('_mean',np.mean),('_std',np.std))}))
    calibration = []
    transforms = {'T0.5':lambda p:p**2/(p**2+(1-p)**2),
                  'T2':lambda p:np.sqrt(p)/(np.sqrt(p)+np.sqrt(1-p)),
                  'prob_smoothing0.05':lambda p:.95*p+.025}
    for label, transform in transforms.items():
        values = []
        for seed in (41,42,43):
            predictions = json.loads((out/f'mlp_ls0_lr0_seed{seed}_test_predictions.json').read_text())
            p = transform(np.array([r['MUSIC_FAKE_PROB'] for r in predictions]))
            for row, value in zip(predictions,p,strict=True):
                row['MUSIC_FAKE_PROB'] = float(value)
                row['FILE_FAKE_PROB'] = max(row['VOICE_PRESENT_PROB']*row['VOICE_FAKE_PROB'],
                                           row['MUSIC_PRESENT_PROB']*row['MUSIC_FAKE_PROB'])
            m = official_metrics(split['test'], predictions)['metrics']
            values.append(dict(seed=seed, music_eer=m['music_eer'],file_eer=m['file_eer'],
                               bce=music_metrics(split['test'],p)['balanced_bce']))
        calibration.append(dict(name=label,seeds=values, **{k:float(np.mean([v[k] for v in values]))
                                                           for k in ('music_eer','file_eer','bce')}))
    result = dict(status='verified',runs=48,settings=16,winner=selection['winner']['name'],
        groups=sorted(summaries,key=lambda r:r['tune_eer_mean']),postprocessing=calibration,
        early_stopping_verified=True,checkpoint_reload_verified_on_A100=True,
        official_score=None,independent_test=False, final_submission_changed=False)
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
