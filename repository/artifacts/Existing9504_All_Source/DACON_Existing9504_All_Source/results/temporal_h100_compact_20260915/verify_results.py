"""Reconcile H100 trials, tune-only selection and historical evaluation artifacts."""

import hashlib
import json
import math
from pathlib import Path
import sys

import nbformat
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from graph_scoring import official_metrics
from music_temporal import load_temporal_head
from temporal_experiment import music_metrics
from v6_data import assert_disjoint


def main() -> None:
    unpacked = ROOT/'unpacked'
    out = unpacked/'results'
    prior = ROOT.parent/'temporal_h100_simple_20260915/unpacked/results'

    def read(path: Path):
        return json.loads(path.read_text())

    assert read(unpacked/'run_state.json')['stage'] == 'complete'
    notebook = nbformat.read(unpacked/'Temporal_H100_Compact_Tuning_Executed.ipynb', as_version=4)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(code) == 5 and all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type == 'error']
    for name, digest in read(unpacked/'input_manifest.json')['sha256'].items():
        assert hashlib.sha256((unpacked/name).read_bytes()).hexdigest() == digest, name
    split = read(out/'split.json')
    assert split == read(prior/'split.json')
    for a, b in [('train','tune'), ('train','test'), ('tune','test')]:
        assert_disjoint(split[a], split[b])
    protocol, shortlist, selection = [read(out/f'{name}.json') for name in
                                      ('search_protocol','shortlist','selection')]
    trials = read(out/'tuning.json')
    assert len({r['name'] for r in trials}) == len(trials)
    screened = [r for r in trials if r['seed'] == 41 and r['setting'] != 'S128']
    assert len(screened) == 24
    assert protocol['created_at'] <= shortlist['frozen_at'] <= selection['frozen_at']
    assert not shortlist['test_used'] and not selection['test_used']
    ranked = sorted(screened, key=lambda r: (r['music_eer'], r['balanced_bce'], r['setting']))
    assert shortlist['candidates'] == [r['setting'] for r in ranked[:4]]
    for trial in trials:
        name = trial['name']
        trace = read(out/name/'training.json')
        expected_recipe = dict(protocol['candidates']['C00' if trial['setting'] == 'S128' else trial['setting']])
        expected_recipe.update(seed=trial['seed'], compact=trial['setting'] != 'S128')
        assert trace['recipe'] == trial['recipe'] == expected_recipe
        assert trace['train_ids'] == [r['ID'] for r in split['train']]
        assert trace['validation_ids'] == [r['ID'] for r in split['tune']]
        assert trace['optimizer_steps'] == math.ceil(len(split['train'])/expected_recipe['batch_size'])*trace['epochs_completed']
        assert trace['selected_epoch'] == int(np.argmin(trace['validation_losses']))+1 == trial['selected_epoch']
        checkpoint = out/name/'temporal_music.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == trace['checkpoint_sha256']
        model = load_temporal_head(checkpoint)
        assert model.compact == expected_recipe['compact']
        assert sum(p.numel() for p in model.parameters()) == trace['parameters'] == trial['parameters']
        assert trace['parameters'] == (88257 if model.compact else 199169)
        with np.load(out/name/'tune_predictions.npz') as saved:
            assert saved['ids'].tolist() == trace['validation_ids']
            measured = music_metrics(split['tune'], saved['probability'])
        for key, value in measured.items():
            np.testing.assert_allclose(value, trial[key], atol=1e-12, rtol=0)
        np.testing.assert_allclose(measured['balanced_bce'], trace['best_validation_loss'], atol=2e-6, rtol=0)
        if trial['setting'] == 'S128':
            old = read(prior/f"S_simple_crop_seed{trial['seed']}"/'training.json')
            for field in ('train_losses', 'validation_losses'):
                np.testing.assert_allclose(trace[field], old[field], atol=1e-6, rtol=0)
    for summary in selection['summaries']:
        members = [r for r in trials if r['setting'] == summary['name']]
        assert {r['seed'] for r in members} == {41,42,43}
        assert set(summary['candidates']) == {r['name'] for r in members}
        for metric in ('music_eer','balanced_bce'):
            values = [r[metric] for r in members]
            np.testing.assert_allclose(summary[metric+'_mean'], np.mean(values), atol=1e-12, rtol=0)
            if metric == 'music_eer':
                np.testing.assert_allclose(summary[metric+'_std'], np.std(values), atol=1e-12, rtol=0)
    eligible = [s for s in selection['summaries'] if s['name'] in shortlist['candidates']]
    winner = min(eligible, key=lambda s: (s['music_eer_mean'], s['music_eer_std'], s['balanced_bce_mean'], s['name']))
    assert selection['winner'] == winner
    assert set(selection['comparison_groups']) == {'S128','C00',winner['name']}
    report = read(out/'test_report.json')
    assert report['selection_sha256'] == hashlib.sha256((out/'selection.json').read_bytes()).hexdigest()
    assert {r['name'] for r in report['results'][2:]} == set(selection['evaluation_candidates'])
    baseline = read(out/'Official_baseline_test_predictions.json')
    for result in report['results']:
        name = result['name']
        predictions = read(out/f'{name}_test_predictions.json')
        measured = official_metrics(split['test'], predictions)
        for metric, value in result['metrics']['metrics'].items():
            if value is None:
                assert measured['metrics'][metric] is None
            else:
                np.testing.assert_allclose(measured['metrics'][metric], value, atol=1e-12, rtol=0)
        for a, b in zip(predictions, baseline, strict=True):
            for key in ('ID','VOICE_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB'):
                assert a[key] == b[key]
            if name != 'Official_baseline':
                assert a['FILE_FAKE_PROB'] == max(a['VOICE_FAKE_PROB']*a['VOICE_PRESENT_PROB'],
                                                 a['MUSIC_FAKE_PROB']*a['MUSIC_PRESENT_PROB'])
        if name in ('Official_baseline','Linear_same_split') or name.startswith('S128_'):
            old_name = name.replace('S128_', 'S_simple_crop_')
            old = read(prior/f'{old_name}_test_predictions.json')
            for a,b in zip(predictions,old,strict=True):
                assert a['ID'] == b['ID']
                for key in a.keys()-{'ID'}:
                    np.testing.assert_allclose(a[key],b[key],atol=1e-6,rtol=0)
    summary = read(out/'comparison_summary.json')
    for row in summary['historical_test']:
        members = [r for r in report['results'] if r['name'] in selection['comparison_groups'][row['name']]]
        assert len(members) == 3
        for metric in ('music_eer','file_eer','file_music_partial'):
            values = [(.45*(1-r['metrics']['metrics']['file_eer'])+.27*(1-r['music_eer'])
                       if metric == 'file_music_partial' else r['metrics']['metrics'][metric]) for r in members]
            np.testing.assert_allclose(row[metric+'_mean'],np.mean(values),atol=1e-12,rtol=0)
            np.testing.assert_allclose(row[metric+'_std'],np.std(values),atol=1e-12,rtol=0)
    evidence = dict(executed_code_cells=len(code), notebook_errors=0, trials_reconciled=len(trials),
        screened=24, shortlisted=4, winner=winner['name'], selection_recomputed=True,
        checkpoint_reload_and_hashes=len(trials), tune_metrics_recomputed=len(trials),
        historical_metrics_recomputed=len(report['results']), fixed_voice_and_presence=True,
        old_simple_curves_and_predictions_reproduced_atol=1e-6, metrics_atol=1e-12,
        split_and_source_hashes_verified=True, cache_unchanged=read(out/'cache_integrity.json')['unchanged'])
    (ROOT/'result_verification.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
