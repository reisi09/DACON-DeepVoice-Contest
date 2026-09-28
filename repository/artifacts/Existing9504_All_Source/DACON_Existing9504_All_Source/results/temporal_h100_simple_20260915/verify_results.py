"""Reconcile the real H100 run against checkpoints, predictions and prior controls."""

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


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    unpacked = ROOT/'unpacked'
    results = unpacked/'results'
    prior = ROOT.parent/'temporal_h100_aug_20260915/unpacked/results'
    assert read(unpacked/'run_state.json')['stage'] == 'complete'
    notebook = nbformat.read(unpacked/'Temporal_H100_Simplification_Executed.ipynb', as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(cells) == 5 and all(c.execution_count is not None for c in cells)
    assert not [o for c in cells for o in c.outputs if o.output_type == 'error']
    split = read(results/'split.json')
    assert split == read(prior/'split.json')
    manifest = read(unpacked/'input_manifest.json')['sha256']
    for name, digest in manifest.items():
        assert hashlib.sha256((unpacked/name).read_bytes()).hexdigest() == digest, name
    for name in ('full_data_plan.json','locked_split.json','baseline_predictions.json','requirements-head.txt'):
        assert (unpacked/name).read_bytes() == (prior.parent/name).read_bytes(), name
    selection = read(results/'selection.json')
    reports = read(results/'test_report.json')['results']
    assert len(reports) == 8
    assert not selection['test_used']
    assert set(selection['evaluation_candidates']) == {r['name'] for r in reports[2:]}
    checkpoints = 0
    for result in reports:
        name = result['name']
        predictions = read(results/f'{name}_test_predictions.json')
        measured = official_metrics(split['test'], predictions)
        for key, expected in result['metrics'].items():
            if key == 'metrics':
                for metric, value in expected.items():
                    if value is None:
                        assert measured[key][metric] is None
                    else:
                        np.testing.assert_allclose(measured[key][metric],value,atol=1e-12,rtol=0)
            else:
                assert measured[key] == expected, (name,key)
        if name in ('Official_baseline','Linear_same_split'):
            before = read(prior/f'{name}_test_predictions.json')
            assert [r['ID'] for r in predictions] == [r['ID'] for r in before]
            for a,b in zip(predictions,before,strict=True):
                for key in a:
                    if key != 'ID':
                        np.testing.assert_allclose(a[key],b[key],atol=1e-6,rtol=0)
            continue
        trace = read(results/name/'training.json')
        checkpoint = results/name/'temporal_music.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == trace['checkpoint_sha256']
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        simple = name.startswith('S_')
        assert payload['architecture'] == ('mert-residual-tcn1-max-v1' if simple else 'mert-residual-tcn3-gated-max-v1')
        assert sum(t.numel() for t in payload['state_dict'].values()) == trace['parameters']
        assert trace['parameters'] == (199169 if simple else 414017)
        recipe = trace['recipe']
        assert recipe['simplified'] == simple and recipe['batch_size'] == 128
        assert recipe['crop_probability'] == .7 and recipe['feature_mask_probability'] == 0
        assert trace['optimizer_steps'] == math.ceil(len(split['train'])/128)*trace['epochs_completed']
        assert trace['selected_epoch'] == int(np.argmin(trace['validation_losses']))+1
        if not simple:
            old = read(prior/f"A1_crop_seed{recipe['seed']}"/'training.json')
            for key in ('train_losses','validation_losses'):
                np.testing.assert_allclose(trace[key],old[key],atol=1e-6,rtol=0)
            old_prediction = read(prior/f"A1_crop_seed{recipe['seed']}_test_predictions.json")
            for a,b in zip(predictions,old_prediction,strict=True):
                assert a['ID'] == b['ID']
                for key in a:
                    if key != 'ID':
                        np.testing.assert_allclose(a[key],b[key],atol=1e-6,rtol=0)
        checkpoints += 1
    for seed in (41,42,43):
        deep = read(results/f'D_deep_crop_seed{seed}'/'training.json')
        simple = read(results/f'S_simple_crop_seed{seed}'/'training.json')
        count = min(deep['epochs_completed'],simple['epochs_completed'])
        assert deep['augmentation_history'][:count] == simple['augmentation_history'][:count]
        a,b = deep['recipe'].copy(),simple['recipe'].copy()
        a.pop('simplified'); b.pop('simplified')
        assert a == b
    summary = read(results/'comparison_summary.json')
    for row in summary['historical_test']:
        members = [r for r in reports if r['name'].startswith(row['name']+'_seed')]
        assert len(members) == 3
        for metric in ('music_eer','file_eer','file_music_partial'):
            values = []
            for member in members:
                m = member['metrics']['metrics']
                values.append(.45*(1-m['file_eer'])+.27*(1-m['music_eer'])
                              if metric == 'file_music_partial' else m[metric])
            np.testing.assert_allclose(row[metric+'_mean'],np.mean(values),atol=1e-12,rtol=0)
            np.testing.assert_allclose(row[metric+'_std'],np.std(values),atol=1e-12,rtol=0)
    evidence = dict(executed_code_cells=len(cells),notebook_errors=0,
        checkpoint_hashes=checkpoints,metrics_recomputed=len(reports),
        metric_recompute_atol=1e-12,
        prior_deep_curves_and_predictions_reproduced_atol=1e-6,
        control_predictions_unchanged=True,paired_crop_counts_and_recipes=True,
        source_manifests_and_split_unchanged=True,summary_recomputed=True,
        cache_unchanged=read(results/'cache_integrity.json')['unchanged'])
    (ROOT/'result_verification.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
