"""Use all approved music across five group-disjoint, early-stopped MLP folds."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import torch

from c3_data import component_ids
from music_temporal import fit_temporal_head, load_temporal_head, predict_temporal_music
from temporal_backbone import REGULARIZED_C08
from temporal_experiment import training_masks
from temporal_resolution import load_pair
from v6_data import assert_disjoint, sha256, write_json
from v7_ablation import source_weights


def train_cv(plan, records, output: Path, deadline: float, *, recipe=REGULARIZED_C08,
             device: str = 'cuda'):
    """Every approved file trains four folds and validates exactly one fold."""
    rows = [r for r in plan['rows'] if r['music_training'] == '1']
    if (output/'training.json').exists():
        raise FileExistsError('An ensemble result already exists; inspect before training')
    output.mkdir(parents=True, exist_ok=True)
    groups = component_ids(rows)
    labels = np.array([int(r['MUSIC_FAKE']) for r in rows])
    masked = training_masks(rows, records, plan['quality'])
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=41)
    folds = []
    oof = np.full(len(rows), np.nan)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        train, valid = [rows[i] for i in train_idx], [rows[i] for i in valid_idx]
        assert_disjoint(train, valid)
        assert np.all(source_weights(train, recipe.nsynth_mass) > 0)
        write_json(output/'progress.json', dict(stage='fold_training', fold=fold+1, total=5,
                                               train_files=len(train), validation_files=len(valid), utc=time.time()))
        folder = output/f'fold{fold}'
        model, trace = fit_temporal_head(train, masked, (valid, records), folder,
                                        deadline, recipe=replace(recipe, seed=41+fold), device=device)
        probabilities = predict_temporal_music(model, valid, records, device)
        loaded = load_temporal_head(folder/'temporal_music.pt', device)
        np.testing.assert_allclose(probabilities, predict_temporal_music(loaded, valid, records, device),
                                   atol=1e-6, rtol=0)
        oof[valid_idx] = probabilities
        folds.append(dict(fold=fold, checkpoint=f'fold{fold}/temporal_music.pt',
                          checkpoint_sha256=trace['checkpoint_sha256'],
                          train_ids=trace['train_ids'], validation_ids=trace['validation_ids'],
                          selected_epoch=trace['selected_epoch'], epochs_completed=trace['epochs_completed']))
        write_json(output/'folds.json', folds)
    assert np.isfinite(oof).all()
    ids = [r['ID'] for r in rows]
    for identifier in ids:
        assert sum(identifier in f['train_ids'] for f in folds) == 4
        assert sum(identifier in f['validation_ids'] for f in folds) == 1
    np.savez_compressed(output/'oof_predictions.npz', ids=np.array(ids), probability=oof)
    metadata = dict(status='complete', training_mode='grouped_5fold_early_stopping',
        all_data_used_across_folds=True, ensemble=True, files=len(rows), train_ids=ids,
        nsynth_loss_mass=recipe.nsynth_mass, folds=folds, epochs_max=recipe.epochs,
        patience=recipe.patience, independent_evaluation=None,
        aggregation='Arithmetic mean of five music probabilities, then original file fusion')
    write_json(output/'training.json', metadata)
    write_json(output/'progress.json', dict(stage='complete', utc=time.time()))
    return metadata


def main() -> None:
    root = Path.cwd()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert 'A100' in torch.cuda.get_device_name()
    plan = json.loads((root/'full_data_plan.json').read_text())
    rows = [r for r in plan['rows'] if r['music_training'] == '1']
    assert len(rows) == len({r['ID'] for r in rows}) == 1206
    assert len({r['pcm_sha256'] for r in rows}) == len(rows)
    manifest = json.loads((root/'reference330_features_manifest.json').read_text())
    records = {}
    for row in rows:
        item = manifest['records'][row['ID']]
        p = root/'feature_cache'/item['file']
        assert item['identity'] == row['audio_sha256']+':'+manifest['cache_id']
        assert sha256(p) == item['sha256']
        records[row['ID']], _ = load_pair(p, item['identity'], 1024)
    recipe = replace(REGULARIZED_C08, input_dim=1024, nsynth_mass=.05)
    output = root/'final_run'
    result = train_cv(plan, records, output, float(os.environ['V7_RESUMED_EPOCH'])+3600, recipe=recipe)
    for name in ('full_data_plan.json','mert330_metadata.json'):
        shutil.copy2(root/name, output/name)
    print('FULL_CV_COMPLETE', result['files'], [f['selected_epoch'] for f in result['folds']], flush=True)


if __name__ == '__main__':
    main()
