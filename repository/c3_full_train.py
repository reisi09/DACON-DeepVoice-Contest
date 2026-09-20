"""Fit the final C3 music head and a source-disjoint out-of-fold calibrator."""

import json
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from c3_data import component_ids
from v6_core import VERSION, Heads, fusion_features
from v6_data import assert_disjoint, sha256, write_json
from v6_runtime import Engine
from v6_train import feature_record
from v7_ablation import Recipe, fit_head


def load_records(plan, project: Path, deadline: float):
    old_work = project / 'v6_run'
    previous = json.loads((old_work / 'training.json').read_text())
    old_split = json.loads((old_work / 'split.json').read_text())
    old_ids = {r['ID'] for p in ('expert', 'fusion', 'dev') for r in old_split[p]}
    records = {}
    for row in plan['rows']:
        if row['ID'] in old_ids:
            records[row['ID']] = feature_record(row, old_work / 'cache', previous['cache_id'])
    engine = Engine(project / 'model/assets', 'cuda')
    if engine.asset_id != previous['asset_id']:
        raise ValueError('Model assets changed since C3 comparison')
    missing = [r for r in plan['rows'] if r['ID'] not in records]
    output = Path.cwd() / 'final_run'
    output.mkdir(exist_ok=True)
    for i, row in enumerate(missing):
        if time.time() >= deadline:
            raise TimeoutError('Missing-feature extraction deadline')
        if sha256(Path(row['path'])) != row['audio_sha256']:
            raise ValueError('Uploaded source hash mismatch')
        records[row['ID']] = engine.cached(Path(row['path']), output / 'new_cache')
        if (i + 1) % 25 == 0 or i + 1 == len(missing):
            write_json(output / 'progress.json', dict(stage='extracting_sealed', completed=i + 1,
                                                       total=len(missing), utc=time.time()))
    return records, engine


def train_final(plan, records, settings, *, recipe: Recipe | None = None):
    recipe = recipe or Recipe('C3_full_clean_es', .05, True, 41, 300)
    output, asset_id, deadline = settings
    rows = plan['rows']
    training_records = dict(records)
    for row in rows:
        if row['music_training'] != '1':
            continue
        record, policy = records[row['ID']], plan['quality'][row['ID']]
        np.testing.assert_array_equal(record['starts'], policy['starts'])
        mask = np.zeros(len(record['starts']))
        mask[policy['usable_indices']] = 1
        training_records[row['ID']] = dict(record, music_audible=record['music_audible'] * mask)
        if not training_records[row['ID']]['music_audible'].any():
            raise ValueError(f'No usable separated music remains: {row["ID"]}')
    groups = component_ids(rows)
    labels = np.array([int(r['FILE_FAKE']) for r in rows])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=recipe.seed)
    matrix = np.full((len(rows), 5), np.nan)
    assignments = np.full(len(rows), -1)
    fold_reports = []
    for fold, (train_indices, hold_indices) in enumerate(folds.split(np.zeros(len(rows)), labels, groups)):
        music_indices = np.array([i for i in train_indices if rows[i]['music_training'] == '1'])
        training = [rows[i] for i in music_indices]
        held = [rows[i] for i in hold_indices]
        assert_disjoint(training, held)
        write_json(output / 'progress.json', dict(stage='oof_head', fold=fold + 1, total=5, utc=time.time()))
        # Outer held files cannot influence normalization, optimization or epoch selection.
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=recipe.seed)
        inner_train, inner_val = next(inner.split(np.zeros(len(training)),
            [int(r['MUSIC_FAKE']) for r in training], groups[music_indices]))
        selection_train = [training[i] for i in inner_train]
        selection_val = [training[i] for i in inner_val]
        assert_disjoint(selection_train, selection_val)
        _, _, selection = fit_head(selection_train, training_records, recipe, deadline,
                                    validation=(selection_val, records))
        selection.update(train_ids=[r['ID'] for r in selection_train],
                         validation_ids=[r['ID'] for r in selection_val])
        head, losses, counts = fit_head(training, training_records,
            replace(recipe, epochs=selection['selected_epoch']), deadline)
        head.save(output / f'oof_head_{fold}.npz')
        for i in hold_indices:
            record = records[rows[i]['ID']]
            music = head.music(record['music'], record['music_audible'])
            matrix[i] = fusion_features(record['presence'], record['voice'], music, record['raw'])
            assignments[i] = fold
        fold_reports.append(dict(fold=fold, music_training_files=len(training), held_files=len(held),
                                 held_ids=[r['ID'] for r in held], epoch_selection=selection,
                                 composition=counts, initial_loss=losses[0], final_loss=losses[-1]))
        write_json(output / 'oof_folds.json', fold_reports)
    if not np.isfinite(matrix).all() or (assignments < 0).any():
        raise ValueError('Incomplete out-of-fold coverage')
    fusion = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=recipe.seed)
    fusion.fit(matrix, labels)
    if fusion.n_iter_.max() >= fusion.max_iter:
        raise RuntimeError('Final calibrator did not converge')
    music_rows = [r for r in rows if r['music_training'] == '1']
    write_json(output / 'progress.json', dict(stage='full_music_head', files=len(music_rows), utc=time.time()))
    selected_epochs = [f['epoch_selection']['selected_epoch'] for f in fold_reports]
    final_epochs = int(np.median(selected_epochs))
    head, losses, counts = fit_head(music_rows, training_records,
                                    replace(recipe, epochs=final_epochs), deadline)
    final = Heads(head.mean, head.scale, head.music_weight, head.music_bias,
                  fusion.coef_[0], np.array(fusion.intercept_[0]))
    final.save(output / 'heads.npz')
    np.savez_compressed(output / 'oof_features.npz', features=matrix, fold=assignments,
                        ids=np.array([r['ID'] for r in rows]), file_labels=labels)
    write_json(output / 'oof_folds.json', fold_reports)
    write_json(output / 'losses.json', losses)
    metadata = dict(status='complete', purpose='final', version=VERSION, model_id='C3-full-clean-oof-es',
        asset_id=asset_id, heads_sha256=sha256(output / 'heads.npz'), epochs_requested=final_epochs,
        epochs_completed=len(losses), music_files=len(music_rows), fusion_files=len(rows), seed=recipe.seed,
        recipe=asdict(recipe),
        early_stopping=dict(monitor='class-balanced validation file-max BCE',
            max_epochs=recipe.epochs, patience=recipe.patience, min_delta=recipe.min_delta,
            restore_best_weights=True, fold_selected_epochs=selected_epochs,
            final_refit='Median selected epoch; retrain on all retained music without validation'),
        source_weighting='Real/fake balanced; NSynth mass from recipe; fake equally split across generators',
        window_policy=plan['policy'], composition=counts, oof_folds=5,
        data_plan_sha256=sha256(Path.cwd() / 'full_data_plan.json'),
        final_evaluation='Unavailable: all eligible public train/dev/sealed data used for final fitting',
        calibration='Trained on source-component-disjoint OOF music predictions',
        architecture='Frozen encoders; C3 linear music head and logistic file calibrator',
        initial_loss=losses[0], final_loss=losses[-1])
    write_json(output / 'training.json', metadata)
    write_json(output / 'dev_metrics.json', dict(score=None, metrics={},
        reason='Full-data final fit; no independent development evaluation. See the earlier C3 comparison separately.'))
    sample = records[rows[0]['ID']]
    np.testing.assert_array_equal(final.predict(sample), Heads.load(output / 'heads.npz').predict(sample))
    write_json(output / 'progress.json', dict(stage='trained', utc=time.time()))
    return metadata
