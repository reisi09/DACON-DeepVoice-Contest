"""Bounded tuning on cached MERT features with a locked, historically reused test."""

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import time
from typing import TypedDict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from c3_data import component_ids
from graph_scoring import official_metrics
from music_temporal import (Records, Rows, TemporalRecipe, fit_temporal_head,
                            load_temporal_head, predict_temporal_music)
from v6_core import Heads
from v6_data import assert_disjoint, sha256, write_json
from v6_train import feature_record
from v6_runtime import load_cache
from v7_ablation import Recipe, fit_head, source_weights


class Split(TypedDict):
    train: Rows
    tune: Rows
    test: Rows


def split_experiment(rows: Rows, baseline_ids: set[str]) -> Split:
    """Keep the old dev comparison panel fixed; isolate transitive parent groups."""
    groups = component_ids(rows)
    held_groups = {g for r, g in zip(rows, groups, strict=True) if r['ID'] in baseline_ids}
    indices = np.array([i for i, g in enumerate(groups) if g not in held_groups])
    available = [rows[i] for i in indices]
    train, tune = next(StratifiedGroupKFold(5, shuffle=True, random_state=41).split(
        np.zeros(len(indices)), [r['FILE_FAKE'] for r in available], groups[indices]))
    split = Split(train=[available[i] for i in train if available[i]['music_training'] == '1'],
        tune=[available[i] for i in tune if available[i]['MUSIC_PRESENT'] == '1'
              and available[i]['MUSIC_FAKE'] in ('0', '1')],
        test=[r for r in rows if r['ID'] in baseline_ids])
    for left, right in [('train', 'tune'), ('train', 'test'), ('tune', 'test')]:
        assert_disjoint(split[left], split[right])
    source_weights(split['train'], .05)
    source_weights(split['tune'], None)
    return split


def load_features(rows: Rows, project: Path, final: Path) -> Records:
    """Verify original audio and cache identities without loading any encoder."""
    portable = Path.cwd() / 'features_manifest.json'
    if portable.exists():
        manifest = json.loads(portable.read_text())
        records = {}
        for row in rows:
            item = manifest['records'][row['ID']]
            path = (portable.parent / item['path']).resolve()
            expected = row['audio_sha256'] + ':' + manifest['cache_id']
            if (not path.is_relative_to(portable.parent.resolve())
                    or item['audio_sha256'] != row['audio_sha256'] or item['identity'] != expected
                    or sha256(path) != item['sha256']):
                message = f'Portable feature provenance mismatch: {row["ID"]}'
                raise ValueError(message)
            record = load_cache(path, expected)
            if record is None:
                message = f'Missing or stale portable cache: {row["ID"]}'
                raise ValueError(message)
            records[row['ID']] = record
        return records
    old = project / 'v6_run'
    metadata = json.loads((old / 'training.json').read_text())
    old_split = json.loads((old / 'split.json').read_text())
    old_ids = {r['ID'] for part in ('expert', 'fusion', 'dev') for r in old_split[part]}
    records = {}
    for row in rows:
        cache = old / 'cache' if row['ID'] in old_ids else final / 'final_run/new_cache'
        records[row['ID']] = feature_record(row, cache, metadata['cache_id'])
    return records


def training_masks(rows: Rows, records: Records, quality) -> Records:
    """Apply the existing C3 quality gate only to training inputs."""
    result = dict(records)
    for row in rows:
        record, policy = records[row['ID']], quality[row['ID']]
        np.testing.assert_array_equal(record['starts'], policy['starts'])
        mask = np.zeros(len(record['starts']))
        mask[policy['usable_indices']] = 1
        result[row['ID']] = dict(record, music_audible=record['music_audible'] * mask)
    return result


def music_metrics(rows: Rows, probabilities: np.ndarray) -> dict[str, float]:
    """Use the project's official discrete EER; BCE is balanced by file class."""
    selected = [i for i, r in enumerate(rows) if r['MUSIC_PRESENT'] == '1'
                and r['MUSIC_FAKE'] in ('0', '1')]
    labels = [rows[i] for i in selected]
    p = np.asarray(probabilities)[selected]
    predictions = [dict(ID=r['ID'], FILE_FAKE_PROB=.5, VOICE_FAKE_PROB=.5,
        MUSIC_FAKE_PROB=float(value), VOICE_PRESENT_PROB=.5, MUSIC_PRESENT_PROB=.5)
        for r, value in zip(labels, p, strict=True)]
    eer = official_metrics(labels, predictions)['metrics']['music_eer']
    if eer is None:
        message = 'Both eligible music classes are required for model comparison'
        raise ValueError(message)
    y = np.array([int(r['MUSIC_FAKE']) for r in labels])
    bce = -np.log(np.where(y, p, 1 - p).clip(1e-7, 1)) @ source_weights(labels, None)
    return dict(music_eer=float(eer), balanced_bce=float(bce))


def run_search(train: Rows, tune: Rows, records: Records, masked: Records,
               output: Path, deadline: float, *, epochs: int = 300,
               device: str = 'cuda', batch_size: int = TemporalRecipe().batch_size) -> list[dict]:
    """Receive no test rows or labels; freeze the winner before test inference."""
    output.mkdir(parents=True, exist_ok=True)
    results = []
    candidates = list(product((3e-4, 1e-4), (.2, .4), (.01, .05)))
    for index, (lr, dropout, decay) in enumerate(candidates):
        name = f'T{index}'
        recipe = TemporalRecipe(epochs=epochs, learning_rate=lr, dropout=dropout,
                                weight_decay=decay, batch_size=batch_size)
        write_json(output / 'progress.json', dict(stage='tuning', candidate=name, total=8, utc=time.time()))
        started = time.time()
        model, report = fit_temporal_head(train, masked, (tune, records), output / name,
            deadline, recipe=recipe, device=device)
        probabilities = predict_temporal_music(model, tune, records, device, recipe.batch_size)
        metrics = music_metrics(tune, probabilities)
        result = dict(name=name, recipe=asdict(recipe), **metrics,
                      selected_epoch=report['selected_epoch'], epochs_completed=report['epochs_completed'],
                      optimizer_steps=report['optimizer_steps'],
                      seconds=time.time() - started)
        results.append(result)
        write_json(output / 'tuning.json', results)
        np.savez_compressed(output / name / 'tune_predictions.npz',
                            ids=np.array([r['ID'] for r in tune]), probability=probabilities)
        print(name, metrics, 'epochs', report['epochs_completed'], flush=True)
        del model
    winner = min(results, key=lambda r: (r['music_eer'], r['balanced_bce'], r['name']))
    write_json(output / 'selection.json', dict(winner=winner,
        criterion='Tuning music EER, then balanced BCE, then candidate name',
        test_used=False, frozen_at=time.time()))
    for seed in (42, 43):
        name = f'{winner["name"]}_seed{seed}'
        recipe = replace(TemporalRecipe(**winner['recipe']), seed=seed)
        write_json(output / 'progress.json', dict(stage='seed_repeat', candidate=name, utc=time.time()))
        fit_temporal_head(train, masked, (tune, records), output / name,
                          deadline, recipe=recipe, device=device)
    linear, _, stats = fit_head(train, masked, Recipe('linear_control', .05, True,
        epochs=epochs), deadline, validation=(tune, records))
    linear.save(output / 'linear_control.npz')
    write_json(output / 'linear_control.json', stats)
    return results


def evaluate_locked_test(rows: Rows, records: Records, baseline, output: Path,
                         device: str = 'cuda', *, fine_records: Records | None = None) -> list[dict]:
    """Compare frozen selection, three seeds and fresh linear with identical truth."""
    selection_path = output / 'selection.json'
    selection_hash = sha256(selection_path)
    selected = json.loads(selection_path.read_text())['winner']
    winner = selected['name']
    batch_size = selected['recipe']['batch_size']
    names = json.loads(selection_path.read_text()).get(
        'evaluation_candidates', [winner, f'{winner}_seed42', f'{winner}_seed43'])
    lookup = {r['ID']: r for r in baseline}
    reference = [lookup[r['ID']] for r in rows]
    controls = {'Official_baseline': np.array([r['MUSIC_FAKE_PROB'] for r in reference])}
    linear = Heads.load(output / 'linear_control.npz')
    controls['Linear_same_split'] = np.array([
        linear.music(records[r['ID']]['music'], records[r['ID']]['music_audible']).max()
        for r in rows])
    for name in names:
        model = load_temporal_head(output / name / 'temporal_music.pt', device)
        model_records = records if model.temporal_bins == 1 else fine_records
        if model_records is None:
            raise ValueError('Fine-resolution checkpoint requires fine-resolution records')
        probability = predict_temporal_music(model, rows, model_records, device, batch_size)
        reloaded = load_temporal_head(output / name / 'temporal_music.pt', device)
        np.testing.assert_allclose(probability, predict_temporal_music(
            reloaded, rows, model_records, device, batch_size), atol=1e-6, rtol=0)
        controls[name] = probability
    reports = []
    for name, probability in controls.items():
        predictions = []
        for original, value in zip(reference, probability, strict=True):
            prediction = dict(original)
            if name != 'Official_baseline':
                prediction['MUSIC_FAKE_PROB'] = float(value)
                prediction['FILE_FAKE_PROB'] = max(
                    prediction['VOICE_PRESENT_PROB'] * prediction['VOICE_FAKE_PROB'],
                    prediction['MUSIC_PRESENT_PROB'] * prediction['MUSIC_FAKE_PROB'])
            predictions.append(prediction)
        panels = {}
        for panel in sorted({r['panel'] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r['panel'] == panel
                       or (panel in ('music_seen', 'music_ood') and r['panel'] == 'music_real')]
            panels[panel] = official_metrics([rows[i] for i in indices],
                                             [predictions[i] for i in indices])
        generators = {}
        for source in sorted({r['source'].split('/')[1] for r in rows
                              if r['source'].startswith('Echoes/')}):
            indices = [i for i, r in enumerate(rows) if r['panel'] == 'music_real'
                       or r['source'].startswith(f'Echoes/{source}/')]
            generators[source] = official_metrics([rows[i] for i in indices],
                                                   [predictions[i] for i in indices])
        report = dict(name=name, **music_metrics(rows, probability),
                      metrics=official_metrics(rows, predictions), panels=panels, generators=generators)
        reports.append(report)
        write_json(output / f'{name}_test_predictions.json', predictions)
    assert sha256(selection_path) == selection_hash
    write_json(output / 'test_report.json', dict(results=reports, selection_sha256=selection_hash,
        checkpoint_reload_tolerance=1e-6,
        limitation='Historical dev reused; no newly untouched external evaluation or leaderboard proof',
        comparison='Only music probability and baseline file-max combination change; voice/presence fixed'))
    write_json(output / 'progress.json', dict(stage='complete', utc=time.time()))
    return reports
