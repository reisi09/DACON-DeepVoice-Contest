"""Bounded compact-head search using tune-only screening and three-seed confirmation."""

from dataclasses import asdict, replace
from pathlib import Path
import time
from typing import TypedDict

import numpy as np

from music_temporal import Records, Rows, TemporalRecipe, fit_temporal_head, predict_temporal_music
from temporal_experiment import music_metrics
from v6_data import write_json
from v7_ablation import Recipe, fit_head


class Trial(TypedDict):
    name: str
    setting: str
    seed: int
    recipe: dict
    parameters: int
    music_eer: float
    balanced_bce: float
    selected_epoch: int
    epochs_completed: int
    seconds: float


class Summary(TypedDict):
    name: str
    recipe: dict
    music_eer_mean: float
    music_eer_std: float
    balanced_bce_mean: float
    candidates: list[str]


def compact_candidates() -> dict[str, TemporalRecipe]:
    """Cover each search dimension evenly in 23 joint settings plus the untuned control."""
    base = TemporalRecipe(simplified=True, compact=True, crop_probability=.7)
    choices = dict(learning_rate=[3e-5, 1e-4, 3e-4, 1e-3],
                   weight_decay=[0., .001, .01, .05, .1], dropout=[0., .1, .2, .35, .5],
                   batch_size=[32, 64, 128, 256], crop_probability=[0., .35, .7, 1.],
                   feature_mask_probability=[0., .15, .3], nsynth_mass=[0., .025, .05, .1])
    rng = np.random.default_rng(20260915)
    columns = {key: rng.permutation(np.resize(values, 23)).tolist() for key, values in choices.items()}
    return {'C00': base, **{f'C{i+1:02d}': replace(base, **{k: v[i] for k, v in columns.items()})
                           for i in range(23)}}


def run_compact_search(train: Rows, tune: Rows, records: Records, masked: Records,
                       output: Path, deadline: float, *, device: str = 'cuda',
                       candidates: dict[str, TemporalRecipe] | None = None,
                       shortlist_size: int = 4) -> list[Trial]:
    """Freeze the shortlist before repeats and final comparisons before test access."""
    candidates = compact_candidates() if candidates is None else candidates
    if ('C00' not in candidates or not 1 <= shortlist_size <= len(candidates)
            or any(not r.compact or not r.simplified or r.seed != 41 for r in candidates.values())):
        message = 'Require compact seed-41 candidates, C00 control and a valid shortlist size'
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'tuning.json').exists():
        message = 'Use a fresh output directory to avoid mixing trials from separate searches'
        raise FileExistsError(message)
    write_json(output/'search_protocol.json', dict(candidates={k: asdict(v) for k, v in candidates.items()},
        screening_seed=41, confirm_seeds=[41, 42, 43], shortlist_size=shortlist_size,
        architecture='768->64 projection; three kernel-3 dilation-1 convs in one residual block; max; linear',
        activation='GELU unchanged; BCEWithLogitsLoss for training; sigmoid for inference',
        selection='Tune music EER; not full official score; no test labels enter search',
        epoch_selection='Minimum tune class-balanced BCE; max 300 epochs, patience 20 by default',
        search_method='Fixed stratified random joint configurations; not exhaustive or globally optimal',
        created_at=time.time()))
    results: list[Trial] = []

    def fit(setting: str, recipe: TemporalRecipe) -> None:
        name = f'{setting}_seed{recipe.seed}'
        write_json(output/'progress.json', dict(stage='compact_tuning', candidate=name,
                                               completed=len(results), utc=time.time()))
        started = time.time()
        model, report = fit_temporal_head(train, masked, (tune, records), output/name, deadline,
                                          recipe=recipe, device=device)
        p = predict_temporal_music(model, tune, records, device, recipe.batch_size)
        metrics = music_metrics(tune, p)
        result = Trial(name=name, setting=setting, seed=recipe.seed, recipe=asdict(recipe),
                       parameters=report['parameters'], music_eer=metrics['music_eer'],
                       balanced_bce=metrics['balanced_bce'], selected_epoch=report['selected_epoch'],
                       epochs_completed=report['epochs_completed'], seconds=time.time()-started)
        np.savez_compressed(output/name/'tune_predictions.npz',
                            ids=np.array([r['ID'] for r in tune]), probability=p)
        results.append(result)
        write_json(output/'tuning.json', results)
        print(name, metrics, 'epoch', report['selected_epoch'], flush=True)

    for setting, recipe in candidates.items():
        fit(setting, recipe)
    ranked = sorted(results, key=lambda r: (r['music_eer'], r['balanced_bce'], r['setting']))
    shortlist = [r['setting'] for r in ranked[:shortlist_size]]
    write_json(output/'shortlist.json', dict(candidates=shortlist, screening=ranked,
        criterion='Seed 41 tune EER, then BCE, then name', test_used=False, frozen_at=time.time()))
    confirmed = list(dict.fromkeys(['C00', *shortlist]))
    for setting in confirmed:
        for seed in (42, 43):
            fit(setting, replace(candidates[setting], seed=seed))
    for seed in (41, 42, 43):
        fit('S128', replace(candidates['C00'], compact=False, seed=seed))
    summaries: list[Summary] = []
    for setting in [*confirmed, 'S128']:
        members = [r for r in results if r['setting'] == setting]
        summaries.append(Summary(name=setting, recipe=members[0]['recipe'],
            music_eer_mean=float(np.mean([r['music_eer'] for r in members])),
            music_eer_std=float(np.std([r['music_eer'] for r in members])),
            balanced_bce_mean=float(np.mean([r['balanced_bce'] for r in members])),
            candidates=[r['name'] for r in members]))
    winner = min([s for s in summaries if s['name'] in shortlist],
                 key=lambda s: (s['music_eer_mean'], s['music_eer_std'], s['balanced_bce_mean'], s['name']))
    groups = {s['name']: s['candidates'] for s in summaries if s['name'] in ('S128', 'C00', winner['name'])}
    write_json(output/'selection.json', dict(winner=winner, summaries=summaries,
        criterion='Shortlisted three-seed tune EER mean, then population std, then BCE, then name',
        test_used=False, frozen_at=time.time(), comparison_groups=groups,
        evaluation_candidates=[name for names in groups.values() for name in names]))
    write_json(output/'progress.json', dict(stage='linear_control', utc=time.time()))
    linear, _, stats = fit_head(train, masked, Recipe('linear_control', .05, True,
        epochs=candidates['C00'].epochs), deadline, validation=(tune, records))
    linear.save(output/'linear_control.npz')
    write_json(output/'linear_control.json', stats)
    return results
