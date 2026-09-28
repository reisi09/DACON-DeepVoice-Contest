"""Compare three training-only augmentation settings with three paired seeds."""

from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from music_temporal import Records, Rows, TemporalRecipe, fit_temporal_head, predict_temporal_music
from temporal_augmentation import CROP_LENGTHS, MASK_FRACTION
from temporal_experiment import music_metrics
from v6_data import write_json
from v7_ablation import Recipe, fit_head


def run_augmentation_search(train: Rows, tune: Rows, records: Records, masked: Records,
                            output: Path, deadline: float, *, epochs: int = 300,
                            device: str = 'cuda', batch_size: int = 128,
                            compare_architectures: bool = False) -> list[dict]:
    """Rank configuration means on tune only; retain every seed for later reporting."""
    settings = ([('D_deep_crop', .7, 0., False), ('S_simple_crop', .7, 0., True)]
                if compare_architectures else
                [('A0_none', 0., 0., False), ('A1_crop', .7, 0., False), ('A2_crop_mask', .7, .3, False)])
    results, summaries = [], []
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'augmentation_protocol.json', dict(settings=settings,
        comparison='architecture' if compare_architectures else 'augmentation',
        seeds=[41, 42, 43], crop_lengths=CROP_LENGTHS, mask_fraction=MASK_FRACTION,
        crop_labels='Only whole-track FMA/NSynth real or Echoes fake; partial/unknown skipped',
        input='Immutable cached MERT; no waveform synthesis or encoder retraining',
        validation='No augmentation; early stopping uses class-balanced BCE',
        randomness='Shuffle, crop and feature mask use independent seeded NumPy streams'))
    for setting, crop_probability, mask_probability, simplified in settings:
        members = []
        for seed in (41, 42, 43):
            name = f'{setting}_seed{seed}'
            recipe = replace(TemporalRecipe(), epochs=epochs, batch_size=batch_size, seed=seed,
                             crop_probability=crop_probability, feature_mask_probability=mask_probability,
                             simplified=simplified)
            write_json(output / 'progress.json', dict(stage='augmentation_training',
                candidate=name, completed=len(results), total=len(settings)*3, utc=time.time()))
            started = time.time()
            model, report = fit_temporal_head(train, masked, (tune, records), output / name,
                                               deadline, recipe=recipe, device=device)
            probability = predict_temporal_music(model, tune, records, device, batch_size)
            panels = {}
            for panel, allowed in [('pure_music', {'music_seen', 'music_ood', 'music_real'}),
                                   ('mixture', {'mixture'}), ('splice', {'splice'})]:
                indices = [i for i, row in enumerate(tune) if row['panel'] in allowed]
                rows = [tune[i] for i in indices]
                if len({r['MUSIC_FAKE'] for r in rows}) == 2:
                    panels[panel] = music_metrics(rows, probability[indices])
            result = dict(name=name, setting=setting, seed=seed, recipe=asdict(recipe),
                parameters=report['parameters'],
                **music_metrics(tune, probability), panels=panels,
                selected_epoch=report['selected_epoch'], epochs_completed=report['epochs_completed'],
                optimizer_steps=report['optimizer_steps'], seconds=time.time()-started,
                augmentation_totals={k:sum(e[k] for e in report['augmentation_history'])
                                     for k in report['augmentation_history'][0]})
            np.savez_compressed(output / name / 'tune_predictions.npz',
                                ids=np.array([r['ID'] for r in tune]), probability=probability)
            results.append(result)
            members.append(result)
            write_json(output / 'tuning.json', results)
            print(name, {k:result[k] for k in ('music_eer','balanced_bce','selected_epoch','seconds')}, flush=True)
            del model
        summaries.append(dict(name=setting, recipe=members[0]['recipe'],
            music_eer_mean=float(np.mean([r['music_eer'] for r in members])),
            music_eer_std=float(np.std([r['music_eer'] for r in members])),
            balanced_bce_mean=float(np.mean([r['balanced_bce'] for r in members])),
            candidates=[r['name'] for r in members]))
    winner = min(summaries, key=lambda r:(r['music_eer_mean'],r['music_eer_std'],r['balanced_bce_mean'],r['name']))
    write_json(output / 'selection.json', dict(winner=winner, summaries=summaries,
        criterion='Three-seed tune EER mean, then population std, then BCE mean, then name',
        test_used=False, frozen_at=time.time(), evaluation_candidates=[r['name'] for r in results]))
    write_json(output / 'progress.json', dict(stage='linear_control', utc=time.time()))
    linear, _, stats = fit_head(train, masked, Recipe('linear_control', .05, True,
        epochs=epochs), deadline, validation=(tune, records))
    linear.save(output / 'linear_control.npz')
    write_json(output / 'linear_control.json', stats)
    return results
