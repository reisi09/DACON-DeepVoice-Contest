"""Matched MERT330 classifier study; reuse sealed features and retain the original split."""
from dataclasses import asdict, replace
from itertools import product
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from music_temporal import fit_temporal_head, predict_temporal_music
from temporal_backbone import REGULARIZED_C08
from temporal_experiment import evaluate_locked_test, music_metrics, training_masks
from temporal_resolution import load_pair, select_resolution
from v6_data import assert_disjoint, sha256, write_json


def candidates():
    base = replace(REGULARIZED_C08, input_dim=1024)
    settings = {}
    for cosine, smoothing, plateau in product((False, True), (0., .05), (False, True)):
        name = f'{"cosine" if cosine else "mlp"}_ls{int(smoothing*100)}_lr{int(plateau)}'
        settings[name] = replace(base, mlp_output=not cosine, cosine_output=cosine,
                                 label_smoothing=smoothing, lr_plateau=plateau)
    for head, regularized in product(('prototype', 'attention', 'conv_mean', 'conv_lme'), (False, True)):
        settings[f'{head}_reg{int(regularized)}'] = replace(base, alternative_output=head,
            label_smoothing=.05 if regularized else 0., lr_plateau=regularized)
    return settings


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert 'A100' in torch.cuda.get_device_name()
    for name, digest in json.loads(Path('input_manifest.json').read_text())['sha256'].items():
        assert sha256(Path(name)) == digest, name
    split = json.loads(Path('locked_split.json').read_text())
    for a, b in (('train', 'tune'), ('train', 'test'), ('tune', 'test')):
        assert_disjoint(split[a], split[b])
    rows = [r for part in ('train', 'tune', 'test') for r in split[part]]
    records = []
    for filename, folder, dimension in (
            ('reference330_features_manifest.json', 'feature_cache', 1024),
            ('reference95_features_manifest.json', 'reference95_cache', 768)):
        manifest = json.loads(Path(filename).read_text())
        selected = {}
        for row in rows:
            item = manifest['records'][row['ID']]
            path = Path(folder)/item['file']
            assert item['identity'] == row['audio_sha256']+':'+manifest['cache_id']
            assert item['audio_sha256'] == row['audio_sha256'] and sha256(path) == item['sha256']
            selected[row['ID']], _ = load_pair(path, item['identity'], dimension)
        records.append(selected)
    wide, coarse = records
    output = Path('results'); output.mkdir(exist_ok=True)
    masked = training_masks(split['train'], wide, json.loads(Path('full_data_plan.json').read_text())['quality'])
    deadline = float(os.environ['V7_RESUMED_EPOCH'])+3600-120
    settings = candidates()
    write_json(output/'search_protocol.json', dict(settings={k:asdict(v) for k,v in settings.items()},
        seeds=[41,42,43], split_sha256=sha256(Path('locked_split.json')), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, selection='tune mean EER, std, BCE, name; frozen before historical test',
        early_stopping='unsmoothed validation loss, patience20, min_delta1e-4; max300 safety limit',
        final_submission_changed=False, train_counts={k:len(v) for k,v in split.items()},
        limitation='historical test already used in previous studies; no independent official score',
        nsynth_mass=0., nsynth_note='Keep the original experimental weighting; submission uses positive mass0.05'))
    trials = []
    summaries = []
    for setting, recipe in settings.items():
        metrics = []
        for seed in (41,42,43):
            name = f'{setting}_seed{seed}'
            chosen = replace(recipe, seed=seed)
            write_json(output/'progress.json', dict(stage='training', candidate=name, completed=len(trials),
                                                    total=3*len(settings), utc=time.time()))
            started = time.time()
            model, report = fit_temporal_head(split['train'], masked, (split['tune'], wide),
                                              output/name, deadline, recipe=chosen, device='cuda')
            p = predict_temporal_music(model, split['tune'], wide, 'cuda', chosen.batch_size)
            metric = music_metrics(split['tune'], p); metrics.append(metric)
            trials.append(dict(name=name, setting=setting, seed=seed, recipe=asdict(chosen),
                selected_epoch=report['selected_epoch'], epochs_completed=report['epochs_completed'],
                parameters=report['parameters'], seconds=time.time()-started, **metric))
            np.savez_compressed(output/name/'tune_predictions.npz',
                                ids=np.array([r['ID'] for r in split['tune']]), probability=p)
            write_json(output/'tuning.json', trials)
            print(trials[-1], flush=True)
            del model
        summaries.append(dict(name=setting, recipe=asdict(recipe), candidates=[f'{setting}_seed{s}' for s in (41,42,43)],
            music_eer_mean=float(np.mean([m['music_eer'] for m in metrics])),
            music_eer_std=float(np.std([m['music_eer'] for m in metrics])),
            balanced_bce_mean=float(np.mean([m['balanced_bce'] for m in metrics]))))
    write_json(output/'selection.json', dict(winner=select_resolution(summaries), summaries=summaries,
        test_used=False, frozen_at=time.time(), comparison_groups={s['name']:s['candidates'] for s in summaries},
        evaluation_candidates=[n for s in summaries for n in s['candidates']]))
    evaluate_locked_test(split['test'], coarse, json.loads(Path('baseline_predictions.json').read_text()),
                         output, wide_records=wide)


if __name__ == '__main__':
    main()
