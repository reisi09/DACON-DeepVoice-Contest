"""Reproduce the overfit audit from verified cached inputs; no cloud or encoder use."""

from collections import Counter
from contextlib import chdir
import csv
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from music_temporal import load_temporal_head, predict_temporal_music
from c3_data import component_ids
from temporal_experiment import load_features, music_metrics, training_masks
from v6_core import Heads
from v6_data import assert_disjoint, sha256, write_json
from v7_ablation import source_weights

RUN = ROOT / 'results/temporal_h100_20260915/unpacked'
OUT = Path(__file__).resolve().parent
CACHE = ROOT / '_workspace/h100_portable_backup/features.zip'


def main():
    torch.set_num_threads(4)
    split = json.loads((RUN / 'locked_split.json').read_text())
    rows = sum(split.values(), [])
    assert sha256(CACHE) == 'ddd3d481269e95172762f3ec988ccb15b541424d10ee05f9925deed463dcb13c'
    with TemporaryDirectory() as directory, zipfile.ZipFile(CACHE) as archive:
        base = Path(directory).resolve()
        assert all((base / n).resolve().is_relative_to(base) for n in archive.namelist())
        archive.extractall(base)
        with chdir(base):
            records = load_features(rows, ROOT, ROOT)
    for a, b in [('train', 'tune'), ('train', 'test'), ('tune', 'test')]:
        assert_disjoint(split[a], split[b])
    quality = json.loads((RUN / 'full_data_plan.json').read_text())['quality']
    masked = training_masks(split['train'], records, quality)
    train = split['train']
    weights = source_weights(train, .05)
    fma = np.array([r['source'].startswith('FMA/') for r in train])
    batches = {}
    for size in (16, 64, 128):
        # Same seed/order stream as the training loop, including the short final batch.
        rng = np.random.default_rng(41)
        counts, masses = [], []
        for _ in range(100):
            order = rng.permutation(len(train))
            for start in range(0, len(train), size):
                idx = order[start:start + size]
                counts.append(int(fma[idx].sum()))
                masses.append(float(weights[idx].sum() * len(train) / len(idx)))
        batches[size] = dict(steps_per_epoch=math.ceil(len(train) / size),
            expected_fma_full_batch=float(size * fma.mean()),
            probability_no_fma_full_batch=math.comb(len(train)-int(fma.sum()), size)
                / math.comb(len(train), size),
            observed_zero_fma_batches=int(np.sum(np.array(counts) == 0)),
            observed_batches=len(counts), loss_mass_cv=float(np.std(masses)/np.mean(masses)))
    result = dict(scope='Saved best-checkpoint diagnosis, not a new independent evaluation',
        cache_sha256=sha256(CACHE), verified_records=len(records),
        train_parent_components=len(set(component_ids(train))),
        train_weight_effective_sample_size=float(1/(weights @ weights)), batches=batches,
        sources={}, curves={}, metrics={}, input_perturbations={})
    for part, selected in split.items():
        result['sources'][part] = []
        for source in sorted({r['source'].split('/')[0] for r in selected}):
            chosen = [r for r in selected if r['source'].split('/')[0] == source]
            lengths = [len(records[r['ID']]['music']) for r in chosen]
            audible = [int(records[r['ID']]['music_audible'].sum()) for r in chosen]
            result['sources'][part].append(dict(source=source, files=len(chosen),
                groups=len({r['group_id'] for r in chosen}),
                labels=dict(Counter(r['MUSIC_FAKE'] for r in chosen)),
                windows_min=int(min(lengths)), windows_median=float(np.median(lengths)),
                windows_max=int(max(lengths)), audible_median=float(np.median(audible))))
    for name in ('T0', 'T7', 'T7_seed42', 'T7_seed43'):
        history = json.loads((RUN / 'results' / name / 'training.json').read_text())
        result['curves'][name] = {k: history[k] for k in ('recipe', 'parameters',
            'selected_epoch', 'epochs_completed', 'train_losses', 'validation_losses')}
        assert sha256(RUN / 'results' / name / 'temporal_music.pt') == history['checkpoint_sha256']
    model = load_temporal_head(RUN / 'results/T7/temporal_music.pt')
    linear = Heads.load(RUN / 'results/linear_control.npz')
    prediction_rows = []
    for part, selected in split.items():
        inputs = masked if part == 'train' else records
        probabilities = {
            'T7': predict_temporal_music(model, selected, inputs, batch_size=128),
            'Linear': np.array([linear.music(inputs[r['ID']]['music'],
                inputs[r['ID']]['music_audible']).max() for r in selected])}
        result['metrics'][part] = {}
        eligible = [i for i, r in enumerate(selected) if r['MUSIC_FAKE'] in ('0', '1')
                    and r['MUSIC_PRESENT'] == '1']
        length = np.array([len(inputs[selected[i]['ID']]['music']) for i in eligible])
        labels = np.array([int(selected[i]['MUSIC_FAKE']) for i in eligible])
        result['metrics'][part]['length_only_auc_positive_longer'] = roc_auc_score(labels, length)
        for name, p in probabilities.items():
            groups = {}
            for source in sorted({r['source'].split('/')[0] for r in selected}):
                idx = [i for i in eligible if selected[i]['source'].split('/')[0] == source]
                if not idx:
                    continue
                y = np.array([int(selected[i]['MUSIC_FAKE']) for i in idx])
                groups[source] = dict(files=len(idx), mean_fake_probability=float(p[idx].mean()),
                    bce=float(-np.log(np.where(y, p[idx], 1-p[idx]).clip(1e-7, 1)).mean()),
                    error_at_point5=float(np.mean((p[idx] >= .5) != y)))
            result['metrics'][part][name] = dict(**music_metrics(selected, p), by_source=groups)
            for r, value in zip(selected, p, strict=True):
                prediction_rows.append(dict(split=part, model=name, ID=r['ID'],
                    source=r['source'], label=r['MUSIC_FAKE'], probability=float(value)))
        if part == 'tune':
            with np.load(RUN / 'results/T7/tune_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['ids'], [r['ID'] for r in selected])
                np.testing.assert_allclose(probabilities['T7'], saved['probability'], atol=1e-6, rtol=0)
            for mode in ('reverse', 'repeat_mean'):
                changed = {}
                for r in selected:
                    record = inputs[r['ID']]
                    x, mask = record['music'].copy(), record['music_audible'] > 0
                    if mask.any():
                        x[mask] = x[mask][::-1] if mode == 'reverse' else x[mask].mean(0)
                    changed[r['ID']] = dict(record, music=x)
                p = predict_temporal_music(model, selected, changed, batch_size=128)
                result['input_perturbations'][mode] = dict(**music_metrics(selected, p),
                    mean_absolute_probability_change=float(np.abs(p-probabilities['T7']).mean()))
    write_json(OUT / 'diagnosis.json', result)
    with (OUT / 'predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(prediction_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    curve = result['curves']['T7']
    epochs = np.arange(1, len(curve['train_losses']) + 1)
    axes[0].plot(epochs, curve['train_losses'], label='Train: source-weighted BCE', color='#167d9a')
    axes[0].plot(epochs, curve['validation_losses'], label='Tune: class-balanced BCE', color='#c95337')
    axes[0].axvline(curve['selected_epoch'], linestyle=':', color='#777777', label='Saved best epoch')
    axes[0].set(title='T7: validation deteriorates after epoch 1', xlabel='Epoch', ylabel='BCE')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.15)
    for offset, source, color in [(-.2, 'FMA', '#167d9a'), (.2, 'Echoes', '#c95337')]:
        values = [next(g['windows_median'] for g in result['sources'][p] if g['source']==source)
                  for p in ('train', 'tune', 'test')]
        bars = axes[1].bar(np.arange(3)+offset, values, width=.35, label=source, color=color)
        axes[1].bar_label(bars, padding=3)
    axes[1].set(xticks=np.arange(3), xticklabels=['Train', 'Tune', 'Historical test'],
                title='Sequence lengths differ across splits', ylabel='Median cached windows per file', ylim=(0,65))
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / 'overfit.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('curves', 'sources')}, indent=2))


if __name__ == '__main__':
    main()
