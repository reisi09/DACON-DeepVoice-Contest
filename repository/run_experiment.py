"""A bounded pilot; every GPU result is checkpointed to the mounted Drive."""

import argparse
import csv
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from graph_features import Extractor, atomic_json, extract_cached
from graph_fusion import fit_fusion, graph_predictions
from graph_scoring import official_metrics, validate_predictions


def read_csv(path: Path) -> list:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_parity(extractor: Extractor, path: Path, feature) -> dict:
    """Compare the shared graph with untouched official functions on real audio."""
    base = extractor.base
    audio = base.load_audio(path)
    vp, mp = base.predict_presence(extractor.panns, extractor.vi, extractor.mi, audio)
    extractor.demucs.cpu()
    voice, music = base.separate_voice_and_music(path, extractor.demucs, extractor.device)
    extractor.demucs.to(extractor.device)
    vf = base.predict_fake(extractor.df, extractor.fake_index, voice, extractor.device)
    mf = base.predict_fake(extractor.df, extractor.fake_index, music, extractor.device)
    expected = np.array([base.combine_file_fake_score(vf, mf, vp, mp), vf, mf, vp, mp])
    actual = np.array(list(graph_predictions(feature)['G0'].values())[1:])
    delta = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        raise AssertionError(f'G0 differs from official inference by {delta}: {path}')
    return {'ID': path.stem, 'max_probability_delta': delta, 'reference': expected.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if start >= args.deadline:
        raise TimeoutError('The existing GPU budget has expired; no new run was started')
    labels = read_csv(args.validation / 'manifest.csv')
    if len({r['ID'] for r in labels}) != len(labels):
        raise ValueError('Duplicate validation IDs')
    paths = {r['ID']: args.validation / r['path'] for r in labels}
    if any(p.stem != key or not p.is_file() for key, p in paths.items()):
        raise ValueError('Manifest ID/path mismatch or missing audio')
    atomic_json(out / 'environment.json', {
        'python': sys.version, 'gpu': torch.cuda.get_device_name(),
        'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3,
        'started_unix': start, 'deadline_unix': args.deadline,
        'packages': {k: importlib.metadata.version(k) for k in
            ('torch', 'torchaudio', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'demucs', 'librosa')},
    })
    extractor = Extractor(args.baseline / 'script.py')
    try:
        dummy = sorted((args.baseline / 'data' / 'test').glob('*.wav'))
        dummy_features = extract_cached(extractor, dummy, out / 'cache', args.deadline)
        atomic_json(out / 'dummy_features.json', dummy_features)
        ordered = list(paths.values())
        random.Random(20260913).shuffle(ordered)
        first = extract_cached(extractor, ordered[:1], out / 'cache', args.deadline)
        if not first:
            raise TimeoutError('No validation feature completed within the budget')
        atomic_json(out / 'reference_parity.json', reference_parity(extractor, ordered[0], first[0]))
        stress_path = args.baseline / 'stress_60s.wav'
        signal = extractor.base.load_audio(ordered[0])
        sf.write(stress_path, np.tile(signal, 960000 // len(signal) + 1)[:960000], 16000)
        try:
            stress = extract_cached(extractor, [stress_path], out / 'stress_cache', args.deadline)
            atomic_json(out / 'stress60.json', {'completed': bool(stress),
                'purpose': '60-second runtime and memory only; repeated source is not validation evidence'})
        except torch.cuda.OutOfMemoryError as error:
            atomic_json(out / 'stress60.json', {'completed': False, 'error': str(error),
                'submission_supported': False})
            torch.cuda.empty_cache()
        features = extract_cached(extractor, ordered, out / 'cache', args.deadline)
    finally:
        extractor.close()
    atomic_json(out / 'features.json', features)
    available = {f['ID'] for f in features}
    completed_labels = [r for r in labels if r['ID'] in available]
    fusion = fit_fusion(features, labels)
    atomic_json(out / 'fusion.json', fusion)
    predictions = {name: [] for name in ('G0', 'G1', 'G2', 'G3', 'G4')}
    for feature in features:
        for name, row in graph_predictions(feature, fusion).items():
            predictions[name].append(row)
    report = {'complete': len(features) == len(labels), 'completed': len(features),
        'planned': len(labels), 'elapsed_seconds': time.time() - start,
        'scope': 'external_source_group_holdout_not_DACON_leaderboard', 'graphs': {}}
    heldout = [r for r in completed_labels if r['split'] == 'holdout']
    heldout_ids = {r['ID'] for r in heldout}
    heldout_groups = {g for r in heldout for g in r['group_id'].split('|')}
    report['holdout'] = {'clips': len(heldout), 'atomic_source_groups': len(heldout_groups),
        'limitation': 'Transforms and mixtures are correlated; source groups are the independent units'}
    weights = {'file_eer': .45, 'voice_eer': .18, 'music_eer': .27,
        'voice_presence_auc': .05, 'music_presence_auc': .05}
    for name, rows in predictions.items():
        validate_predictions(rows, [f['ID'] for f in features])
        write_csv(out / f'{name}_predictions.csv', rows)
        by_id = {r['ID']: r for r in rows if r['ID'] in heldout_ids}
        selected = [by_id[r['ID']] for r in heldout]
        result = official_metrics(heldout, selected) if heldout else {
            'score': None, 'metrics': {k: None for k in weights},
            'invalid': {k: 'no completed holdout examples' for k in weights},
            'partial': {}, 'counts': {k: 0 for k in weights},
            'unknown_counts': {k: 0 for k in weights}, 'coverage': {k: 0.0 for k in weights}}
        # ponytail: partial-label proxy only selects this pilot; it is never an official score.
        result['selection_proxy'] = sum(weight * ((1 - result['metrics'][metric]) if metric.endswith('eer')
            else result['metrics'][metric]) for metric, weight in weights.items()
            if result['metrics'][metric] is not None)
        report['graphs'][name] = result
        print(name, json.dumps(result, ensure_ascii=False), flush=True)
    ranked = sorted(report['graphs'], key=lambda g: report['graphs'][g]['selection_proxy'], reverse=True)
    best = ranked[0]
    improvement = report['graphs'][best]['selection_proxy'] - report['graphs']['G0']['selection_proxy']
    stress_ok = json.loads((out / 'stress60.json').read_text())['completed']
    eligible = report['complete'] and stress_ok and len(heldout_groups) >= 20 and improvement >= .01
    report['selection'] = {'graph': best if eligible else 'G0', 'best_observed_graph': best,
        'proxy_improvement_over_G0': improvement, 'new_submission_supported': eligible,
        'reason': 'complete_pilot_and_proxy_gain_at_least_0.01' if eligible else 'retain_reference_without_sufficient_evidence'}
    records = [json.loads(p.read_text()) for p in (out / 'cache').glob('*.json')]
    timings = [r['seconds'] for r in records]
    report['runtime'] = {'unique_cached_files': len(records), 'total_extraction_seconds': sum(timings),
        'mean_seconds_per_file': float(np.mean(timings)),
        'max_allocated_gib': max(r['max_allocated_gib'] for r in records),
        'max_reserved_gib': max(r['max_reserved_gib'] for r in records),
        'L4_60_minute_compliance': 'unverified; A100 measurements are not L4 evidence'}
    report['runtime']['stress60'] = [
        {k: v for k, v in json.loads(p.read_text()).items() if k != 'feature'}
        for p in (out / 'stress_cache').glob('*.json')]
    atomic_json(out / 'comparison.json', report)
    for name in predictions:
        dummy_rows = [graph_predictions(f, fusion)[name] for f in dummy_features]
        write_csv(out / f'{name}_dummy_submission.csv', dummy_rows)
    atomic_json(out / 'status.json', {'status': 'comparison_finished', 'unix': time.time(),
        'selected_graph': report['selection']['graph'], 'completed': len(features)})
    print('COMPARISON_FINISHED', json.dumps(report['selection']), flush=True)


if __name__ == '__main__':
    main()
