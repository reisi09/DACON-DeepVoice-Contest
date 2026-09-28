"""Prepare auditable full-data C3 inputs without deleting original recordings."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from v6_data import sha256, write_json


def usable_windows(audio: np.ndarray) -> tuple[list[int], list[int]]:
    """Match C3's 64600-sample grid; require >=20% active 100ms frames."""
    last = max(0, len(audio) - 64600)
    starts = list(range(0, last + 1, 64600))
    if starts[-1] != last:
        starts.append(last)
    usable = []
    for index, start in enumerate(starts):
        crop = audio[start:start + 64600]
        count = len(crop) // 1600
        if not count:
            continue
        frames = crop[:count * 1600].reshape(count, 1600).astype('float64')
        active = np.sqrt((frames ** 2).mean(1)) >= 10 ** (-50 / 20)
        if active.mean() >= .2:
            usable.append(index)
    return starts, usable


def component_ids(rows: list[dict[str, str]]) -> np.ndarray:
    """Connect all parent groups transitively before assigning calibration folds."""
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for group in row['group_id'].split('|'):
            if not group:
                raise ValueError('Missing parent group')
            if group in seen:
                parents[root(i)] = root(seen[group])
            seen[group] = i
    return np.array([root(i) for i in range(len(rows))])


def prepare(root: Path, output: Path) -> dict:
    audited = {r['ID']: r for r in csv.DictReader(
        (root / 'results/data_audit_20260915/audio_features.csv').open(encoding='utf-8-sig'))}
    old = json.loads((root / 'results/v6_l40s_20260915/v6_run/split.json').read_text())
    old_rows = {r['ID']: r for p in ('expert', 'fusion', 'dev') for r in old[p]}
    kept, excluded, quality, seen = [], [], {}, {}
    for partition in ('train', 'dev', 'sealed'):
        with (root / 'v4_validation' / f'{partition}.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        for source in rows:
            path = root / 'v4_validation' / source['path']
            audio, rate = sf.read(path, dtype='float32')
            if rate != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError(f'Corrupted source requires review: {source["ID"]}')
            pcm = hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()
            if pcm != audited[source['ID']]['pcm_sha256']:
                raise ValueError(f'Audio changed since quality audit: {source["ID"]}')
            if pcm in seen:
                first = seen[pcm]
                for key in ('FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT'):
                    if source[key] and first[key] and source[key] != first[key]:
                        raise ValueError('Conflicting labels on duplicate PCM')
                excluded.append(dict(ID=source['ID'], reason='duplicate_pcm', retained=first['ID']))
                continue
            seen[pcm] = source
            pure = source['MUSIC_PRESENT'] == '1' and source['source'].lower().startswith(('fma/', 'nsynth', 'echoes/'))
            if pure and source['MUSIC_FAKE'] not in ('0', '1'):
                raise ValueError('Unknown pure-music label')
            if pure:
                starts, usable = usable_windows(audio)
                quality[source['ID']] = dict(starts=starts, usable_indices=usable,
                                              removed_windows=len(starts) - len(usable))
                if not usable:
                    excluded.append(dict(ID=source['ID'], reason='no_window_with_20pct_active_frames', retained=''))
                    continue
            remote = (old_rows[source['ID']]['path'] if source['ID'] in old_rows else
                      '/root/C3_Final/v4_validation/' + source['path'])
            row = dict(source, path=remote, audio_sha256=sha256(path), pcm_sha256=pcm,
                       music_training='1' if pure else '0')
            if source['ID'] in old_rows and row['audio_sha256'] != old_rows[source['ID']]['audio_sha256']:
                raise ValueError('Cached source changed since V6')
            kept.append(row)
    for row in old['fusion']:
        if row['ID'].startswith('v6_mix_'):
            excluded.append(dict(ID=row['ID'], reason='legacy_tiled_synthetic_mix_without_parent_file_ids', retained=''))
    plan = dict(purpose='full_training', partitions=['train', 'dev', 'sealed'], rows=kept,
                music_files=sum(r['music_training'] == '1' for r in kept), fusion_files=len(kept),
                quality=quality, exclusions=excluded, evaluation='No untouched holdout after full-data fit',
                policy=dict(window_samples=64600, frame_samples=1600, active_dbfs=-50,
                            minimum_active_fraction=.2, nsynth_loss_mass=.05,
                            synthetic_mixes='exclude legacy V6 supplementary mixtures'))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'full_data_plan.json', plan)
    with (output / 'exclusions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['ID', 'reason', 'retained'], lineterminator='\n')
        writer.writeheader(); writer.writerows(excluded)
    return plan
