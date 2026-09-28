"""Verified V6 cache import and bounded four-second MERT crop extraction."""

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from v6_core import COLUMNS
from v6_data import assert_disjoint, sha256, write_json
from v6_runtime import Engine
from v6_train import feature_record


def read_existing(project: Path):
    work = project / 'v6_run'
    metadata = json.loads((work / 'training.json').read_text())
    split = json.loads((work / 'split.json').read_text())
    if hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest() != metadata['split_sha256']:
        raise ValueError('V6 split no longer matches its recorded training run')
    assert_disjoint(split['expert'], split['fusion'])
    assert_disjoint(split['expert'] + split['fusion'], split['dev'])
    rows = [r for p in ('expert', 'fusion', 'dev') for r in split[p]]
    records = {r['ID']: feature_record(r, work / 'cache', metadata['cache_id']) for r in rows}
    return split, records, metadata


def crop_starts(length: int, identity: str) -> np.ndarray:
    last = max(0, length - 64000)
    seed = int(hashlib.sha256(('41:' + identity).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, last + 1, 5).astype(int)
    return np.unique([rng.integers(lo, max(lo + 1, hi)) for lo, hi in zip(edges[:-1], edges[1:])]).astype(int)


def fresh_features(context, assets: Path, deadline: float):
    """Preserve voice/presence/fusion partition; change only music waveform crops."""
    import librosa
    import torch

    split, existing, output = context
    engine = Engine(assets, 'cuda')
    base = engine.baseline
    expert_ids = {r['ID'] for r in split['expert']}
    dev_ids = {r['ID'] for r in split['dev']}
    rows = [r for p in ('dev', 'fusion', 'expert') for r in split[p]
            if p != 'expert' or r['MUSIC_PRESENT'] == '1']
    updated = dict(existing)
    baseline = []
    cache = output / 'four_second_cache'
    cache.mkdir(parents=True, exist_ok=True)
    version = sha256(Path(__file__)) + ':' + engine.cache_id
    audit = []
    for index, row in enumerate(rows):
        if time.time() >= deadline:
            raise TimeoutError('Feature extraction deadline; partial cache is retained')
        key = hashlib.sha256((version + row['audio_sha256'] + str(row['ID'] in expert_ids)).encode()).hexdigest()
        target = cache / (key + '.npz')
        if not target.exists():
            if sha256(Path(row['path'])) != row['audio_sha256']:
                raise ValueError('Audio changed after input verification')
            original = base.load_audio(Path(row['path']))
            seed = int(row['audio_sha256'][:8], 16)
            random.seed(seed)
            torch.manual_seed(seed)
            voice, music = base.separate_voice_and_music(Path(row['path']), engine.demucs, engine.device)
            music = np.pad(music[:len(original)], (0, max(0, len(original) - len(music))))
            starts = (crop_starts(len(original), row['ID']) if row['ID'] in expert_ids
                      else existing[row['ID']]['starts'])
            vectors, audible, quiet = [], [], []
            with torch.inference_mode():
                for start in starts:
                    crop = music[int(start):int(start) + 64000]
                    crop = np.pad(crop, (0, max(0, 64000 - len(crop))))
                    audible.append(float(base.calculate_rms(crop) >= base.SILENCE_RMS))
                    rms = np.sqrt(np.mean(crop.reshape(40, 1600).astype('float64') ** 2, axis=1))
                    quiet.append(float(np.mean(rms < 10 ** (-50 / 20))))
                    waveform = librosa.resample(crop, orig_sr=16000, target_sr=24000, res_type='soxr_hq')
                    inputs = engine.processor(waveform, sampling_rate=24000, return_tensors='pt')
                    vectors.append(engine.mert(inputs.input_values.to(engine.device)).last_hidden_state.mean(1)[0].cpu().numpy())
            reference = np.array([])
            if row['ID'] in dev_ids:
                # Recompute the organizer's component scores with its original segmentation.
                vf = base.predict_fake(engine.raw, engine.fake_index, voice, engine.device)
                mf = base.predict_fake(engine.raw, engine.fake_index, music, engine.device)
                vp, mp = existing[row['ID']]['presence'].max(0)
                reference = np.round([max(vp * vf, mp * mf), vf, mf, vp, mp], 10)
            with target.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, music=np.asarray(vectors, dtype='float32'),
                                    music_audible=np.asarray(audible), starts=starts,
                                    quiet=np.asarray(quiet), baseline=reference)
            target.with_suffix('.tmp').replace(target)
        with np.load(target, allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('Nonfinite four-second features')
        updated[row['ID']] = dict(existing[row['ID']], **{k: result[k] for k in ('music', 'music_audible', 'starts')})
        audit.append(dict(ID=row['ID'], starts=result['starts'].tolist(),
                          quiet_fraction=result['quiet'].tolist(), cache=target.name))
        if row['ID'] in dev_ids:
            baseline.append(dict(ID=row['ID'], **dict(zip(COLUMNS, result['baseline'].tolist()))))
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            state = dict(stage='four_second_features', completed=index + 1, total=len(rows), utc=time.time())
            write_json(output / 'progress.json', state)
            write_json(output / 'crop_audit.json', audit)
            write_json(output / 'baseline_predictions.json', baseline)
            print(json.dumps(state), flush=True)
    return updated, baseline
