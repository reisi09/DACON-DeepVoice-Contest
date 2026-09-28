"""Measure MPS Demucs against the locked CPU stem cache, without adopting it."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from decision_heads import presence, stem_features, tagger_features
from mert12_submission import pin_torchaudio_loader
from panns_adaptation_cache import baseline, load_demucs, readout_assets
from panns_adaptation_data import load_cache, read_cache_index, sha256, validate_config, write_json
from panns_adaptation_run import check_budget, resource_usage


def probe(config):
    package = validate_config(config)
    index = read_cache_index(config)
    out = Path(config['output'])/'stem_mps_probe.json'
    if out.exists():
        raise FileExistsError('Preserve the previous device measurement')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.5)
    original = baseline(package)
    pin_torchaudio_loader(original)
    heads, labels, groups = readout_assets(package)
    model = load_demucs(package).to('mps')
    rows = {r['ID']: r for r in map(json.loads, Path(config['rows']).open())}
    records = []
    started = time.monotonic()
    with Path(config['archive']).open('rb') as archive:
        for entry in index['entries']:
            check_budget(config, time.monotonic()-started, resource_usage('mps'))
            row = rows[entry['ID']]
            audio = row['verified_audio']
            if 'local_path' in audio:
                path = Path(audio['local_path']).resolve()
                if not path.is_relative_to(Path(config['recovered_audio_root']).resolve()):
                    raise ValueError('Audio outside recovery namespace')
                payload = path.read_bytes()
            else:
                archive.seek(audio['tar_offset'])
                payload = archive.read(audio['bytes'])
            if hashlib.sha256(payload).hexdigest() != audio['sha256']:
                raise ValueError('Audio hash mismatch')
            with tempfile.TemporaryDirectory(dir=out.parent) as directory:
                source = Path(directory)/('input'+Path(row['path']).suffix)
                source.write_bytes(payload)
                raw = original.load_audio(source)
                peak = float(np.abs(raw).max())
                if peak > .6:
                    raw = (raw*(.6/peak)).astype(np.float32)
                normalized = Path(directory)/'normalized.wav'
                sf.write(normalized, raw, 16000, subtype='FLOAT')
                t = time.monotonic()
                with torch.inference_mode():
                    voice, _ = original.separate_voice_and_music(normalized, model, torch.device('mps'))
                torch.mps.synchronize()
                voice = np.pad(voice[:len(raw)], (0, max(0, len(raw)-len(voice))))
                stems = stem_features(raw, voice, 16000)
                seconds = time.monotonic()-t
            features = tagger_features(load_cache(entry)['tags'], labels, groups)
            reference = np.array(presence(heads, features | entry['stems'], (0., 0.)))
            measured = np.array(presence(heads, features | stems, (0., 0.)))
            if not np.isfinite(measured).all():
                raise ValueError('Nonfinite MPS stem readout')
            usage = resource_usage('mps')
            check_budget(config, time.monotonic()-started, usage)
            record = dict(ID=entry['ID'], audio_seconds=entry['seconds'], cpu_seconds=entry['stem_seconds'],
                mps_seconds=seconds, presence_max_abs=float(np.abs(reference-measured).max()),
                stem_max_abs=max(abs(stems[k]-entry['stems'][k]) for k in stems), **usage)
            records.append(record)
            print(json.dumps(record), flush=True)
    report = dict(records=records, config_identity=config['identity'], cache_index_sha256=sha256(out.parent/'cache/index.json'),
        producer_sha256=sha256(__file__), tolerance=1e-5,
        all_presence_within_tolerance=all(r['presence_max_abs']<=1e-5 for r in records),
        cpu_stem_seconds=sum(r['cpu_seconds'] for r in records), mps_stem_seconds=sum(r['mps_seconds'] for r in records),
        total_seconds=time.monotonic()-started, adopted=False, scope='engineering panel only; CPU cache unchanged')
    write_json(out, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    probe(json.loads(args.run.read_text()))
