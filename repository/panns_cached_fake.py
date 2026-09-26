# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Run with the existing pinned Python: panns_cached_fake.py --help."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Final

import numpy as np
import torch

from decision_heads import file_score, presence
from graph_scoring import validate_predictions
from localization_models import WindowMIL
from panns_adaptation_cache import read_stem_cache
from panns_adaptation_data import identity, sha256, validate_config, write_json
from panns_fixed_predictions import configure_numerics, load_fixed_predictions

RUNTIME_SHA: Final = 'f0b14ebe7ac0050d77b5fe7c40a69318105ca378ba1f9c844d321b076a9f3211'
FAKE_COLUMNS: Final = ('VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB')
TOLERANCE: Final = 1e-5


def before_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError('Frozen-head evaluation deadline reached')


def head_scores(heads, data, device: str):
    music, voice, fake_index = heads
    features = torch.from_numpy(data['mert_windows']).to(device)
    embeddings = torch.from_numpy(data['voice_embeddings']).to(device)
    if (not len(features) or not len(embeddings) or not torch.isfinite(features).all()
            or not torch.isfinite(embeddings).all()):
        raise ValueError('Invalid cached encoder values')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        mf = []
        for head in music:
            logits, _ = head.head((features-head.mean)/head.scale)
            mf.append(float((logits.float().logsumexp(0)-np.log(len(features))).sigmoid()))
        vf = [float(head(embeddings).float().softmax(-1)[:, fake_index].max()) for head in voice]
    return dict(VOICE_FAKE_PROB=0. if bool(data['voice_silent']) else float(np.mean(vf)),
                MUSIC_FAKE_PROB=float(np.mean(mf)))


def compare_gate(cached, fresh):
    if not cached or cached.keys() != fresh.keys():
        raise ValueError('Parity requires the same nonempty audio IDs')
    differences = {rid: {name: abs(cached[rid][name]-fresh[rid][name]) for name in FAKE_COLUMNS}
                   for rid in cached}
    maximum = max(value for item in differences.values() for value in item.values())
    return dict(passed=bool(all(np.isfinite(value) for item in differences.values() for value in item.values())
                            and maximum <= TOLERANCE),
                tolerance=TOLERANCE, max_absolute_difference=maximum, differences=differences)


def raw_predictions(config, identifiers: list[str], output: Path, deadline: float, device: str):
    before_deadline(deadline)
    probe = dict(config, panel_ids=identifiers, output=str(output),
                 fixed_predictions=str(output/'fixed_predictions.json'))
    probe['identity'] = identity({k: v for k, v in probe.items() if k != 'identity'})
    run = output/'run.json'
    write_json(run, probe)
    with (output/'execution.log').open('a') as log:
        producer = import_module('panns_fixed_predictions').__file__
        subprocess.run([sys.executable, producer,
            '--run', str(run), '--device', device], check=True, stdout=log, stderr=subprocess.STDOUT,
            timeout=max(1., deadline-time.time()))
    return {entry['ID']: entry for entry in load_fixed_predictions(probe, device)['entries']}


def cached_predictions(config, options: argparse.Namespace):
    package = validate_config(config)
    reference = json.loads(Path(config['reference']).read_text())
    if reference['members']['model/runtime/full_bank_submission.py'] != RUNTIME_SHA:
        raise ValueError('Frozen Engine arithmetic differs from the audited DataV4 runtime')
    for name, digest in reference['members'].items():
        if sha256(package/name) != digest:
            raise ValueError('Frozen package changed: '+name)
    for name in ('localization_models.py', 'decision_heads.py', 'eight_heads.py'):
        if sha256(import_module(Path(name).stem).__file__) != reference['members']['model/runtime/'+name]:
            raise ValueError('Cached-head implementation differs from the frozen package: '+name)
    rows = {r['ID']: r for r in map(json.loads, Path(config['rows']).read_text().splitlines())}
    stems = read_stem_cache(config, rows)
    selected = [rid for rid in config['panel_ids'] if rows[rid]['role'] == options.role]
    if not selected:
        raise ValueError('Requested evaluation role is empty')
    receipt = json.loads(Path(config['stem_cache']).read_text())['source_receipt']
    before_deadline(options.deadline_epoch)
    if sha256(options.legacy_tar) != receipt['archive_sha256']:
        raise ValueError('Legacy archive hash mismatch')
    mapping = {f"{stems[rid]['source_cache']}/{stems[rid]['source_bank_id']}.npz": rid
               for rid in selected if rid in stems}
    if len(mapping) != sum(rid in stems for rid in selected):
        raise ValueError('Legacy row mapping is not one-to-one')
    music, voice, indices = [], [], set()
    for seed in (41, 42, 43):
        for route, collection in (('music', music), ('voice', voice)):
            saved = torch.load(package/f'model/trained/{route}_{seed}.pt', map_location='cpu', weights_only=True)
            if saved['route'] != route or saved['seed'] != seed:
                raise ValueError('Frozen checkpoint route or seed mismatch')
            state = saved['model']
            model = WindowMIL(state['mean'], state['scale']) if route == 'music' else torch.nn.Linear(1280, 2)
            model.load_state_dict(state, strict=True)
            collection.append(model.to(options.device).eval().requires_grad_(False))
            indices.add(saved['fake_label_index'])
    if indices != {0}:
        raise ValueError('Frozen fake-label mapping changed')
    presence_heads = json.loads((package/'model/trained/presence_head.json').read_text())
    ensemble = json.loads((package/'model/trained/file_ensemble.json').read_text())
    predictions, metadata, seen = {}, {}, set()
    with tarfile.open(options.legacy_tar, 'r:') as archive:
        for member in archive:
            before_deadline(options.deadline_epoch)
            if member.name not in mapping:
                continue
            if member.name in seen or not member.isfile() or not 0 < member.size <= 64*1024**2:
                raise ValueError('Invalid legacy cache member')
            seen.add(member.name)
            rid = mapping[member.name]
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != stems[rid]['source_sha256']:
                raise ValueError('Legacy feature bytes changed: '+rid)
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                fake = head_scores((music, voice, 0), data, options.device)
                vp, mp = presence(presence_heads, json.loads(str(data['presence_features'])), data['presence'])
                predictions[rid] = dict(ID=rid, audio_sha256=rows[rid]['verified_audio']['sha256'], **fake,
                    VOICE_PRESENT_PROB=vp, MUSIC_PRESENT_PROB=mp,
                    FILE_FAKE_PROB=file_score(ensemble, dict(Pv=vp, Pm=mp,
                        Fv=fake['VOICE_FAKE_PROB'], Fm=fake['MUSIC_FAKE_PROB'])))
                metadata[rid] = dict(samples=int(data['samples']), silent=bool(data['voice_silent']),
                                     source=rows[rid]['source'])
    if seen != set(mapping):
        raise ValueError('Legacy archive is missing selected cache entries')
    del music, voice, model, collection
    gc.collect()
    if options.device == 'cuda':
        torch.cuda.empty_cache()
    probe = set()
    for source in sorted({item['source'] for item in metadata.values()})[:4]:
        group = sorted((rid for rid in metadata if metadata[rid]['source'] == source),
                       key=lambda rid: (metadata[rid]['samples'], rid))
        probe.update((group[0], group[-1]))
    for silent in (False, True):
        group = sorted(rid for rid in metadata if metadata[rid]['silent'] == silent)
        if group:
            probe.add(group[0])
    missing = set(selected)-predictions.keys()
    raw_ids = [rid for rid in selected if rid in missing or rid in probe]
    fresh = raw_predictions(config, raw_ids, options.output/'raw_gate', options.deadline_epoch, options.device)
    gate = compare_gate({rid: predictions[rid] for rid in probe}, {rid: fresh[rid] for rid in probe})
    gate['coverage'] = {rid: metadata[rid] for rid in sorted(probe)}
    write_json(options.output/'parity_gate.json', gate)
    if not gate['passed']:
        remaining = [rid for rid in selected if rid not in fresh]
        fresh.update(raw_predictions(config, remaining, options.output/'raw_fallback',
                                     options.deadline_epoch, options.device))
        predictions = fresh
    else:
        predictions.update(fresh)
    entries = [predictions[rid] for rid in selected]
    validate_predictions([{k: v for k, v in e.items() if k != 'audio_sha256'} for e in entries], selected)
    return dict(config_identity=config['identity'], rows_sha256=sha256(config['rows']),
        reference_sha256=sha256(config['reference']), legacy_archive_sha256=receipt['archive_sha256'],
        stem_cache_sha256=sha256(config['stem_cache']), producer_sha256=sha256(__file__),
        role=options.role, device=options.device, gate=gate, legacy_used=gate['passed'],
        entries=entries, entries_sha256=identity(entries), independent_evaluation=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--legacy-tar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--role', choices=['selection', 'development'], default='selection')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    options = parser.parse_args()
    if not time.time() < options.deadline_epoch <= time.time()+12*3600:
        parser.error('Deadline must be in the next 12 hours')
    configure_numerics(options.device)
    result = cached_predictions(json.loads(options.run.read_text()), options)
    write_json(options.output/'fixed.json', result)
    print(json.dumps(dict(fixed_prediction_rows=len(result['entries']), role=options.role,
                         legacy_used=result['legacy_used'], parity=result['gate']['passed'])), flush=True)


if __name__ == '__main__':
    main()
