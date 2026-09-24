"""Audited, file-level PANNs v1 experiments. No provisioning or network operations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

import numpy as np
import soundfile as sf

from data_v4.build_manifest import components

TARGETS = ('VOICE_PRESENT', 'MUSIC_PRESENT')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def quarantine_roles(rows):
    component = components(rows)
    roles = defaultdict(set)
    for row in rows:
        roles[component[row['ID']]].add(row['role'])
    return {row['ID']: dict(component_id=component[row['ID']],
                           quarantined=len(roles[component[row['ID']]]) != 1)
            for row in rows}


def safe_member(member):
    path = PurePosixPath(member.name)
    if not member.isfile() or path.is_absolute() or '..' in path.parts or str(path) != member.name or '\\' in member.name:
        raise ValueError(f'Unsafe archive member: {member.name}')
    if member.size > 64 * 1024**2:
        raise ValueError(f'Archive member exceeds resource limit: {member.name}')


def audit_archive(archive, rows):
    by_path = {row['path']: row for row in rows}
    if len(by_path) != len(rows) or len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest IDs or paths')
    verified, seen = {}, set()
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            safe_member(member)
            if member.name in seen:
                raise ValueError('Duplicate archive member')
            seen.add(member.name)
            if member.name == 'SHA256SUMS.json':
                continue
            if member.name not in by_path:
                raise ValueError(f'Unlisted archive member: {member.name}')
            row = by_path[member.name]
            payload = tar.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row['sha256'] or ('bytes' in row and len(payload) != row['bytes']):
                raise ValueError(f'Audio hash/size mismatch: {row["ID"]}')
            try:
                with sf.SoundFile(io.BytesIO(payload)) as source:
                    if not 8000 <= source.samplerate <= 192000 or not 1 <= source.channels <= 2 or source.frames/source.samplerate > 600:
                        raise ValueError('Audio exceeds decode bounds')
                    sr = source.samplerate
                    audio = source.read(dtype='float32', always_2d=True)
            except Exception as exc:
                raise ValueError(f'Audio decode failed: {row["ID"]}: {exc}') from exc
            if not len(audio) or len(audio)/sr > 600 or not np.isfinite(audio).all():
                raise ValueError(f'Invalid decoded audio: {row["ID"]}')
            verified[row['ID']] = dict(sha256=digest, bytes=len(payload), sample_rate=sr,
                decoded_samples=len(audio), channels=audio.shape[1], seconds=len(audio)/sr,
                tar_offset=member.offset_data, path=member.name)
            if len(verified) % 5000 == 0:
                print(f'Validated {len(verified)} audio payloads', flush=True)
    return verified, [r['ID'] for r in rows if r['path'] not in seen]


def audit(config):
    out = Path(config['output'])
    rows = [json.loads(line) for line in Path(config['manifest']).open()]
    role_rows = list(csv.DictReader(Path(config['roles']).open()))
    roles = {r['ID']: r for r in role_rows}
    if len(roles) != len(role_rows) or set(roles) != {r['ID'] for r in rows}:
        raise ValueError('Role manifest identity mismatch')
    for row in rows:
        if row['role'] != roles[row['ID']]['role']:
            raise ValueError('Conflicting role assignment')
    closure = quarantine_roles(rows)
    actual = sha256(config['archive'])
    if actual != config['archive_sha256']:
        raise ValueError('Archive hash mismatch')
    verified, missing = audit_archive(config['archive'], rows)
    selected, excluded = [], Counter()
    for row in rows:
        reason = ('Dplus_not_D0' if row['source'] in ('LibriSpeech', 'MLAAD', 'MusicArena') else
                  'missing_audio' if row['ID'] not in verified else
                  'cross_role_component' if closure[row['ID']]['quarantined'] else None)
        if reason:
            excluded[f'{reason}|{row["source"]}|{row["role"]}'] += 1
            continue
        if any(row[k] not in (0, 1, None) for k in TARGETS):
            raise ValueError('Unexpected presence target')
        if not any(row[k] is not None for k in TARGETS):
            excluded['both_targets_unknown'] += 1
            continue
        selected.append({**row, 'verified_audio': verified[row['ID']], **closure[row['ID']]})
    train_rows = [r for r in selected if r['role'] == 'train']
    fraction = [sum(r[k] is not None for r in train_rows)/len(train_rows) for k in TARGETS]
    if any({r[k] for r in train_rows if r[k] is not None} != {0, 1} for k in TARGETS):
        raise ValueError('Training needs known positives and negatives for both targets')
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out/'presence_rows.jsonl'
    if rows_path.exists():
        raise FileExistsError('Use a fresh data audit namespace')
    rows_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    missing_set = set(missing)
    report = dict(rows=len(rows), verified=len(verified), missing=len(missing),
        missing_by_source=dict(Counter(r['source'] for r in rows if r['ID'] in missing_set)),
        eligible_D0=len(selected), excluded=dict(excluded), known_fraction=fraction,
        presence_by_role=dict(Counter(f'{r["role"]}|{r[TARGETS[0]]}|{r[TARGETS[1]]}' for r in selected)),
        input_hashes={key: sha256(config[key]) for key in ('manifest', 'roles')},
        archive_sha256=actual, rows_sha256=sha256(rows_path),
        complete_presence_panel=any(r['VOICE_PRESENT']==0 and r['MUSIC_PRESENT']==0 for r in selected),
        independent_evaluation_available=False, soundfile_version=sf.__version__)
    write_json(out/'data_audit.json', report)
    print(json.dumps(report), flush=True)
    return report


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(config):
    if identity({k: v for k, v in config.items() if k != 'identity'}) != config['identity']:
        raise ValueError('Run manifest identity mismatch')
    for path, digest in config['inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'Pinned input hash mismatch: {path}')
    reference = json.loads(Path(config['reference']).read_text())
    package = Path(reference['package'])
    for name, digest in reference['members'].items():
        if '/panns/' in name or '/htdemucs/' in name or name in (
                'model/trained/presence_head.json', 'model/assets/baseline/script.py'):
            if sha256(package/name) != digest:
                raise ValueError(f'Reference asset hash mismatch: {name}')
    return package


def read_cache_index(config):
    path = Path(config['output'])/'cache/index.json'
    if sha256(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Cache index hash mismatch')
    index = json.loads(path.read_text())
    if index['config_identity'] != config['identity']:
        raise ValueError('Cache/config identity mismatch')
    if [e['ID'] for e in index['entries']] != config['panel_ids']:
        raise ValueError('Cache panel IDs or order differ from locked manifest')
    return index


def load_cache(entry):
    if sha256(entry['path']) != entry['sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(entry['path'], allow_pickle=False) as data:
        if str(data['identity'].item()) != entry['identity']:
            raise ValueError('Cache identity mismatch')
        result = {key: data[key].copy() for key in data.files}
    if result['block5'].dtype != np.float32 or not np.isfinite(result['block5']).all():
        raise ValueError('Invalid cached activations')
    return result


def load_base(package):
    import torch
    from panns_inference.models import Cnn14
    base = Cnn14(32000, 1024, 320, 64, 50, 14000, 527)
    record = torch.load(package/'model/assets/baseline/model/panns/Cnn14_mAP=0.431.pth',
                        map_location='cpu', weights_only=True)
    base.load_state_dict(record['model'], strict=True)
    return base.eval()


def readout_assets(package):
    panns = package/'model/assets/baseline/model/panns'
    labels = [r['display_name'] for r in csv.DictReader((panns/'class_labels_indices.csv').open())]
    return (json.loads((package/'model/trained/presence_head.json').read_text()), labels,
            json.loads((panns/'component_labels.json').read_text()))


def baseline(package):
    import importlib.util
    path = package/'model/assets/baseline/script.py'
    spec = importlib.util.spec_from_file_location('panns_reference_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_demucs(package):
    import torch
    from fractions import Fraction
    from demucs.htdemucs import HTDemucs
    from demucs.pretrained import get_model
    original_load = torch.load
    def restricted_load(*args, **kwargs):
        kwargs['weights_only'] = True
        return original_load(*args, **kwargs)
    allowed = [HTDemucs, np.dtype, Fraction,
               (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        torch.load = restricted_load
        try:
            model = get_model('htdemucs', repo=package/'model/assets/baseline/model/htdemucs')
        finally:
            torch.load = original_load
    return model.cpu().eval().requires_grad_(False)


def cache(config):
    import tempfile
    import torch
    from panns_adaptation import AdaptedCnn14, presence_logits, tag_features
    from decision_heads import stem_features, tagger_features, presence
    from mert12_submission import pin_torchaudio_loader
    package = validate_config(config)
    torch.set_num_threads(4)
    out = Path(config['output'])/'cache'
    out.mkdir(parents=True, exist_ok=True)
    index_path = out/'index.json'
    if index_path.exists():
        saved = read_cache_index(config)
        for entry in saved['entries']:
            load_cache(entry)
        return saved
    rows = {r['ID']: r for r in (json.loads(s) for s in Path(config['rows']).open())}
    selected = [rows[rid] for rid in config['panel_ids']]
    base = load_base(package)
    model = AdaptedCnn14(base, train_blocks=())
    original = baseline(package)
    pin_torchaudio_loader(original)
    demucs = load_demucs(package)
    heads, labels, groups = readout_assets(package)
    entries = []
    start = time.monotonic()
    with Path(config['archive']).open('rb') as archive:
        for row in selected:
            t = time.monotonic()
            audio = row['verified_audio']
            archive.seek(audio['tar_offset'])
            payload = archive.read(audio['bytes'])
            if hashlib.sha256(payload).hexdigest() != audio['sha256']:
                raise ValueError('Selected audio hash mismatch')
            with tempfile.TemporaryDirectory(dir=out) as temporary:
                source = Path(temporary)/('input'+Path(row['path']).suffix)
                source.write_bytes(payload)
                raw = original.load_audio(source)
                peak = float(np.abs(raw).max())
                if peak > .6:
                    raw = (raw * (.6/peak)).astype(np.float32)
                segments = original.make_panns_segments(raw)
                starts = original.get_segment_starts(len(raw))
                normalized = Path(temporary)/'normalized.wav'
                sf.write(normalized, raw, 16000, subtype='FLOAT')
                stem_start = time.monotonic()
                with torch.inference_mode():
                    voice, _ = original.separate_voice_and_music(normalized, demucs, torch.device('cpu'))
                voice = np.pad(voice[:len(raw)], (0, max(0, len(raw)-len(voice))))
                stems = stem_features(raw, voice, 16000)
                stem_seconds = time.monotonic()-stem_start
            features, reference, adapted = [], [], []
            with torch.no_grad():
                for segment in segments:
                    waveform = torch.from_numpy(segment[None])
                    features.append(model.prefix(waveform))
                    reference.append(base(waveform)['clipwise_output'])
                    adapted.append(model.tags_from_block5(features[-1]))
                features = torch.cat(features)
                reference, adapted = torch.cat(reference), torch.cat(adapted)
                tag_error = float((reference-adapted).abs().max())
                numpy_feats = tagger_features(reference.numpy(), labels, groups)
                torch_feats = tag_features(adapted, labels, groups)
                stats_error = max(abs(numpy_feats[k]-float(torch_feats[k])) for k in numpy_feats)
                old_presence = np.array(presence(heads, numpy_feats | stems, (0., 0.)))
                new_presence = presence_logits(adapted, stems, heads, labels, groups).sigmoid().numpy()
                presence_error = float(np.abs(old_presence-new_presence).max())
            if tag_error > 1e-6 or stats_error > 2*np.finfo(np.float32).eps or presence_error > 1e-6:
                raise ValueError(f'Initial parity failed: tags={tag_error}, stats={stats_error}, presence={presence_error}')
            key = identity(dict(config=config['identity'], audio=audio['sha256'], starts=starts,
                                preprocessing='preserved baseline gain .6, soxr_hq', dtype='float32'))
            path = out/(hashlib.sha256(row['ID'].encode()).hexdigest()+'.npz')
            if path.exists():
                raise FileExistsError('Partial cache requires explicit inspection before replacement')
            np.savez(path, identity=key, block5=features.numpy(), tags=reference.numpy(), starts=np.array(starts))
            entry = dict(ID=row['ID'], role=row['role'], component_id=row['component_id'],
                path=str(path), sha256=sha256(path), identity=key, stems=stems,
                targets=[row[k] for k in TARGETS], segments=len(segments), seconds=len(raw)/16000,
                tag_error=tag_error, stats_error=stats_error, presence_error=presence_error,
                stem_seconds=stem_seconds, total_seconds=time.monotonic()-t)
            entries.append(entry)
            write_json(out/'partial_index.json', dict(config_identity=config['identity'], entries=entries))
            print(f'Cached {len(entries)}/{len(selected)}: {len(segments)} segments, {entry["total_seconds"]:.2f}s', flush=True)
    report = dict(config_identity=config['identity'], entries=entries, seconds=time.monotonic()-start,
                  stem_reuse=False, incomplete_types=['none/none'], quality_evaluation=False)
    write_json(index_path, report)
    index_path.with_suffix('.sha256').write_text(sha256(index_path)+'\n')
    return report


def model_for_arm(package, arm, seed, device):
    import torch
    from panns_adaptation import AdaptedCnn14
    if arm not in ('R0', 'R1', 'R2', 'R3', 'R5'):
        raise ValueError('Unapproved v1 arm')
    torch.manual_seed(seed)
    model = AdaptedCnn14(load_base(package), channel_se=arm in ('R2', 'R3'),
                        temporal_pool=arm=='R5', train_blocks=() if arm in ('R0', 'R2') else (6,))
    return model.to(device).train()


def frozen_digest(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def restore_checkpoint(path, expected_identity, model, optimizer):
    import torch
    record = torch.load(path, map_location='cpu', weights_only=True)
    if record['run_identity'] != expected_identity:
        raise ValueError('Checkpoint identity mismatch')
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(record['model']) != expected:
        raise ValueError('Checkpoint trainable parameter set mismatch')
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in expected:
                value = record['model'][name]
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite checkpoint')
                param.copy_(value)
    optimizer.load_state_dict(record['optimizer'])
    return record


def resource_usage(device):
    import resource
    import torch
    driver = torch.mps.driver_allocated_memory() if device == 'mps' else 0
    return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mps_driver_bytes=driver)


def optimization_step(model, optimizer, entries, heads, labels, groups, fraction, device, microbatch, capture_gradients=False):
    import torch
    from panns_adaptation import masked_file_loss, presence_logits
    optimizer.zero_grad(set_to_none=True)
    loss_sum, predictions = 0., []
    for entry in entries:
        features = torch.from_numpy(load_cache(entry)['block5'])
        tags = torch.cat([model.tags_from_block5(part.to(device)) for part in features.split(microbatch)])
        logits = presence_logits(tags, entry['stems'], heads, labels, groups)
        known = torch.tensor([[y is not None for y in entry['targets']]])
        targets = torch.tensor([[float('nan') if y is None else y for y in entry['targets']]], dtype=torch.float64)
        loss = masked_file_loss(logits[None], targets, known, fraction) / len(entries)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite file loss')
        loss.backward()
        loss_sum += float(loss.detach())
        predictions.append(logits.detach().sigmoid().tolist())
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise ValueError(f'Missing or nonfinite gradient: {name}')
            if capture_gradients:
                gradients[name] = param.grad.detach().cpu().clone()
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1.)
    optimizer.step()
    return dict(loss=loss_sum, probabilities=predictions, IDs=[r['ID'] for r in entries]), gradients


def pilot(config, arm, device, microbatch):
    import gc
    import random
    import torch
    package = validate_config(config)
    if arm not in config['arms'] or microbatch not in (1, 2, 4) or device not in ('cpu', 'mps'):
        raise ValueError('Pilot condition outside fixed plan')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(.5)
    cache_index = read_cache_index(config)
    entries = cache_index['entries']
    if len(entries) != len(config['panel_ids']) or any(e['role'] != 'train' for e in entries):
        raise ValueError('Engineering pilot requires exact train-only panel')
    training_rows = [r for r in (json.loads(line) for line in Path(config['rows']).open()) if r['role']=='train']
    fraction = [sum(r[k] is not None for r in training_rows)/len(training_rows) for k in TARGETS]
    out = Path(config['output'])/f'pilot_{arm}_{device}_mb{microbatch}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'pilot.json').exists():
        raise FileExistsError('Completed pilot must not be overwritten')
    run_identity = identity(dict(config=config['identity'], arm=arm, device=device,
                                 microbatch=microbatch, code=sha256(__file__), seed=41,
                                 cache_index=sha256(Path(config['output'])/'cache/index.json')))
    model = model_for_arm(package, arm, 41, device)
    frozen_before = frozen_digest(model)
    initial = trainable_state(model)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    sampler = random.Random(41)
    heads, labels, groups = readout_assets(package)
    initial_tag_error = 0.
    with torch.no_grad():
        for entry in entries:
            cached = load_cache(entry)
            tags = torch.cat([model.tags_from_block5(part.to(device)).cpu()
                              for part in torch.from_numpy(cached['block5']).split(microbatch)])
            initial_tag_error = max(initial_tag_error, float(np.abs(tags.numpy()-cached['tags']).max()))
    if initial_tag_error > (1e-6 if device == 'cpu' else 1e-5):
        raise ValueError(f'Adapted initial tags fail parity: {initial_tag_error}')
    records = []
    start = time.monotonic()
    steps = config['pilot_steps']
    if not 3 <= steps <= 200:
        raise ValueError('Pilot needs 3-200 steps')
    for step in range(steps):
        usage = resource_usage(device)
        if sum(usage.values()) > config['memory_limit_bytes'] or time.monotonic()-start > config['pilot_timeout_seconds']:
            raise RuntimeError('Pilot resource budget exceeded; no quality claim')
        selected = sampler.sample(entries, 4)
        t = time.monotonic()
        record, gradients = optimization_step(model, optimizer, selected, heads, labels, groups, fraction,
                    device, microbatch, capture_gradients=step==0)
        if device == 'mps':
            torch.mps.synchronize()
        record.update(step=step+1, seconds=time.monotonic()-t, **resource_usage(device))
        records.append(record)
        if step == 0:
            torch.save(dict(gradients=gradients, record=record), out/'first_gradient.pt')
        del gradients
        if step == 1:
            torch.save(dict(run_identity=run_identity, step=2, model=trainable_state(model),
                            optimizer=optimizer.state_dict(), sampler=sampler.getstate()), out/'resume_step2.pt')
        print(json.dumps(record), flush=True)
        write_json(out/'progress.json', records)
    final = trainable_state(model)
    if frozen_digest(model) != frozen_before:
        raise ValueError('Frozen parameter or BN buffer changed')
    changes = {name: float((final[name]-value).double().norm()) for name, value in initial.items()}
    if any(value <= 0 for value in changes.values()):
        raise ValueError(f'Intended parameters did not update: {changes}')
    del model, optimizer, initial
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    model = model_for_arm(package, arm, 41, device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    resumed = restore_checkpoint(out/'resume_step2.pt', run_identity, model, optimizer)
    sampler.setstate(resumed['sampler'])
    heads, labels, groups = readout_assets(package)
    resume_prob_error = 0.
    for step in range(resumed['step'], steps):
        record, gradients = optimization_step(model, optimizer, sampler.sample(entries, 4),
                         heads, labels, groups, fraction, device, microbatch)
        del gradients
        if record['IDs'] != records[step]['IDs']:
            raise ValueError('Resume sampler order mismatch')
        resume_prob_error = max(resume_prob_error, float(np.abs(np.array(record['probabilities'])-
                                                 np.array(records[step]['probabilities'])).max()))
    resumed_final = trainable_state(model)
    resume_weight_error = max(float((resumed_final[k]-v).abs().max()) for k, v in final.items())
    if resume_weight_error > (0 if device == 'cpu' else 1e-6) or resume_prob_error > 1e-6:
        raise ValueError('Resume changed continuation')
    if frozen_digest(model) != frozen_before:
        raise ValueError('Resume changed frozen state')
    report = dict(arm=arm, seed=41, device=device, microbatch=microbatch, steps=steps,
        config_identity=config['identity'], run_identity=run_identity, records=records,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_sha256=frozen_before, frozen_unchanged=True, initial_tag_max_abs=initial_tag_error, parameter_update_norms=changes,
        resume_weight_max_abs=resume_weight_error, resume_probability_max_abs=resume_prob_error,
        total_seconds=time.monotonic()-start, engineering_pilot_only=True,
        memory_limit_respected=all(r['peak_rss_bytes']+r['mps_driver_bytes'] <= config['memory_limit_bytes'] for r in records))
    if not report['memory_limit_respected']:
        raise RuntimeError('Pilot memory limit exceeded')
    torch.save(dict(run_identity=run_identity, arm=arm, seed=41, model=final,
                    config_identity=config['identity']), out/'pilot_checkpoint.pt')
    write_json(out/'pilot.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','parameter_update_norms')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'cache', 'pilot'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=['R1', 'R2', 'R3', 'R5'], default='R3')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--microbatch', type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    config = json.loads(args.run.read_text())
    if args.command == 'pilot':
        pilot(config, args.arm, args.device, args.microbatch)
    else:
        {'audit': audit, 'cache': cache}[args.command](config)


if __name__ == '__main__':
    main()
