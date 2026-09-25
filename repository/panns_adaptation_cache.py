"""Pinned reference loading and CPU prefix/stem extraction."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from panns_adaptation_data import (TARGETS, cache_provenance, identity, load_cache,
    read_cache_index, sha256, validate_config, write_json)


def load_base(package):
    import torch
    label_file = Path.home()/'panns_data/class_labels_indices.csv'
    if not label_file.exists() or sha256(label_file) != sha256(package/'model/assets/baseline/model/panns/class_labels_indices.csv'):
        raise ValueError('Offline PANNs label table missing or mismatched')
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


def cache_key(config, row, starts):
    return identity(dict(config=config['identity'], provenance=cache_provenance(),
        audio=row['verified_audio']['sha256'], starts=starts,
        preprocessing='preserved baseline gain .6, soxr_hq', dtype='float32'))


def validate_cache_numerics(arrays, stems, errors):
    if (any(not np.isfinite(array).all() for array in arrays)
            or not np.isfinite(list(stems.values())).all() or not np.isfinite(errors).all()):
        raise ValueError('Nonfinite cache activations, predictions, stems or parity errors')


def validate_cache_entry(config, row, entry):
    if (entry['ID'] != row['ID'] or entry['role'] != row['role']
            or entry['component_id'] != row['component_id']
            or entry['targets'] != [row[k] for k in TARGETS]):
        raise ValueError('Cache row identity, role or target mismatch')
    values = load_cache(entry)
    validate_cache_numerics((values['block5'], values['tags']), entry['stems'],
        [entry[key] for key in ('tag_error', 'stats_error', 'presence_error')])
    if entry['identity'] != cache_key(config, row, values['starts'].tolist()):
        raise ValueError('Incompatible cache key for current audio and runtime')
    return entry


def validate_partial_entry(config, row, saved):
    if saved.get('provenance') != cache_provenance():
        raise ValueError('Partial cache preprocessing/runtime provenance mismatch')
    entry = saved['entry']
    if saved['config_identity'] != config['identity'] or saved['entry_sha256'] != identity(entry):
        raise ValueError('Partial cache identity mismatch')
    return validate_cache_entry(config, row, entry)


def read_stem_cache(config, rows):
    if 'stem_cache' not in config:
        return {}
    path = Path(config['stem_cache'])
    if config['inputs'].get(str(path)) != sha256(path):
        raise ValueError('Stem cache must be pinned in config inputs')
    saved = json.loads(path.read_text())
    row_identity = identity(sorted([[r['ID'], r['verified_audio']['sha256']] for r in rows.values()]))
    if saved['version'] != 1 or saved['rows_audio_identity'] != row_identity:
        raise ValueError('Stem cache row/audio identity mismatch')
    assets = saved['source_receipt']['assets']
    reference = json.loads(Path(config['reference']).read_text())['members']
    if not assets or any(reference.get(name) != digest for name, digest in assets.items()):
        raise ValueError('Stem cache reference assets mismatch')
    from decision_heads import stem_features
    unit = np.ones(1, np.float32)
    expected = stem_features(unit, unit, 16000).keys()
    for rid, entry in saved['entries'].items():
        if rid not in rows or entry['audio_sha256'] != rows[rid]['verified_audio']['sha256']:
            raise ValueError('Stem cache entry audio identity mismatch')
        if (entry['stems'].keys() != expected or any(type(v) not in (int, float) or not np.isfinite(v)
                for v in entry['stems'].values())):
            raise ValueError('Stem cache requires the exact finite scalar feature set')
        digest = entry['source_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Stem cache source digest is invalid')
    return saved['entries']


def cache(config, resume=False):
    import tempfile
    import torch
    from panns_adaptation import AdaptedCnn14, presence_logits, tag_features
    from decision_heads import stem_features, tagger_features, presence
    from mert12_submission import pin_torchaudio_loader
    package = validate_config(config)
    torch.set_num_threads(4)
    device = config.get('cache_device', 'cpu')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested cache device unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = Path(config['output'])/'cache'
    out.mkdir(parents=True, exist_ok=True)
    index_path = out/'index.json'
    rows = {r['ID']: r for r in (json.loads(s) for s in Path(config['rows']).open())}
    selected = [rows[rid] for rid in config['panel_ids']]
    stem_cache = read_stem_cache(config, rows)
    if index_path.exists():
        marker_path = index_path.with_suffix('.sha256')
        marker = marker_path.read_bytes().strip() if marker_path.exists() else b''
        if len(marker) == 64 and all(c in b'0123456789abcdef' for c in marker):
            saved = read_cache_index(config)
            for entry in saved['entries']:
                validate_cache_entry(config, rows[entry['ID']], entry)
            return saved
        if not resume:
            raise FileExistsError('Unpublished cache index exists; use --resume')
    base = demucs = None
    entries = []
    start = time.monotonic()
    with Path(config['archive']).open('rb') as archive:
        for row in selected:
            entry_path = out/(hashlib.sha256(row['ID'].encode()).hexdigest()+'.json')
            if entry_path.exists():
                if not resume:
                    raise FileExistsError('Partial cache exists; use --resume')
                entry = validate_partial_entry(config, row, json.loads(entry_path.read_text()))
                entries.append(entry)
                continue
            if base is None:
                base = load_base(package).to(device)
                model = AdaptedCnn14(base, train_blocks=())
                original = baseline(package)
                pin_torchaudio_loader(original)
                heads, labels, groups = readout_assets(package)
            t = time.monotonic()
            audio = row['verified_audio']
            if 'local_path' in audio:
                source_path = Path(audio['local_path']).resolve()
                if not source_path.is_relative_to(Path(config['recovered_audio_root']).resolve()):
                    raise ValueError('Audio outside admitted recovery namespace')
                payload = source_path.read_bytes()
            else:
                archive.seek(audio['tar_offset'])
                payload = archive.read(audio['bytes'])
            if hashlib.sha256(payload).hexdigest() != audio['sha256']:
                raise ValueError('Selected audio hash mismatch')
            with tempfile.TemporaryDirectory(prefix='panns_decode_') as temporary:
                source = Path(temporary)/('input'+Path(row['path']).suffix)
                source.write_bytes(payload)
                raw = original.load_audio(source)
                peak = float(np.abs(raw).max())
                if peak > .6:
                    raw = (raw * (.6/peak)).astype(np.float32)
                segments = original.make_panns_segments(raw)
                starts = original.get_segment_starts(len(raw))
                stem_start = time.monotonic()
                reused = stem_cache.get(row['ID'])
                if reused is not None:
                    stems = reused['stems']
                else:
                    if demucs is None:
                        demucs = load_demucs(package).to(device)
                    normalized = Path(temporary)/'normalized.wav'
                    sf.write(normalized, raw, 16000, subtype='FLOAT')
                    with torch.inference_mode():
                        voice, _ = original.separate_voice_and_music(normalized, demucs, torch.device(device))
                    voice = np.pad(voice[:len(raw)], (0, max(0, len(raw)-len(voice))))
                    stems = stem_features(raw, voice, 16000)
                stem_seconds = time.monotonic()-stem_start
            features, reference, adapted = [], [], []
            with torch.no_grad():
                for segment in segments:
                    waveform = torch.from_numpy(segment[None]).to(device)
                    features.append(model.prefix(waveform).cpu())
                    reference.append(base(waveform)['clipwise_output'].cpu())
                    adapted.append(model.tags_from_block5(features[-1].to(device)).cpu())
                features = torch.cat(features)
                reference, adapted = torch.cat(reference), torch.cat(adapted)
                tag_error = float((reference-adapted).abs().max())
                numpy_feats = tagger_features(reference.numpy(), labels, groups)
                torch_feats = tag_features(adapted, labels, groups)
                stats_error = float(np.max([abs(numpy_feats[k]-float(torch_feats[k])) for k in numpy_feats]))
                old_presence = np.array(presence(heads, numpy_feats | stems, (0., 0.)))
                new_presence = presence_logits(adapted, stems, heads, labels, groups).sigmoid().numpy()
                presence_error = float(np.abs(old_presence-new_presence).max())
            validate_cache_numerics((features.numpy(), reference.numpy(), adapted.numpy(), old_presence, new_presence),
                                   stems, [tag_error, stats_error, presence_error])
            if tag_error > (1e-6 if device == 'cpu' else 1e-5) or stats_error > 2*np.finfo(np.float32).eps or presence_error > (1e-6 if device == 'cpu' else 1e-5):
                raise ValueError(f'Initial parity failed: tags={tag_error}, stats={stats_error}, presence={presence_error}')
            key = cache_key(config, row, starts)
            path = out/(hashlib.sha256(row['ID'].encode()).hexdigest()+'.npz')
            if path.exists():
                if not resume:
                    raise FileExistsError('Orphan cache exists; use --resume to preserve and recompute')
                path.replace(path.with_suffix('.orphan.'+str(time.time_ns())+'.npz'))
            with path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez(stream, identity=key, block5=features.numpy(), tags=reference.numpy(), starts=np.array(starts))
            path.with_suffix('.npz.tmp').replace(path)
            entry = dict(ID=row['ID'], role=row['role'], component_id=row['component_id'],
                path=str(path), sha256=sha256(path), identity=key, stems=stems,
                targets=[row[k] for k in TARGETS], segments=len(segments), seconds=len(raw)/16000,
                tag_error=tag_error, stats_error=stats_error, presence_error=presence_error,
                stem_origin='legacy_cache' if reused is not None else 'demucs',
                stem_source_sha256=reused['source_sha256'] if reused is not None else None,
                stem_seconds=stem_seconds, total_seconds=time.monotonic()-t)
            entries.append(entry)
            write_json(entry_path, dict(config_identity=config['identity'], provenance=cache_provenance(),
                                       entry_sha256=identity(entry), entry=entry))
            print(f'Cached {len(entries)}/{len(selected)}: {len(segments)} segments, {entry["total_seconds"]:.2f}s', flush=True)
    reused_count = sum(e.get('stem_origin') == 'legacy_cache' for e in entries)
    report = dict(config_identity=config['identity'], provenance=cache_provenance(), entries=entries, seconds=time.monotonic()-start,
                  stem_reuse=bool(reused_count), stem_reused_files=reused_count, stem_computed_files=len(entries)-reused_count,
                  incomplete_types=[f'{v}/{m}' for v, m in ((0,0),(0,1),(1,0),(1,1))
                      if not any(r[TARGETS[0]]==v and r[TARGETS[1]]==m for r in selected)], quality_evaluation=False)
    write_json(index_path, report)
    marker_path = index_path.with_suffix('.sha256')
    temporary_marker = marker_path.with_suffix('.sha256.tmp')
    temporary_marker.write_text(sha256(index_path)+'\n')
    temporary_marker.replace(marker_path)
    return report
