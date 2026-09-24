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
            key = identity(dict(config=config['identity'], provenance=cache_provenance(), audio=audio['sha256'], starts=starts,
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
    report = dict(config_identity=config['identity'], provenance=cache_provenance(), entries=entries, seconds=time.monotonic()-start,
                  stem_reuse=False, incomplete_types=[f'{v}/{m}' for v, m in ((0,0),(0,1),(1,0),(1,1))
                      if not any(r[TARGETS[0]]==v and r[TARGETS[1]]==m for r in selected)], quality_evaluation=False)
    write_json(index_path, report)
    index_path.with_suffix('.sha256').write_text(sha256(index_path)+'\n')
    return report


