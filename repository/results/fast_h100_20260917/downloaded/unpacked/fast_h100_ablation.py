"""Bounded frozen-MERT330 ablations; select on validation, then evaluate once."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchaudio", "transformers", "librosa", "soundfile", "scikit-learn"]
# ///
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

from graph_scoring import official_metrics
from music_temporal import fit_temporal_head, predict_temporal_music, load_temporal_head
from temporal_backbone import REGULARIZED_C08
from v6_data import assert_disjoint, mixtures, sha256, write_json
from v6_runtime import baseline_module


ROOT = Path('/root/fast_h100_20260917')
OUT = ROOT / 'results'
DEADLINE = float(os.environ.get('FAST_DEADLINE', '0'))


def progress(stage: str, **values: object) -> None:
    if time.time() >= DEADLINE:
        raise TimeoutError('Authorized GPU deadline reached')
    state = dict(stage=stage, utc=time.time(), **values)
    write_json(OUT / 'progress.json', state)
    print(json.dumps(state), flush=True)


def prepare() -> dict:
    """Recover hash-checked audio and create mixtures only inside each partition."""
    split = json.loads((ROOT / 'split.json').read_text())
    mapping = json.loads((ROOT / 'existing_audio_map.json').read_text())
    rows = [r for part in split.values() for r in part]
    audio = ROOT / 'audio'
    audio.mkdir(exist_ok=True)
    archives = {}
    try:
        for row in rows:
            target = audio / (row['ID'] + '.wav')
            if not target.exists():
                archive, member = mapping[row['ID']]
                if archive not in archives:
                    archives[archive] = zipfile.ZipFile(Path('/root/fullft8/inputs') / archive)
                target.write_bytes(archives[archive].read(member))
            if sha256(target) != row['audio_sha256']:
                raise ValueError(f'Audio hash mismatch: {row["ID"]}')
            row['path'] = str(target)
        for index, (part, items) in enumerate(split.items()):
            generated = mixtures(items, audio / part, 710 + index,
                                 count=320 if part == 'train' else 40)
            for row in generated:
                row['source'] = 'mixture/fresh'
                row['music_training'] = '0'
                if row['MUSIC_FAKE'] == '0' and row['VOICE_FAKE'] == '':
                    row['VOICE_FAKE'] = '0'
            items.extend(generated)
        for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            assert_disjoint(split[left], split[right])
        write_json(OUT / 'split.json', split)
    finally:
        for archive in archives.values():
            archive.close()
    return split


def prepare_assets() -> Path:
    """Reuse existing persistent model weights; never download an unpinned model."""
    assets = ROOT / 'assets'
    base_model = assets / 'baseline/model'
    base_model.mkdir(parents=True, exist_ok=True)
    for archive_path in Path('/root/fullft8/inputs').rglob('*.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                for family in ('df_arena_1b', 'panns', 'htdemucs'):
                    token = family + '/'
                    if token not in name or name.endswith('/'):
                        continue
                    relative = token + name.split(token, 1)[1]
                    target = base_model / relative
                    if not target.resolve().is_relative_to(base_model.resolve()):
                        raise ValueError('Unsafe model archive member')
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name) as source, target.open('wb') as output:
                            shutil.copyfileobj(source, output)
    mert = assets / 'mert'
    if not mert.exists():
        mert.symlink_to('/root/fullft8/assets/models/mert', target_is_directory=True)
    metadata = json.loads((ROOT / 'mert330_metadata.json').read_text())
    if sha256(mert / 'pytorch_model.bin') != metadata['weight_sha256']:
        raise ValueError('Persistent MERT weights differ from submitted M330')
    manifest = json.loads((ROOT / 'baseline_asset_hashes.json').read_text())
    for name, digest in manifest.items():
        if sha256(assets / 'baseline' / name) != digest:
            raise ValueError(f'Baseline model differs from submitted M330: {name}')
    return assets


def extract(split: dict, assets: Path) -> tuple[dict, dict, dict]:
    """Batch shared frozen features; keep full-file baseline voice/presence semantics."""
    base = baseline_module(assets)
    device = torch.device('cuda')
    demucs = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    processor = AutoFeatureExtractor.from_pretrained(assets / 'mert', local_files_only=True)
    mert, info = AutoModel.from_pretrained(assets / 'mert', trust_remote_code=True,
                                          local_files_only=True, output_loading_info=True)
    if any(info.values()) or mert.config.hidden_size != 1024:
        raise ValueError(f'MERT330 incomplete load: {info}')
    mert.to(device).eval().requires_grad_(False)
    df, fake_index = base.load_df_arena_model(device)
    df.requires_grad_(False)
    panns, voice_indices, music_indices = base.load_panns_model(device)
    rows = [r for part in split.values() for r in part]
    stem_records, raw_records, evidence = {}, {}, {}
    cache = ROOT / 'feature_cache'
    cache.mkdir(exist_ok=True)
    protocol = dict(extractor_sha256=sha256(Path(__file__)),
                    model_sha256=json.loads((ROOT / 'mert330_metadata.json').read_text())['weight_sha256'],
                    dtype='float32', window=64600, views=['native_demucs_music', 'raw'],
                    torch=torch.__version__, gpu=torch.cuda.get_device_name())
    cache_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(OUT / 'extraction_protocol.json', dict(cache_id=cache_id, **protocol))
    audit = []
    began = time.time()
    for index, row in enumerate(rows):
        identifier = row['ID']
        progress('extracting', completed=index, total=len(rows), elapsed=time.time() - began,
                 estimated_remaining=(time.time() - began) / max(index, 1) * (len(rows) - index))
        target = cache / (identifier + '.npz')
        identity = row['audio_sha256'] + ':' + cache_id
        if target.exists():
            with np.load(target) as archive:
                if archive['identity'].item() != identity:
                    raise ValueError(f'Stale cache: {identifier}')
                values = {k: archive[k] for k in archive.files if k != 'identity'}
        else:
            path = Path(row['path'])
            audio = base.load_audio(path)
            with torch.inference_mode():
                vocal, backing = base.separate_voice_and_music(path, demucs, device)
                backing = np.pad(backing[:len(audio)], (0, max(0, len(audio) - len(backing))))
                starts = np.asarray(base.get_segment_starts(len(audio)))
                vectors = {}
                masks = {}
                for view, waveform in [('stem', backing), ('raw', audio)]:
                    vectors[view], masks[view] = [], []
                    for offset in range(0, len(starts), 16):
                        pieces = [base.extract_segment(waveform, int(s)) for s in starts[offset:offset + 16]]
                        masks[view].extend(float(base.calculate_rms(p) >= base.SILENCE_RMS) for p in pieces)
                        inputs = processor([librosa.resample(p, orig_sr=16000, target_sr=24000,
                                            res_type='soxr_hq') for p in pieces],
                                           sampling_rate=24000, return_tensors='pt')
                        vectors[view].extend(mert(inputs.input_values.to(device)).last_hidden_state.mean(1).cpu().numpy())
                voice = base.predict_fake(df, fake_index, vocal, device)
                vp, mp = base.predict_presence(panns, voice_indices, music_indices, audio)
                values = dict(stem=np.asarray(vectors['stem']), raw=np.asarray(vectors['raw']),
                              stem_audible=np.asarray(masks['stem']), raw_audible=np.asarray(masks['raw']),
                              starts=starts, evidence=np.asarray([voice, vp, mp]))
                # Same recordings and weights isolate the changed S2 separator input path.
                if len(audit) < 24 and row['music_training'] == '1':
                    from s2_eat_submission import separate
                    s2_voice, s2_music = separate(audio, demucs, device)
                    s2_p = base.predict_fake(df, fake_index, s2_voice, device)
                    audio_info = sf.info(path)
                    audit.append(dict(ID=identifier, rate=audio_info.samplerate, channels=audio_info.channels,
                                      original_voice=voice, s2_voice=s2_p, delta=s2_p - voice))
                    write_json(OUT / 'preprocessing_audit.json', audit)
            if any(not np.isfinite(x).all() for x in values.values()):
                raise FloatingPointError(f'Nonfinite encoder output: {identifier}')
            with target.with_suffix('.partial').open('wb') as stream:
                np.savez_compressed(stream, identity=np.array(identity), **values)
            target.with_suffix('.partial').replace(target)
        stem_records[identifier] = dict(music=values['stem'], music_audible=values['stem_audible'], starts=values['starts'])
        raw_records[identifier] = dict(music=values['raw'], music_audible=values['raw_audible'], starts=values['starts'])
        evidence[identifier] = values['evidence']
    write_json(OUT / 'feature_manifest.json', {
        r['ID']: dict(audio_sha256=r['audio_sha256'], sha256=sha256(cache / (r['ID'] + '.npz')))
        for r in rows})
    del mert, df, panns, demucs
    gc.collect()
    torch.cuda.empty_cache()
    return stem_records, raw_records, evidence


def predictions(rows: list, probability: np.ndarray, evidence: dict,
                file_probability: np.ndarray | None = None) -> list:
    result = []
    for i, (row, music) in enumerate(zip(rows, probability, strict=True)):
        voice, vp, mp = map(float, evidence[row['ID']])
        result.append(dict(ID=row['ID'], FILE_FAKE_PROB=float(file_probability[i]) if file_probability is not None else max(vp * voice, mp * music),
                           VOICE_FAKE_PROB=voice, MUSIC_FAKE_PROB=float(music),
                           VOICE_PRESENT_PROB=vp, MUSIC_PRESENT_PROB=mp))
    return result


def file_head(split: dict, stem: dict, raw: dict, evidence: dict, seed: int) -> tuple[dict, dict]:
    """Train the direct File classifier without using music-head in-sample predictions."""
    torch.manual_seed(seed)
    features, targets = {}, {}
    for part, rows in split.items():
        features[part] = torch.tensor(np.stack([
            np.r_[stem[r['ID']]['music'].mean(0), raw[r['ID']]['music'].mean(0), evidence[r['ID']]]
            for r in rows]), dtype=torch.float32, device='cuda')
        if part != 'test':
            targets[part] = torch.tensor([int(r['FILE_FAKE']) for r in rows], device='cuda')
    model = nn.Sequential(nn.LayerNorm(2051), nn.Linear(2051, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 2)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    weights = {part: len(y) / (2 * torch.bincount(y, minlength=2).float()) for part, y in targets.items()}
    best, state, stale, history = float('inf'), None, 0, []
    for epoch in range(1, 301):
        progress('file_head', seed=seed, epoch=epoch)
        model.train()
        for indices in torch.randperm(len(targets['train']), device='cuda').split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features['train'][indices]), targets['train'][indices], weight=weights['train'])
            loss.backward()
            gradient = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            if not np.isfinite(gradient) or gradient <= 0:
                raise RuntimeError('Direct File head is disconnected from its loss')
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(nn.functional.cross_entropy(model(features['validation']), targets['validation'], weight=weights['validation']))
        history.append(validation_loss)
        if validation_loss < best - 1e-4:
            best, stale = validation_loss, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 20:
            break
    model.load_state_dict(state)
    model.eval()
    torch.save(state, OUT / f'file_seed{seed}.pt')
    with torch.inference_mode():
        result = {part: model(x).softmax(-1)[:, 1].cpu().numpy() for part, x in features.items()}
        ablated = features['validation'].clone()
        ablated[:, 1024:2048] = 0
        sensitivity = float((model(ablated).softmax(-1)[:, 1] - torch.tensor(result['validation'], device='cuda')).abs().mean())
    return result, dict(seed=seed, validation_losses=history, selected_loss=best,
                        final_gradient_l1=gradient, raw_branch_ablation_probability_delta=sensitivity)


def train(split: dict, stem: dict, raw: dict, evidence: dict) -> None:
    """Predeclared 2x2 ablation, three seeds, validation-only ensemble weight."""
    torch.use_deterministic_algorithms(True)
    music_train = [r for r in split['train'] if r['music_training'] == '1']
    music_valid = [r for r in split['validation'] if r['music_training'] == '1']
    scores, traces = {}, []
    for view, records in [('stem', stem), ('raw', raw)]:
        for seed in (41, 42, 43):
            name = f'{view}_seed{seed}'
            progress('music_head', candidate=name)
            eligible = [r for r in music_train if np.any(records[r['ID']]['music_audible'])]
            write_json(OUT / name / 'excluded_silent_training.json', [r['ID'] for r in music_train if r not in eligible])
            model, trace = fit_temporal_head(eligible, records, (music_valid, records), OUT / name, DEADLINE,
                recipe=replace(REGULARIZED_C08, input_dim=1024, nsynth_mass=.05, seed=seed), device='cuda')
            loaded = load_temporal_head(OUT / name / 'temporal_music.pt', 'cuda')
            np.testing.assert_allclose(predict_temporal_music(model, music_valid, records, 'cuda'),
                                       predict_temporal_music(loaded, music_valid, records, 'cuda'), atol=1e-6, rtol=0)
            scores[name] = {part: predict_temporal_music(model, rows, records, 'cuda') for part, rows in split.items()}
            traces.append(dict(name=name, selected_epoch=trace['selected_epoch'], parameters=trace['parameters']))
    means = {view: {part: np.mean([scores[f'{view}_seed{s}'][part] for s in (41, 42, 43)], axis=0)
                    for part in split} for view in ('stem', 'raw')}
    choices = []
    for weight in (0., .25, .5, .75, 1.):
        p = (1 - weight) * means['stem']['validation'] + weight * means['raw']['validation']
        report = official_metrics(split['validation'], predictions(split['validation'], p, evidence))
        choices.append(dict(raw_weight=weight, music_eer=report['metrics']['music_eer']))
    weight = min(choices, key=lambda item: (item['music_eer'], item['raw_weight']))['raw_weight']
    direct, file_traces = [], []
    for seed in (41, 42, 43):
        result, trace = file_head(split, stem, raw, evidence, seed)
        direct.append(result)
        file_traces.append(trace)
    write_json(OUT / 'selection.json', dict(raw_weight=weight, candidates=choices, music_heads=traces,
        file_heads=file_traces, selection_partition='validation', test_used_for_selection=False,
        limitation='Historically reused source pool; fresh heads only; not unseen leaderboard data', frozen_at=time.time()))
    summaries = {}
    for part in ('validation', 'test'):
        rows = split[part]
        blend = (1 - weight) * means['stem'][part] + weight * means['raw'][part]
        file_p = np.mean([r[part] for r in direct], axis=0)
        variants = {'M330_stem_max': predictions(rows, means['stem'][part], evidence),
                    'M330_raw_max': predictions(rows, means['raw'][part], evidence),
                    'M330_dual_max': predictions(rows, blend, evidence),
                    'M330_stem_direct': predictions(rows, means['stem'][part], evidence, file_p),
                    'M330_dual_direct': predictions(rows, blend, evidence, file_p)}
        summaries[part] = {}
        for name, prediction in variants.items():
            write_json(OUT / f'{part}_{name}_predictions.json', prediction)
            groups = {'all': list(range(len(rows))),
                      'natural': [i for i, r in enumerate(rows) if r['source'] != 'mixture/fresh'],
                      'mixed': [i for i, r in enumerate(rows) if r['source'] == 'mixture/fresh'],
                      'music_without_nsynth': [i for i, r in enumerate(rows) if r['music_training'] == '1' and r['source'] != 'nsynth']}
            summaries[part][name] = {group: official_metrics([rows[i] for i in indices], [prediction[i] for i in indices])
                                     for group, indices in groups.items()}
        np.savez_compressed(OUT / f'{part}_seed_probabilities.npz', **{name: value[part] for name, value in scores.items()})
    write_json(OUT / 'metrics.json', summaries)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'metrics.json').exists():
        raise FileExistsError('Completed experiment exists; refusing to retrain')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if 'H100' not in torch.cuda.get_device_name():
        raise RuntimeError('This experiment requires the authorized H100')
    progress('preparing')
    split = prepare()
    assets = prepare_assets()
    records = extract(split, assets)
    train(split, *records)
    progress('complete')


if __name__ == '__main__':
    main()
