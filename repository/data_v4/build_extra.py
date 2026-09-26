"""S2 extra rows: AI-Hub real domains (processed only on this machine, AI-Hub terms clause 3), Jamendo real songs,
Music Arena AI songs and AI-Hub-channel phone fakes/reals. Labels, crop/gain/codec and vocal evidence reuse
build_manifest; bank rows reuse export_bank, so extra rows are indistinguishable in format from data v4.

  python data_v4/build_extra.py collect  --out OUT      # items.jsonl (+ staged channel/pcm audio), split per item
  python data_v4/build_extra.py materialize --out OUT   # label-independent crop/gain/codec (build_manifest)
  python data_v4/build_extra.py vocals --out OUT        # HTDemucs vocal evidence on the final crop (songs only)
  python data_v4/build_extra.py export --out OUT        # bank_extra_train.csv / bank_extra_dev.csv
Dev rows (split 'val' in the CSV) are never trained on: unseen AI-Hub sessions/calls, KtelSpeech (a second phone
source), held-out songs/pieces, the Sonauto family of Music Arena and two MLAAD-ko systems through held-out channels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_manifest import lab, item, materialize_one, vocal_ratio, apply_vocal_evidence, SR  # noqa: E402
from export_bank import export, COLUMNS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / '_workspace'
AIHUB = W / 'aihub'
SEED = 20260927
HELDOUT_CALLS = W / 'aihub_probe/heldout.csv'          # probe held-out calls stay dev-only
DEV_ARENA = ('sonauto',)                                 # whole generator family held out
DEV_MLAAD = ('Higgs-Audio-V2', 'minimax_speech-02-turbo')
AIHUB_LICENSE = 'AI-Hub (NIA) {} - training use only, no export abroad'


def seconds(path: str) -> float:
    import soundfile as sf
    try:
        return sf.info(path).duration
    except Exception:
        return 0.


def pick_sessions(paths: list[str], per: int, rng: random.Random, min_s=4.0) -> list[str]:
    """<= per utterances of >= min_s per session directory (sessions = calls/conversations)."""
    by: dict[str, list[str]] = {}
    for p in paths:
        by.setdefault(str(Path(p).parent), []).append(p)
    out = []
    for s in sorted(by):
        ok = [p for p in sorted(by[s]) if seconds(p) >= min_s]
        rng.shuffle(ok)
        out += ok[:per]
    return out


def split_by(keys: list[str], dev_share: float, salt: str) -> dict[str, str]:
    return {k: 'val' if int(hashlib.sha256((salt + k).encode()).hexdigest()[:8], 16) / 16**8 < dev_share else 'train'
            for k in keys}


# ---------------------------------------------------------------- AI-Hub-matched telephone channel
def _stft(y):
    from scipy.signal import stft
    return stft(y, SR, nperseg=512, noverlap=384)[2]


def _istft(Y, n):
    from scipy.signal import istft
    y = istft(Y, SR, nperseg=512, noverlap=384)[1]
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def channel_profile(wavs: list[str]) -> dict:
    """Long-term spectrum of speech frames and the spectrum/level of the quietest frames of one real call."""
    import librosa
    power = np.concatenate([np.abs(_stft(librosa.load(w, sr=SR)[0])) ** 2 for w in wavs], 1)
    energy = power.sum(0)
    speech, quiet = power[:, energy >= np.median(energy)], power[:, energy <= np.quantile(energy, .1)]
    return {'ltas': 10 * np.log10(speech.mean(1) + 1e-12), 'noise': quiet.mean(1),
            'snr': float(speech.sum(0).mean() / (quiet.sum(0).mean() + 1e-12))}


def transplant(y: np.ndarray, prof: dict, rng: random.Random) -> np.ndarray:
    """EQ the source to the call's speech spectrum, band-limit through 8 kHz, add the call's own noise colour/level.
    ponytail: static per-call EQ + stationary noise; no AGC/packet loss modelling."""
    import librosa
    Y = _stft(y)
    p = np.abs(Y) ** 2
    src = 10 * np.log10(p[:, p.sum(0) >= np.median(p.sum(0))].mean(1) + 1e-12)
    eq = np.clip(np.convolve(prof['ltas'] - src, np.ones(9) / 9, 'same'), -40, 15)
    out = _istft(Y * 10 ** (eq / 20)[:, None], len(y))
    out = librosa.resample(librosa.resample(out, orig_sr=SR, target_sr=8000, res_type='soxr_hq'), orig_sr=8000, target_sr=SR,
                           res_type='soxr_hq')[:len(y)]
    phase = np.exp(2j * np.pi * np.random.default_rng(rng.randrange(2**32)).random((len(prof['noise']), Y.shape[1])))
    noise = _istft(np.sqrt(prof['noise'])[:, None] * phase, len(out))
    level = np.sqrt(np.mean(out ** 2) / (prof['snr'] * np.mean(noise ** 2) + 1e-12))
    return (out + noise * level).astype(np.float32)


def calls_571() -> dict[str, list[str]]:
    wav = {'/'.join(Path(w).parts[-4:]): w for w in glob.glob(str(AIHUB / '571/**/*.wav'), recursive=True)}
    calls = {}
    for f in sorted(glob.glob(str(AIHUB / '571/**/*.json'), recursive=True)):
        d = json.load(open(f))['dataSet']
        paths = [wav[k] for x in d['dialogs'] if (k := '/'.join(Path(x['audioPath']).parts[-4:])) in wav]
        if paths:
            calls['id:aihub571:' + '/'.join(Path(paths[0]).parts[-4:-1])] = paths
    return calls


# ---------------------------------------------------------------- adapters
def collect(out: Path) -> list[dict]:
    import librosa
    import soundfile as sf
    rng = random.Random(SEED)
    stage = out / 'staged'
    stage.mkdir(parents=True, exist_ok=True)
    items = []

    def add(source, rid, path, labels, split, groups, lic, **extra):
        items.append({**item(source, rid, path, labels, 'source_metadata', lic, groups, **extra), 'split': split})

    speech_real = lab(1, 0, 0, None)
    # 571 real phone calls: probe held-out calls remain the phone dev set (already featurized by the probe)
    held = {r['component_id'] for r in csv.DictReader(open(HELDOUT_CALLS))}
    calls = calls_571()
    train_calls = [c for c in calls if c not in held]
    rng.shuffle(train_calls)
    for p in pick_sessions([p for c in train_calls[:900] for p in calls[c]], 2, rng)[:1500]:
        add('aihub_571', Path(p).relative_to(AIHUB).as_posix(), p, speech_real, 'train',
            ['id:aihub571:' + '/'.join(Path(p).parts[-4:-1])], AIHUB_LICENSE.format(571), language='ko')
    # 100 KtelSpeech (second phone source): dev only; 87 KrespSpeech: train/dev by session
    for key, per, cap, dev_only in (('100', 1, 300, True), ('87', 2, 950, False)):
        paths = pick_sessions(sorted(glob.glob(str(AIHUB / key / '**/*.wav'), recursive=True)), per, rng)
        rng.shuffle(paths)
        sessions = split_by(sorted({str(Path(p).parent) for p in paths}), .16, key)
        for p in paths[:cap]:
            add(f'aihub_{key}', Path(p).relative_to(AIHUB).as_posix(), p, speech_real,
                'val' if dev_only else sessions[str(Path(p).parent)], [f'id:aihub{key}:' + str(Path(p).parent.relative_to(AIHUB))],
                AIHUB_LICENSE.format(key), language='ko')
    # 123 KsponSpeech eval: raw 16 kHz PCM -> staged wav, one utterance per component
    pcms = sorted(glob.glob(str(AIHUB / '123/**/*.pcm'), recursive=True))
    rng.shuffle(pcms)
    n = 0
    for p in pcms:
        a = np.fromfile(p, dtype='<i2')
        if len(a) < 4 * SR:
            continue
        dst = stage / 'kspon' / (Path(p).stem + '.wav')
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, a.astype(np.float32) / 32768, SR)
        add('aihub_123', Path(p).stem, dst, speech_real, 'val' if n < 150 else 'train', [f'id:aihub123:{Path(p).stem}'],
            AIHUB_LICENSE.format(123), language='ko')
        n += 1
        if n >= 950:
            break
    # 465 real singing (voice route, music presence unknown), 71470 gugak and 71767/644 instrument real music
    for key, labels, cap, dev_share, lic_key in (('465', lab(1, None, 0, 0), 600, .17, 465),
                                                 ('71470', lab(None, 1, 0, 0), 500, .2, 71470),
                                                 ('71767', lab(0, 1, 0, 0), 380, .2, 71767)):
        files = sorted(f for e in ('wav', 'mp3', 'flac') for f in glob.glob(str(AIHUB / key / f'**/*.{e}'), recursive=True)
                       if seconds(f) >= 4.0)
        rng.shuffle(files)
        groups = split_by(sorted({str(Path(f).parent) for f in files[:cap]}), dev_share, key)
        for f in files[:cap]:
            add(f'aihub_{key}', Path(f).relative_to(AIHUB).as_posix(), f, labels, groups[str(Path(f).parent)],
                [f'id:aihub{key}:' + str(Path(f).relative_to(AIHUB))], AIHUB_LICENSE.format(lic_key),
                vocal_evidence=key == '71470')
    inst = sorted(f for e in ('wav', 'mp3') for f in glob.glob(str(AIHUB / f'644/**/*.{e}'), recursive=True)
                  if '악기' in f and seconds(f) >= 4.0)
    rng.shuffle(inst)
    for i, f in enumerate(inst[:200]):
        add('aihub_644', Path(f).relative_to(AIHUB).as_posix(), f, lab(0, 1, 0, 0), 'val' if i < 50 else 'train',
            [f'id:aihub644:{Path(f).stem}'], AIHUB_LICENSE.format(644))
    # Jamendo real songs (CC BY / BY-SA): one track per artist, vocal presence from evidence
    jam = W / 'lb_collect_20260924/payload/jamendo-v1'
    tracks = [json.loads(l) for l in open(jam / 'verified_sources.jsonl')]
    artists = split_by(sorted({t['artist_id'] for t in tracks}), .15, 'jamendo')
    for t in tracks:
        add('jamendo', t['track_id'], jam / t['path'], lab(None, 1, 0, 0), artists[t['artist_id']], [f"artist:jamendo:{t['artist_id']}"],
            t['license'], vocal_evidence=True, credit=f"{t['artist_name']} - {t['title']} ({t['track_url']})")
    # Music Arena AI songs (CC BY 4.0 released audio): Sonauto family held out
    arena = W / 'lb_collect_20260924/payload/music-arena-v1'
    for c in (json.loads(l) for l in open(W / 'lb_admission_20260924/prepared/normalized_candidates.jsonl')):
        if c.get('source') != 'MusicArena':
            continue
        src = arena / c['original_source_path'].split('music-arena-v1/', 1)[-1]
        add('music_arena', c['ID'].split(':', 1)[1], src, lab(None, 1, 1, 1),
            'val' if c['generator'].startswith(DEV_ARENA) else 'train', [f"battle:music_arena:{c['battle_id']}"],
            c['license'], generator=c['generator'], vocal_evidence=True)
    # AI-Hub-channel phone versions of Korean fakes (MLAAD-ko) and Korean reals (Zeroth): train channels from train
    # calls, dev channels from the held-out calls, so the dev tests exactly the 'real call channel = real' shortcut
    prof = {name: [channel_profile(calls[c][:6]) for c in cs] for name, cs in
            (('train', train_calls[:120]), ('val', sorted(held)[:60]))}
    mlaad = W / 'v4_archives/mlaad_ko/mlaad_ko/fake/ko'
    fakes = []
    for system in sorted(p.name for p in mlaad.iterdir() if p.is_dir()):
        files = sorted(glob.glob(str(mlaad / system / '*.wav')))
        rng.shuffle(files)
        fakes += [(system, f) for f in files[:125 if system in DEV_MLAAD else 110]]
    import pandas as pd
    import io
    zer = []
    for split, pattern, cap in (('train', 'train-*.parquet', 500), ('val', 'test-*.parquet', 100)):
        frame = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(W / f'v4_archives/zeroth/**/{pattern}'), recursive=True))])
        frame = frame.sample(frac=1, random_state=SEED).head(cap)
        zer += [(split, r) for r in frame.itertuples()]
    for kind, rows in (('fake', fakes), ('real', zer)):
        for x in rows:
            if kind == 'fake':
                system, f = x
                split = 'val' if system in DEV_MLAAD else 'train'
                y = librosa.load(f, sr=SR)[0]
                rid, gen, labels, groups = f'{system}/{Path(f).stem}', f'{system}|aihub_channel', lab(1, 0, 1, None), [f'mlaad:{system}:{Path(f).stem}']
                lic = 'CC-BY-NC-4.0 (MLAAD) + AI-Hub-derived channel profile'
            else:
                split, r = x
                y = sf.read(io.BytesIO(r.audio['bytes']), dtype='float32')[0]
                y = y if y.ndim == 1 else y.mean(1)
                rid, gen, labels, groups = f'{r.id}', 'aihub_channel', speech_real, [f'zeroth:{r.id}']
                lic = 'CC-BY-4.0 (Zeroth) + AI-Hub-derived channel profile'
            dst = stage / 'channel' / kind / (hashlib.sha256(rid.encode()).hexdigest()[:20] + '.wav')
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, transplant(y, rng.choice(prof[split]), rng), SR, subtype='FLOAT')
            add(f'channel_{kind}', rid, dst, labels, split, groups, lic, generator=gen, language='ko')
    for r in items:
        r['path'] = str(r['path'])
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['collect', 'materialize', 'vocals', 'export'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, manifest = a.out / 'items.jsonl', a.out / 'manifest.jsonl'
    if a.stage == 'collect':
        rows = collect(a.out)
        items.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        count = {}
        for r in rows:
            count[(r['source'], r['split'])] = count.get((r['source'], r['split']), 0) + 1
        print(json.dumps({f'{s}/{p}': n for (s, p), n in sorted(count.items())}, ensure_ascii=False, indent=1))
    elif a.stage == 'materialize':
        from concurrent.futures import ProcessPoolExecutor
        rows = [json.loads(l) for l in items.open()]
        done = {json.loads(l)['ID'] for l in manifest.open()} if manifest.exists() else set()
        todo = [r for r in rows if r['ID'] not in done]
        with manifest.open('a') as f, ProcessPoolExecutor(a.workers) as pool:
            for i, r in enumerate(pool.map(materialize_one, todo, [a.out] * len(todo), [SEED] * len(todo), chunksize=8)):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if i % 500 == 0:
                    print(f'materialize {i}/{len(todo)}', flush=True)
    elif a.stage == 'vocals':
        import torch
        import librosa
        sys.path.insert(0, str(ROOT))
        from v6_runtime import baseline_module   # same trusted local HTDemucs loader the feature extractor uses
        model = baseline_module(W / 'assets_e9504').load_htdemucs_model().to(a.device).eval()
        rows = [json.loads(l) for l in manifest.open()]
        todo = [r for r in rows if r.get('vocal_evidence') and 'vocal_ratio' not in r]
        with torch.inference_mode():
            for i, r in enumerate(todo):
                w = librosa.load(a.out / r['path'], sr=model.samplerate, mono=False)[0]
                r.update(apply_vocal_evidence(r, vocal_ratio(model, w, a.device), .35, .08))
                if i % 200 == 0:
                    print(f'vocals {i}/{len(todo)}', flush=True)
        manifest.with_suffix('.tmp').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifest.with_suffix('.tmp').replace(manifest)
    else:
        rows = [json.loads(l) for l in manifest.open()]
        bank, _, skipped = export(rows, {r['ID']: r['split'] for r in rows})
        for name, split in (('bank_extra_train.csv', 'train'), ('bank_extra_dev.csv', 'val')):
            with (a.out / name).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(r for r in bank if r['split'] == split)
        print(json.dumps({'train': sum(r['split'] == 'train' for r in bank), 'dev': sum(r['split'] == 'val' for r in bank),
                          'skipped': len(skipped)}))


if __name__ == '__main__':
    main()
