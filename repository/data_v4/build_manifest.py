"""Data v4: raw sources -> labeled items -> 16 kHz training audio + manifest.

Stages, in order (run on VESSL, see DESIGN.md §8):
  ingest       RAW/<source>/... -> items.jsonl (labels, masks, evidence, lineage, license)
  materialize  decode -> 16 kHz -> crop -> loudness/channel -> codec chain -> OUT/audio + manifest.jsonl
  vocals       GPU HTDemucs vocal-energy evidence on each materialized music crop (updates manifest)
  mix          voice x music mixtures from materialized pools, component-paired (DESIGN.md §6)
Label table (DACON rules + operator answers): DESIGN.md §4. None = unknown -> mask 0, never a 0 label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from functools import lru_cache
import sys
from pathlib import Path

LABELS = ('VOICE_PRESENT', 'MUSIC_PRESENT', 'VOICE_FAKE', 'MUSIC_FAKE', 'FILE_FAKE')
SR = 16000


# ---------------------------------------------------------------- labels
def lab(vp, mp, vf, mf, ff=None) -> dict:
    """Presence + *conditional* authenticity ("fake if present") -> five labels.
    A component contributes 0 if absent, its authenticity if present, and when presence is
    unknown it contributes 0 only if it would be real anyway (e.g. any voice in a real recording).
    FILE_FAKE = OR of contributions; unknown stays None. ff overrides (audited legacy labels)."""
    def part(p, f):
        return 0 if p == 0 else f if p == 1 else (0 if f == 0 else None)
    c = (part(vp, vf), part(mp, mf))
    if ff is None:
        ff = 1 if 1 in c else 0 if c == (0, 0) else None
    return dict(VOICE_PRESENT=vp, MUSIC_PRESENT=mp, VOICE_FAKE=vf if vp == 1 else None,
                MUSIC_FAKE=mf if mp == 1 else None, FILE_FAKE=ff, VOICE_FAKE_IF_PRESENT=vf)


def item(source, rid, path, labels, evidence, license, groups, **extra) -> dict:
    return dict(ID=f'{source}:{rid}', source=source, path=str(path), evidence=evidence, license=license,
                group_keys=sorted(set(groups)), **labels, **extra)


# ---------------------------------------------------------------- ingest adapters
def ing_zeroth(raw: Path):
    import pyarrow.parquet as pq
    out = raw.parent.parent / 'extracted' / 'zeroth_korean'
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((raw / 'data').glob('*.parquet')):
        for r in pq.read_table(f).to_pylist():
            uid = str(r.get('id') or Path(r['audio']['path']).stem)
            p = out / f'{uid}.flac'
            if not p.exists():
                p.write_bytes(r['audio']['bytes'])
            yield item('zeroth', uid, p, lab(1, 0, 0, None), 'dataset:real_read_speech', 'CC-BY-4.0',
                       [f"speaker:zeroth:{r['speaker_id']}"], language='ko', subset=f.name.split('-')[0])


def ing_fleurs(raw: Path):
    base = raw / 'data/ko_kr'
    for tsv in sorted(base.glob('*.tsv')):
        split = tsv.stem
        for row in csv.reader(tsv.open(encoding='utf-8'), delimiter='\t'):
            p = base / 'audio' / split / row[1]
            if p.exists():  # tar extracted by runbook
                # FLEURS has no speaker id; the same sentence id is read by several speakers -> group by sentence
                yield item('fleurs_ko', row[1], p, lab(1, 0, 0, None), 'dataset:real_read_speech', 'CC-BY-4.0',
                           [f'fleurs_sentence:{row[0]}'], language='ko')


def ing_asvspoof2021(raw: Path, track: str):
    keys = next(raw.rglob(f'keys/{track}/CM/trial_metadata.txt'))
    audio = {p.stem: p for p in raw.rglob('*.flac')}
    for line in keys.open():
        f = line.split()
        spk, trial, codec, trans_or_src, attack, key = f[:6]
        if trial not in audio:
            continue  # DF: only part00 downloaded
        fake = int(key == 'spoof')
        yield item(f'asvspoof2021_{track.lower()}', trial, audio[trial], lab(1, 0, fake, None),
                   'dataset:key_file', 'ODC-By-1.0' if track == 'LA' else 'ODbL-1.0',
                   [f'speaker:asvspoof:{spk}'], generator=attack if fake else 'bonafide',
                   channel=f'{codec}|{trans_or_src}', language='en')


def ing_mlaad(raw: Path):
    for meta in sorted(raw.rglob('meta.csv')):
        for r in csv.DictReader(meta.open(encoding='utf-8'), delimiter='|'):
            p = meta.parent / Path(r['path']).name
            if p.exists():
                yield item('mlaad', f"{meta.parent.name}/{p.stem}", p, lab(1, 0, 1, None), 'dataset:all_fake',
                           'CC-BY-NC-4.0', [f"mailabs_text:{r.get('original_file', p.stem)}"],
                           generator=r.get('model_name', meta.parent.name), language=r.get('language'))


def ing_vocalset(raw: Path):
    for p in sorted(raw.rglob('*.wav')):
        singer = next((x for x in p.parts if x[:1] in 'fm' and x[1:].isdigit()), p.parent.name)
        yield item('vocalset', p.stem, p, lab(1, 0, 0, None), 'dataset:real_acapella_singing', 'CC-BY-4.0',
                   [f'speaker:vocalset:{singer}'], kind='singing')


def ing_csd(raw: Path):
    for p in sorted(raw.rglob('*.wav')):
        yield item('csd', p.stem, p, lab(1, 0, 0, None), 'dataset:real_acapella_singing', 'CC-BY-NC-SA-4.0',
                   ['speaker:csd:single_singer'], kind='singing', language='ko' if 'korean' in str(p).lower() else 'en')


HAIM_SUBSETS = {  # dir prefix -> (music real/fake rule, note)
    'A1': 'real', 'A2': 'fake', 'B1': 'real_ai_mastered', 'B2': 'fake', 'B3': 'fake', 'B4': 'fake',
    'B6': 'fake', 'B7': 'fake_partial', 'B8': 'fake_partial', 'B9': 'fake_partial', 'C1': 'segments', 'C2': 'segments'}


def ing_haim(raw: Path):
    for js in sorted(raw.rglob('*.json')):
        subset = next((x[:2] for x in js.parts if x[:2] in HAIM_SUBSETS), None)
        if subset is None:
            continue
        m = json.loads(js.read_text())
        audio = js.with_name(m.get('filename', js.stem + '.mp3'))
        if not audio.exists():
            # sidecar may name .wav while the repo ships .mp3 (B1); never pick up *.download.lock
            audio = next((q for q in sorted(js.parent.glob(js.stem + '.*'))
                          if q.suffix in ('.mp3', '.wav', '.flac', '.ogg')), None)
            if audio is None:
                continue
        rule = HAIM_SUBSETS[subset]
        lyrics = (m.get('lyrics') or '').strip()
        # vocal presence from metadata only when lyrics text exists; otherwise unknown until separator check
        vp = 1 if lyrics and 'instrumental' not in lyrics.lower() else None
        groups = [f"haim_track:{m.get('input_source', m.get('track_id'))}".split('.')[0]]
        segments = None
        if rule == 'segments':
            segments = [(s['output_start'], s['output_end'], int(s['type'] == 'ai')) for s in m['segments']]
            groups += [f"haim_track:{s['source'].split('.')[0]}" for s in m['segments']]
            mf = int(any(f for *_, f in segments))
        else:
            mf = 0 if rule.startswith('real') else 1
        # full generations: any vocal is generated with the music; real/AI-mastered real: any vocal is real;
        # partial edits and mixsets: vocal authenticity unknown
        vf = 0 if rule.startswith('real') else 1 if rule == 'fake' else None
        yield item('haim', f'{subset}/{js.stem}', audio, lab(vp, 1, vf, mf), f'dataset:haim_{rule}',
                   'CC-BY-NC-4.0; MTG-Jamendo/SONICS credits', groups, subset=subset,
                   generator=m.get('model_name') or m.get('method') or rule, segments=segments,
                   ai_postprocessed_real=rule == 'real_ai_mastered')


def ing_sonics(raw: Path):
    for r in csv.DictReader((raw / 'fake_songs.csv').open(encoding='utf-8')):
        p = next(iter(raw.rglob(r['filename'] if r['filename'].endswith('.mp3') else r['filename'] + '.mp3')), None)
        if p is None:
            continue
        # SONICS fakes are lyric-conditioned songs; vocal presence verified by the separator stage
        yield item('sonics', p.stem, p, lab(None, 1, 1, 1), 'dataset:all_fake_song', 'CC-BY-NC-4.0',
                   [f"sonics_lyrics:{r.get('lyrics_id', r.get('id', p.stem))}"], generator=r.get('source', r.get('algorithm')))


def ing_openmic(raw: Path):
    root = next(raw.rglob('openmic-2018-aggregated-labels.csv')).parent
    voice = {}
    for r in csv.DictReader((root / 'openmic-2018-aggregated-labels.csv').open()):
        if r['instrument'] == 'voice':
            voice[r['sample_key']] = float(r['relevance'])
    for p in sorted((root / 'audio').rglob('*.ogg')):
        rel = voice.get(p.stem)
        vp = None if rel is None else int(rel >= 0.5)       # unannotated voice -> unknown (not absent)
        yield item('openmic', p.stem, p, lab(vp, 1, 0, 0), 'dataset:real_music+voice_annotation',
                   'CC-BY-4.0', [f"track:fma:{p.stem.split('_')[0]}"], voice_relevance=rel)


def ing_musan(raw: Path):
    root = next(raw.rglob('musan'))
    for ann in sorted((root / 'music').rglob('ANNOTATIONS')):
        for line in ann.read_text().splitlines():
            f = line.split()
            if len(f) < 3:
                continue
            p = ann.parent / f'{f[0]}.wav'
            vp = {'Y': 1, 'N': 0}.get(f[2])
            yield item('musan_music', f[0], p, lab(vp, 1, 0, 0), 'dataset:real_music+vocals_flag',
                       'CC-BY-4.0', [f'artist:musan:{f[3] if len(f) > 3 else f[0]}'])
    for p in sorted((root / 'speech').rglob('*.wav')):
        yield item('musan_speech', p.stem, p, lab(1, 0, 0, None), 'dataset:real_speech', 'CC-BY-4.0',
                   [f'musan_speech:{p.stem.rsplit("-", 1)[0]}'])
    for p in sorted((root / 'noise').rglob('*.wav')):
        yield item('musan_noise', p.stem, p, lab(0, 0, None, None), 'dataset:noise', 'CC-BY-4.0',
                   [f'musan_noise:{p.stem}'])


def ing_generated(raw: Path):
    for ledger in sorted(raw.rglob('rows.jsonl')):
        for line in ledger.open():
            r = json.loads(line)
            groups = r.get('group_keys', []) + [f'speaker:{s}' for s in r.get('speaker_ids', [])]
            labels = lab(r['VOICE_PRESENT'], r['MUSIC_PRESENT'], r['VOICE_FAKE'], r['MUSIC_FAKE'])
            yield item(r['source'], r['ID'], ledger.parent.parent / r['path'], labels, r['label_evidence'],
                       r['license'], groups, generator=r['generator'], language=r.get('language'), kind=r.get('kind'),
                       parent_ids=r.get('parent_ids', []))


def ing_existing(csv_path: Path, audio_root: Path):
    """Previous 16,042-row bank (already licensed/audited). Masks are carried over, not re-guessed."""
    for r in csv.DictReader(csv_path.open(encoding='utf-8')):
        v = {n: (int(float(r[n])) if r.get(n) not in (None, '', 'nan') and r.get(n + '_mask', '1') in ('1', '1.0', 'True') else None)
             for n in LABELS}
        groups = [f"legacy_component:{r['component_id']}"] if r.get('component_id') else []
        groups += [f"speaker:{r['source']}:{r['speaker_id']}"] if r.get('speaker_id') else []
        groups += [f"artist:{r['source']}:{r['artist_id']}"] if r.get('artist_id') else []
        groups += [f"track:fma:{r['track_id']}"] if r.get('track_id') and 'FMA' in r['source'] else []
        # an audited real file has only real components: its (possibly unlabeled) voice is real if present
        vf = v['VOICE_FAKE'] if v['VOICE_FAKE'] is not None else 0 if v['FILE_FAKE'] == 0 else None
        yield item('existing:' + r['source'], r['ID'], audio_root / r['path'], lab(v['VOICE_PRESENT'], v['MUSIC_PRESENT'],
                   vf, v['MUSIC_FAKE'], ff=v['FILE_FAKE']), 'existing_bank', r.get('license', ''), groups,
                   file_fake_override=v['FILE_FAKE'],
                   generator=r.get('generator') or r.get('attack_id'), prior_split=r.get('prior_split'),
                   prior_seen=r.get('prior_seen'))


ADAPTERS = {'zeroth_korean': ing_zeroth, 'fleurs_ko': ing_fleurs, 'mlaad_ko': ing_mlaad, 'mlaad_other': ing_mlaad,
            'asvspoof2021_la': lambda r: ing_asvspoof2021(r, 'LA'), 'asvspoof2021_df': lambda r: ing_asvspoof2021(r, 'DF'),
            'vocalset': ing_vocalset, 'csd': ing_csd, 'haim': ing_haim, 'sonics': ing_sonics, 'openmic': ing_openmic,
            'musan': ing_musan, 'gen': ing_generated}


# ---------------------------------------------------------------- vocal presence check (GPU)
def vocal_ratio(model, wav, device) -> float:
    """Max over 3 s windows of vocal-stem RMS / mixture RMS. Evidence, not ground truth (review P1)."""
    import torch
    from demucs.apply import apply_model
    x = torch.from_numpy(wav).float().to(device)
    x = x[None].repeat(2, 1) if x.ndim == 1 else x
    stems = apply_model(model, x[None], split=True, overlap=0.1, progress=False)[0]
    voc = stems[model.sources.index('vocals')].pow(2).mean(0)
    mix = x.pow(2).mean(0)
    w = int(3 * model.samplerate)
    n = max(1, voc.numel() // w)
    r = [(voc[i*w:(i+1)*w].mean() / (mix[i*w:(i+1)*w].mean() + 1e-10)).sqrt().item() for i in range(n)]
    return float(max(r))


def apply_vocal_evidence(row: dict, ratio: float, hi: float, lo: float) -> dict:
    """Agreeing evidence confirms; disagreement or the grey zone masks VOICE_* (never flips to a new label)."""
    row = dict(row, vocal_ratio=ratio)
    detected = 1 if ratio >= hi else 0 if ratio <= lo else None
    vp = row['VOICE_PRESENT']
    if vp is None:
        vp = detected
    elif detected is not None and detected != vp:
        vp = None
    return {**row, **lab(vp, row['MUSIC_PRESENT'], row['VOICE_FAKE_IF_PRESENT'], row['MUSIC_FAKE'],
                         ff=row.get('file_fake_override'))}


# ---------------------------------------------------------------- materialize
LENGTH_BINS = ((4, 10, .30), (10, 30, .40), (30, 60, .30))    # test: 4 s .. 60 s
CODECS = (('pcm_wav', .30), ('flac', .15), ('mp3_64', .06), ('mp3_128', .12), ('mp3_192', .07),
          ('opus_32', .05), ('aac_96', .05), ('tel_ulaw', .07), ('tel_alaw', .05), ('tel_gsm', .04), ('tel_opus8', .04))


def draw(rng: random.Random, table):
    x, acc = rng.random(), 0.
    for value, p in table:
        acc += p
        if x < acc:
            return value
    return table[-1][0]


def crop_window(rng: random.Random, dur: float) -> tuple[float, float]:
    """Label-independent target length; sources shorter than the draw are used whole."""
    lo, hi = draw(rng, [((a, b), p) for a, b, p in LENGTH_BINS])
    target = rng.uniform(lo, hi)
    if dur <= target:
        return 0., dur
    start = rng.uniform(0, dur - target)
    return start, target


def crop_labels(row: dict, start: float, length: float) -> dict:
    """Segment-labelled sources: the crop is fake only if it keeps >= 1 s of an AI segment."""
    if not row.get('segments'):
        return {n: row[n] for n in LABELS + ('VOICE_FAKE_IF_PRESENT',)}
    end = start + length
    fake = any(f and min(end, b) - max(start, a) >= 1.0 for a, b, f in row['segments'])
    real_only = all(not f for a, b, f in row['segments'] if min(end, b) - max(start, a) > 0)
    mf = 1 if fake else 0 if real_only else None                   # <1 s AI overlap -> unknown
    return lab(row['VOICE_PRESENT'], row['MUSIC_PRESENT'], row['VOICE_FAKE_IF_PRESENT'], mf)


FFMPEG = {  # encode args, final container; telephony = 8 kHz band-limited round trip
    'pcm_wav': (['-c:a', 'pcm_s16le'], 'wav'), 'flac': (['-c:a', 'flac'], 'flac'),
    'mp3_64': (['-c:a', 'libmp3lame', '-b:a', '64k'], 'mp3'), 'mp3_128': (['-c:a', 'libmp3lame', '-b:a', '128k'], 'mp3'),
    'mp3_192': (['-c:a', 'libmp3lame', '-b:a', '192k'], 'mp3'),
    'opus_32': (['-c:a', 'libopus', '-b:a', '32k'], 'ogg'), 'aac_96': (['-c:a', 'aac', '-b:a', '96k'], 'm4a'),
    'tel_ulaw': (['-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-c:a', 'pcm_mulaw'], 'wav'),
    'tel_alaw': (['-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-c:a', 'pcm_alaw'], 'wav'),
    'tel_gsm': (['-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-ac', '1', '-c:a', 'libgsm_ms'], 'wav'),
    'tel_opus8': (['-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-c:a', 'libopus', '-b:a', '8k'], 'ogg'),
}
FINAL = {'wav', 'flac', 'mp3'}   # test containers; others are decoded back to 16 kHz WAV/FLAC


@lru_cache(maxsize=1)
def usable_codecs(table=CODECS) -> tuple:
    """Drop codecs whose encoder this ffmpeg lacks and renormalise, so the draw stays one fixed
    label-independent table per run (recorded in the manifest via codec_chain)."""
    out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout.split()
    keep = [(c, p) for c, p in table if FFMPEG[c][0][FFMPEG[c][0].index('-c:a') + 1] in out]
    total = sum(p for _, p in keep)
    return tuple((c, p / total) for c, p in keep)


def encode(src: Path, dst_stem: Path, codec: str) -> Path:
    args, ext = FFMPEG[codec]
    tmp = dst_stem.with_suffix('.' + ext)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), *args, str(tmp)], check=True)
    if ext in FINAL and not codec.startswith('tel'):
        return tmp
    final = dst_stem.with_suffix('.flac' if codec.startswith('tel') else '.wav')  # telephony: back to 16 kHz
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(tmp), '-ar', str(SR), str(final)], check=True)
    tmp.unlink()
    return final


def materialize_one(row: dict, out: Path, seed: int) -> dict:
    import numpy as np
    import soundfile as sf
    import librosa
    rng = random.Random(f"{seed}|{row['ID']}")   # draws depend on ID only: label/source independent
    wav, sr = librosa.load(row['path'], sr=SR, mono=False)
    wav = wav if wav.ndim == 1 else wav.T
    dur = len(wav) / SR
    start, length = crop_window(rng, dur)
    seg = wav[int(start * SR):int((start + length) * SR)]
    mono = seg if seg.ndim == 1 else seg.mean(1)
    rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
    gain = 10 ** ((-23 + rng.uniform(-6, 4)) / 20) / rms          # ~-23 dBFS RMS +-, label independent
    seg = seg * gain
    peak = float(np.abs(seg).max())
    if peak > 0.97:
        seg = seg * (0.97 / peak)
    if seg.ndim == 2 and rng.random() < 0.5:
        seg = seg.mean(1)
    elif seg.ndim == 1 and rng.random() < 0.2:
        seg = np.stack([seg, seg], 1)
    codec = draw(rng, usable_codecs())
    h = hashlib.sha256(row['ID'].encode()).hexdigest()[:20]
    stem = out / 'audio' / h[:2] / h
    stem.parent.mkdir(parents=True, exist_ok=True)
    pcm = stem.with_suffix('.src.wav')
    sf.write(pcm, seg, SR, subtype='FLOAT')
    final = encode(pcm, stem, codec)
    pcm.unlink()
    data = final.read_bytes()
    return {**{k: v for k, v in row.items() if k not in LABELS}, **crop_labels(row, start, length),
            'orig_path': row['path'], 'path': str(final.relative_to(out)), 'crop_start_s': round(start, 3),
            'crop_len_s': round(len(seg) / SR, 3), 'orig_duration_s': round(dur, 3), 'codec_chain': codec,
            'gain_db': round(20 * np.log10(gain), 2), 'channels': 1 if seg.ndim == 1 else 2,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


# ---------------------------------------------------------------- mixtures
def components(rows: list[dict]) -> dict[str, str]:
    """Union-find over group_keys: any shared speaker/track/prompt/parent joins rows."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        keys = ['id:' + r['ID']] + r['group_keys'] + ['id:' + p for p in r.get('parent_ids', [])]
        for k in keys[1:]:
            a, b = find(keys[0]), find(k)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {r['ID']: find('id:' + r['ID']) for r in rows}


def plan_mixes(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Pair voice components with music components one-to-one, so a mixture joins exactly two
    existing components and never chains them into a giant one."""
    comp = components(rows)
    voice = [r for r in rows if r['VOICE_PRESENT'] == 1 and r['MUSIC_PRESENT'] == 0 and r['VOICE_FAKE'] is not None]
    music = [r for r in rows if r['MUSIC_PRESENT'] == 1 and r['VOICE_PRESENT'] == 0 and r['MUSIC_FAKE'] is not None]
    rng = random.Random(seed)
    by = lambda rs: {c: [r for r in rs if comp[r['ID']] == c] for c in sorted({comp[r['ID']] for r in rs})}
    vc, mc = by(voice), by(music)
    vkeys, mkeys = list(vc), [k for k in mc if k not in vc]
    rng.shuffle(vkeys)
    rng.shuffle(mkeys)
    jobs = []
    for v, m in zip(vkeys, mkeys):
        for _ in range(min(3, len(vc[v]), len(mc[m]))):
            a, b = rng.choice(vc[v]), rng.choice(mc[m])
            jobs.append(dict(voice=a, music=b, mode=rng.choice(['simultaneous'] * 3 + ['sequential'] * 2),
                             snr_db=rng.uniform(-5, 15), voice_first=rng.random() < .5))
            if len(jobs) >= n:
                return jobs
    return jobs


def mix_one(job: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf
    import librosa
    v, m = job['voice'], job['music']
    a = librosa.load(out / v['path'], sr=SR, mono=True)[0]
    b = librosa.load(out / m['path'], sr=SR, mono=True)[0]
    if job['mode'] == 'simultaneous':
        n = max(len(a), min(len(b), len(a) + 4 * SR))
        b = np.resize(b, n)
        off = random.Random(v['ID'] + m['ID']).randint(0, n - len(a))
        va = np.zeros(n, np.float32)
        va[off:off + len(a)] = a
        pv, pm = np.mean(a ** 2) + 1e-10, np.mean(b ** 2) + 1e-10
        y = va + b * np.sqrt(pv / pm / 10 ** (job['snr_db'] / 10))
    else:
        gap = np.zeros(int(.2 * SR), np.float32)
        y = np.concatenate([a, gap, b] if job['voice_first'] else [b, gap, a])
    y = y / max(1., float(np.abs(y).max()) / .97)
    rid = 'mix:' + hashlib.sha256((v['ID'] + '|' + m['ID'] + job['mode']).encode()).hexdigest()[:20]
    stem = out / 'audio' / 'mix' / rid.split(':')[1]
    stem.parent.mkdir(parents=True, exist_ok=True)
    pcm = stem.with_suffix('.src.wav')
    sf.write(pcm, y, SR, subtype='FLOAT')
    codec = draw(random.Random(rid), usable_codecs())   # same label-independent table as single items
    p = encode(pcm, stem, codec)
    pcm.unlink()
    labels = lab(1, 1, v['VOICE_FAKE'], m['MUSIC_FAKE'])
    return dict(ID=rid, source='constructed_mix', path=str(p.relative_to(out)), evidence='construction',
                license=f"{v['license']} + {m['license']}", group_keys=v['group_keys'] + m['group_keys'],
                parent_ids=[v['ID'], m['ID']], mix_mode=job['mode'], snr_db=round(job['snr_db'], 2),
                codec_chain=codec, crop_len_s=round(len(y) / SR, 3),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(), **labels)


# ---------------------------------------------------------------- CLI
def jsonl(path: Path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['ingest', 'vocals', 'materialize', 'mix'])
    ap.add_argument('--raw', type=Path, default=Path('/root/data_v4/raw'))
    ap.add_argument('--out', type=Path, default=Path('/root/data_v4/out'))
    ap.add_argument('--existing-csv', type=Path)
    ap.add_argument('--existing-audio', type=Path)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--vocal-hi', type=float, default=0.35)
    ap.add_argument('--vocal-lo', type=float, default=0.08)
    ap.add_argument('--mixes', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, manifest = a.out / 'items.jsonl', a.out / 'manifest.jsonl'
    if a.stage == 'ingest':
        with items.open('w') as f:
            counts = {}
            for name, fn in ADAPTERS.items():
                if not (a.raw / name).exists():
                    print(f'skip {name}: {a.raw / name} missing', flush=True)
                    continue
                for r in fn(a.raw / name):
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    counts[name] = counts.get(name, 0) + 1
            if a.existing_csv:
                for r in ing_existing(a.existing_csv, a.existing_audio):
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    counts['existing'] = counts.get('existing', 0) + 1
        print(json.dumps(counts, indent=1))
    elif a.stage == 'vocals':
        # runs on the *materialized crop*: the label describes exactly the audio that is trained on
        import torch
        import librosa
        from demucs.pretrained import get_model
        model = get_model('htdemucs').cuda().eval()
        rows = jsonl(manifest)
        todo = [r for r in rows if r['MUSIC_PRESENT'] == 1 and r['source'] != 'constructed_mix' and 'vocal_ratio' not in r]
        with torch.inference_mode():
            for i, r in enumerate(todo):
                w = librosa.load(a.out / r['path'], sr=model.samplerate, mono=False)[0]
                r.update(apply_vocal_evidence(r, vocal_ratio(model, w, 'cuda'), a.vocal_hi, a.vocal_lo))
                if i % 500 == 0:
                    print(f'vocals {i}/{len(todo)}', flush=True)
        tmp = manifest.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        tmp.replace(manifest)
    elif a.stage == 'materialize':
        from concurrent.futures import ProcessPoolExecutor
        done = {r['ID'] for r in jsonl(manifest)}
        todo = [r for r in jsonl(items) if r['ID'] not in done]
        with manifest.open('a') as f, ProcessPoolExecutor(a.workers) as pool:
            futs = {pool.submit(materialize_one, r, a.out, a.seed): r['ID'] for r in todo}
            for i, fut in enumerate(futs):
                try:
                    f.write(json.dumps(fut.result(), ensure_ascii=False) + '\n')
                except Exception as exc:
                    print(f'fail {futs[fut]}: {type(exc).__name__}: {exc}'[:300], flush=True)
                if i % 1000 == 0:
                    f.flush()
                    print(f'materialize {i}/{len(todo)}', flush=True)
    else:
        rows = [r for r in jsonl(manifest) if r['source'] != 'constructed_mix']
        done = {r['ID'] for r in jsonl(manifest) if r['source'] == 'constructed_mix'}
        with manifest.open('a') as f:
            for job in plan_mixes(rows, a.mixes, a.seed):
                r = mix_one(job, a.out)
                if r['ID'] not in done:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')


def self_check() -> None:
    assert lab(1, 0, 1, None)['FILE_FAKE'] == 1 and lab(1, 0, 0, None)['FILE_FAKE'] == 0
    assert lab(1, 1, 0, None)['FILE_FAKE'] is None, 'unknown music authenticity must not become real'
    assert lab(0, 1, 1, 0)['VOICE_FAKE'] is None, 'absent voice -> fake label masked'
    assert lab(None, 1, None, 1)['FILE_FAKE'] == 1 and lab(0, 0, None, None)['FILE_FAKE'] == 0
    assert lab(None, 1, 0, 0)['FILE_FAKE'] == 0, 'real recording with unknown vocals is still a real file'
    assert lab(None, 1, 1, 0)['FILE_FAKE'] is None, 'possible fake vocal of unknown presence -> unknown'
    legacy = dict(evidence='existing_bank', file_fake_override=0, **lab(None, 1, None, 0, ff=0))
    assert apply_vocal_evidence(legacy, .6, .35, .08)['FILE_FAKE'] == 0
    seg = dict(segments=[(0, 10, 0), (10, 20, 1)], **lab(None, 1, None, 1))
    assert crop_labels(seg, 0, 9)['MUSIC_FAKE'] == 0 and crop_labels(seg, 5, 10)['MUSIC_FAKE'] == 1
    assert crop_labels(seg, 0, 10.5)['MUSIC_FAKE'] is None and crop_labels(seg, 0, 10.5)['FILE_FAKE'] is None
    base = dict(evidence='dataset:all_fake_song', **lab(None, 1, 1, 1))
    assert apply_vocal_evidence(base, .6, .35, .08)['VOICE_FAKE'] == 1
    assert apply_vocal_evidence(base, .2, .35, .08)['VOICE_PRESENT'] is None
    real = dict(evidence='dataset:real_music+vocals_flag', **lab(0, 1, 0, 0))
    assert apply_vocal_evidence(real, .6, .35, .08)['VOICE_PRESENT'] is None, 'disagreement masks, never flips'
    assert apply_vocal_evidence(dict(real, **lab(None, 1, 0, 0)), .6, .35, .08)['VOICE_FAKE'] == 0
    rng = random.Random(1)
    assert all(4 <= crop_window(rng, 120)[1] <= 60 for _ in range(200))
    rows = [dict(ID=f'v{i}', group_keys=[f'spk:{i % 3}'], **lab(1, 0, i % 2, None)) for i in range(12)]
    rows += [dict(ID=f'm{i}', group_keys=[f'trk:{i % 4}'], **lab(0, 1, None, i % 2)) for i in range(12)]
    jobs = plan_mixes(rows, 100, 1)
    comp = components(rows)
    pairs = {(comp[j['voice']['ID']], comp[j['music']['ID']]) for j in jobs}
    assert len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs}), 'component pairing must be 1:1'
    print('build_manifest self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
