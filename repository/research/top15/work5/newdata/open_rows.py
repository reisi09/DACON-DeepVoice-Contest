"""Open datasets as bank-format training rows + prepared audio (research/top15/work5/FINAL_PLAN_20260928.md data additions).
  haim_b6 / haim_b3 / haim_b4  HAIM AI songs: ACE-Step 1.5 on K-pop lyrics (Korean/English, sung = fake vocal), Ozone-mastered,
                               professionally mixed: fake music (20 s crop at 30 % of the track)
  musdb                        MUSDB18-HQ professional real mixes, 3 crops of 15 s per song, vocal presence from the vocals stem
  openbmat                     AI-OpenBMAT 60 s broadcast mixes: AI (fake) or human (real) music under broadcast audio
  gtsinger_ko(_speech)         GTSinger Korean a cappella singing and its paired speech: real
  sdrr                         SD-RR Jamendo real songs (no ND licences), 20 s crop
  remix_d1                     D1 2x2 remixes at 16 kHz: real vocals (MUSDB stems, GTSinger-ko, AT-ADD real singing) or fake
                               vocals (AT-ADD SVC train pool) over real (MUSDB drums+bass+other) or fake (AT-ADD TTM train pool)
                               accompaniment, vocal/accompaniment active-RMS ratio -12/-6/0/+6 dB
Every row is pool 'train'; groups follow lineage (song, prompt, singer, source track).
  python open_rows.py [OUT_DIR]   -> OUT_DIR/open_rows.csv, OUT_DIR/open_manifest.csv; audio under /root/newdata/prepared"""
import csv
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from atadd_rows import COLUMNS

N = Path('/root/newdata')
PREP, FEATS = N / 'prepared', N / 'feats/open'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else N / 'rows'
FFMPEG = '/root/s7pkg/x/model/runtime/bin/ffmpeg'
SEED = 20260928


def mk(ID, path, route, group, source, generator, license_, *, vp='', vf='', mp='', mf='', language='', parents=(), variant=''):
    """Bank row; '' = unknown (mask 0). FILE = fake if any known component is fake."""
    known = [x for x in (vf if vp == '1' else '', mf if mp == '1' else '') if x != '']
    assert known, f'{ID}: no known component label'
    ff = str(int('1' in known))
    return {'ID': ID, 'orig_id': ID, 'split': 'train', 'route': route, 'path': str(path), 'audio_sha256': '', 'source_sha256': '',
            'group_id': group, 'component_id': group, 'VOICE_PRESENT': vp, 'VOICE_PRESENT_mask': '1' if vp else '0',
            'MUSIC_PRESENT': mp, 'MUSIC_PRESENT_mask': '1' if mp else '0', 'VOICE_FAKE': vf if vp == '1' else '',
            'VOICE_FAKE_mask': '1' if vp == '1' and vf else '0', 'MUSIC_FAKE': mf if mp == '1' else '',
            'MUSIC_FAKE_mask': '1' if mp == '1' and mf else '0', 'FILE_FAKE': ff, 'FILE_FAKE_mask': '1', 'source': source,
            'generator': generator, 'language': language, 'codec_chain': Path(path).suffix.lstrip('.'), 'event_intervals': '[]',
            'parent_ids': json.dumps(list(parents)), 'variant': variant, 'license': license_,
            'historical_development_exposure': 'False', 'npz': str(FEATS / f'{ID}.npz'), 'dataset': source}


def duration(path) -> float:
    err = subprocess.run([FFMPEG, '-nostdin', '-i', str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r'Duration: (\d+):(\d+):([\d.]+)', err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def crop(src, dst: Path, frac=.3, seconds=20.):
    """seconds-long FLAC crop starting at frac of the track (whole track when shorter)."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    d = duration(src)
    start = max(0., min(d * frac, d - seconds))
    subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-y', '-ss', f'{start:.3f}', '-t', f'{seconds}', '-i', str(src),
                    '-c:a', 'flac', str(dst.with_suffix('.part.flac'))], check=True)
    dst.with_suffix('.part.flac').rename(dst)
    return dst


def haim_one(args):
    sub, f = args
    tag = {'B6_human_lyrics_ai_gen': 'b6', 'B3_mastered': 'b3', 'B4_professional_human_mix': 'b4'}[sub]
    stem = f.stem
    dst = crop(f, PREP / 'haim' / tag / f'{stem}.flac')
    if tag == 'b6':   # B6_{prompt}_{en|kr}: sung lyrics, Korean or English renditions of one prompt share a group
        prompt, lang = stem.split('_')[1], stem.split('_')[2]
        return mk(f'haim_b6_{stem}', dst, 'music', f'haim_b6:{prompt}', 'haim_b6', 'acestep1.5', 'CC-BY-NC-4.0',
                  vp='1', vf='1', mp='1', mf='1', language='ko' if lang == 'kr' else 'en')
    return mk(f'haim_{tag}_{stem}', dst, 'music', f'haim_b34:{stem}', f'haim_{tag}', {'b3': 'ozone9_mastered_ai', 'b4': 'human_mixed_ai'}[tag],
              'CC-BY-NC-4.0', mp='1', mf='1')


def musdb_one(name):
    """3 crops of 15 s (25/50/75 %) of one song: mixture FLAC (44.1 kHz stereo), plus 16 kHz mono vocal and accompaniment
    stems for the D1 donors; vocal presence = vocals-stem 1 s frames above -40 dBFS on >= 30 % of the crop."""
    with zipfile.ZipFile(N / 'musdb18hq/musdb18hq.zip') as z:
        stems = {s: sf.read(io.BytesIO(z.read(f'{name}/{s}.wav')), dtype='float32')[0] for s in ('mixture', 'vocals', 'drums', 'bass', 'other')}
    import librosa
    sr, rows, donors = 44100, [], []
    n, song = len(stems['mixture']), name.split('/', 1)[1]
    key = hashlib.sha1(name.encode()).hexdigest()[:10]
    for k, frac in enumerate((.25, .5, .75)):
        a = int(max(0, min(n * frac, n - 15 * sr)))
        seg = {s: x[a:a + 15 * sr] for s, x in stems.items()}
        dst = PREP / 'musdb' / f'{key}_{k}.flac'
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, seg['mixture'], sr, subtype='PCM_16')
        v = seg['vocals'].mean(1)
        rms = np.sqrt(np.mean(v[:len(v) // sr * sr].reshape(-1, sr) ** 2, 1))
        vp = '1' if (rms > .01).mean() >= .3 else '0'
        rows.append(mk(f'musdb_{key}_{k}', dst, 'music', f'musdb:{song}', 'musdb', 'bonafide', 'MUSDB18-HQ-research-NC',
                       vp=vp, vf='0', mp='1', mf='0'))
        acc = (seg['drums'] + seg['bass'] + seg['other']).mean(1)
        for kind, x in (('vocal', v), ('acc', acc)):
            if kind == 'vocal' and vp != '1':
                continue
            p = PREP / 'donors' / f'musdb_{kind}_{key}_{k}.wav'
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, librosa.resample(x, orig_sr=sr, target_sr=16000, res_type='soxr_hq'), 16000, subtype='FLOAT')
            donors.append({'kind': 'realV' if kind == 'vocal' else 'realM', 'path': str(p), 'ID': f'musdb_{key}_{k}', 'group': f'musdb:{song}'})
    return rows, donors


def openbmat_rows(rng, per_class=300):
    z = zipfile.ZipFile(N / 'ai_openbmat/AI-OpenBMAT.zip')
    rows = []
    for art in ('ai', 'human'):
        names = sorted(n for n in z.namelist() if n.startswith(f'AI-OpenBMAT/{art}/') and n.endswith('.wav'))
        for n in sorted(rng.sample(names, min(per_class, len(names)))):
            meta = json.loads(z.read(n[:-4] + '.json'))
            music = [Path(s['file']).stem for seg in meta['segments'] for s in seg['sources'] if s['type'] == 'music']
            dst = PREP / 'openbmat' / art / Path(n).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(z.read(n))
            fake = '1' if art == 'ai' else '0'
            rows.append(mk(f'openbmat_{art}_{Path(n).stem}', dst, 'music', f"openbmat:{art}:{music[0] if music else Path(n).stem}",
                           'openbmat', 'ai_music_broadcast' if fake == '1' else 'bonafide', 'AI-OpenBMAT', mp='1', mf=fake))
    return rows


def gtsinger_rows():
    rows, donors = [], []
    for f in sorted((N / 'gtsinger/Korean').rglob('*.wav')):
        parts = f.relative_to(N / 'gtsinger/Korean').parts   # singer / technique / song / group / file
        singer, song, grp = parts[0], parts[2], parts[3]
        speech = grp.startswith('Paired_Speech')
        ID = 'gtsinger_' + hashlib.sha1(str(f).encode()).hexdigest()[:16]
        rows.append(mk(ID, f, 'voice', f'gtsinger:{singer}', 'gtsinger_ko_speech' if speech else 'gtsinger_ko', 'bonafide',
                       'CC-BY-NC-SA-4.0+GTSinger-terms', vp='1', vf='0', mp='0', language='ko', variant='/'.join(parts[1:4])))
        if not speech:
            donors.append({'kind': 'realV', 'path': str(f), 'ID': ID, 'group': f'gtsinger:{singer}'})
    return rows, donors


def sdrr_rows():
    rows = []
    for a in csv.DictReader((N / 'sdrr/attribution.csv').open()):
        if a['license_has_noderivatives'] == 'True':
            continue
        src = N / 'sdrr/references' / f"{a['reference_id']}.mp3"
        if not src.exists():
            continue
        dst = crop(src, PREP / 'sdrr' / f"{a['reference_id']}.flac")
        rows.append(mk(f"sdrr_{a['reference_id']}", dst, 'music', f"jamendo:{a['source_track_id']}", 'sdrr', 'bonafide',
                       a['license_family'], mp='1', mf='0'))
    return rows


def load16(path, seconds=None):
    import librosa
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    x = librosa.resample(x, orig_sr=sr, target_sr=16000, res_type='soxr_hq') if sr != 16000 else x
    return x[:int(seconds * 16000)] if seconds else x


def active_rms(x):
    frames = x[:len(x) // 1600 * 1600].reshape(-1, 1600)
    r = np.sqrt((frames ** 2).mean(1))
    act = r[r > max(r.max() * .05, 1e-5)]
    return float(np.sqrt((act ** 2).mean())) if len(act) else 0.


def remix_one(args):
    """One D1 mix: accompaniment (<= 10 s) + vocal placed at 0.5 s, vocal/accompaniment active-RMS ratio = db, peak <= 0.95."""
    ID, voc, acc, db = args
    dst = PREP / 'remix' / f'{ID}.wav'
    if not dst.exists():
        a, v = load16(acc['path'], 10.), load16(voc['path'])
        v = v[:max(0, len(a) - 8000)]
        mix = a.copy()
        ra, rv = active_rms(a), active_rms(v)
        if ra > 0 and rv > 0:
            mix[8000:8000 + len(v)] += v * (ra / rv) * 10 ** (db / 20)
        mix *= min(1., .95 / max(np.abs(mix).max(), 1e-9))
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, mix, 16000, subtype='FLOAT')
    vf, mf = str(int(voc['kind'] == 'fakeV')), str(int(acc['kind'] == 'fakeM'))
    return mk(ID, dst, 'music', f"remix:{voc['group']}", 'remix_d1', f"{voc['kind']}+{acc['kind']}", 'derived-NC-research',
              vp='1', vf=vf, mp='1', mf=mf, parents=(voc['ID'], acc['ID']), variant=f'{db:+d}dB')


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(24) as ex:
        haim = [(sub, f) for sub in ('B6_human_lyrics_ai_gen', 'B3_mastered', 'B4_professional_human_mix')
                for f in sorted((N / 'haim/B_hybrid' / sub).glob('*.mp3'))]
        rows += list(ex.map(haim_one, haim, chunksize=8))
        print('haim', len(rows), flush=True)
        with zipfile.ZipFile(N / 'musdb18hq/musdb18hq.zip') as z:
            songs = sorted({'/'.join(n.split('/')[:2]) for n in z.namelist() if n.endswith('/mixture.wav')})
        donors = []
        for r, d in ex.map(musdb_one, songs):
            rows += r
            donors += d
        print('musdb songs', len(songs), 'rows', len(rows), flush=True)
    rows += openbmat_rows(rng)
    g_rows, g_donors = gtsinger_rows()
    rows += g_rows
    donors += g_donors
    rows += sdrr_rows()
    print('before remix', len(rows), flush=True)
    atadd = [r for r in csv.DictReader((OUT / 'atadd_rows.csv').open()) if r['split'] == 'train']
    donors += [{'kind': 'realV', 'path': r['path'], 'ID': r['ID'], 'group': r['group_id']} for r in atadd if r['source'] == 'atadd_singing' and r['FILE_FAKE'] == '0']
    donors += [{'kind': 'fakeV', 'path': r['path'], 'ID': r['ID'], 'group': r['group_id']} for r in atadd if r['source'] == 'atadd_singing' and r['FILE_FAKE'] == '1']
    donors += [{'kind': 'fakeM', 'path': r['path'], 'ID': r['ID'], 'group': r['group_id']} for r in atadd if r['source'] == 'atadd_music' and r['FILE_FAKE'] == '1']
    pools = {k: [d for d in donors if d['kind'] == k] for k in ('realV', 'fakeV', 'realM', 'fakeM')}
    print({k: len(v) for k, v in pools.items()}, flush=True)
    jobs = []
    for vk in ('realV', 'fakeV'):
        for mk_ in ('realM', 'fakeM'):
            for i in range(300):
                jobs.append((f'remix_{vk}_{mk_}_{i:03d}', rng.choice(pools[vk]), rng.choice(pools[mk_]), (-12, -6, 0, 6)[i % 4]))
    with ProcessPoolExecutor(24) as ex:
        rows += list(ex.map(remix_one, jobs, chunksize=8))
    ids = [r['ID'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate IDs'
    with (OUT / 'open_rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'open_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in rows)
    from collections import Counter
    print('rows', len(rows), Counter(r['source'] for r in rows), flush=True)


if __name__ == '__main__':
    main()
