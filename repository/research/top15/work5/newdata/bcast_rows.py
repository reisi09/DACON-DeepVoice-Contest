"""Broadcast low-SNR rows (Codex rounds 24-27, research/top15/work5/debate_20260928/codex_round27.md): real speech with a short
real or AI instrumental excerpt far below the speech level. AI-OpenBMAT is our weakest slice (E2 FILE EER 0.34-0.39) and the
D1 remixes put the music at or above the vocal level (open_rows.remix_one), so this condition is missing from training.
  grid   music real|fake x music/speech active-RMS ratio -20/-15/-10 dB x excerpt 2/6 s; each real/fake pair shares the 20 s
         speech clip, the excerpt position, length and fades, so the pair differs only in the music class
  train  100 rows per (class, ratio, length) cell = 1,200 rows from train-split donors (bank 'train', new 'train')
  panel  30 rows per cell = 360 rows from held-out donors (bank 'test', new 'newtest'), split 'bpanel': never trained on,
         scored with score_split.py
Labels: VOICE real (1/0), MUSIC present with its class, FILE = MUSIC_FAKE; the music interval is kept in event_intervals.
Speech donors: real speech without music; music donors: known no-voice music rows (VOICE_PRESENT=0 with mask 1).
AI-Hub rows never reach VESSL; OpenBMAT is evaluation only and never a donor.
  python bcast_rows.py BANK_DIR AUDIO_ROOT [OUT_DIR]   -> OUT_DIR/bcast_rows.csv, bcast_panel.csv, bcast_manifest.csv
  python bcast_rows.py --selftest"""
import csv
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

SR, SPEECH_S, FADE = 16000, 20.0, 0.1
RATIOS, LENGTHS = (-20, -15, -10), (2.0, 6.0)
N = Path('/root/newdata')
PREP, FEATS = N / 'prepared/bcast_v3', N / 'feats/bcast'   # v3: own-speaker clips, donor lineage recorded, zero-RMS rejected
DIRTY = os.environ.get('INTEGRITY_JSON', '')   # integrity_check.py output: leak-flagged rows are never panel donors
SPEECH_SOURCES = {'zeroth', 'speechfake', 'existing:librispeech_test-clean', 'musan_speech', 'asvspoof2021_la',
                  'existing:asvspoof2019_la', 'asvspoof2021_df', 'atadd_speech', 'gtsinger_ko_speech', 'fleurs_ko'}
CLEAN_CODECS = {'pcm_wav', 'flac', 'wav', 'mp3_192', 'mp3_128'}
# temporal human+AI mixes: a 2-6 s excerpt may fall in the human part, so their FAKE label is unreliable for an excerpt
MIXED_GENERATORS = {'concat', 'crossfade'}


def load16(path, seconds=None):
    import librosa
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type='soxr_hq') if sr != SR else x
    return x[:int(seconds * SR)] if seconds else x


def frame_rms(x, hop=1600):
    f = x[:len(x) // hop * hop].reshape(-1, hop)
    return np.sqrt((f ** 2).mean(1))


def active_rms(x):
    r = frame_rms(x)
    act = r[r > max(r.max() * .05, 1e-5)]
    return float(np.sqrt((act ** 2).mean())) if len(act) else 0.


def speech_clip(paths):
    """20 s of speech: the donor group's utterances (then same-source top-up) joined with 0.3 s gaps, and how many of the
    listed utterances were used; (None, used) when shorter than 12 s."""
    parts, n, used = [], 0, 0
    for p in paths:
        x = load16(p)
        parts += [x, np.zeros(int(.3 * SR), np.float32)]
        n += len(x) + int(.3 * SR)
        used += 1
        if n >= SPEECH_S * SR:
            break
    x = np.concatenate(parts)[:int(SPEECH_S * SR)] if parts else np.zeros(0, np.float32)
    return (x, used) if len(x) >= 12 * SR else (None, used)


def excerpt(path, seconds, rng):
    """A seconds-long active window of a music donor with FADE s raised-cosine fades, or None."""
    x = load16(path, 60.)
    n = int(seconds * SR)
    if len(x) < n + SR:
        return None
    r = frame_rms(x)
    thr = max(r.max() * .05, 1e-5)
    starts = [s for s in range(0, len(x) - n, SR // 4) if (r[s // 1600:(s + n) // 1600] > thr).mean() >= .8]
    if not starts:
        return None
    s = rng.choice(starts)
    e = x[s:s + n].copy()
    k = int(FADE * SR)
    ramp = .5 - .5 * np.cos(np.linspace(0, np.pi, k, dtype=np.float32))
    e[:k] *= ramp
    e[-k:] *= ramp[::-1]
    return e


def place(speech, n, rng):
    """Excerpt start where >= 70 % of the frames are speech-active (else the most active window)."""
    r = frame_rms(speech)
    act = r > max(r.max() * .05, 1e-5)
    cand = list(range(int(.5 * SR), len(speech) - n - int(.5 * SR), SR // 10))
    score = [act[s // 1600:(s + n) // 1600].mean() for s in cand]
    good = [s for s, a in zip(cand, score) if a >= .7]
    return rng.choice(good) if good else cand[int(np.argmax(score))]


def mix(speech, music, t0, ratio_db):
    """speech + music at t0, music/speech active-RMS ratio = ratio_db over the overlapped speech (no peak cap here);
    None when either RMS is 0 (the music could not be placed, so its labels would be wrong)."""
    rs, rm = active_rms(speech[t0:t0 + len(music)]), active_rms(music)
    if not (rs > 0 and rm > 0):
        return None
    out = speech.copy()
    out[t0:t0 + len(music)] += music * (rs / rm) * 10 ** (ratio_db / 20)
    return out


def build_pair(job):
    """Two rows (real music, fake music) that share the speech clip, the excerpt position, length and ratio."""
    ID, split, sp, real_m, fake_m, ratio, length, seed = job
    rng = random.Random(seed)
    speech, used = speech_clip(sp['paths'])
    if speech is None:
        return []
    ex = {k: excerpt(d['path'], length, rng) for k, d in (('real', real_m), ('fake', fake_m))}
    if any(v is None for v in ex.values()):
        return []
    t0 = place(speech, len(ex['real']), rng)
    mixed = {k: mix(speech, ex[k], t0, ratio) for k in ex}
    if any(v is None for v in mixed.values()):
        return []
    gain = min(1., .95 / max(max(np.abs(m).max() for m in mixed.values()), 1e-9))   # one peak cap for the pair
    rows = []
    for kind, donor in (('real', real_m), ('fake', fake_m)):
        rid = f'{ID}_{kind}'
        dst = PREP / split / f'{rid}.wav'
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, mixed[kind] * gain, SR, subtype='FLOAT')
        mf = '1' if kind == 'fake' else '0'
        rows.append({
            'ID': rid, 'orig_id': rid, 'split': split, 'route': 'music', 'path': str(dst), 'audio_sha256': '', 'source_sha256': '',
            'group_id': f"bcast:{sp['group']}", 'component_id': f"bcast:{sp['group']}",
            'VOICE_PRESENT': '1', 'VOICE_PRESENT_mask': '1', 'MUSIC_PRESENT': '1', 'MUSIC_PRESENT_mask': '1',
            'VOICE_FAKE': '0', 'VOICE_FAKE_mask': '1', 'MUSIC_FAKE': mf, 'MUSIC_FAKE_mask': '1', 'FILE_FAKE': mf,
            'FILE_FAKE_mask': '1', 'source': 'bcast', 'generator': f"realV+{kind}M:{donor['generator']}",
            'language': sp['language'], 'codec_chain': 'wav',
            'event_intervals': json.dumps([{'component': 'music', 'start': t0 / SR, 'end': (t0 + len(ex[kind])) / SR}]),
            'parent_ids': json.dumps(sp['ids'][:used] + [donor['ID']]), 'variant': f'{ratio:+d}dB_{length:g}s',
            '_speech_groups': sp['groups'][:used], '_music_group': donor['group'],
            'license': 'derived-NC-research', 'historical_development_exposure': 'False',
            'npz': str(FEATS / f'{rid}.npz'), 'dataset': 'bcast'})
    return rows


def pools(rows, splits, banned_groups=frozenset(), banned_ids=frozenset()):
    """Speech donors grouped by speaker lineage, and no-voice music donors split by class (fake balanced by generator).
    Rows in banned_groups / banned_ids (for the panel: the training lineage and the leak-flagged rows) are skipped."""
    sp, real_m, fake_m = defaultdict(list), [], defaultdict(list)
    for r in rows:
        if r['split'] not in splits or 'aihub' in (r['source'] + r['path']).lower() or r['source'] == 'openbmat':
            continue
        if r['group_id'] in banned_groups or r['ID'] in banned_ids:
            continue
        if (r['source'] in SPEECH_SOURCES and r['VOICE_PRESENT'] == '1' and r['VOICE_FAKE'] == '0'
                and r['VOICE_FAKE_mask'] == '1' and r['MUSIC_PRESENT'] == '0' and r['MUSIC_PRESENT_mask'] == '1'
                and r.get('codec_chain', '') in CLEAN_CODECS):
            sp[r['group_id']].append(r)
        elif (r['MUSIC_PRESENT'] == '1' and r['MUSIC_FAKE_mask'] == '1' and r['VOICE_PRESENT'] == '0'
              and r['VOICE_PRESENT_mask'] == '1' and r['source'] not in ('bcast', 'remix_d1')
              and r.get('codec_chain', '') in CLEAN_CODECS and r['generator'] not in MIXED_GENERATORS):
            d = {'ID': r['ID'], 'path': r['path'], 'generator': r['generator'] or r['source'], 'group': r['group_id']}
            (fake_m[d['generator']].append(d) if r['MUSIC_FAKE'] == '1' else real_m.append(d))
    # v3: a clip uses only its own speaker group's utterances (a cross-speaker top-up joined whole sources into one
    # lineage component: 8 train / 3 panel components in v2); a group below 12 s of speech is skipped by speech_clip
    def enough(v):   # the group holds >= 12 s of speech (stop reading headers at 20 s)
        t = 0.
        for x in v:
            try:
                t += sf.info(x['path']).duration
            except Exception:
                continue
            if t >= SPEECH_S:
                break
        return t >= 12.
    speakers = [{'group': g, 'paths': [x['path'] for x in v], 'ids': [x['ID'] for x in v],
                 'groups': [x['group_id'] for x in v], 'language': v[0].get('language', '')} for g, v in sp.items() if enough(v)]
    return speakers, real_m, fake_m


def jobs_for(split, speakers, real_m, fake_m, per_cell, rng, tag):
    gens = sorted(fake_m)
    rng.shuffle(speakers)
    jobs, k = [], 0
    for ratio in RATIOS:
        for length in LENGTHS:
            for i in range(per_cell):
                sp = speakers[k % len(speakers)]
                g = gens[k % len(gens)]
                jobs.append((f'bcast_{tag}_{k:04d}', split, sp, rng.choice(real_m), rng.choice(fake_m[g]), ratio, length,
                             rng.randrange(1 << 30)))
                k += 1
    return jobs


def load_rows(bank_dir: Path, audio_root: Path):
    rows = []
    for f in ('bank_trainval.csv', 'bank_test.csv'):
        for r in csv.DictReader((bank_dir / f).open()):
            r['path'] = str(audio_root / r['path'])
            rows.append(r)
    rows += list(csv.DictReader((N / 'rows/new_rows.csv').open()))
    return rows


def main(bank_dir: Path, audio_root: Path, out: Path):
    from atadd_rows import COLUMNS
    assert DIRTY, 'INTEGRITY_JSON (integrity_check.py output) is required: its leak-flagged rows are never panel donors'
    rng = random.Random(20260929)
    rows = load_rows(bank_dir, audio_root)
    tr = pools(rows, {'train'})
    train_groups = frozenset(r['group_id'] for r in rows if r['split'] == 'train')
    dirty = frozenset(json.load(open(DIRTY))['dirty_ids']) if DIRTY else frozenset()
    pa = pools(rows, {'test', 'newtest'}, train_groups, dirty)
    for name, (s, r, f) in (('train', tr), ('panel', pa)):
        print(name, 'speakers', len(s), 'real music', len(r), 'fake music', sum(map(len, f.values())), 'generators', sorted(f), flush=True)
    # spare pairs cover donors that fail (too short, silent); the first per_cell good pairs per cell are kept
    todo = jobs_for('train', tr[0], tr[1], tr[2], 200, rng, 'tr') + jobs_for('bpanel', pa[0], pa[1], pa[2], 80, rng, 'pa')
    with ProcessPoolExecutor(16) as ex:
        built = list(ex.map(build_pair, todo, chunksize=4))
    keep = {'train': 100, 'bpanel': 30}
    cells, out_rows = defaultdict(int), []
    for job, pair in zip(todo, built):
        cell = (job[1], job[5], job[6])
        if pair and cells[cell] < keep[job[1]]:
            cells[cell] += 1
            out_rows += pair
    print('pairs per cell', dict(cells), flush=True)
    assert all(v == keep[c[0]] for c, v in cells.items()) and len(cells) == 12, 'a cell is short of donors'
    # lineage components over every speech group used and the music donor group -> group_id (bootstrap / weighting unit)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in out_rows:
        for gname in r['_speech_groups'] + [r['_music_group']]:
            parent[find(('pair', r['ID'].rsplit('_', 1)[0]))] = find(('g', gname))
    comp = {}
    for r in out_rows:
        root = find(('pair', r['ID'].rsplit('_', 1)[0]))
        r['group_id'] = r['component_id'] = f"bcast:{r['split']}:c{comp.setdefault(root, len(comp))}"
        del r['_speech_groups'], r['_music_group']
    split_of = defaultdict(set)   # checked on the union-find roots, before the split is written into group_id
    for r in out_rows:
        split_of[find(('pair', r['ID'].rsplit('_', 1)[0]))].add(r['split'])
    assert all(len(v) == 1 for v in split_of.values()), 'a lineage component spans train and panel'
    used_ids = lambda split: {i for r in out_rows if r['split'] == split for i in json.loads(r['parent_ids'])}
    assert not used_ids('train') & used_ids('bpanel'), 'a donor row is used by both train and panel'
    print('components train', len({r['group_id'] for r in out_rows if r['split'] == 'train'}),
          'panel', len({r['group_id'] for r in out_rows if r['split'] == 'bpanel'}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (('bcast_rows.csv', 'train'), ('bcast_panel.csv', 'bpanel')):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, COLUMNS)
            w.writeheader()
            w.writerows(r for r in out_rows if r['split'] == split)
    with (out / 'bcast_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in out_rows)
    print('rows', len(out_rows), flush=True)


def selftest():
    """One pair from synthetic audio: labels, shared speech outside the excerpt, and the music/speech ratio."""
    import tempfile
    global PREP
    d = Path(tempfile.mkdtemp())
    PREP = d
    t = np.arange(int(SPEECH_S * SR)) / SR
    sf.write(d / 's.wav', (.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * .7 * t) > -.5)).astype(np.float32), SR)
    for name, hz in (('mr.wav', 440), ('mf.wav', 660)):
        sf.write(d / name, (.9 * np.sin(2 * np.pi * hz * t[:10 * SR])).astype(np.float32), SR)
    for ratio in RATIOS:
        rows = build_pair((f'st{ratio}', 'train', {'paths': [str(d / 's.wav')], 'group': 'g', 'ids': ['s'], 'groups': ['g'], 'language': 'ko'},
                           {'ID': 'mr', 'path': str(d / 'mr.wav'), 'generator': 'bonafide', 'group': 'gm0'},
                           {'ID': 'mf', 'path': str(d / 'mf.wav'), 'generator': 'gen', 'group': 'gm1'}, ratio, 2.0, 7))
        assert [(r['MUSIC_FAKE'], r['FILE_FAKE'], r['VOICE_FAKE']) for r in rows] == [('0', '0', '0'), ('1', '1', '0')]
        iv = json.loads(rows[0]['event_intervals'])[0]
        a, b = (sf.read(r['path'], dtype='float32')[0] for r in rows)
        i0, i1 = int(iv['start'] * SR), int(iv['end'] * SR)
        outside = np.r_[0:i0, i1:len(a)]
        assert np.abs(a[outside] - b[outside]).max() < 1e-6, 'pair differs outside the excerpt'
        speech = sf.read(d / 's.wav', dtype='float32')[0]
        g = a[outside][np.argmax(np.abs(speech[outside]))] / speech[outside][np.argmax(np.abs(speech[outside]))]
        got = 20 * np.log10(active_rms(a[i0:i1] - g * speech[i0:i1]) / active_rms(g * speech[i0:i1]))
        assert abs(got - ratio) < 1.0, (ratio, got)
    print('selftest ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else N / 'rows')
