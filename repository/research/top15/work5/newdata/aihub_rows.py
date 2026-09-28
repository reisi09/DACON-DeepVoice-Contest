"""AI-Hub Korean real-audio evaluation rows (E3), prepared and scored ONLY on this Mac: AI-Hub audio, crops, features and
row lists never leave Korea (AI-Hub terms: no export abroad, no third-party provision); only aggregate metrics are shared.
  aihub_71544  음악 유사성 판별 (Validation): human-produced Korean pop covers, 25 songs per genre, 10 s at 30 %: real music
  aihub_473    다음색 가이드보컬 (Validation): a cappella guide vocals, 75 across singers, 10 s at 30 %: real singing
  aihub_465    다화자 가창 (Validation): singing recordings, 75 across singers, 10 s at 30 %: real singing
  python aihub_rows.py   -> _workspace/aihub_prep/aihub_rows.csv (+ manifest), crops under _workspace/aihub_prep/"""
import csv
import io
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atadd_rows import COLUMNS  # noqa: E402

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace')
OUT = ROOT / 'aihub_prep'
Z71544 = next((ROOT / 'aihub_new/71544').rglob('VS.zip'))
Z473 = next((ROOT / 'aihub_new/473').rglob('VS1.zip'))
D465 = ROOT / 'aihub/465'


def crop(data: bytes | Path, dst: Path, seconds=10., frac=.3):
    x, sr = sf.read(io.BytesIO(data) if isinstance(data, bytes) else data, dtype='float32', always_2d=True)
    n = int(seconds * sr)
    a = max(0, min(int(len(x) * frac), len(x) - n))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x[a:a + n], sr, format='FLAC', subtype='PCM_16')


def row(ID, path, source, group, voice_route):
    r = {c: '' for c in COLUMNS}
    r.update({'ID': ID, 'orig_id': ID, 'split': 'e3', 'route': 'voice' if voice_route else 'music', 'path': str(path),
              'group_id': group, 'component_id': group, 'VOICE_PRESENT': '1', 'VOICE_PRESENT_mask': '1', 'VOICE_FAKE': '0',
              'VOICE_FAKE_mask': '1', 'MUSIC_PRESENT': '0' if voice_route else '1', 'MUSIC_PRESENT_mask': '1',
              'MUSIC_FAKE': '' if voice_route else '0', 'MUSIC_FAKE_mask': '0' if voice_route else '1', 'FILE_FAKE': '0',
              'FILE_FAKE_mask': '1', 'source': source, 'generator': 'bonafide', 'language': 'ko', 'codec_chain': 'flac',
              'event_intervals': '[]', 'parent_ids': '[]', 'license': 'AI-Hub (domestic research use)',
              'historical_development_exposure': 'False', 'npz': str(OUT / 'feats' / f'{ID}.npz'), 'dataset': source})
    return r


def spread(groups: dict, n: int, rng) -> list:
    """n items, round-robin over the sorted groups (one item per group per round)."""
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    out = []
    while len(out) < n and any(pools.values()):
        for k in list(pools):
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


def main():
    rng = random.Random(20260928)
    rows = []
    with zipfile.ZipFile(Z71544) as z:
        by_genre = defaultdict(list)
        for n in z.namelist():
            if n.endswith('.wav'):
                by_genre[n.split('/')[0]].append(n)
        for genre, names in sorted(by_genre.items()):
            songs = defaultdict(list)
            for n in names:
                songs[n.split('/')[1]].append(n)
            for song in sorted(rng.sample(sorted(songs), min(25, len(songs)))):
                n = rng.choice(sorted(songs[song]))
                ID = f"aihub71544_{genre}_{song}_{Path(n).stem.rsplit('_', 2)[-2]}{Path(n).stem.rsplit('_', 1)[-1]}"
                dst = OUT / '71544' / f'{ID}.flac'
                if not dst.exists():
                    crop(z.read(n), dst)
                rows.append(row(ID, dst, 'aihub_71544', f'aihub71544:{genre}/{song}', False))
    with zipfile.ZipFile(Z473) as z:
        singers = defaultdict(list)
        for n in z.namelist():
            if n.endswith('.wav'):
                singers[[p for p in n.split('/') if p.startswith('SINGER')][0] if 'SINGER' in n else n.split('/')[0]].append(n)
        for i, n in enumerate(spread(singers, 75, rng)):
            ID = f'aihub473_{i:03d}'
            dst = OUT / '473' / f'{ID}.flac'
            if not dst.exists():
                crop(z.read(n), dst)
            singer = [p for p in n.split('/') if p.startswith('SINGER')]
            rows.append(row(ID, dst, 'aihub_473', f"aihub473:{n.split('/')[0]}:{singer[0] if singer else 'x'}", True))
    singers = defaultdict(list)
    for f in D465.rglob('*.wav'):
        singers[f.parent.name].append(f)
    for i, f in enumerate(spread(singers, 75, rng)):
        ID = f'aihub465_{i:03d}'
        dst = OUT / '465' / f'{ID}.flac'
        if not dst.exists():
            crop(f, dst)
        rows.append(row(ID, dst, 'aihub_465', f'aihub465:{f.parent.name}', True))
    with (OUT / 'aihub_rows.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'aihub_manifest.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in rows)
    from collections import Counter
    print(len(rows), Counter(r['source'] for r in rows))


if __name__ == '__main__':
    main()
