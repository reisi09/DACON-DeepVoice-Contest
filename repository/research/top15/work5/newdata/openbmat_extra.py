"""Codex round 13 (c): 200 AI + 200 human AI-OpenBMAT mixes more as TRAINING rows, from files not used yet and sharing no
music source track with any held-out (newval / newtest) AI-OpenBMAT row, so the existing E2 check stays clean.
  python openbmat_extra.py [ROWS_DIR]   -> ROWS_DIR/openbmat_extra_rows.csv, ROWS_DIR/openbmat_extra_manifest.csv"""
import csv
import json
import random
import sys
import zipfile
from pathlib import Path

from atadd_rows import COLUMNS
from open_rows import N, PREP, SEED, mk

D = Path(sys.argv[1]) if len(sys.argv) > 1 else N / 'rows'


def tracks(meta):
    return {Path(s['file']).stem for seg in meta['segments'] for s in seg.get('sources', []) if s.get('type') == 'music'}


def main():
    rows = list(csv.DictReader((D / 'new_rows.csv').open()))
    used = {r['path'].rsplit('/', 1)[-1] for r in rows if r['source'] == 'openbmat'}
    z = zipfile.ZipFile(N / 'ai_openbmat/AI-OpenBMAT.zip')
    names = [n for n in z.namelist() if n.startswith('AI-OpenBMAT/') and n.endswith('.wav') and '/__MACOSX' not in n]
    meta = {n: json.loads(z.read(n[:-4] + '.json')) for n in names}
    held = set()
    for n in names:
        if Path(n).name in used:
            r = next(x for x in rows if x['source'] == 'openbmat' and x['path'].endswith('/' + Path(n).name))
            if r['split'] != 'train':
                held |= tracks(meta[n])
    rng = random.Random(SEED + 1)
    out = []
    for art in ('ai', 'human'):
        pool = sorted(n for n in names if f'/{art}/' in n and Path(n).name not in used and not (tracks(meta[n]) & held))
        for n in sorted(rng.sample(pool, min(200, len(pool)))):
            dst = PREP / 'openbmat' / art / Path(n).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(z.read(n))
            music = sorted(tracks(meta[n]))
            fake = '1' if art == 'ai' else '0'
            out.append(mk(f'openbmat_{art}_{Path(n).stem}', dst, 'music', f"openbmat:{art}:{music[0] if music else Path(n).stem}",
                          'openbmat', 'ai_music_broadcast' if fake == '1' else 'bonafide', 'AI-OpenBMAT', mp='1', mf=fake))
        print(art, 'pool', len(pool), 'taken', sum(r['ID'].startswith(f'openbmat_{art}_') for r in out), flush=True)
    assert not ({r['ID'] for r in out} & {r['ID'] for r in rows}), 'extra rows collide with existing IDs'
    with (D / 'openbmat_extra_rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(out)
    with (D / 'openbmat_extra_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in out)
    print('held-out music tracks excluded', len(held), 'rows', len(out))


if __name__ == '__main__':
    main()
