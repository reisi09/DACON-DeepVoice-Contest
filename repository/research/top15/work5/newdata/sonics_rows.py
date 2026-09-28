"""X3 (2026-09-28 night): SONICS end-to-end AI songs (Suno chirp-v3.5 / Udio udio-120s, SONICS train split, all with
vocals) as training rows: a 20 s FLAC crop at 30 % of each song (open_rows.crop), music present and fake, voice fake if
present. One song per SONICS id (lyrics lineage); an id referenced anywhere in the bank or the new rows (any split, any
column: constructed mixes and songs carry SONICS donors in parent_ids) is never picked; ids compare as integers (the bank
writes fake_02205_*, the metadata id 2205). Stops unless each algorithm reaches N_PER unique ids.
  python sonics_rows.py ZIP_DIR FAKE_SONGS_CSV BANK_DIR NEW_ROWS_CSV OUT_DIR [N_PER_ALGORITHM]
  python sonics_rows.py relabel OUT_DIR   # after the feature dump (review round 5): the crop's voice presence is the
      # packaged presence estimate stored in each npz; below 0.5 the voice labels become unknown (music labels stay)"""
import csv
import hashlib
import random
import re
import sys
import zipfile
from pathlib import Path

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
from atadd_rows import COLUMNS  # noqa: E402
from open_rows import crop, mk  # noqa: E402

N = Path('/root/newdata')
PREP, FEATS = N / 'prepared/sonics', N / 'feats/sonics'
ALGOS = ('chirp-v3.5', 'udio-120s')
REF = re.compile(r'fake_(\d+)_(?:suno|udio)|sonics_lyrics:(\d+)')


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def referenced(csvs):
    """SONICS ids (int strings) referenced in any column of any row, and every audio/source hash of those rows."""
    ids, hashes = set(), set()
    for f in csvs:
        for r in csv.DictReader(Path(f).open()):
            for m in REF.finditer(' '.join(v or '' for v in r.values())):
                ids.add(str(int(m.group(1) or m.group(2))))
            hashes |= {r.get('audio_sha256') or '', r.get('source_sha256') or ''}
    return ids, hashes - {''}


def write(out: Path, name: str, rows):
    with (out / name).open('w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)


def build(zip_dir: Path, meta_csv: Path, bank: Path, new_rows: Path, out: Path, n_per: int):
    csv.field_size_limit(1 << 30)
    rng = random.Random(20260929)
    used, hashes = referenced([bank / 'bank_trainval.csv', bank / 'bank_test.csv', new_rows])
    print('referenced SONICS ids', len(used), 'hashes', len(hashes), flush=True)
    meta = {r['filename']: r for r in csv.DictReader(meta_csv.open()) if r['algorithm'] in ALGOS and r['split'] == 'train'
            and r['no_vocal'] == 'False' and float(r['duration'] or 0) >= 40}
    members = {}
    for z in sorted(zip_dir.glob('part_*.zip')):
        for n in zipfile.ZipFile(z).namelist():
            if Path(n).stem in meta:
                members[Path(n).stem] = (z, n)
    print('metadata songs', len(meta), 'found in zips', len(members), flush=True)
    pick, seen = [], set()
    for algo in ALGOS:
        pool = [s for s in members if meta[s]['algorithm'] == algo and str(int(meta[s]['id'])) not in used]
        rng.shuffle(pool)
        k = 0
        for s in pool:
            i = str(int(meta[s]['id']))
            if i in seen:
                continue
            seen.add(i)
            pick.append(s)
            k += 1
            if k == n_per:
                break
        print(algo, 'picked', k, 'of', len(pool), flush=True)
        assert k == n_per, f'{algo}: only {k} of {n_per} songs'
    assert len(pick) == len(seen) == n_per * len(ALGOS)
    rows = []
    for s in pick:
        z, n = members[s]
        src = PREP / 'full' / Path(n).name
        src.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            src.write_bytes(zipfile.ZipFile(z).read(n))
        dst = crop(src, PREP / f'{s}.flac', frac=.3, seconds=20.)
        m = meta[s]
        r = mk(f'sonics_x3_{s}', dst, 'music', f"sonics:id:{int(m['id'])}", 'sonics_x3', m['algorithm'],
               'CC-BY-NC-4.0 (SONICS)', vp='1', vf='1', mp='1', mf='1', language='en')
        r.update(npz=str(FEATS / f"{r['ID']}.npz"), source_sha256=sha(src), audio_sha256=sha(dst))
        assert not {r['source_sha256'], r['audio_sha256']} & hashes, f'{s}: audio already in the bank or new rows'
        rows.append(r)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'sonics_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in rows)
    write(out, 'sonics_rows_all.csv', rows)   # every picked row (the runner's npz list); training uses sonics_rows.csv
    print('rows', len(rows), flush=True)


def relabel(out: Path):
    import numpy as np
    rows, unknown = list(csv.DictReader((out / 'sonics_rows_all.csv').open())), 0
    for r in rows:
        with np.load(r['npz']) as d:
            voice = float(d['presence'][0])
        if voice < .5:   # no audible vocal in this crop: voice labels unknown, FILE stays fake through the music label
            r.update(VOICE_PRESENT='', VOICE_PRESENT_mask='0', VOICE_FAKE='', VOICE_FAKE_mask='0')
            unknown += 1
    write(out, 'sonics_rows.csv', rows)
    print('rows', len(rows), 'voice unknown', unknown, flush=True)


if __name__ == '__main__':
    if ARGS[0] == 'relabel':
        relabel(Path(ARGS[1]))
    else:
        build(*map(Path, ARGS[:5]), int(ARGS[5]) if len(ARGS) > 5 else 750)
