"""AT-ADD Track 2 subset as bank-format rows (research/top15/work5/FINAL_PLAN_20260928.md data additions).
Pools: 'train' = AT-ADD train+dev (training rows), 'newtest' = AT-ADD eval (its generators are unseen in train/dev: the
new-domain held-out of the DB protocol; training rows only in P1). Types speech / singing (a cappella) / music; sound skipped.
Stratified sample: fake rows spread evenly over generators, real rows over source corpora (fixed seed); the selected flac
members are extracted from the tars.
  python atadd_rows.py [ATADD_DIR] [OUT_DIR]   -> OUT_DIR/atadd_rows.csv, OUT_DIR/atadd_manifest.csv"""
import csv
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/root/newdata/atadd')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/root/newdata/rows')
FEATS = Path('/root/newdata/feats/atadd')
TARGET = {'train': {('speech', 'fake'): 5000, ('speech', 'real'): 3000, ('singing', 'fake'): 6000, ('singing', 'real'): 3000,
                    ('music', 'fake'): 4000, ('music', 'real'): 3000},
          'newtest': {('speech', 'fake'): 2000, ('speech', 'real'): 1000, ('singing', 'fake'): 2500, ('singing', 'real'): 1000,
                      ('music', 'fake'): 2000, ('music', 'real'): 1500}}
POOLS = {'train': ('train', 'dev'), 'newtest': ('eval',)}
LANG = {'AISHELL-3': 'zh', 'LJSpeech': 'en', 'LibriTTS-R': 'en', 'M4Singer': 'zh', 'OpenCpop': 'zh', 'KiSing': 'zh', '3D-Speaker': 'zh'}
COLUMNS = ['ID', 'orig_id', 'split', 'route', 'path', 'audio_sha256', 'source_sha256', 'group_id', 'component_id', 'VOICE_PRESENT',
           'VOICE_PRESENT_mask', 'MUSIC_PRESENT', 'MUSIC_PRESENT_mask', 'VOICE_FAKE', 'VOICE_FAKE_mask', 'MUSIC_FAKE', 'MUSIC_FAKE_mask',
           'FILE_FAKE', 'FILE_FAKE_mask', 'source', 'generator', 'language', 'codec_chain', 'event_intervals', 'parent_ids', 'variant',
           'license', 'historical_development_exposure', 'npz', 'dataset']


def spread(cells: dict, n: int, rng) -> list:
    """n rows spread as evenly as possible over the cells (a small cell gives all it has, the rest is re-spread)."""
    cells = {k: rng.sample(v, len(v)) for k, v in sorted(cells.items())}
    take, left = {k: 0 for k in cells}, n
    while left > 0:
        open_ = [k for k in cells if take[k] < len(cells[k])]
        if not open_:
            break
        share = max(1, left // len(open_))
        for k in open_:
            add = min(share, len(cells[k]) - take[k], left)
            take[k] += add
            left -= add
            if left == 0:
                break
    return [r for k in cells for r in cells[k][:take[k]]]


def row(m: dict, pool: str, split_dir: str) -> dict:
    fake = m['label'] == 'fake'
    ty = m['type']
    sid, spk = m['source_id'], m['speaker_id']
    lang = sid.split('_')[2] if sid.startswith('common_voice_') else LANG.get(m['source'], '')
    group = f"atadd:{spk}" if spk not in ('', '-') else (f"atadd:{m['source']}:{sid}" if sid not in ('', '-') else f"atadd:{m['name']}")
    voice = ty in ('speech', 'singing')
    stem = m['name'].rsplit('.', 1)[0]
    return {'ID': f'atadd_{stem}', 'orig_id': f"atadd:{m['name']}", 'split': pool, 'route': 'voice' if voice else 'music',
            'path': str(A / 'audio' / split_dir / m['name']), 'audio_sha256': '', 'source_sha256': '', 'group_id': group, 'component_id': group,
            'VOICE_PRESENT': '1' if voice else '', 'VOICE_PRESENT_mask': '1' if voice else '0',
            'MUSIC_PRESENT': '0' if voice else '1', 'MUSIC_PRESENT_mask': '1',
            'VOICE_FAKE': str(int(fake)) if voice else '', 'VOICE_FAKE_mask': '1' if voice else '0',
            'MUSIC_FAKE': '' if voice else str(int(fake)), 'MUSIC_FAKE_mask': '0' if voice else '1',
            'FILE_FAKE': str(int(fake)), 'FILE_FAKE_mask': '1', 'source': f'atadd_{ty}',
            'generator': m['generator'] if fake else 'bonafide', 'language': lang, 'codec_chain': 'flac', 'event_intervals': '[]',
            'parent_ids': '[]', 'variant': f"{m['source']}|{m['generation_task']}", 'license': 'AT-ADD-NC-research',
            'historical_development_exposure': 'False', 'npz': str(FEATS / f'atadd_{stem}.npz'), 'dataset': 'atadd'}


def main():
    rng = random.Random(20260928)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, wanted = [], defaultdict(set)
    for pool, splits in POOLS.items():
        meta = []
        for s in splits:
            meta += [{**m, '_split': s} for m in csv.DictReader((A / 'meta_label' / f't2_{s}_meta.csv').open(encoding='utf-8-sig'))]
        for (ty, lab), n in TARGET[pool].items():
            cells = defaultdict(list)
            for m in meta:
                if m['type'] == ty and m['label'] == lab:
                    cells[m['generator'] if lab == 'fake' else m['source']].append(m)
            picked = spread(cells, n, rng)
            print(pool, ty, lab, len(picked), 'from', {k: len(v) for k, v in sorted(cells.items())}, flush=True)
            for m in picked:
                rows.append(row(m, pool, m['_split']))
                wanted[m['_split']].add(m['name'])
    ids = [r['ID'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate IDs'
    with (OUT / 'atadd_rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'atadd_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in rows)
    from multiprocessing import Pool
    with Pool(len(wanted)) as pool:   # the .tar files are gzip streams: one process per tar
        for s, missing in pool.starmap(extract, sorted(wanted.items())):
            print(s, 'extracted; missing', missing, flush=True)
            assert not missing, f'{s}: {missing} selected files not in the tar'


def extract(s: str, names: set):
    """Stream one tar once and write only the selected members."""
    dest = A / 'audio' / s
    dest.mkdir(parents=True, exist_ok=True)
    todo = {n for n in names if not (dest / n).exists()}
    with tarfile.open(A / f'AT-ADD_T2_{s}.tar', 'r|gz') as t:
        for member in t:
            base = member.name.rsplit('/', 1)[-1]
            if member.isfile() and base in todo:
                (dest / (base + '.part')).write_bytes(t.extractfile(member).read())
                (dest / (base + '.part')).rename(dest / base)
                todo.discard(base)
    return s, len(todo)

if __name__ == '__main__':
    main()
