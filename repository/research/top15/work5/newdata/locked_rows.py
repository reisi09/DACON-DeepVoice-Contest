"""Locked final-check set (design 2026-09-28, DESIGN_KO.md 4.4 / 9): rows that no model, selection or diagnosis has used.
  AT-ADD T2 eval pool files never sampled into new_rows.csv, minus every speaker/source group that any used new row has
    (train, newval, newtest): speech / singing / music x real / fake, fakes spread over generators (atadd_rows.spread)
  AI-OpenBMAT excerpts never used (not in new_rows.csv nor openbmat_extra_rows.csv), minus every excerpt that shares a
    music track or a speech source with a used excerpt: human / AI music under broadcast speech (VOICE label unknown)
Writes rows (split 'locked'), a manifest, and LOCK.json (sha256 of the sorted IDs, of every audio file and of the label
columns). No metric is computed here; the set is scored once, after the final candidate is fixed.
  python locked_rows.py OUT_DIR"""
import csv
import hashlib
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1])
sys.argv = sys.argv[:1]                      # atadd_rows / open_rows read their own argv at import
import atadd_rows as AR  # noqa: E402
from atadd_rows import COLUMNS  # noqa: E402
from open_rows import mk  # noqa: E402

N = Path('/root/newdata')
TARGET = {('speech', 'fake'): 150, ('speech', 'real'): 150, ('singing', 'fake'): 100, ('singing', 'real'): 100,
          ('music', 'fake'): 100, ('music', 'real'): 100}
OPENBMAT_PER_CLASS = 100
PREP = N / 'prepared/locked'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def used_rows():
    rows = list(csv.DictReader((N / 'rows/new_rows.csv').open()))
    extra = N / 'rows/openbmat_extra_rows.csv'
    return rows + (list(csv.DictReader(extra.open())) if extra.exists() else [])


def atadd_locked(rng, used):
    used_ids = {r['ID'] for r in used}
    used_groups = {r['group_id'] for r in used}
    meta = list(csv.DictReader((AR.A / 'meta_label' / 't2_eval_meta.csv').open(encoding='utf-8-sig')))
    rows, names = [], set()
    for (ty, lab), n in TARGET.items():
        cells = defaultdict(list)
        for m in meta:
            if m['type'] == ty and m['label'] == lab:
                r = AR.row(m, 'locked', 'eval')
                if r['ID'] not in used_ids and r['group_id'] not in used_groups:
                    cells[m['generator'] if lab == 'fake' else m['source']].append(m)
        picked = AR.spread(cells, n, rng)
        print('atadd', ty, lab, len(picked), 'from', {k: len(v) for k, v in sorted(cells.items())}, flush=True)
        for m in picked:
            r = AR.row(m, 'locked', 'eval_locked')
            r['npz'] = str(N / 'feats/locked' / f"{r['ID']}.npz")
            rows.append(r)
            names.add(m['name'])
    return rows, names


def openbmat_locked(rng, used):
    z = zipfile.ZipFile(N / 'ai_openbmat/AI-OpenBMAT.zip')
    names = set(z.namelist())

    def sources(n):
        m = json.loads(z.read(n[:-4] + '.json'))
        music = {Path(s['file']).stem for seg in m['segments'] for s in seg.get('sources', []) if s.get('type') == 'music'}
        other = {Path(s['file']).stem for seg in m['segments'] for s in seg.get('sources', []) if s.get('type') != 'music'}
        return music, other

    used_stems = {Path(r['path']).stem for r in used if 'openbmat' in (r['source'] + r.get('dataset', '')).lower()}
    used_music, used_other = set(), set()
    for art in ('ai', 'human'):
        for stem in used_stems:
            n = f'AI-OpenBMAT/{art}/{stem}.wav'
            if n in names:
                mu, ot = sources(n)
                used_music |= mu
                used_other |= ot
    rows = []
    for art in ('ai', 'human'):
        pool = []
        for n in sorted(x for x in names if x.startswith(f'AI-OpenBMAT/{art}/') and x.endswith('.wav')):
            stem = Path(n).stem
            if stem in used_stems:
                continue
            mu, ot = sources(n)
            # music lineage only: broadcast speech sources recur across AI and human excerpts alike (no class signal),
            # and excluding them too leaves 10 excerpts per class (22:14 run)
            if mu & used_music:
                continue
            pool.append((n, mu))
        picked = rng.sample(pool, min(OPENBMAT_PER_CLASS, len(pool)))
        print('openbmat', art, len(picked), 'from', len(pool), 'clean of', sum(1 for x in names if x.startswith(f'AI-OpenBMAT/{art}/') and x.endswith('.wav')), flush=True)
        for n, mu in sorted(picked):
            dst = PREP / 'openbmat' / art / Path(n).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(z.read(n))
            r = mk(f'locked_openbmat_{art}_{Path(n).stem}', dst, 'music', f"openbmat:{art}:{'|'.join(sorted(mu)) or Path(n).stem}",
                   'openbmat', 'ai_music_broadcast' if art == 'ai' else 'bonafide', 'AI-OpenBMAT', mp='1', mf='1' if art == 'ai' else '0')
            r.update(split='locked', npz=str(N / 'feats/locked' / f"{r['ID']}.npz"))
            rows.append(r)
    return rows


def main():
    rng = random.Random(20260929)
    used = used_rows()
    at_rows, at_names = atadd_locked(rng, used)
    ob_rows = openbmat_locked(rng, used)
    AR.A.joinpath('audio/eval_locked').mkdir(parents=True, exist_ok=True)
    # extract the selected eval members into audio/eval_locked (atadd_rows.extract writes into audio/<split>)
    import tarfile
    todo, dest = set(at_names), AR.A / 'audio' / 'eval_locked'
    todo = {n for n in todo if not (dest / n).exists()}
    with tarfile.open(AR.A / 'AT-ADD_T2_eval.tar', 'r|gz') as t:
        for member in t:
            base = member.name.rsplit('/', 1)[-1]
            if member.isfile() and base in todo:
                (dest / (base + '.part')).write_bytes(t.extractfile(member).read())
                (dest / (base + '.part')).rename(dest / base)
                todo.discard(base)
    assert not todo, f'{len(todo)} selected eval files not in the tar'
    rows = at_rows + ob_rows
    ids = [r['ID'] for r in rows]
    assert len(ids) == len(set(ids)) and not set(ids) & {r['ID'] for r in used}, 'locked IDs collide with used rows'
    for r in rows:
        r['audio_sha256'] = sha(r['path'])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'locked_rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'locked_manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'path'])
        w.writerows([r['ID'], r['path']] for r in rows)
    label_cols = ['ID', 'VOICE_PRESENT', 'VOICE_PRESENT_mask', 'MUSIC_PRESENT', 'MUSIC_PRESENT_mask', 'VOICE_FAKE',
                  'VOICE_FAKE_mask', 'MUSIC_FAKE', 'MUSIC_FAKE_mask', 'FILE_FAKE', 'FILE_FAKE_mask', 'group_id']
    from datetime import datetime
    from zoneinfo import ZoneInfo
    lock = {'created_kst': datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M'),
            'n': len(rows), 'ids_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'audio_sha256_of_sorted': hashlib.sha256('\n'.join(sorted(r['audio_sha256'] for r in rows)).encode()).hexdigest(),
            'labels_sha256': hashlib.sha256(json.dumps([[r[c] for c in label_cols] for r in sorted(rows, key=lambda r: r['ID'])]).encode()).hexdigest(),
            'counts': {f"{r['source']}|FILE={r['FILE_FAKE']}": 0 for r in rows}, 'rule': 'score once after the final candidate is fixed'}
    for r in rows:
        lock['counts'][f"{r['source']}|FILE={r['FILE_FAKE']}"] += 1
    (OUT / 'LOCK.json').write_text(json.dumps(lock, indent=1))
    print(json.dumps(lock, indent=1))


if __name__ == '__main__':
    main()
