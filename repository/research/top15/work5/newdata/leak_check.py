"""Codex r14 item 2: newtest rows linked to training rows (AT-ADD speaker/source groups; AI-OpenBMAT ANY shared music track,
incl. openbmat_extra), then E2 recomputed on the unlinked rows for s7_ref / W_new / W_newB / s7_new / s7_new_ob."""
import csv, json, os, sys, zipfile
from collections import Counter
from pathlib import Path
rows = list(csv.DictReader(open('/root/newdata/rows/new_rows.csv')))
extra = list(csv.DictReader(open('/root/newdata/rows/openbmat_extra_rows.csv')))
train = [r for r in rows if r['split'] in ('train', 'newval')] + extra
test = [r for r in rows if r['split'] == 'newtest']
tg = {r['group_id'] for r in train}
linked = set()
c = Counter()
for r in test:
    if r['dataset'] == 'atadd' and r['group_id'] in tg:
        linked.add(r['ID']); c['atadd_group'] += 1
z = zipfile.ZipFile('/root/newdata/ai_openbmat/AI-OpenBMAT.zip')
def tracks(name):
    m = json.loads(z.read(f'AI-OpenBMAT/{name}'[:-4] + '.json'))
    return {Path(s['file']).stem for seg in m['segments'] for s in seg.get('sources', []) if s.get('type') == 'music'}
def zname(r):
    art = 'ai' if '/ai/' in r['path'] else 'human'
    return f"{art}/{r['path'].rsplit('/', 1)[-1]}"
train_tracks = set()
for r in train:
    if r['source'] == 'openbmat':
        train_tracks |= tracks(zname(r))
for r in test:
    if r['source'] == 'openbmat' and tracks(zname(r)) & train_tracks:
        linked.add(r['ID']); c['openbmat_track'] += 1
print('newtest', len(test), 'linked', len(linked), dict(c), 'atadd newtest', sum(r['dataset'] == 'atadd' for r in test), 'openbmat newtest', sum(r['source'] == 'openbmat' for r in test))
json.dump(sorted(linked), open('/root/nd_eval/linked_newtest.json', 'w'))
sys.argv = ['new_eval.py', '/root/nd_eval_all', 's7_ref', 's7_ref', 'W_new', 'W_newB', 's7_new', 's7_new_ob']
os.environ.setdefault('NEW_ROWS', '/root/newdata/rows/new_rows.csv'); os.environ.setdefault('EVAL_SPLITS', 'newtest')
__file__ = '/root/proj/research/top15/work5/newdata/new_eval.py'
src = open(__file__).read()
exec(compile(src[:src.index("preds = {c: predictions(c) for c in CONFIGS}")] + "preds = {c: predictions(c) for c in CONFIGS}\n", __file__, 'exec'))
for name, keep in (('E2_all', [i for i in range(len(newtest))]), ('E2_unlinked', [i for i, r in enumerate(newtest) if r['ID'] not in linked]),
                   ('E2_unlinked_non_atadd', [i for i, r in enumerate(newtest) if r['ID'] not in linked and r['dataset'] != 'atadd']),
                   ('E2_unlinked_atadd', [i for i, r in enumerate(newtest) if r['ID'] not in linked and r['dataset'] == 'atadd'])):
    rs = [newtest[i] for i in keep]; boot = safe(F.Boot, rs)
    line = f'{name:24s} n={len(keep):5d} '
    for c in CONFIGS:
        p = [preds[c]['newtest'][i] for i in keep]; a = safe(F.ads_of, rs, p)
        d = boot and c != REF and safe(boot.delta, [preds[REF]['newtest'][i] for i in keep], p)
        line += f"| {c} {a['ads'] if a else float('nan'):.4f}" + (f" ({d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}])" if d else '') + ' '
    print(line, flush=True)
