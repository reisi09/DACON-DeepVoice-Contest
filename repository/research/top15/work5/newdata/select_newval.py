"""Round-16 selection (pre-registered): s7_new vs mix voice s7_new / music W_new on newval (never used for training or
early stopping by either), lineage-linked rows removed; then the E1 / E2 vetoes with a paired group bootstrap vs s7_new.
Adopt the mix iff newval dADS >= +0.002 and CI low > 0, and E1 >= 0, E2-unlinked >= 0, E2-open >= -0.002."""
import csv, json, os, sys, zipfile
from pathlib import Path
rows = list(csv.DictReader(open('/root/newdata/rows/new_rows.csv')))
extra = list(csv.DictReader(open('/root/newdata/rows/openbmat_extra_rows.csv')))
train = [r for r in rows if r['split'] == 'train'] + extra
tg = {r['group_id'] for r in train}
z = zipfile.ZipFile('/root/newdata/ai_openbmat/AI-OpenBMAT.zip')
def tracks(r):
    art = 'ai' if '/ai/' in r['path'] else 'human'
    m = json.loads(z.read(f"AI-OpenBMAT/{art}/{r['path'].rsplit('/', 1)[-1][:-4]}.json"))
    return {Path(s['file']).stem for seg in m['segments'] for s in seg.get('sources', []) if s.get('type') == 'music'}
train_tracks = set().union(*[tracks(r) for r in train if r['source'] == 'openbmat'])
linked = {r['ID'] for r in rows if r['split'] in ('newval', 'newtest') and
          (r['group_id'] in tg or (r['source'] == 'openbmat' and tracks(r) & train_tracks))}
linked_newtest = set(json.load(open('/root/nd_eval/linked_newtest.json')))
print('linked newval', sum(1 for r in rows if r['split'] == 'newval' and r['ID'] in linked), 'linked newtest (leak_check)', len(linked_newtest), flush=True)

def run(runs, split, configs, subsets):
    sys.argv = ['new_eval.py', runs, 's7_new', *configs]
    os.environ['NEW_ROWS'] = '/root/newdata/rows/new_rows.csv'; os.environ['EVAL_SPLITS'] = split
    g = {'__file__': '/root/proj/research/top15/work5/newdata/new_eval.py', '__name__': 'x'}
    src = open(g['__file__']).read()
    exec(compile(src[:src.index("preds = {c: predictions(c) for c in CONFIGS}")] + "preds = {c: predictions(c) for c in CONFIGS}\n", g['__file__'], 'exec'), g)
    F, safe = g['F'], g['safe']
    for name, rows_, pick in subsets(g):
        boot = F.Boot(rows_, reps=2000)
        base = pick(g['preds']['s7_new'])
        line = f'{name:22s} n={len(rows_):5d} | s7_new {F.ads_of(rows_, base)["ads"]:.5f}'
        for c in configs:
            if c == 's7_new':
                continue
            p = pick(g['preds'][c]); d = boot.delta(base, p)
            line += f' | {c} {F.ads_of(rows_, p)["ads"]:.5f} ({d["delta"]:+.4f} [{d["ci95"][0]:+.4f},{d["ci95"][1]:+.4f}] p>0 {d["p_gt_0"]:.3f})'
        print(line, flush=True)

run('/root/nd_newval', 'newval', ['s7_new', 'mix_s7_new_W_new'],
    lambda g: [('newval_unlinked', [r for r in g['newtest'] if r['ID'] not in linked],
                lambda p, g=g: [x for r, x in zip(g['newtest'], p['newtest']) if r['ID'] not in linked])])
run('/root/nd_virtual', 'newtest', ['s7_new', 'mix_s7_new_W_new', 'mix_s7_new_W_newB'],
    lambda g: [('E1', g['test'], lambda p: p['test']),
               ('E2_unlinked', [r for r in g['newtest'] if r['ID'] not in linked_newtest],
                lambda p, g=g: [x for r, x in zip(g['newtest'], p['newtest']) if r['ID'] not in linked_newtest]),
               ('E2_open_unlinked', [r for r in g['newtest'] if r['ID'] not in linked_newtest and r['dataset'] != 'atadd'],
                lambda p, g=g: [x for r, x in zip(g['newtest'], p['newtest']) if r['ID'] not in linked_newtest and r['dataset'] != 'atadd'])])
