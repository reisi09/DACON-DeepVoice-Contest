"""Root-level lineage check of the generated broadcast rows (review round 3b): every donor ID in parent_ids is mapped to
its source row's group_id; a union-find over rows and donor groups must have no component with both train and bpanel rows."""
import csv, json, sys
from collections import defaultdict
rows = list(csv.DictReader(open('/root/x2/rows/bcast_rows_all.csv')))
grp = {}
for f in ('/root/data_v4/out/bank/bank_trainval.csv', '/root/data_v4/out/bank/bank_test.csv', '/root/newdata/rows/new_rows.csv'):
    for r in csv.DictReader(open(f)):
        grp[r['ID']] = r['group_id']
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
missing = 0
for r in rows:
    for d in json.loads(r['parent_ids']):
        g = grp.get(d)
        if g is None:
            missing += 1; continue
        parent[find(('row', r['ID'].rsplit('_', 1)[0]))] = find(('g', g))
splits = defaultdict(set)
for r in rows:
    splits[find(('row', r['ID'].rsplit('_', 1)[0]))].add(r['split'])
cross = [k for k, v in splits.items() if len(v) > 1]
by = defaultdict(lambda: defaultdict(int))
for r in rows:
    by[r['split']][find(('row', r['ID'].rsplit('_', 1)[0]))] += 1
print(json.dumps({'rows': len(rows), 'donor_ids_without_group': missing, 'components_spanning_train_and_panel': len(cross),
                  'components': {s: len(v) for s, v in by.items()},
                  'largest_component_share': {s: round(max(v.values()) / sum(v.values()), 3) for s, v in by.items()}}, indent=1))
sys.exit(1 if cross or missing else 0)
