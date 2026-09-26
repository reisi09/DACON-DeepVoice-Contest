#!/usr/bin/env bash
# Pull all member score shards, merge by ID, check integrity, write merged CSV (+ a copy without xlsr2b columns).
set -euo pipefail
cd /Users/choiwuseck/Documents/ChatGPT/데이콘
D=_workspace/members_20260927; mkdir -p $D/raw
for f in $(ssh -n -p 31082 root@capella.cloud.vessl.ai 'ls /root/members_scores/'); do
  ssh -n -p 31082 root@capella.cloud.vessl.ai "cat /root/members_scores/$f" > $D/raw/$f
done
.venv/bin/python - <<'PY'
import csv, glob, math, collections
D='_workspace/members_20260927'
rows=[r for f in sorted(glob.glob(D+'/raw/scores_*.csv')) for r in csv.DictReader(open(f))]
want=[r['ID'] for r in csv.DictReader(open('/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/vessl/bundle/members_manifest.csv'))]
by=collections.defaultdict(list)
for r in rows: by[r['ID']].append(r)
cols=[c for c in rows[0] if c!='seconds']
bad=[i for i,rs in by.items() if any(r[c] in ('',None) or (c!='ID' and not math.isfinite(float(r[c]))) for r in rs for c in cols)]
incons=[i for i,rs in by.items() if len(rs)>1 and any(abs(float(a[c])-float(rs[0][c]))>1e-6 for a in rs for c in cols if c!='ID')]
missing=[i for i in want if i not in by]
print('raw rows',len(rows),'unique',len(by),'wanted',len(want),'missing',len(missing),'bad',len(bad),'inconsistent dups',len(incons),'dup ids',sum(len(v)>1 for v in by.values()))
assert not missing and not bad and not incons, 'integrity failure'
for name,keep in (('members_all.csv',cols+['seconds']),('members_1b.csv',[c for c in cols+['seconds'] if not c.startswith('xlsr2b')])):
    with open(f'{D}/{name}','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keep,extrasaction='ignore'); w.writeheader(); w.writerows(by[i][0] for i in want)
print('written', D)
PY
