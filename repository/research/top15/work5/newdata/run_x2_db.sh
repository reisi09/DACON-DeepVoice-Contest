#!/bin/bash
# X2 GPU phase v2 (review round 3): dump features for every expected ID that lacks a valid npz (bcast v2 rows, panel,
# locked v2), with worker exit codes and a wall-clock limit; then validate every npz (loadable, presence key) before the
# 4 DB-protocol member jobs; a failed job or an invalid npz stops the script (non-zero exit), nothing is masked.
set -euo pipefail
N=/root/proj/research/top15/work5/newdata; PY=/root/envs/data/bin/python
todo_manifest() {  # MANIFEST -> MANIFEST.todo with the IDs whose npz is missing or invalid
  $PY - "$1" <<'EOF'
import csv, sys
import numpy as np
from pathlib import Path
m = sys.argv[1]
rows = list(csv.DictReader(open(m.replace('_manifest.csv', '_rows_all.csv'))))
bad = []
for r in rows:
    p = Path(r['npz'])
    try:
        with np.load(p) as d:
            assert 'presence' in d.files and np.all(np.isfinite(d['presence']))
    except Exception:
        bad.append(r)
        if p.exists():   # an invalid npz is moved aside: the extractor skips any existing npz (review round 3b)
            q = p.parent / '_invalid'
            q.mkdir(exist_ok=True)
            p.rename(q / p.name)
with open(m + '.todo', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'path']); w.writerows([r['ID'], r['path']] for r in bad)
print(m, 'rows', len(rows), 'need dump', len(bad))
EOF
}
dump() {  # MANIFEST OUT: 2 workers, 45 min limit, both must exit 0
  local todo=$1.todo n; n=$(($(wc -l < "$todo") - 1)); [ "$n" -le 0 ] && { echo "dump $2: nothing to do"; return 0; }
  mkdir -p $2/_logs; rm -f $2/*.lock
  pids=()
  for i in 0 1; do timeout 2700 $PY $N/dump_features.py /root/s7pkg/x "$todo" $2 --shard $i/2 --device cuda:0 > $2/_logs/v2_w$i.log 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || { echo "dump worker $p failed"; exit 3; }; done
  echo "dump $2 done $(date +%T)"
}
echo "start $(date +%T)"
for m in /root/x2/rows/bcast_manifest.csv /root/x2/locked/locked_manifest.csv; do todo_manifest $m; done
dump /root/x2/rows/bcast_manifest.csv /root/newdata/feats/bcast
dump /root/x2/locked/locked_manifest.csv /root/newdata/feats/locked
for m in /root/x2/rows/bcast_manifest.csv /root/x2/locked/locked_manifest.csv; do
  todo_manifest $m; [ "$(($(wc -l < $m.todo) - 1))" -eq 0 ] || { echo "invalid npz remain for $m"; exit 4; }
done
cd /root/proj/research/top15/work5/w
STATUS=/root/x2/db_status.json PAR=4 NGPU=1 timeout 5400 $PY run_cmds.py /root/x2/jobs_x2.txt > /root/x2/db_runner.log 2>&1
echo "db jobs done $(date +%T)"
