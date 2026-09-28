#!/bin/bash
# X3 GPU phase (the X2 v2 runner): dump features for every SONICS row that lacks a valid npz, with worker exit codes and
# a wall-clock limit; validate every npz (all trainer keys, finite encoder features, one window count; review round 5),
# set the crop voice labels from the stored presence (sonics_rows.py relabel), then the 4 DB-protocol member jobs; a failed
# step or an invalid npz stops the script (non-zero exit), nothing is masked.
set -euo pipefail
N=/root/proj/research/top15/work5/newdata; PY=/root/envs/data/bin/python
todo_manifest() {  # MANIFEST -> MANIFEST.todo with the IDs whose npz is missing or invalid
  $PY - "$1" <<'EOF'
import csv, sys
import numpy as np
from pathlib import Path
m = sys.argv[1]
FINITE = ('mert_layers', 'eat_backing', 'sslam_pt_backing', 'wavlm_voice', 'voice_embeddings', 'ad_embeddings', 'presence')
KEYS = FINITE + ('samples', 'window_truth', 'layers', 'eat_layers', 'sslam_layers', 'wavlm_layers', 'voice_silent')
rows = list(csv.DictReader(open(m.replace('_manifest.csv', '_rows_all.csv'))))
bad = []
for r in rows:
    p = Path(r['npz'])
    try:
        with np.load(p) as d:
            assert set(KEYS) <= set(d.files) and all(np.all(np.isfinite(d[k])) for k in FINITE)
            assert len({len(d[k]) for k in ('window_truth', 'mert_layers', 'eat_backing', 'sslam_pt_backing', 'wavlm_voice')}) == 1
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
for m in /root/x3/rows/sonics_manifest.csv; do todo_manifest $m; done
dump /root/x3/rows/sonics_manifest.csv /root/newdata/feats/sonics
for m in /root/x3/rows/sonics_manifest.csv; do
  todo_manifest $m; [ "$(($(wc -l < $m.todo) - 1))" -eq 0 ] || { echo "invalid npz remain for $m"; exit 4; }
done
(cd $N && $PY sonics_rows.py relabel /root/x3/rows)
[ "$(($(wc -l < /root/x3/rows/sonics_rows.csv) - 1))" -eq 1500 ] || { echo "sonics_rows.csv must hold 1500 rows"; exit 5; }
cd /root/proj/research/top15/work5/w
STATUS=/root/x3/db_status.json PAR=4 NGPU=1 timeout 5400 $PY run_cmds.py /root/x3/jobs_x3.txt > /root/x3/db_runner.log 2>&1
echo "db jobs done $(date +%T)"
