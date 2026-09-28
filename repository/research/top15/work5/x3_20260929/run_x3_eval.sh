#!/bin/bash
# X3 gates (MODE=x3, x2_gates.py frozen before any X3 result): control s7_new vs candidate s7_sn, both DB protocol
set -euo pipefail
N=/root/proj/research/top15/work5/newdata; PY=/root/envs/data/bin/python
rm -rf /root/x3/x3_gates.json /root/x3/x3_gates.log /root/x2/seed_runs/x3   # fresh results only (review round 5c)
python3 -c "import json,sys; s=json.load(open('/root/x3/db_status.json')); bad={k:v for k,v in s.items() if v['exit']!=0}; print('db status', s); sys.exit(1 if bad or len(s)<4 else 0)"
mkdir -p /root/x3/runs_sel && cd /root/x3/runs_sel && ln -sfn /root/nd_runs/s7_new s7_new && ln -sfn /root/nd_runs/s7_sn s7_sn
cd $N && MODE=x3 $PY x2_gates.py /root/x3/runs_sel s7_new s7_sn /root/newdata/rows/new_rows.csv NONE NONE \
  /root/x2/integrity.json /root/nd_eval/linked_newtest.json /root/x3/x3_gates.json > /root/x3/x3_gates.log 2>&1
echo "gates done $(date +%T)"
python3 -c "import json; d=json.load(open('/root/x3/x3_gates.json')); print(json.dumps(d['verdicts'])); print(d['adoption'])"
