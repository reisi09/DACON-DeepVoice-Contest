#!/bin/bash
# X2 evaluation on the CPU workspace after the DB jobs: panel scoring (saved DB models), then the pre-registered gates
set -euo pipefail
N=/root/proj/research/top15/work5/newdata; PY=/root/envs/data/bin/python
python3 -c "import json,sys; s=json.load(open('/root/x2/db_status.json')); bad={k:v for k,v in s.items() if v['exit']!=0}; print('db status', s); sys.exit(1 if bad or len(s)<4 else 0)"
mkdir -p /root/x2/runs_sel && cd /root/x2/runs_sel && ln -sfn /root/nd_runs/s7_new s7_new && ln -sfn /root/nd_runs/s7_bc s7_bc
rm -rf /root/x2/nd_bpanel
cd $N && $PY score_split.py /root/x2/runs_sel /root/nd_models /root/x2/rows/bcast_panel.csv /root/x2/nd_bpanel bpanel > /root/x2/score_bpanel.log 2>&1
echo "bpanel scored $(date +%T)"
MODE=x2 $PY x2_gates.py /root/x2/runs_sel s7_new s7_bc /root/newdata/rows/new_rows.csv /root/x2/nd_bpanel /root/x2/rows/bcast_panel.csv \
  /root/x2/integrity.json /root/nd_eval/linked_newtest.json /root/x2/x2_gates.json > /root/x2/x2_gates.log 2>&1
echo "gates done $(date +%T)"
python3 -c "import json; d=json.load(open('/root/x2/x2_gates.json')); print(json.dumps(d['verdicts'])); print(d['adoption'])"
