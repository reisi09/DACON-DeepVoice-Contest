#!/usr/bin/env bash
# Pre-declared fusion protocol: select on AD-unseen rows (val/test mean) with heads never trained on test (DB),
# check no harm on all rows, report 2B variant for reference, then transfer weights to P1 heads (tables refit on P1 val).
set -euo pipefail
cd /Users/choiwuseck/Documents/ChatGPT/데이콘
M=_workspace/members_20260927; F=_workspace/fusion_20260927; DB=_keep_best_20260926/train_duration_balanced; P1=_workspace/p1_refit_cpu
mkdir -p $F
run() { .venv/bin/python data_v4/fusion_fit.py "$@" 2>&1 | grep -v -i warn | grep -E "^(baseline|fused)_|voice alone" ; }
echo "## 1 DB unseen 1B select valtest"; run --heads $DB --members $M/members_1b.csv --out $F/db_unseen_1b --unseen --select valtest
echo "## 2 DB all rows, weights of 1";   run --heads $DB --members $M/members_1b.csv --out $F/db_all_w1 --weights $F/db_unseen_1b/component_fusion.json
echo "## 3 DB unseen with 2B (reference)"; run --heads $DB --members $M/members_all.csv --out $F/db_unseen_all --unseen --select valtest
echo "## 4 P1 unseen, weights of 1";     run --heads $P1 --members $M/members_1b.csv --out $F/p1_unseen_w1 --unseen --weights $F/db_unseen_1b/component_fusion.json
echo "## 5 P1 all rows, weights of 1";   run --heads $P1 --members $M/members_1b.csv --out $F/p1_all_w1 --weights $F/db_unseen_1b/component_fusion.json
for d in db_unseen_1b db_unseen_all; do python3 -c "import json;r=json.load(open('$F/$d/fusion_report.json'));print('$d', 'voice', r['chosen_voice'], 'music', r['chosen_music'], 'rows', r['rows'])"; done
