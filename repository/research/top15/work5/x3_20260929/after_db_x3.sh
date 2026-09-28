#!/bin/bash
# after the X3 DB jobs (review round 5c): P1 for the candidate in its own session (PGID recorded, killed on rejection);
# the gates run alongside. Outcome lines: DB_FAILED, EVAL_OK or EVAL_FAILED (never a bare 'done').
until grep -q "db jobs done" /root/x3/x3_run.log; do
  pgrep -f "[r]un_x3_db.sh" > /dev/null || { echo "DB_FAILED: db chain ended without 'db jobs done' $(date +%T)"; exit 1; }
  sleep 30
done
cd /root
rm -f /root/x3/p1.pgid
# the new session's leader writes its own PID (= PGID) and execs final_new.sh (review round 5d)
setsid bash -c 'echo $$ > /root/x3/p1.pgid; export NEW_ROWS=/root/newdata/rows/new_rows.csv,/root/x3/rows/sonics_rows.csv; exec /root/final_new.sh s7_sn s7 0 0' > /root/final_s7_sn.out 2>&1 < /dev/null &
for i in $(seq 20); do [ -s /root/x3/p1.pgid ] && break; sleep 0.5; done
g=$(cat /root/x3/p1.pgid); echo "P1 s7_sn started pgid $g (ps pgid $(ps -o pgid= -p $g | tr -d ' ')) $(date +%T)"
if bash /root/x3/run_x3_eval.sh > /root/x3/eval.out 2>&1; then echo "EVAL_OK $(date +%T)"; else echo "EVAL_FAILED $(date +%T)"; fi
