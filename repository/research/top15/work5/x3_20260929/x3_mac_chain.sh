#!/bin/bash
# Mac side of X3 (review round 5c): every step checks its exit code and completion evidence, every wait has a deadline
# and the budget cap, and every stop kills only the recorded P1 process group, pauses x1 and confirms the stopped state.
# E3 stays on this Mac (AI-Hub never leaves it; only aggregates are printed).
P=/Users/choiwuseck/Documents/ChatGPT/데이콘; n=s7_sn; V=$P/_workspace/bin/vesslctl; WS=wsp-9683432r72m7
SSH="ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=30 -p 31082 root@capella.cloud.vessl.ai"
FLOOR=4.85   # balance at the cumulative $14 cap ($18.75 - $14.00) plus $0.10 for pausing
kst() { TZ=Asia/Seoul date '+%H:%M KST'; }
bal() {  # live VESSL balance; 3 tries, then nothing (fail-closed: budget() then fails; review round 5d)
  for i in 1 2 3; do
    python3 -c "import json, subprocess, sys; print(float(json.loads(subprocess.run([sys.argv[1], 'billing', 'show', '-o', 'json'], capture_output=True, timeout=30).stdout)['balance']))" "$V" 2>/dev/null && return 0
    sleep 10
  done
  return 1
}
stop_x1() {
  $SSH 'g=$(cat /root/x3/p1.pgid 2>/dev/null) && kill -- -$g 2>/dev/null; true'
  for i in $(seq 10); do
    $V workspace pause $WS > /dev/null 2>&1
    sleep 30
    st=$($V workspace show $WS -o json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['workspace']['state'])" 2>/dev/null)
    [ "$st" = stopped ] && { echo "x1 stopped $(kst), balance \$$(bal) ($1)"; exit 0; }
  done
  echo "X1 PAUSE NOT CONFIRMED (state '$st') $(kst) ($1)"; exit 2
}
budget() { local b; b=$(bal) || { echo "balance unavailable $(kst)"; return 1; }; python3 -c "import sys; sys.exit(0 if $b - $1 >= $FLOOR else 1)"; }   # $1: cost still needed
at() { TZ=Asia/Seoul date -j -f '%Y-%m-%d %H:%M' "$1" +%s; }
wait_for() {  # REMOTE_CONDITION DEADLINE_EPOCH LABEL: 60 s polls; ssh failures only retry; deadline or cap stops x1
  budget 0 || stop_x1 "budget cap before: $3"
  until $SSH "$1" 2>/dev/null; do
    [ "$(date +%s)" -lt "$2" ] || stop_x1 "deadline: $3"
    budget 0 || stop_x1 "budget cap while waiting: $3"
    sleep 60
  done
}
SETSID=(python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])')   # macOS has no setsid(1)
guarded() {  # LABEL DEADLINE_EPOCH CMD...: deadline and budget checked BEFORE the start and every 30 s; CMD runs in its own
  local label=$1 dl=$2 pid; shift 2   # session, so a violation kills its whole process group; returns CMD's exit code
  [ "$(date +%s)" -lt "$dl" ] || stop_x1 "deadline before: $label"
  budget 0 || stop_x1 "budget cap before: $label"
  [ "$(date +%s)" -lt "$dl" ] || stop_x1 "deadline before: $label"   # again: the balance query may take up to 90 s (review 5f)
  "${SETSID[@]}" "$@" & pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    if [ "$(date +%s)" -ge "$dl" ] || ! budget 0; then kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null; stop_x1 "deadline or budget cap during: $label"; fi
  done
  wait $pid
}
e3_step() {   # run as: guarded ... bash -c e3_step (exported below)
  .venv/bin/python research/top15/work5/newdata/score_e3.py _workspace/nd_runs_x3 _workspace/nd_models _workspace/aihub_prep/aihub_rows.csv \
    _workspace/nd_runs_e3_x3 > _workspace/x3_e3_score.log 2>&1 || { echo "score_e3 failed"; return 1; }
  (cd research/top15/work5/newdata && NEW_ROWS=$P/_workspace/aihub_prep/aihub_rows.csv EVAL_SPLITS=e3 ../../../../.venv/bin/python new_eval.py \
    $P/_workspace/nd_runs_e3_x3 s7_new s7_new $n > $P/_workspace/x3_e3_eval.log 2>&1) || { echo "E3 new_eval failed"; return 1; }
}
export -f e3_step bal kst; export P n V
cd $P
echo "chain start $(kst), balance \$$(bal)"
wait_for "grep -q 'EVAL_OK\|EVAL_FAILED\|DB_FAILED' /root/x3/after_db.out" "$(at '2026-09-29 02:30')" gates
$SSH "cat /root/x3/after_db.out; tail -2 /root/x3/eval.out"
$SSH "grep -q EVAL_OK /root/x3/after_db.out" || stop_x1 "gates did not complete"
adopt=$($SSH "python3 -c \"import json; print(json.load(open('/root/x3/x3_gates.json'))['adoption'])\"")
echo "gates: $adopt"
case "$adopt" in pending*) ;; *) stop_x1 "gates rejected";; esac
# E3 on this Mac
mkdir -p _workspace/nd_runs/$n _workspace/nd_models/$n _workspace/nd_runs_x3
guarded download "$(at '2026-09-29 02:40')" scp -q -o ConnectTimeout=20 -P 31082 "root@capella.cloud.vessl.ai:/root/nd_runs/$n/*.json" _workspace/nd_runs/$n/ || stop_x1 "run download failed"
guarded download "$(at '2026-09-29 02:40')" scp -q -o ConnectTimeout=20 -P 31082 "root@capella.cloud.vessl.ai:/root/nd_models/$n/*" _workspace/nd_models/$n/ || stop_x1 "model download failed"
ln -sfn ../nd_runs/s7_new _workspace/nd_runs_x3/s7_new; ln -sfn ../nd_runs/$n _workspace/nd_runs_x3/$n
rm -rf _workspace/nd_runs_e3_x3
guarded E3 "$(at '2026-09-29 02:45')" bash -c e3_step || stop_x1 "E3 scoring failed"
.venv/bin/python - "$P/_workspace/nd_runs_e3_x3/new_eval_e3.json" <<'PY' || stop_x1 "E3 rejected"
import json, sys
d = json.load(open(sys.argv[1]))
a, b = d['s7_new']['E2_by_source'], d['s7_sn']['E2_by_source']
inc = [b[s][k] - a[s][k] for s in a for k in ('FILE_real_fpr', 'VOICE_real_fpr') if k in a[s]]
assert inc and set(a) == set(b), 'no E3 aggregates'
print('E3 aggregate real FPR increase, worst over', len(a), 'sources x FILE/VOICE:', round(max(inc), 4), '(limit 0.01)')
sys.exit(0 if max(inc) <= .01 else 1)
PY
# P1: exact completion line and every head file
wait_for "grep -qE '^done $n [0-9:]+\$|failed' /root/final_$n.out || ! pgrep -f '[f]inal_new.sh $n' > /dev/null" "$(at '2026-09-29 03:05')" P1
$SSH "cat /root/final_$n.out"
$SSH "grep -qE '^done $n [0-9:]+\$' /root/final_$n.out && cd /root/proj/_workspace/final_${n}_20260928 && ls file_ensemble.json component_fusion.json music_layers.json music_eat_layers.json music_sslam_layers.json voice_wavlm_layer.json {music,voice,voice_ad,wavlm}_4{1,2,3}.pt > /dev/null" \
  || stop_x1 "P1 failed or incomplete"
# locked set: once, candidate fixed, G6 must be PASS against both references
guarded locked "$(at '2026-09-29 03:05')" $SSH "cd /root/proj/research/top15/work5/newdata && /root/envs/data/bin/python score_locked.py /root/x2/locked/locked_rows.csv /root/x2/locked/LOCK_v2.json /root/x3/locked_scores.json $n=/root/proj/_workspace/final_${n}_20260928 S7=/root/proj/_workspace/final_s7_20260928 s7_new=/root/proj/_workspace/final_s7_new_20260928 > /root/x3/locked_scores.log 2>&1" \
  || stop_x1 "locked scorer failed"
$SSH "grep -v -i warn /root/x3/locked_scores.log | tail -45"
g6=$($SSH "python3 -c \"import json; print(json.load(open('/root/x3/locked_scores.json'))['G6'])\"")
echo "locked G6: $g6"
[ "$g6" = PASS ] || stop_x1 "locked set G6 $g6"
note="S7 recipe retrained on bank v4 P1 rows + new rows + 1,500 SONICS song crops (X3: Suno chirp-v3.5 / Udio udio-120s); s7 weights; FILE fixed 0/9/1"
guarded package "$(at '2026-09-29 03:05')" bash -c 'PORT=31082 research/top15/work5/newdata/pkg_chain.sh $n datav4_${n}_newdata_sonics "$0" > submissions/DataV4_$n.chain.log 2>&1' "$note" \
  || stop_x1 "package chain failed"
tail -4 submissions/DataV4_$n.chain.log
budget 0.75 || stop_x1 "PACKAGE READY, CUDA check NOT RUN (budget cap)"
guarded CUDA "$(at '2026-09-29 03:08')" $SSH "cd /root/pkg_$n && rm -rf changed v && mkdir -p changed && tar xzf changed.tgz -C changed && cd /root/s8pkg && /root/envs/data/bin/python verify_s8_vessl.py /root/pkg_$n/submit.json /root/s7pkg/submit.zip /root/pkg_$n/changed /root/s8pkg/s8par /root/pkg_$n/v cuda > /root/pkg_$n/verify_cuda.log 2>&1 && test -f /root/pkg_$n/v/TREE_OK" \
  || stop_x1 "CUDA check failed"
$SSH "cat /root/pkg_$n/v/TREE_OK; grep -E 'network_connections_forbidden|file_independence' /root/pkg_$n/verify_cuda.log"
echo "X3 SUBMITTABLE: submissions/DataV4_$n/submit.zip"
stop_x1 "X3 chain complete"
