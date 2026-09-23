#!/usr/bin/env bash
# Phase C (cache only, no feature re-extraction): waits for PHASE_B_DONE, then
#   1) retrains the heads with --duration-balance (length-shortcut control) + its presence/ensemble layer,
#   2) scores the held-out test with the previous packages' heads, the phase B heads and the balanced heads,
#      auditing the length shortcut for each (compare_runs.py),
#   3) zips the comparison into the object volume.
#   setsid bash -c "nohup bash /shared/data-v4-20260923/code/data_v4/phase_c.sh >> .../phase_c.log 2>&1 < /dev/null &"
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
S=/shared/data-v4-20260923; LOGS=$S/run_logs
DATA=${DATA:-/data/data_v4}; OUT=${OUT:-/root/data_v4/out}; BANK=$OUT/bank; TRAIN=${TRAIN:-/root/data_v4/train}
BAL=${BAL:-/root/data_v4/train_bal}; CMP=${CMP:-/root/data_v4/compare}; PREV=${PREV:-/data/prev_heads}
PY=/root/envs/data/bin/python
step() { echo "[$(date -u +%FT%TZ)] == $*" | tee -a "$LOGS/phase_c.steps"; }
until grep -q PHASE_B_DONE "$LOGS/phase_b.steps" 2>/dev/null; do
  grep -q FAILED "$LOGS/phase_b.steps" 2>/dev/null && { step "phase B failed, not starting"; exit 1; }
  sleep 60
done
# own code copy: phase B's /data/code stays untouched
rm -rf /data/code_c && cp -r "$S/code" /data/code_c && cd /data/code_c
mkdir -p "$BAL" "$CMP"
step train_balanced
$PY full_bank_train.py --manifest "$BANK/bank_trainval.csv" --cache "$DATA/cache_tv" --output "$BAL" \
  --max-epochs 50 --patience 5 --duration-balance > "$LOGS/train_balanced.log" 2>&1
step file_ensemble_balanced
$PY data_v4/file_ensemble.py --bank "$BANK/bank_trainval.csv" --test-bank "$BANK/bank_test.csv" --cache "$DATA/cache_tv" \
  --test-cache "$DATA/cache_test" --train-dir "$BAL" --presence-head "$TRAIN/presence_head.json" > "$LOGS/file_ensemble_balanced.log" 2>&1
step compare
$PY data_v4/compare_runs.py --test-bank "$BANK/bank_test.csv" --cache "$DATA/cache_test" \
  --run "prev_existing9504=$PREV/Existing9504_MIL" --run "prev_balanced7144=$PREV/Balanced7144_MIL" \
  --run "new=$TRAIN" --run "new_duration_balanced=$BAL" --out "$CMP/compare_report.json" > "$LOGS/compare.log" 2>&1
step results_zip
TS=$(date -u +%Y%m%dT%H%MZ); Z=$S/results/data_v4_phase_c_$TS.zip; mkdir -p "$S/results"
$PY - "$Z" "$BAL" "$CMP" "$LOGS" <<'EOF'
import sys, zipfile, pathlib
z, bal, cmp, logs = sys.argv[1], *map(pathlib.Path, sys.argv[2:])
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(bal.glob('*')):
        if p.is_file() and not p.name.endswith('_resume.pt'): zf.write(p, 'train_duration_balanced/' + p.name)
    for p in sorted(cmp.glob('*')): zf.write(p, 'compare/' + p.name)
    for n in ('train_balanced.log', 'file_ensemble_balanced.log', 'compare.log', 'phase_c.steps'):
        if (logs / n).exists(): zf.write(logs / n, 'logs/' + n)
print(z)
EOF
echo "$Z" > "$LOGS/RESULTS_ZIP_C"
step PHASE_C_DONE
