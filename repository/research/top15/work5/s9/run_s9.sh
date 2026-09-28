#!/usr/bin/env bash
# S9 raw-mix EAT/MERT features (extract_s9.py) for the 42,039 backbone rows on H100xN. Single launch at a time (lock);
# per-worker exit codes are collected; DONE is written only if every worker exits 0 AND validate_s9.py passes.
set -euo pipefail
PY=/root/envs/data/bin/python; D=/root/s9; OUT=/root/s9_cache; LOG=/root/s9_logs; TREE=/root/s8pkg/v2/tree
NGPU=${NGPU:-4}; PER=${PER:-4}; MINUTES=${MINUTES:-120}; N=$((NGPU*PER))
mkdir -p $LOG $OUT
exec 9>$LOG/launch.lock; flock -n 9 || { echo "another launch is running"; exit 1; }
rm -f $LOG/DONE $LOG/FAILED $LOG/workers.txt $LOG/validate.txt
find $OUT -name '*.tmp.npz' -delete   # no worker can be alive under the lock
DEADLINE=$(( $(date +%s) + MINUTES*60 ))
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pids=()
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY $D/extract_s9.py --ids /root/s6/manifest_s6.txt --manifest /root/ft/backbone_manifest.csv \
    --audio /root/data_v4/out --assets $TREE/model/assets --runtime $TREE/model/runtime --bb-assets /root/bb_assets \
    --code-root /root/ft/code --out $OUT --shard $k --shards $N --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); done
echo "workers failed: $fail" > $LOG/workers.txt
if [ $fail -eq 0 ] && $PY $D/validate_s9.py > $LOG/validate.txt 2>&1; then
  echo "ok $(date -u +%FT%TZ)" > $LOG/DONE
else
  echo "FAILED $(date -u +%FT%TZ)" > $LOG/FAILED
  exit 1
fi
