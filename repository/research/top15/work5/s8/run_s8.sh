#!/usr/bin/env bash
# S8 candidate features (extract_s8.py) for the 42,039 backbone rows on H100x4. Resumable (finished npz are skipped).
set -euo pipefail
PY=/root/envs/data/bin/python; D=/root/s8; OUT=/root/s8_cache; LOG=/root/s8_logs
NGPU=${NGPU:-4}; PER=${PER:-4}; MINUTES=${MINUTES:-200}; N=$((NGPU*PER))
mkdir -p $LOG $OUT
find $OUT -name '*.tmp.npz' -delete
DEADLINE=$(( $(date +%s) + MINUTES*60 ))
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS="--manifest /root/ft/backbone_manifest.csv --audio /root/data_v4/out --stems /root/stems --models /root/models --assets /root/bb_assets
      --code-root /root/ft/code --eat-core /root/newbb/vendor --s6-code /root/s6 --out $OUT"
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY $D/extract_s8.py --ids /root/s6/manifest_s6.txt $ARGS --shard $k --shards $N --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
