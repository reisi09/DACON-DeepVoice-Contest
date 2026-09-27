#!/usr/bin/env bash
# Resume the S6 candidate-encoder extraction (choice_lock.json) on H100x4 after the 18:14 KST stop: weights already in
# /root/models/s6 (sha-checked against the first run's weights_sha.txt, no re-download), finished npz are skipped.
set -euo pipefail
PY=/root/envs/data/bin/python; D=/root/s6; M=/root/models/s6; OUT=/root/s6_cache; LOG=/root/s6_logs
NGPU=${NGPU:-4}; PER=${PER:-3}; MINUTES=${MINUTES:-80}; N=$((NGPU*PER))
mkdir -p $LOG/run1 && mv $LOG/shard_*.log $LOG/run1/ 2>/dev/null || true
sha256sum $M/*/model.safetensors $M/*/pytorch_model.bin > $LOG/weights_sha_resume.txt
cmp $LOG/weights_sha.txt $LOG/weights_sha_resume.txt
find $OUT -name '*.tmp.npz' -delete
DEADLINE=$(( $(date +%s) + MINUTES*60 ))
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY $D/extract_s6.py --ids $D/manifest_s6.txt --stems /root/stems --models $M --code-root /root/ft/code \
    --eat-core /root/newbb/vendor --out $OUT --shard $k --shards $N --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
