#!/usr/bin/env bash
# SingMOS-Pro wavs already at /root/singmos_pro/wav (uploaded from the local parquet decode; bytes equal the repo files) -> voice-route features.
# Copy extract_voice_feats.py, singmos_manifest.csv and this file to /root/work4/ (not via /shared).
set -euo pipefail
PY=/root/envs/data/bin/python; W=/root/work4; OUT=/root/singmos_feats; LOG=/root/work4_logs
A=/root/dacon_existing9504_h100_20260920/assets   # baseline/ with HTDemucs + DF-Arena (sha-checked against the package assets)
N=${N:-3}
mkdir -p $LOG $OUT
ls /root/singmos_pro/wav | wc -l > $LOG/wav_count.txt
head -3 $W/singmos_manifest.csv > $LOG/smoke.csv
$PY $W/extract_voice_feats.py --manifest $LOG/smoke.csv --input /root/singmos_pro --assets $A --antideepfake /root/models/xlsr1b \
  --code-root /root/ft/code --out $LOG/smoke_out > $LOG/smoke.log 2>&1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in $(seq 0 $((N-1))); do
  $PY $W/extract_voice_feats.py --manifest $W/singmos_manifest.csv --input /root/singmos_pro --assets $A --antideepfake /root/models/xlsr1b \
    --code-root /root/ft/code --out $OUT --shard $k --shards $N > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
