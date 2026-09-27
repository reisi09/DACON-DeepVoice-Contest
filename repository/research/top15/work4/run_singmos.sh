#!/usr/bin/env bash
# SingMOS-Pro (rev 9e83332, CC BY 4.0) download + voice-route features (runtime path) on H100x1. choice_lock.json variant S.
# Copy extract_voice_feats.py, singmos_manifest.csv and this file to /root/work4/ (not via /shared).
set -euo pipefail
PY=/root/envs/data/bin/python; W=/root/work4; OUT=/root/singmos_feats; LOG=/root/work4_logs
A=/root/dacon_existing9504_h100_20260920/assets   # baseline/ with HTDemucs + DF-Arena (sha-checked against the package assets)
N=${N:-6}
mkdir -p $LOG $OUT
$PY - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('TangRain/SingMOS-Pro', repo_type='dataset', revision='9e83332dc481752ce17146ced86aa68da0bd84aa',
                        local_dir='/root/singmos_pro'), flush=True)
PY
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
