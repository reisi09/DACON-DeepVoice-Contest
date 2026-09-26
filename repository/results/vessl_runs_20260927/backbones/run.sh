#!/usr/bin/env bash
# New-backbone features for trained heads (AntiDeepfake-1B voice-stem window embeddings + MERT multi-layer backing-stem
# means) for all data v4 rows (non-AI-Hub data only). Hard 150-min extraction deadline (user-approved ~$9 budget).
# Launch: setsid bash -c "nohup bash /shared/backbones-20260927/run.sh > /shared/backbones-20260927/logs/run.log 2>&1 < /dev/null &"
set -euo pipefail
S=/shared/backbones-20260927; PY=/root/envs/data/bin/python; M=/root/models; OUT=/root/backbone_cache; LOG=$S/logs
ASSETS=/shared/members-20260927/assets
mkdir -p $LOG $OUT
sha256sum $ASSETS/mert/pytorch_model.bin > $LOG/mert_sha.txt
DEADLINE=$(( $(date +%s) + 150*60 ))
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=5
for k in $(seq 0 $((N-1))); do
  $PY $S/code/data_v4/extract_backbones.py --manifest $S/backbone_manifest.csv --input /root/data_v4/out --assets $ASSETS \
    --antideepfake $M/xlsr1b --out $OUT --shard $k --shards $N --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
