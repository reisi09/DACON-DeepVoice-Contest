#!/usr/bin/env bash
# Zero-shot member scoring (AntiDeepfake XLS-R 1B/2B on the voice stem, SONICS on the mix) for data v4 val+test.
# Non-AI-Hub data only. Launch: setsid bash -c "nohup bash /shared/members-20260927/run.sh > /shared/members-20260927/logs/run.log 2>&1 < /dev/null &"
set -euo pipefail
S=/shared/members-20260927; PY=/root/envs/data/bin/python; HF=/root/envs/data/bin/hf
OUT=/root/data_v4/out; ASSETS=/root/members_assets; M=/root/models; LOG=$S/logs; RES=/root/members_scores
mkdir -p $LOG $M $RES
ASSETS=$S/assets
$PY -c "import timm, pandas" 2>/dev/null || uv pip install -q -p $PY timm==1.0.15 pandas || $PY -m pip install -q timm==1.0.15 pandas
[ -f $M/xlsr1b/model.safetensors ] || $HF download nii-yamagishilab/xls-r-1b-anti-deepfake --revision 60c96543210117bdb256636e7961e16463f99291 --local-dir $M/xlsr1b
[ -f $M/xlsr2b/model.safetensors ] || $HF download nii-yamagishilab/xls-r-2b-anti-deepfake --local-dir $M/xlsr2b || echo "xlsr2b download failed"
[ -f $M/sonics/pytorch_model.bin ] || $HF download awsaf49/sonics-spectttra-alpha-120s --local-dir $M/sonics
AD=""; for d in xlsr1b xlsr2b; do [ -f $M/$d/model.safetensors ] && cp $M/$d/config.json $M/$d/antideepfake_config.json && AD="$AD $d=$M/$d"; done
echo "detectors:$AD"
echo "setup done $(date -u +%FT%TZ)"
N=5; export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3   # container quota is 15 vCPU but torch sees 128 cores
for k in $(seq 0 $((N-1))); do
  $PY $S/code/data_v4/score_members.py --manifest $S/members_manifest.csv --input $OUT --assets $ASSETS \
    --antideepfake $AD --sonics $M/sonics --sonics-code $S/code \
    --out $RES/scores_$k.csv --shard $k --shards $N > $LOG/shard_$k.log 2>&1 &
done
wait
cp $RES/scores_*.csv $S/logs/ && echo "all done $(date -u +%FT%TZ)" > $S/logs/DONE
