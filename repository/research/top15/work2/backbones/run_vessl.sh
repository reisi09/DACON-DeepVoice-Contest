#!/usr/bin/env bash
# MuQ (both stems) + EAT-large (backing) window features for the 31,270 R2 backbone rows, on H100(s).
# Modeled on the S3 launcher (scratchpad/vessl/bb/run.sh): same env, audio, assets, 5 shards per GPU, hard deadline.
# Manifest order: val+test (8,502) -> train music-labelled (6,981) -> other train rows (15,787), so a deadline stop after
# ~50% still leaves the complete music study (rule A1/A2/B); the voice (muqv) study needs all rows.
# Upload this folder's extract_new_backbones.py, vendor/, manifest_backbone_rows.csv and run_vessl.sh to $S/newbb/.
# Launch: setsid bash -c "nohup bash /shared/newbackbones-20260927/newbb/run_vessl.sh > /shared/newbackbones-20260927/logs/run.log 2>&1 < /dev/null &"
# Knobs: NGPU (GPUs on this machine), N (shards here), SHARDS/FIRST (split rows across machines), MINUTES, EAT=0 to skip EAT.
set -euo pipefail
S=/shared/newbackbones-20260927; PY=/root/envs/data/bin/python; M=/root/models; OUT=/root/newbackbone_cache; LOG=$S/logs
CODE=/shared/backbones-20260927/code      # project root of the S3 extraction upload (mert12_submission.py, v6_runtime.py, ...)
ASSETS=/shared/members-20260927/assets    # baseline/ (HTDemucs)
NGPU=${NGPU:-1}; N=${N:-$((5*NGPU))}; SHARDS=${SHARDS:-$N}; FIRST=${FIRST:-0}; MINUTES=${MINUTES:-150}; EAT=${EAT:-1}
mkdir -p $LOG $OUT
$PY -c "import torch, torchaudio, transformers; print('versions', torch.__version__, torchaudio.__version__, transformers.__version__)" | tee $LOG/versions.txt
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('OpenMuQ/MuQ-large-msd-iter', revision='0562a57814f6f8bbd9fdea0a25921a2fce1a841a',
                  local_dir='/root/models/muq', allow_patterns=['config.json', 'model.safetensors'])
snapshot_download('worstchan/EAT-large_epoch20_finetune_AS2M', revision='53b1251196e3dabb991dfa7629fa89394c7f8335',
                  local_dir='/root/models/eat', allow_patterns=['config.json', 'model.safetensors'])
EOF
# fp16 weight files once, before the shards start (these exact files ship in the package); local smoke: muq 7cecd3b9..., eat 36833579...
$PY -c "import sys; sys.path.insert(0, '$S/newbb'); from pathlib import Path; import extract_new_backbones as X; [print(p, X.sha256(X.fp16_weights(Path(p)))) for p in ('$M/muq', '$M/eat')]" | tee $LOG/weights_sha.txt
EATARG=""; [ "$EAT" = 1 ] && EATARG="--eat $M/eat"
# 2-row CUDA smoke before the fleet (fails fast on an import / CUDA problem)
head -3 $S/newbb/manifest_backbone_rows.csv > $LOG/smoke_manifest.csv
$PY $S/newbb/extract_new_backbones.py --manifest $LOG/smoke_manifest.csv --input /root/data_v4/out --assets $ASSETS \
  --muq $M/muq $EATARG --code-root $CODE --out $LOG/smoke_cache > $LOG/smoke.log 2>&1
DEADLINE=$(( $(date +%s) + MINUTES*60 ))
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 0 $((N-1))); do
  k=$((FIRST+i))
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) $PY $S/newbb/extract_new_backbones.py --manifest $S/newbb/manifest_backbone_rows.csv \
    --input /root/data_v4/out --assets $ASSETS --muq $M/muq $EATARG --code-root $CODE --out $OUT \
    --shard $k --shards $SHARDS --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
