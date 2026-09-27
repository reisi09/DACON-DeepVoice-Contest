#!/usr/bin/env bash
# S6 candidate encoders (choice_lock.json) on the stored stems, H100x4: SSLAM pt/ft (backing), WavLM-L / HuBERT-L (voice).
# Copy extract_s6.py, manifest_s6.txt and this file to /root/s6/ (not via /shared: FUSE served stale files before).
set -euo pipefail
PY=/root/envs/data/bin/python; D=/root/s6; M=/root/models/s6; OUT=/root/s6_cache; LOG=/root/s6_logs
NGPU=${NGPU:-4}; PER=${PER:-3}; MINUTES=${MINUTES:-75}; N=$((NGPU*PER))
mkdir -p $LOG $OUT $M
$PY - <<'PY'
from huggingface_hub import snapshot_download
for name, repo, rev, pats in (('sslam_pt', 'ta012/SSLAM_pretrain', 'af71a4b50473c14efdfaeeffc8636e37692576f7', ['config.json', 'model.safetensors']),
                              ('sslam_ft', 'ta012/SSLAM_AS2M_Finetuned', '4500c5d989436b57b0621418fa49a54ca2ed8dff', ['config.json', 'model.safetensors']),
                              ('wavlm', 'microsoft/wavlm-large', 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c', ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']),
                              ('hubert', 'facebook/hubert-large-ll60k', 'ff022d095678a2995f3c49bab18a96a9e553f782', ['config.json', 'pytorch_model.bin', 'preprocessor_config.json'])):
    print(name, snapshot_download(repo, revision=rev, local_dir=f'/root/models/s6/{name}', allow_patterns=pats), flush=True)
PY
sha256sum $M/*/model.safetensors $M/*/pytorch_model.bin | tee $LOG/weights_sha.txt
head -2 $D/manifest_s6.txt > $LOG/smoke_ids.txt
$PY $D/extract_s6.py --ids $LOG/smoke_ids.txt --stems /root/stems --models $M --code-root /root/ft/code --eat-core /root/newbb/vendor \
  --out $LOG/smoke_cache > $LOG/smoke.log 2>&1
DEADLINE=$(( $(date +%s) + MINUTES*60 ))
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY $D/extract_s6.py --ids $D/manifest_s6.txt --stems /root/stems --models $M --code-root /root/ft/code \
    --eat-core /root/newbb/vendor --out $OUT --shard $k --shards $N --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
