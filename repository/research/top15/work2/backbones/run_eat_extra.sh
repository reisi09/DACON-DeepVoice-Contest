#!/usr/bin/env bash
# EAT-large (backing stem) features only, for the S4 P1 rows missing from the MuQ/EAT pass: every val row and every
# music-labelled train/test row outside the 31,270 R2 rows (manifest_eat_extra.csv, 4,339 rows). Same extractor and
# weights as run_vessl.sh (/root/models/eat, fp16 file sha 36833579...), MuQ skipped. Copy to /root/newbb/ with the
# extractor and manifest (not via /shared: the FUSE mount served stale files before).
set -euo pipefail
PY=/root/envs/data/bin/python; NB=/root/newbb; OUT=/root/eat_extra_cache; LOG=/root/eat_extra_logs
CODE=/shared/backbones-20260927/code; ASSETS=/root/bb_assets
N=${N:-5}; MINUTES=${MINUTES:-60}
mkdir -p $LOG $OUT
$PY -c "import sys; sys.path.insert(0, '$NB'); from pathlib import Path; import extract_new_backbones as X; print(X.sha256(X.fp16_weights(Path('/root/models/eat'))))" | tee $LOG/weights_sha.txt
head -3 $NB/manifest_eat_extra.csv > $LOG/smoke_manifest.csv
$PY $NB/extract_new_backbones.py --manifest $LOG/smoke_manifest.csv --input /root/data_v4/out --assets $ASSETS \
  --eat /root/models/eat --code-root $CODE --out $LOG/smoke_cache > $LOG/smoke.log 2>&1
DEADLINE=$(( $(date +%s) + MINUTES*60 ))
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in $(seq 0 $((N-1))); do
  $PY $NB/extract_new_backbones.py --manifest $NB/manifest_eat_extra.csv --input /root/data_v4/out --assets $ASSETS \
    --eat /root/models/eat --code-root $CODE --out $OUT --shard $k --shards $N --deadline $DEADLINE > $LOG/shard_$k.log 2>&1 &
done
wait
echo "files $(ls $OUT | grep -c '\.npz$') $(date -u +%FT%TZ)" > $LOG/DONE
