#!/bin/bash
# S2-B only (music data: Jamendo, Music Arena, AI-Hub 71470), one MPS process at a time. A parts are dropped (S2a refuted).
set -uo pipefail
cd "/Users/choiwuseck/Documents/ChatGPT/데이콘"
A=_workspace/assets_e9504
for part in B_dev B_train; do
  HF_HOME=$PWD/_workspace/diag_aihub/empty_hf PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python local_features.py \
    --manifest _workspace/s2_extra/bank_$part.csv --input _workspace/s2_extra --cache _workspace/s2_extra/cache_$part \
    --output _workspace/s2_extra/out_$part --assets $A --asset-hashes $A/asset_hashes.json --device mps \
    >> _workspace/s2_extra/features_$part.log 2>&1
  echo "$part exit $? $(TZ=Asia/Seoul date +%H:%M)"
done
