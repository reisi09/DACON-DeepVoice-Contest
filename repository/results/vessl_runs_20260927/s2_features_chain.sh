#!/bin/bash
# one MPS process at a time: wait for vocal evidence, split B rows, then extract A-train, A-dev, B-train, B-dev
set -uo pipefail
cd "/Users/choiwuseck/Documents/ChatGPT/데이콘"
while pgrep -f "build_extra.py vocals" >/dev/null; do sleep 20; done
.venv/bin/python - <<'PY'
import json, csv, sys
sys.path.insert(0,'data_v4')
from export_bank import export, COLUMNS
rows=[json.loads(l) for l in open('_workspace/s2_extra/manifest.jsonl')]
b=[r for r in rows if r.get('vocal_evidence')]
assert all('vocal_ratio' in r for r in b), 'vocal evidence incomplete'
bank,_,skipped=export(b,{r['ID']:r['split'] for r in b})
for name,split in (('bank_B_train.csv','train'),('bank_B_dev.csv','val')):
    with open(f'_workspace/s2_extra/{name}','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS); w.writeheader(); w.writerows(r for r in bank if r['split']==split)
print('B rows',len(b),'train',sum(r['split']=='train' for r in bank),'dev',sum(r['split']=='val' for r in bank),flush=True)
PY
A=_workspace/assets_e9504
for part in A_train A_dev B_train B_dev; do
  HF_HOME=$PWD/_workspace/diag_aihub/empty_hf PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python local_features.py \
    --manifest _workspace/s2_extra/bank_$part.csv --input _workspace/s2_extra --cache _workspace/s2_extra/cache_$part \
    --output _workspace/s2_extra/out_$part --assets $A --asset-hashes $A/asset_hashes.json --device mps \
    >> _workspace/s2_extra/features_$part.log 2>&1
  echo "$part exit $? $(date +%H:%M)"
done
