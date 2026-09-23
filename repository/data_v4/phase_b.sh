#!/usr/bin/env bash
# Phase B on the H100 workspace (after split rules are written from phase A's analysis):
# split apply/verify -> export bank -> features (train+val, test in parallel) -> head training -> presence heads
# -> final FILE ensemble + official metrics -> results zip in the object volume.
#   setsid bash -c "nohup bash /shared/data-v4-20260923/code/data_v4/phase_b.sh >> .../phase_b.log 2>&1 < /dev/null &"
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
S=/shared/data-v4-20260923; CODE=$S/code; LOGS=$S/run_logs
DATA=${DATA:-/data/data_v4}; OUT=${OUT:-/root/data_v4/out}; BANK=$OUT/bank; TRAIN=${TRAIN:-/root/data_v4/train}
PY=/root/envs/data/bin/python; ASSETS=$DATA/assets
mkdir -p "$TRAIN" "$DATA"
step() { echo "[$(date -u +%FT%TZ)] == $*" | tee -a "$LOGS/phase_b.steps"; }
cd "$CODE"
step split_apply
$PY data_v4/split.py apply --manifest "$OUT/manifest.jsonl" --rules data_v4/split_rules.json --out "$OUT/splits.csv" > "$LOGS/split_apply.log" 2>&1
$PY data_v4/split.py verify --manifest "$OUT/manifest.jsonl" --splits "$OUT/splits.csv" >> "$LOGS/split_apply.log" 2>&1
step export_bank
$PY data_v4/export_bank.py --manifest "$OUT/manifest.jsonl" --splits "$OUT/splits.csv" --out "$BANK" > "$LOGS/export_bank.log" 2>&1
# test rows are extracted into their own cache; the extractor only accepts train/val split names
$PY -c "import csv,sys;r=list(csv.DictReader(open(sys.argv[1])));[x.update(split='val') for x in r];w=csv.DictWriter(open(sys.argv[2],'w',newline=''),fieldnames=r[0].keys());w.writeheader();w.writerows(r)" \
  "$BANK/bank_test.csv" "$BANK/bank_test_as_val.csv"
step assets
if [ ! -f "$ASSETS/.ok" ]; then
  mkdir -p "$ASSETS" && $PY -m zipfile -e /shared/existing-only-9504-20260919/assets.zip "$ASSETS" && touch "$ASSETS/.ok"
fi
cp /shared/existing-only-9504-20260919/asset_hashes.json "$DATA/asset_hashes.json"
step features   # train+val and test in parallel on the one H100 (DF-Arena runs one segment at a time)
( $PY full_bank_features.py --manifest "$BANK/bank_trainval.csv" --input "$OUT" --cache "$DATA/cache_tv" \
    --output "$DATA/feat_tv" --assets "$ASSETS" --asset-hashes "$DATA/asset_hashes.json" > "$LOGS/features_tv.log" 2>&1; echo $? > "$LOGS/features_tv.exit" ) &
( $PY full_bank_features.py --manifest "$BANK/bank_test_as_val.csv" --input "$OUT" --cache "$DATA/cache_test" \
    --output "$DATA/feat_test" --assets "$ASSETS" --asset-hashes "$DATA/asset_hashes.json" > "$LOGS/features_test.log" 2>&1; echo $? > "$LOGS/features_test.exit" ) &
( $PY data_v4/presence.py extract --manifest "$OUT/manifest.jsonl" --root "$OUT" --assets "$ASSETS" \
    --out "$OUT/presence_feats.jsonl" --per-cell 200 > "$LOGS/presence_extract.log" 2>&1; echo $? > "$LOGS/presence_extract.exit" ) &
wait
for f in features_tv features_test presence_extract; do [ "$(cat "$LOGS/$f.exit")" = 0 ] || { step "FAILED $f"; exit 1; }; done
step train_heads
$PY full_bank_train.py --manifest "$BANK/bank_trainval.csv" --cache "$DATA/cache_tv" --output "$TRAIN" --max-epochs 50 --patience 5 > "$LOGS/train.log" 2>&1
step presence_heads
$PY data_v4/presence.py train --manifest "$OUT/manifest.jsonl" --feats "$OUT/presence_feats.jsonl" --splits "$OUT/splits.csv" \
  --out "$TRAIN/presence_head.json" > "$LOGS/presence_train.log" 2>&1
step file_ensemble
$PY data_v4/file_ensemble.py --bank "$BANK/bank_trainval.csv" --test-bank "$BANK/bank_test.csv" --cache "$DATA/cache_tv" \
  --test-cache "$DATA/cache_test" --train-dir "$TRAIN" --presence-head "$TRAIN/presence_head.json" > "$LOGS/file_ensemble.log" 2>&1
step results_zip
TS=$(date -u +%Y%m%dT%H%MZ); Z=$S/results/data_v4_results_$TS.zip; mkdir -p "$S/results"
$PY - "$Z" "$OUT" "$TRAIN" "$LOGS" <<'EOF'
import sys, zipfile, pathlib
z, out, train, logs = sys.argv[1], *map(pathlib.Path, sys.argv[2:])
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in [out / n for n in ('manifest.jsonl', 'items_selected.jsonl', 'select_counts.json', 'split_analysis.json',
                                'splits.csv', 'splits.report.json', 'presence_feats.jsonl')] + sorted((out / 'bank').glob('*')):
        if p.exists(): zf.write(p, 'data/' + str(p.relative_to(out)))
    for p in sorted(train.glob('*')):
        if p.is_file() and not p.name.endswith('_resume.pt'): zf.write(p, 'train/' + p.name)
    for p in sorted(logs.glob('*')):
        if p.is_file(): zf.write(p, 'logs/' + p.name)
print(z)
EOF
echo "$Z" > "$LOGS/RESULTS_ZIP"
step PHASE_B_DONE
