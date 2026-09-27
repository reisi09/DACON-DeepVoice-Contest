#!/usr/bin/env bash
# Track C on 4 x H100: fine-tune (train split, select on val), then score val + held-out test with the best checkpoint.
#   bash run_ft.sh OUT [extra finetune_ad.py args, e.g. --max-epochs 8]
# Env overrides: PY NPROC BANK_TV BANK_TEST STEMS MODEL CODE WORKERS. Artifacts in OUT: train.log, metrics.jsonl,
# evals.json, result.json, val_best.csv, best/ (runtime-loadable fp16 AntiDeepfake), score.log, ft_members.csv.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: run_ft.sh OUT [finetune_ad.py args]}; shift
PY=${PY:-/root/envs/data/bin/python}
NPROC=${NPROC:-4}
BANK_TV=${BANK_TV:-/root/ft/bank_trainval.csv}
BANK_TEST=${BANK_TEST:-/root/ft/bank_test.csv}
STEMS=${STEMS:-/root/stems}
MODEL=${MODEL:-/root/models/xlsr1b}
CODE=${CODE:-/root/ft/code}
mkdir -p "$OUT"
export OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpus.txt"

"$PY" -m torch.distributed.run --nnodes 1 --nproc_per_node "$NPROC" --master_addr 127.0.0.1 --master_port 29531 "$HERE/finetune_ad.py" \
  --manifest "$BANK_TV" --stems "$STEMS" --model "$MODEL" --code-root "$CODE" --out "$OUT" \
  --workers "${WORKERS:-10}" "$@" 2>&1 | tee -a "$OUT/train.log"

# hard invariants: checkpoint weights == fp16(trained weights) exactly, frozen tensors untouched; fp16 prob drift is reported
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); t=r['roundtrip']; assert t['weights_exact'] and not r['params']['changed_outside_trainable'], r; print('roundtrip', t, 'best', r['best'])" \
  "$OUT/result.json"

# stems must be complete for train and val (DB protocol row sets)
"$PY" -c "import json,sys; s=[json.loads(l) for l in open(sys.argv[1]) if '\"setup\"' in l][0]; m=s.get('missing_stems', s.get('missing')); assert not m or m in ([0,0],[0,0,0]), s; print('setup ok', {k: s[k] for k in ('train_items','sampler_ess') if k in s})" "$OUT/metrics.jsonl"
"$PY" -m torch.distributed.run --nnodes 1 --nproc_per_node "$NPROC" --master_addr 127.0.0.1 --master_port 29532 "$HERE/score_ft.py" \
  --ckpt "$OUT/best" --manifest "$BANK_TV" "$BANK_TEST" --splits val test --stems "$STEMS" --code-root "$CODE" \
  --out "$OUT/ft_members.csv" 2>&1 | tee -a "$OUT/score.log"

"$PY" -c "import csv,sys; n=sum(1 for _ in csv.DictReader(open(sys.argv[1]))); assert n==11381, n; print('ft_members rows', n)" "$OUT/ft_members.csv"
echo "done $(date -u +%FT%TZ)" > "$OUT/DONE"
