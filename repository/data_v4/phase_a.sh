#!/usr/bin/env bash
# Phase A on the H100 workspace: preflight -> envs -> unpack (+ generation in parallel) -> ingest -> select ->
# materialize -> vocals -> construct -> split analyze. Stops there: split rules are written from the report.
#   nohup bash /shared/data-v4-20260923/code/data_v4/phase_a.sh > /shared/data-v4-20260923/run_logs/phase_a.log 2>&1 &
set -euo pipefail
export PATH=/opt/conda/bin:$PATH      # image python/uv; SSH shells do not have it on PATH
S=/shared/data-v4-20260923; CODE=$S/code; LOGS=$S/run_logs; W=$S/weights
DATA=${DATA:-/data/data_v4}; RAW=$DATA/raw; OUT=${OUT:-/root/data_v4/out}; GEN=$RAW/gen   # adapter "gen" reads RAW/gen
DEADLINE=${DEADLINE:-$(( $(date +%s) + 2*3600 ))}                    # absolute stop for generation (GPU budget)
mkdir -p "$LOGS" "$DATA" "$OUT" "$GEN"
step() { echo "[$(date -u +%FT%TZ)] == $*" | tee -a "$LOGS/phase_a.steps"; }
cd "$CODE"
step preflight; { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h / /root /shared | tail -3; nproc; free -g | head -2; } | tee "$LOGS/preflight.txt"
command -v ffmpeg >/dev/null && command -v pigz >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg pigz >/dev/null; }
step envs; bash data_v4/envs.sh data > "$LOGS/env_data.log" 2>&1
PY=/root/envs/data/bin/python
step smoke; bash data_v4/smoke_vessl.sh "$PY" > "$LOGS/smoke.log" 2>&1 || { step "SMOKE FAILED (see smoke.log)"; exit 1; }
step unpack+generation
( RAW=$RAW PY=$PY bash data_v4/unpack.sh > "$LOGS/unpack.log" 2>&1; echo $? > "$LOGS/unpack.exit" ) &
UNPACK=$!
( set +e
  bash data_v4/envs.sh acestep > "$LOGS/env_acestep.log" 2>&1 && \
    /root/envs/acestep/bin/python data_v4/generate_music.py --out "$GEN/music" --songs 300 --instrumentals 200 \
      --deadline-epoch "$DEADLINE" > "$LOGS/gen_music.log" 2>&1
  echo $? > "$LOGS/gen_music.exit" ) &
GENM=$!
wait $UNPACK; [ "$(cat "$LOGS/unpack.exit")" = 0 ] || { step "UNPACK FAILED"; exit 1; }
( set +e
  bash data_v4/envs.sh cosyvoice2 > "$LOGS/env_cosyvoice2.log" 2>&1 && source /root/envs/cosyvoice2/pythonpath && \
    /root/envs/cosyvoice2/bin/python data_v4/generate_speech.py --backend cosyvoice2 --zeroth "$RAW/zeroth_korean/data" \
      --out "$GEN/speech" --work /root/gen_models --per-speaker 8 --deadline-epoch "$DEADLINE" > "$LOGS/gen_speech.log" 2>&1
  echo $? > "$LOGS/gen_speech.exit" ) &
GENS=$!
wait $GENM $GENS || true      # generation failures are tolerated: SpeechFake/MLAAD cover Korean TTS/VC
step ingest; $PY data_v4/build_manifest.py ingest --raw "$RAW" --out "$OUT" --existing-csv "$CODE/legacy/all_sources.csv" \
  --existing-audio "$RAW/existing" --existing-map "$CODE/legacy/audio_upload.json" > "$LOGS/ingest.log" 2>&1
step select; $PY data_v4/build_manifest.py select --out "$OUT" > "$LOGS/select.log" 2>&1
step materialize; $PY data_v4/build_manifest.py materialize --out "$OUT" --workers "$(( $(nproc) - 1 ))" > "$LOGS/materialize.log" 2>&1
step vocals; $PY data_v4/build_manifest.py vocals --out "$OUT" > "$LOGS/vocals.log" 2>&1
step construct; $PY data_v4/build_manifest.py construct --out "$OUT" > "$LOGS/construct.log" 2>&1
step split_analyze; $PY data_v4/split.py analyze --manifest "$OUT/manifest.jsonl" --report "$OUT/split_analysis" > "$LOGS/split_analyze.log" 2>&1
cp "$OUT/split_analysis.json" "$OUT/select_counts.json" "$LOGS/"
step PHASE_A_DONE
