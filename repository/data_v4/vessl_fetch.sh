#!/usr/bin/env bash
# CPU-only VESSL job: fetch data_v4 sources + pretrained weights into the object volume before any GPU run.
#   vesslctl job create -n data-v4-fetch -r resourcespec-kmswqig4b12w -i quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim \
#     --object-volume objvol-a5gxdg7er70i:/root-1 --secret HF_TOKEN=hf-token \
#     --cmd 'timeout 36000 bash /root-1/data-v4-20260923/code/data_v4/vessl_fetch.sh'
# Each source: download to local ephemeral disk -> checksum-verify (candidate_data_acquire.py) ->
# one tar per source into the object volume (object storage handles one big object better than
# thousands of small ones) -> delete local copy. Finished sources are skipped on rerun.
set -euo pipefail
OBJ=${OBJ:-/root-1/data-v4-20260923}
CODE=$OBJ/code; PLANS=${PLAN_DIR:-$CODE/data_v4/plan}; ARCH=$OBJ/archives; W=$OBJ/weights; LOG=$OBJ/fetch_logs
LOCAL=${LOCAL:-/tmp/bank}; PY=${PY:-python}
SOURCES=${SOURCES:-"zeroth_korean fleurs_ko mlaad_ko csd vocalset openmic asvspoof2021_la haim echoes sonics mlaad_other musan asvspoof2021_df"}
mkdir -p "$ARCH" "$W" "$LOG" "$LOCAL"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/fetch.log"; }
log "start; local disk: $(df -h "$LOCAL" | tail -1)"
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
$PY -c 'import huggingface_hub' 2>/dev/null || $PY -m pip install -q huggingface_hub

for src in $SOURCES; do
  if [ -f "$ARCH/$src.tar.done" ]; then log "skip $src (done)"; continue; fi
  log "fetch $src"
  if ACQUIRE_BANK="$LOCAL" ACQUIRE_OUT="$LOG/receipts" ACQUIRE_RESERVE_GIB=${RESERVE_GIB:-2} \
       $PY "$CODE/candidate_data_acquire.py" "$PLANS/$src.json" >> "$LOG/$src.log" 2>&1; then
    # No rename on the object-storage mount: a job that exits right after `mv` lost the rename (2026-09-23,
    # mlaad_other). The .done marker, written only after tar succeeds, is the commit point.
    rm -f "$ARCH/$src.tar.done"
    tar -cf "$ARCH/$src.tar" -C "$LOCAL" "$src"
    $PY -c "import sys,pathlib;print(sum(f.stat().st_size for f in pathlib.Path(sys.argv[1]).rglob('*') if f.is_file()))" "$LOCAL/$src" > "$ARCH/$src.tar.done"      # marker = verified payload bytes
    rm -rf "$LOCAL/$src"
    log "done $src ($(cat "$ARCH/$src.tar.done") bytes)"
  else
    log "FAIL $src (see $LOG/$src.log); continuing with next source"   # partials stay for a rerun
  fi
done

[ -n "${SKIP_WEIGHTS:-}" ] && { log "finished (weights skipped)"; exit 0; }
for set in ${WEIGHTS:-core}; do log "weights $set"; $PY "$CODE/data_v4/fetch_weights.py" "$W" "$set" 2>&1 | tee -a "$LOG/weights.log"; done
log "finished; archives: $(ls "$ARCH" | tr '\n' ' ')"
