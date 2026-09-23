#!/usr/bin/env bash
# CPU-only VESSL job: SpeechFake Korean subset via HTTP Range (no 115 GB download).
#   MD ko fakes (CosyVoice, EdgeTTS, MeloTTS, OpenVoice, SeedVC: 82,728 files, 13.2 GB) + CommonVoice ko reals (2,500)
#   + metadata.zip (labels/generators). CRC32-verified per member; one tar + .done marker in the object volume.
set -euo pipefail
OBJ=${OBJ:-/root-1/data-v4-20260923}; CODE=$OBJ/code; ARCH=$OBJ/archives; LOG=$OBJ/fetch_logs
LOCAL=${LOCAL:-/tmp/bank_speechfake}/speechfake; PY=${PY:-python}; DS=inclusionAI/SPEECHFAKE
mkdir -p "$ARCH" "$LOG" "$LOCAL"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/fetch.log"; }
[ -f "$ARCH/speechfake.tar.done" ] && { log "skip speechfake (done)"; exit 0; }
log "fetch speechfake"
$PY "$CODE/data_v4/zip_range_extract.py" --modelscope $DS MD/MD.z01 MD/MD.z02 MD/MD.zip --include 'MD/*/ko/*' --out "$LOCAL" --workers 8 2>&1 | tee -a "$LOG/speechfake.log"
mv "$LOCAL/extract_receipt.json" "$LOCAL/extract_receipt_MD.json"
$PY "$CODE/data_v4/zip_range_extract.py" --modelscope $DS Real/CommonVoice.zip --include 'CommonVoice/ko/*' --out "$LOCAL" --workers 8 2>&1 | tee -a "$LOG/speechfake.log"
mv "$LOCAL/extract_receipt.json" "$LOCAL/extract_receipt_CV.json"
curl -fsSL -o "$LOCAL/metadata.zip" "https://www.modelscope.cn/api/v1/datasets/$DS/repo?Revision=master&FilePath=metadata/metadata.zip"
(cd "$LOCAL" && $PY -m zipfile -e metadata.zip .)
rm -f "$ARCH/speechfake.tar.done"
tar -cf "$ARCH/speechfake.tar" -C "$(dirname "$LOCAL")" speechfake
$PY -c "import sys,pathlib;print(sum(f.stat().st_size for f in pathlib.Path(sys.argv[1]).rglob('*') if f.is_file()))" "$LOCAL" > "$ARCH/speechfake.tar.done"
log "done speechfake ($(cat "$ARCH/speechfake.tar.done") bytes)"
