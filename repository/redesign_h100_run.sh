#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${REDESIGN_ROOT:-/root/redesign_h100_20260917}"
SHARED="${REDESIGN_SHARED:-/shared/redesign-h100-20260917}"
PYTHON="${REDESIGN_PYTHON:-/root/fullft8/.venv/bin/python}"
CREATED_UNIX="${REDESIGN_WORKSPACE_CREATED_UNIX:-$(date +%s)}"
SOURCE_ZIP="${REDESIGN_INPUT_ZIP:-$SHARED/input.zip}"
UPLOAD_RECEIPT="${REDESIGN_UPLOAD_RECEIPT:-$SHARED/upload_receipt.json}"

mkdir -p "$ROOT" "$SHARED" "$ROOT/output"
exec > >(tee -a "$ROOT/output/bootstrap.log") 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] bounded redesign H100 bootstrap"
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used --format=csv
TELEMETRY="$ROOT/gpu_telemetry.live.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw --format=csv,noheader,nounits --loop=10 > "$TELEMETRY" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
test -x "$PYTHON"
test -f "$SOURCE_ZIP"
test -f "$UPLOAD_RECEIPT"
read -r EXPECTED_SHA EXPECTED_BYTES EXPECTED_FILES UPLOAD_PREFIX < <("$PYTHON" - "$UPLOAD_RECEIPT" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
print(value["sha256"], value["bytes"], value["files"], value["prefix"])
PY
)
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ ]]
if ! cp --reflink=auto "$SOURCE_ZIP" "$ROOT/input.zip" 2>/dev/null; then cp "$SOURCE_ZIP" "$ROOT/input.zip"; fi
ACTUAL_SHA="$("$PYTHON" -c 'import hashlib,sys; print(hashlib.file_digest(open(sys.argv[1],"rb"),"sha256").hexdigest())' "$ROOT/input.zip")"
test "$ACTUAL_SHA" = "$EXPECTED_SHA"
ACTUAL_BYTES="$("$PYTHON" -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$ROOT/input.zip")"
test "$ACTUAL_BYTES" = "$EXPECTED_BYTES"
[[ "$EXPECTED_FILES" =~ ^[0-9]+$ ]]
test -n "$UPLOAD_PREFIX"
MISSING_NOTEBOOK="$($PYTHON - <<'PY'
import importlib.util
print(' '.join(name for name in ('nbformat','nbclient','ipykernel') if importlib.util.find_spec(name) is None))
PY
)"
if [[ -n "$MISSING_NOTEBOOK" ]]; then
  NOTEBOOK_PACKAGES=()
  for NAME in $MISSING_NOTEBOOK; do
    case "$NAME" in
      nbformat) NOTEBOOK_PACKAGES+=(nbformat==5.10.4) ;;
      nbclient) NOTEBOOK_PACKAGES+=(nbclient==0.10.2) ;;
      ipykernel) NOTEBOOK_PACKAGES+=(ipykernel==6.29.5) ;;
    esac
  done
  if [[ -x /root/.local/bin/uv ]]; then
    /root/.local/bin/uv pip install --python "$PYTHON" "${NOTEBOOK_PACKAGES[@]}"
  else
    python -m pip --python "$PYTHON" install "${NOTEBOOK_PACKAGES[@]}"
  fi
fi
"$PYTHON" -c 'import nbformat,nbclient,ipykernel; print({"nbformat":nbformat.__version__,"nbclient":nbclient.__version__,"ipykernel":ipykernel.__version__})'
if ! "$PYTHON" -c 'import diffq' >/dev/null 2>&1; then
  if [[ -x /root/.local/bin/uv ]]; then
    /root/.local/bin/uv pip install --python "$PYTHON" --no-deps diffq==0.2.4
  else
    python -m pip --python "$PYTHON" install --no-deps diffq==0.2.4
  fi
fi
"$PYTHON" -c 'import diffq; print({"diffq":diffq.__file__})'

RUNNER="$ROOT/redesign_h100_run.py"
"$PYTHON" - "$ROOT/input.zip" "$RUNNER" <<'PY'
import stat, sys, zipfile
from pathlib import Path
archive, target = map(Path, sys.argv[1:])
with zipfile.ZipFile(archive) as packed:
    matches = [item for item in packed.infolist() if item.filename == "redesign_h100_run.py"]
    if len(matches) != 1 or stat.S_ISLNK(matches[0].external_attr >> 16):
        raise SystemExit("input ZIP must contain one regular redesign_h100_run.py")
    target.write_bytes(packed.read(matches[0]))
PY

set +e
HARD_DEADLINE=$((CREATED_UNIX + 14400))
REMAINING=$((HARD_DEADLINE - $(date +%s)))
if (( REMAINING <= 0 )); then REMAINING=1; fi
timeout --signal=TERM --kill-after=30s "${REMAINING}s" "$PYTHON" "$RUNNER" --input-zip "$ROOT/input.zip" --input-sha256 "$EXPECTED_SHA" \
  --root "$ROOT" --shared "$SHARED" --python "$PYTHON" --created-unix "$CREATED_UNIX"
RUN_EXIT=$?
set -e
if [[ ! -s "$SHARED/results.zip" ]]; then
  "$PYTHON" - "$ROOT" "$SHARED" "$CREATED_UNIX" <<'PY'
import hashlib, json, sys, time, zipfile
from pathlib import Path
root, shared, began = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
archive = shared / "results.zip"; files = []
def digest(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
for base in (root / "output", root / "evidence"):
    if base.exists(): files += [(p, f"{base.name}/{p.relative_to(base).as_posix()}") for p in base.rglob("*") if p.is_file() and not p.is_symlink()]
hashes = {name: digest(path) for path, name in files}
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as packed:
    for path, name in sorted(files, key=lambda item: item[1]): packed.write(path, name)
    packed.writestr("EXPORT_SHA256.json", json.dumps(hashes, indent=2, sort_keys=True))
with zipfile.ZipFile(archive) as packed:
    assert packed.testzip() is None
    assert all(hashlib.sha256(packed.read(name)).hexdigest() == expected for name, expected in hashes.items())
archive_digest = digest(archive); now = time.time()
(shared / "results.zip.sha256").write_text(f"{archive_digest}  results.zip\n")
receipt = {"schema_version":1,"archive":str(archive),"archive_sha256":archive_digest,"archive_bytes":archive.stat().st_size,"archive_ready":True,"archive_verified":True,"export_member_count":len(files),"exported_unix":now,"status":"partial_exported"}
(shared / "export_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
state = receipt | {"status":"failed","started_unix":began,"hard_deadline_unix":began+14400,"stage_deadline_unix":began+13500,"updated_unix":now,"finished_unix":now,"error":"runner terminated before normal export","notebook_exit_code":None}
(shared / "run_state.json").write_text(json.dumps(state, indent=2, sort_keys=True)+"\n")
PY
fi
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used --format=csv || true
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] runner exit=$RUN_EXIT"
exit "$RUN_EXIT"
