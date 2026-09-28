#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${RUN_ROOT:-/root/eight_h100_20260917}
SHARED=${RUN_SHARED:-/shared/eight-h100-20260917}
KERNEL=${RUN_KERNEL:-eight-h100}
PYTHON=${RUN_PYTHON:-/root/fullft8/.venv/bin/python}
mkdir -p "$ROOT" "$ROOT/output" "$SHARED"
export PYTHONDONTWRITEBYTECODE=1

if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$0" --worker </dev/null > "$ROOT/launcher.log" 2>&1 &
  echo "$!" > "$ROOT/launcher.pid"
  exit 0
fi

exec > >(tee -a "$ROOT/run.log") 2>&1

partial_export() {
  code=$?
  trap - ERR
  cp "$ROOT/run.log" "$ROOT/output/bootstrap_failure.log" 2>/dev/null || true
  "$PYTHON" - "$ROOT" "$SHARED" "$code" <<'PY' || true
import hashlib, json, sys, time, zipfile
from pathlib import Path
root, shared, exit_code = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
archive = shared / "results.zip"
files = sorted(path for path in (root / "output").rglob("*") if path.is_file() and not path.is_symlink())
manifest = {}
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as packed:
    for path in files:
        name = "output/" + path.relative_to(root / "output").as_posix()
        data = path.read_bytes(); manifest[name] = hashlib.sha256(data).hexdigest(); packed.writestr(name, data)
    packed.writestr("EXPORT_SHA256.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
with zipfile.ZipFile(archive) as packed:
    if packed.testzip() is not None: raise SystemExit("partial archive CRC verification failed")
    names = set(packed.namelist())
    if names != set(manifest) | {"EXPORT_SHA256.json"}: raise SystemExit("partial archive manifest coverage mismatch")
    loaded = json.loads(packed.read("EXPORT_SHA256.json"))
    if loaded != manifest: raise SystemExit("partial archive manifest changed")
    for name, expected in manifest.items():
        if hashlib.sha256(packed.read(name)).hexdigest() != expected:
            raise SystemExit(f"partial archive member hash mismatch: {name}")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(shared / "results.zip.sha256").write_text(f"{digest}  results.zip\n")
receipt = {"status":"partial_exported","archive_ready":True,"archive_verified":True,
           "archive_sha256":digest,"archive_bytes":archive.stat().st_size,
           "export_member_count":len(files),"exported_unix":time.time()}
(shared / "export_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
state = receipt | {"status":"failed","updated_unix":time.time(),"error":f"bootstrap exit {exit_code}"}
(shared / "run_state.json").write_text(json.dumps(state, indent=2, sort_keys=True)+"\n")
PY
  exit "$code"
}
trap partial_export ERR

ATTEMPT="$($PYTHON - "$SHARED/config.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1])).get("attempt", 1)))
PY
)"
if (( ATTEMPT >= 2 )) && [[ ! -f "$SHARED/retry${ATTEMPT}_started.marker" ]]; then
  FAILED="$SHARED/retry$((ATTEMPT - 1))-failed"
  mkdir -p "$FAILED"
  for NAME in results.zip results.zip.sha256 export_receipt.json; do
    [[ ! -f "$SHARED/$NAME" ]] || mv "$SHARED/$NAME" "$FAILED/$NAME"
  done
  if [[ -f "$SHARED/run_state.json" ]] && ! "$PYTHON" - "$SHARED/run_state.json" "$ATTEMPT" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
raise SystemExit(0 if state.get("stage") == "preparing" and state.get("attempt") == int(sys.argv[2]) else 1)
PY
  then
    mv "$SHARED/run_state.json" "$FAILED/run_state.json"
  fi
  date -u +%FT%TZ > "$SHARED/retry${ATTEMPT}_started.marker"
fi

WHEEL="$SHARED/offline_wheels/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl"
printf '%s  %s\n' \
  c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282 "$WHEEL" \
  | sha256sum --check --status
"$PYTHON" - "$WHEEL" "$ROOT/output/dependency_recovery.json" <<'PY'
import hashlib, importlib, importlib.metadata, json, os, site, stat, subprocess, sys, time, zipfile
from pathlib import Path, PurePosixPath
wheel, receipt_path = Path(sys.argv[1]), Path(sys.argv[2])
target = Path(site.getsitepackages()[0])
with zipfile.ZipFile(wheel) as packed:
    if packed.testzip() is not None:
        raise SystemExit("imageio-ffmpeg wheel CRC failure")
    for item in packed.infolist():
        path = PurePosixPath(item.filename)
        if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(item.external_attr >> 16):
            raise SystemExit(f"unsafe wheel member: {item.filename}")
    packed.extractall(target)
binary = target / "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
if not binary.is_file():
    raise SystemExit("fixed wheel ffmpeg binary is missing")
binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
if not os.access(binary, os.X_OK):
    raise SystemExit("fixed wheel ffmpeg binary is not executable")
importlib.invalidate_caches()
import imageio_ffmpeg
executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
if executable.resolve() != binary.resolve():
    raise SystemExit("imageio-ffmpeg selected an unexpected executable")
subprocess.run([str(executable), "-version"], check=True, stdout=subprocess.DEVNULL)
receipt = {
    "status": "passed",
    "package": "imageio-ffmpeg",
    "version": importlib.metadata.version("imageio-ffmpeg"),
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "install_method": "fixed-sha-offline-wheel-extraction",
    "executable_mode": oct(binary.stat().st_mode & 0o777),
    "selected_expected_executable": True,
    "ffmpeg_executable_verified": True,
    "verified_unix": time.time(),
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

if [[ -f "$ROOT/source/INPUT_SHA256.json" ]]; then
  "$PYTHON" - "$ROOT/source" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = root / "INPUT_SHA256.json"
expected_manifest = "c21e12f9fc4b4ad9c29763ef403a8624f8d09d1f7ea43bc1e5e469ba9c0fe2b1"
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest:
    raise SystemExit("extracted source manifest identity mismatch")
manifest = json.loads(manifest_path.read_text())
overrides = {
    "eight_features.py": {
        "original": "17ffe77d2f16019db24061d3c52317a70918d580dbe92ff8960e3d328cfd6760",
        "runtime": "b0304f4f9aa2d13524d56622f494c4b1fab160b5009e09a51fb77987ed23acac",
    },
    "antideepfake_runtime.py": {
        "original": "abf775389fdbe7a97793e9b2ef59f764d5eb312fafd286c054e3ca583b38be31",
        "runtime": "bbf7ec7bc0b20afa416e7a6fac535d8ab92d984621c1e9be3e395bb7f3d0f729",
    },
    "eight_feature_schema.py": {
        "original": "fb9eb35e299b42a3e2646c42fa5bec251a9beb5e50b2781fef63c72acd0f7d7c",
        "runtime": "cdc62ffb8da7773deee4d030c68ca7fda1cd2cb99ce2294a866e88ba79d5e9ff",
    },
    "eight_feature_identity.py": {
        "original": "d5a772fc2198ecbff1f56d6f93e431113d56e7adcd091980cf622042a4b4d62c",
        "runtime": "edb796b9c99b1083ec839743694cf9738230ba0f2eac6b14a00c5886a1d45bb2",
    },
    "eight_probes.py": {
        "original": "30e1079747d683c1f8fcfd6b215db7af4fd812755b305034e3a400f8913ea9fb",
        "runtime": "1661764b7214f5b7cdc2724656aaf8fe865bda11ffec920ec86067fbceb10f52",
    },
    "eight_ablation.py": {
        "original": "b44ca59ffc10cc3bc58a3ff2074c92186ee5f9cb52365253aefc0201d0f1abee",
        "runtime": "5d6c8fc7b214bcae3701a0ead2798fc97dfc1e1fea34023d37a2a41d10d12246",
    },
    "eight_lora.py": {
        "original": "b6504eb5531f0fd3fa683872869b3b4babfd1838c92db27d6899c1b173593716",
        "runtime": "be8828a1eb51dc7a8c8059abfc86ffbd4dda25d27e6b98a0932867a816c1f579",
    },
}
actual = set()
for path in root.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts and relative.suffix == ".pyc":
        continue
    actual.add(relative.as_posix())
if actual != set(manifest) | {"INPUT_SHA256.json"}:
    raise SystemExit("extracted source coverage mismatch")
for name, expected in manifest.items():
    actual_sha = hashlib.sha256((root / name).read_bytes()).hexdigest()
    accepted = {expected}
    if name in overrides:
        if expected != overrides[name]["original"]:
            raise SystemExit(f"override original identity mismatch: {name}")
        accepted.add(overrides[name]["runtime"])
    if actual_sha not in accepted:
        raise SystemExit(f"extracted source member mismatch: {name}")
PY
elif [[ -e "$ROOT/source" ]]; then
  echo "existing source lacks pinned manifest" >&2
  exit 1
else
  read -r SOURCE_SHA _ < "$SHARED/source.zip.sha256"
  printf '%s  %s\n' "$SOURCE_SHA" "$SHARED/source.zip" | sha256sum --check --status
  mkdir -p "$ROOT/source"
  "$PYTHON" - "$SHARED/source.zip" "$ROOT/source" <<'PY'
import hashlib, json, shutil, stat, sys, zipfile
from pathlib import Path, PurePosixPath
archive, target = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(archive) as packed:
    if packed.testzip() is not None: raise SystemExit("source ZIP CRC failure")
    infos = packed.infolist(); names = [item.filename for item in infos]
    if len(set(names)) != len(names): raise SystemExit("duplicate source ZIP member")
    if shutil.disk_usage(target.parent).free <= sum(item.file_size for item in infos) + 5_000_000_000:
        raise SystemExit("insufficient free space for source extraction")
    for item in infos:
        path = PurePosixPath(item.filename)
        if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(item.external_attr >> 16):
            raise SystemExit(f"unsafe source ZIP member: {item.filename}")
    manifest = json.loads(packed.read("INPUT_SHA256.json"))
    if set(manifest) != set(names) - {"INPUT_SHA256.json"}: raise SystemExit("source manifest coverage mismatch")
    for name, expected in manifest.items():
        if hashlib.sha256(packed.read(name)).hexdigest() != expected:
            raise SystemExit(f"source member hash mismatch: {name}")
    packed.extractall(target)
PY
fi

cp "$SHARED/config.json" "$ROOT/config.json"
read -r HARD_DEADLINE STARTED_EPOCH < <("$PYTHON" - "$ROOT/config.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1])); print(int(value["pause_deadline_epoch"]), int(value["started_epoch"]))
PY
)
REMAINING=$((HARD_DEADLINE - $(date +%s)))
(( HARD_DEADLINE - STARTED_EPOCH > 0 && HARD_DEADLINE - STARTED_EPOCH <= 14400 ))
(( REMAINING > 600 ))

export PYTHONPATH="$ROOT/source"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export JUPYTER_PATH=/root/fullft8/.venv/share/jupyter
ANTI_PATH="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["antideepfake"])' "$ROOT/config.json")"
mkdir -p "$ANTI_PATH"
if [[ ! -f "$ANTI_PATH/model.safetensors" ]]; then
  cp /shared/voice-music-h100-20260917/model.safetensors "$ANTI_PATH/model.safetensors"
fi
for NAME in antideepfake_config.json model.sha256 native_reference.json parity_audio.npz antideepfake_README.md; do
  if [[ ! -f "$ANTI_PATH/$NAME" && -f "$ROOT/source/antideepfake/$NAME" ]]; then
    cp "$ROOT/source/antideepfake/$NAME" "$ANTI_PATH/$NAME"
  fi
done
(cd "$ANTI_PATH" && sha256sum --check model.sha256)
"$PYTHON" -m ipykernel install --prefix /root/fullft8/.venv --name "$KERNEL" >/dev/null
"$PYTHON" - "$ROOT/config.json" "$ROOT/output/gpu_preflight.json" <<'PY'
import hashlib, importlib, json, os, subprocess, sys, time
from pathlib import Path
config=json.load(open(sys.argv[1])); output=Path(sys.argv[2]); assets=Path(config["assets"]); anti=Path(config["antideepfake"])
for name in ("imageio_ffmpeg","nbformat","nbclient","ipykernel","torch","transformers",
             "eight_contract","eight_features","eight_ablation","eight_probes","eight_lora_train"):
    importlib.import_module(name)
manifest=json.loads((assets/"asset_hashes.json").read_text())
for relative, expected in manifest.items():
    path=assets/relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f"asset identity mismatch: {relative}")
for name in ("antideepfake_config.json","model.safetensors","native_reference.json","parity_audio.npz"):
    if not (anti/name).is_file(): raise SystemExit(f"missing AntiDeepfake asset: {name}")
names=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.splitlines()
if len(names)!=1 or "H100" not in names[0]: raise SystemExit(f"expected exactly one H100, got {names}")
receipt={"status":"passed","gpu_count":1,"gpu_name":names[0],"asset_files_verified":len(manifest),
         "assets":str(assets),"antideepfake":str(anti),"hf_offline":True,"verified_unix":time.time()}
output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
PY

nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used --format=csv
df -B1 "$ROOT"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits --loop=10 > "$ROOT/gpu_telemetry.live.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=30s "${REMAINING}s" "$PYTHON" "$ROOT/source/eight_h100_run.py" "$ROOT/config.json"
