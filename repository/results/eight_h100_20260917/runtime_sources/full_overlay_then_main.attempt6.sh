#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/eight_h100_20260917
SHARED=/shared/eight-h100-20260917
PYTHON=/root/fullft8/.venv/bin/python
mkdir -p "$ROOT" "$SHARED"

if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$0" --worker </dev/null > "$ROOT/full_overlay_then_main.log" 2>&1 &
  echo "$!" > "$ROOT/full_overlay_then_main.pid"
  exit 0
fi

export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/source"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

fail_state() {
  code=$?
  trap - ERR
  cp "$ROOT/full_overlay_then_main.log" "$SHARED/full_overlay_failure.log" 2>/dev/null || true
  "$PYTHON" - "$SHARED/run_state.json" "$code" <<'PY' || true
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); code=int(sys.argv[2])
try: state=json.loads(p.read_text())
except Exception: state={}
state.update(status='failed',stage='full_overlay',archive_ready=False,updated_unix=time.time(),error=f'full overlay bootstrap exit {code}')
tmp=p.with_suffix('.json.tmp'); tmp.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n'); tmp.replace(p)
PY
  exit "$code"
}
trap fail_state ERR

"$PYTHON" - "$ROOT/source" "$SHARED/runtime_overrides_full" "$SHARED/runtime_override_receipt.json" <<'PY'
import hashlib,json,shutil,sys,time
from pathlib import Path
source,overrides,receipt_path=map(Path,sys.argv[1:])
input_manifest_sha='c21e12f9fc4b4ad9c29763ef403a8624f8d09d1f7ea43bc1e5e469ba9c0fe2b1'
manifest_path=source/'INPUT_SHA256.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(manifest_path)!=input_manifest_sha: raise SystemExit('source INPUT manifest mismatch')
manifest=json.loads(manifest_path.read_text())
declared={
 'eight_features.py':('17ffe77d2f16019db24061d3c52317a70918d580dbe92ff8960e3d328cfd6760','b0304f4f9aa2d13524d56622f494c4b1fab160b5009e09a51fb77987ed23acac',{'f399b991e694e5cbddd69791499829351f5e2d30c69166911f93e79a2e643a4d'}),
 'antideepfake_runtime.py':('abf775389fdbe7a97793e9b2ef59f764d5eb312fafd286c054e3ca583b38be31','bbf7ec7bc0b20afa416e7a6fac535d8ab92d984621c1e9be3e395bb7f3d0f729',set()),
 'eight_feature_schema.py':('fb9eb35e299b42a3e2646c42fa5bec251a9beb5e50b2781fef63c72acd0f7d7c','cdc62ffb8da7773deee4d030c68ca7fda1cd2cb99ce2294a866e88ba79d5e9ff',set()),
 'eight_feature_identity.py':('d5a772fc2198ecbff1f56d6f93e431113d56e7adcd091980cf622042a4b4d62c','edb796b9c99b1083ec839743694cf9738230ba0f2eac6b14a00c5886a1d45bb2',set()),
 'eight_probes.py':('30e1079747d683c1f8fcfd6b215db7af4fd812755b305034e3a400f8913ea9fb','1661764b7214f5b7cdc2724656aaf8fe865bda11ffec920ec86067fbceb10f52',set()),
 'eight_ablation.py':('b44ca59ffc10cc3bc58a3ff2074c92186ee5f9cb52365253aefc0201d0f1abee','5d6c8fc7b214bcae3701a0ead2798fc97dfc1e1fea34023d37a2a41d10d12246',set()),
 'eight_lora.py':('b6504eb5531f0fd3fa683872869b3b4babfd1838c92db27d6899c1b173593716','be8828a1eb51dc7a8c8059abfc86ffbd4dda25d27e6b98a0932867a816c1f579',set()),
}
actual=set()
for path in source.rglob('*'):
    if not path.is_file(): continue
    rel=path.relative_to(source)
    if '__pycache__' in rel.parts and rel.suffix=='.pyc': continue
    actual.add(rel.as_posix())
if actual!=set(manifest)|{'INPUT_SHA256.json'}: raise SystemExit('source member coverage mismatch')
files={}
for name,expected in manifest.items():
    got=sha(source/name)
    if name not in declared:
        if got!=expected: raise SystemExit(f'unchanged source mismatch: {name}')
        continue
    original,runtime,intermediate=declared[name]
    if expected!=original or got not in {original,runtime}|intermediate: raise SystemExit(f'declared source mismatch: {name}')
    candidate=overrides/name
    if sha(candidate)!=runtime: raise SystemExit(f'override readback mismatch: {name}')
    before=got
    if got!=runtime: shutil.copyfile(candidate,source/name)
    after=sha(source/name)
    if after!=runtime: raise SystemExit(f'override application failed: {name}')
    files[name]={'original':original,'runtime':runtime,'before':before,'after':after}
receipt={'status':'passed','source_archive_sha256':'2606dbc7fbdb98b024d781467c17528b95f94e58486d1be77d0be7efff12d363',
         'input_manifest_sha256':input_manifest_sha,'declared_override_count':len(declared),
         'unchanged_members_verified':len(manifest)-len(declared),'files':files,'applied_unix':time.time()}
tmp=receipt_path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n'); tmp.replace(receipt_path)
PY

if [[ -d "$ROOT/output" ]] && find "$ROOT/output" -mindepth 1 -print -quit | grep -q .; then
  [[ ! -e "$ROOT/failed_attempt_05_output" ]]
  mv "$ROOT/output" "$ROOT/failed_attempt_05_output"
fi
mkdir -p "$ROOT/output"
cp "$SHARED/runtime_override_receipt.json" "$ROOT/output/runtime_override_receipt.json"
cp "$SHARED/config.json" "$ROOT/config.json"
exec bash "$SHARED/eight_h100_run_retry.sh" --worker
