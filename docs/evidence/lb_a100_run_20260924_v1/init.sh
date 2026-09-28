#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LB_DEADLINE=1790256172
LB_SHARED=/shared
LB_PREFIX=lb-improvement-20260924/a100-run-v1/
LB_WORK=/root/lb-a100-20260924-v1
LB_BOOT=/tmp/lb-a100-bootstrap-v1
test ! -e "$LB_SHARED/$LB_PREFIX/run_started.json"
python - "$LB_SHARED/$LB_PREFIX/bootstrap_bundle.zip" "$LB_BOOT" <<'PY'
from pathlib import Path,PurePosixPath
import hashlib,sys,zipfile,stat
archive,root=map(Path,sys.argv[1:])
with archive.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!='357f7e387ae6367f29974a3b582fbcb3e874a68b503b4dc42558030dbf7a5162':raise ValueError('Bootstrap hash differs')
with zipfile.ZipFile(archive) as z:
 names=set()
 for i in z.infolist():
  p=PurePosixPath(i.filename)
  if i.is_dir() or p.is_absolute() or '..' in p.parts or str(p)!=i.filename or i.filename in names or stat.S_IFMT(i.external_attr>>16) not in (0,stat.S_IFREG):raise ValueError('Unsafe bootstrap member')
  names.add(i.filename)
 if len(names)!=6:raise ValueError('Wrong bootstrap member count')
 root.mkdir(exist_ok=False)
 z.extractall(root)
PY
set +e
LB_REMAIN=$((LB_DEADLINE - 360 - $(date +%s)))
if [ "$LB_REMAIN" -gt 0 ]; then
 timeout --kill-after=20s "${LB_REMAIN}s" python -u "$LB_BOOT/a100_workspace_bootstrap.py" run --work "$LB_WORK" --shared "$LB_SHARED" --prefix "$LB_PREFIX" --deadline "$LB_DEADLINE"
 LB_RUN_CODE=$?
else
 LB_RUN_CODE=124
fi
LB_REMAIN=$((LB_DEADLINE - 60 - $(date +%s)))
if [ "$LB_REMAIN" -gt 0 ]; then
 timeout --kill-after=20s "${LB_REMAIN}s" python -u "$LB_BOOT/a100_workspace_bootstrap.py" export --work "$LB_WORK" --shared "$LB_SHARED" --prefix "$LB_PREFIX" --deadline "$LB_DEADLINE"
 LB_EXPORT_CODE=$?
else
 LB_EXPORT_CODE=124
fi
python - "$LB_SHARED/$LB_PREFIX/bootstrap_exit.json" "$LB_RUN_CODE" "$LB_EXPORT_CODE" "$LB_DEADLINE" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(run_returncode=int(sys.argv[2]),export_returncode=int(sys.argv[3]),pause_deadline_epoch=int(sys.argv[4]),finished_epoch=time.time()),indent=2))
PY
exit "$LB_RUN_CODE"
