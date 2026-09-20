#!/usr/bin/env bash
set -euo pipefail

STARTED_EPOCH=${1:?original start epoch required}
DEADLINE_EPOCH=${2:?original absolute pause deadline required}
OBJECT=/shared/expanded-h100-20260918
JOB=/root/expanded_h100_20260918_retry1
INPUT=$JOB/input
SOURCE=$INPUT/source
OUTPUT=$JOB/output
PY=/root/fullft8/.venv/bin/python
SYS_PY=$PY
export STARTED_EPOCH DEADLINE_EPOCH

finalize() {
  code=$?
  trap - EXIT
  test -z "${GPU_MONITOR_PID:-}" || kill "$GPU_MONITOR_PID" 2>/dev/null || true
  test -z "${PROGRESS_PID:-}" || kill "$PROGRESS_PID" 2>/dev/null || true
  if [[ "${CREATED_JOB:-false}" == true && -f "$SOURCE/localization_export.py" ]]; then
    EXPORT_SECONDS=$("$PY" -c 'import os,time; print(max(1,int(float(os.environ["DEADLINE_EPOCH"])-time.time()-15)))')
    timeout --signal=TERM --kill-after=5 "$EXPORT_SECONDS" "$PY" "$SOURCE/localization_export.py" --output-dir "$OUTPUT" --object-dir "$OBJECT" \
      --workspace wsp-9683432r72m7 --exit-code "$code" --started-epoch "$STARTED_EPOCH" --deadline-epoch "$DEADLINE_EPOCH" && exit "$code"
  fi
  if [[ ! -x "$PY" ]]; then
    printf '%s\n' '{"status":"failed","archive_ready":false,"reason":"recorded_python_runtime_missing"}' > "$OBJECT/run_state.json"
    exit 1
  fi
  "$SYS_PY" - "$OBJECT" "$STARTED_EPOCH" "$DEADLINE_EPOCH" "$code" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
state={'status':'failed','archive_ready':False,'exit_code':int(sys.argv[4]) or 1,
       'started_epoch':float(sys.argv[2]),'pause_deadline_epoch':float(sys.argv[3]),
       'updated_at':time.time(),'reason':'bootstrap_or_export_failure'}
(root/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
PY
  exit 1
}
trap finalize EXIT

exec 9> /root/expanded_h100_20260918.lock
flock -n 9 || { echo 'another localization bootstrap owns the run lock'; exit 1; }
test ! -e "$JOB"
mkdir -p "$INPUT" "$OUTPUT" "$JOB/cache"
CREATED_JOB=true
test -x "$PY"

"$SYS_PY" - "$OBJECT" "$JOB" <<'PY'
import hashlib,json,math,os,shutil,stat,sys,time,zipfile
from pathlib import Path
obj=Path(sys.argv[1]);job=Path(sys.argv[2])
started=float(os.environ['STARTED_EPOCH']);deadline=float(os.environ['DEADLINE_EPOCH'])
assert math.isfinite(started) and math.isfinite(deadline) and 0 < deadline-started <= 14400
assert time.time() < deadline-600,'insufficient time remaining before export reserve'
run=json.loads((obj/'workspace_run.json').read_text())
assert run['workspace']=='wsp-9683432r72m7' and run['object_prefix']=='expanded-h100-20260918/'
assert abs(run['started_epoch']-started)<1e-4 and abs(run['pause_deadline_epoch']-deadline)<1e-4
manifest=json.loads((obj/'upload_manifest.json').read_text())
assert set(manifest)=={'bootstrap.sh','input.zip'}
for name,record in manifest.items():
    path=obj/name
    assert path.stat().st_size==record['bytes'],name
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert actual==record['sha256'],name
free={str(path):shutil.disk_usage(path).free for path in (job,obj)}
assert min(free.values())>=30_000_000_000,free
with zipfile.ZipFile(obj/'input.zip') as packed:
    assert packed.testzip() is None,'input archive CRC'
    infos=packed.infolist();names=[info.filename for info in infos]
    assert len(names)==len(set(names)),'duplicate archive names'
    assert sum(info.file_size for info in infos)<free[str(job)]-30_000_000_000
    for info in infos:
        path=Path(info.filename)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename
        assert stat.S_IFMT(info.external_attr>>16) in (0,stat.S_IFREG,stat.S_IFDIR)
    packed.extractall(job/'input')
input_root=job/'input';assets=Path('/root/redesign_h100_20260917/assets')
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
member_hashes=json.loads((input_root/'INPUT_SHA256.json').read_text())
for name,expected in member_hashes.items():
    path=input_root/name
    assert path.resolve().is_relative_to(input_root.resolve()) and digest(path)==expected,name
legacy=Path('/shared/localization-a100-20260918/input.zip')
assert digest(legacy)=='00e495257b76e09bcbd91646f98ef9e8286bfc403349679e0018bddaa3dd1c4d'
import csv
rows=list(csv.DictReader((input_root/'dataset/manifest.csv').open()))
needed={'dataset/'+r['path'] for r in rows if not r['ID'].startswith('bank_')}
with zipfile.ZipFile(legacy) as packed:
    for name in packed.namelist():
        if name in needed or name.startswith(('ast_assets/','wheels/')):
            packed.extract(name,input_root)
audio=json.loads((input_root/'audio_upload.json').read_text())
for shard in audio['shards']:
    path=Path('/shared')/shard['prefix']/shard['filename']
    assert path.stat().st_size==shard['bytes'] and digest(path)==shard['sha256']
    with zipfile.ZipFile(path) as packed:
        for info in packed.infolist():
            assert info.filename.startswith('audio/') and '..' not in Path(info.filename).parts
            assert stat.S_IFMT(info.external_attr>>16) in (0,stat.S_IFREG)
        packed.extractall(input_root/'dataset')
for row in rows:
    assert digest(input_root/'dataset'/row['path'])==row['audio_sha256'],row['ID']
required=json.loads((input_root/'data_preparation.json').read_text())['ast_prefix_bytes']+25_000_000_000
assert shutil.disk_usage(job).free>=required,('insufficient cache and export space',required)

asset_hashes=json.loads((input_root/'asset_hashes.json').read_text())
assert len(asset_hashes)==27,'unexpected original model asset manifest'
ast_receipt=json.loads((input_root/'ast_assets_receipt.json').read_text())
assert ast_receipt['repo']=='MIT/ast-finetuned-audioset-10-10-0.4593'
assert ast_receipt['revision']=='f826b80d28226b62986cc218e5cec390b1096902'
for base,hashes in [(assets,asset_hashes),(input_root/'ast_assets',ast_receipt['sha256'])]:
    for name,expected in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()),name
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==expected,name
assert shutil.disk_usage(job).free>=30_000_000_000
shutil.copytree(input_root/'source',job/'output'/'source')
for source,name in [(input_root/'dataset'/'manifest.csv','dataset_manifest.csv'),
                    (input_root/'asset_hashes.json','asset_hashes.json'),
                    (input_root/'ast_assets_receipt.json','ast_assets_receipt.json')]:
    shutil.copyfile(source,job/'output'/name)
(job/'output'/'storage_preflight.json').write_text(json.dumps({
    'free_bytes_before_extract':free,'original_asset_files_verified':len(asset_hashes),
    'ast_revision':ast_receipt['revision'],'ast_asset_files_verified':len(ast_receipt['sha256']),
    'input_sha256':manifest['input.zip']['sha256'],'checked_at':time.time(),
},indent=2)+'\n')
state={**run,'status':'running','archive_ready':False,'phase':'preflight','updated_at':time.time()}
(obj/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
PY

export PYTHONPATH="$SOURCE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
shopt -s nullglob
wheels=("$INPUT"/wheels/*.whl)
test "${#wheels[@]}" -gt 0
"$PY" -m ensurepip --upgrade > "$OUTPUT/pip_setup.log" 2>&1
"$PY" -m pip install --disable-pip-version-check --no-deps --no-index --target "$JOB/runtime" \
  "${wheels[@]}" > "$OUTPUT/runtime_install.log" 2>&1
export PYTHONPATH="$JOB/runtime:$SOURCE"
"$PY" - "$OUTPUT" <<'PY'
import hashlib,importlib.metadata,json,platform,sys,time,torch,torchaudio,transformers
from pathlib import Path
import localization_features,localization_train
assert torch.cuda.is_available() and torch.cuda.device_count()==1
props=torch.cuda.get_device_properties(0)
assert 'H100' in props.name and props.total_memory>=79*1024**3
assert torch.__version__.split('+')[0]==torchaudio.__version__.split('+')[0]
assert transformers.__version__=='4.57.6'
job=Path(sys.argv[1]).parent
assert Path(transformers.__file__).resolve().is_relative_to((job/'runtime').resolve())
packages={}
for name in (
    'torch','torchaudio','torchvision','transformers','numpy','scipy','soundfile','safetensors',
    'scikit-learn','pandas','librosa','demucs','panns-inference','tokenizers','huggingface-hub'):
    try:packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:packages[name]=None
for name,version in [('tokenizers','0.22.2'),('huggingface-hub','0.36.2'),('safetensors','0.8.0')]:
    assert packages[name]==version,(name,packages[name])
wheel_hashes={}
for path in sorted((job/'input'/'wheels').glob('*.whl')):
    with path.open('rb') as stream:wheel_hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
receipt={'python':platform.python_version(),'executable':sys.executable,'packages':packages,
         'gpu':props.name,'gpu_memory_bytes':props.total_memory,'cuda':torch.version.cuda,
         'runtime_overlay':str(job/'runtime'),'wheel_sha256':wheel_hashes,'checked_at':time.time()}
Path(sys.argv[1],'runtime_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 > "$OUTPUT/gpu_telemetry.csv" &
GPU_MONITOR_PID=$!
"$PY" - "$OUTPUT" "$OBJECT" <<'PY' &
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1]);objects=Path(sys.argv[2])
while True:
    progress={'updated_at':time.time()}
    for name,relative in [('features','feature_progress.json'),('training','train/progress.json'),
                          ('mert_batch','mert_batch_benchmark.json'),('ast_batch','ast_prefix_batch_benchmark.json')]:
        path=output/relative
        if path.is_file():
            try:progress[name]=json.loads(path.read_text())
            except json.JSONDecodeError:progress[name]={'read_pending':True}
    logs=[path for path in (output/'features.log',output/'training.log') if path.is_file()]
    if logs:
        latest=max(logs,key=lambda path:path.stat().st_mtime)
        with latest.open('rb') as stream:
            stream.seek(max(0,latest.stat().st_size-2000))
            progress['log_tail']=stream.read().decode('utf-8',errors='replace')
    temporary=objects/'progress.json.partial'
    temporary.write_text(json.dumps(progress,indent=2)+'\n')
    temporary.replace(objects/'progress.json')
    time.sleep(300)
PY
PROGRESS_PID=$!
export EXPANDED_EXECUTE_GPU=1 EXPANDED_JOB_ROOT="$JOB"
export JUPYTER_PATH="$JOB/kernel/share/jupyter:/root/fullft8/.venv/share/jupyter"
"$PY" -m ipykernel install --prefix "$JOB/kernel" --name expanded-h100 > "$OUTPUT/kernel_install.log" 2>&1
REMAINING=$("$PY" -c 'import os,time; print(max(1,int(float(os.environ["DEADLINE_EPOCH"])-600-time.time())))')
timeout --signal=TERM --kill-after=30 "$REMAINING" "$PY" - "$INPUT/DACON_Expanded_H100_Experiments.ipynb" "$OUTPUT/DACON_Expanded_H100_Executed.ipynb" <<'PYNOTEBOOK' > "$OUTPUT/notebook.log" 2>&1
import nbformat,sys
from nbclient import NotebookClient
from pathlib import Path
notebook=nbformat.read(sys.argv[1],as_version=4)
output=Path(sys.argv[2])
client=NotebookClient(notebook,timeout=None,kernel_name='expanded-h100',resources={'metadata':{'path':str(output.parent.parent)}})
client.on_cell_executed=lambda **kwargs: nbformat.write(notebook,output)
try:
    client.execute()
finally:
    nbformat.write(notebook,output)
PYNOTEBOOK
