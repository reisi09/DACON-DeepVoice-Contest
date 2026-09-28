"""Build a self-contained, bounded Colab reproduction of the full-data refit."""
import json
from pathlib import Path
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parent
SOURCES = ('full_refit.py', 'run_full_refit.py', 'train_v2.py', 'ast_multitask.py',
           'v2_inference.py', 'graph_features.py', 'graph_fusion.py', 'graph_scoring.py',
           'run_experiment.py', 'package_submission.py', 'package_v2_submission.py', 'AST_LICENSE.txt')


def main() -> None:
    """Embed exact sources and the already uploaded, hash-locked input archives."""
    cells = [nb.v4.new_markdown_cell('# 전체 공개 데이터 1,029개 최종 학습\n\n'
        '새 AudioSet AST에서 head 8 epoch, 마지막 블록 17 epoch를 고정 학습합니다. '
        '모든 파일이 매 epoch에 참여했는지 기록합니다. 기존 평가 자료도 학습에 포함하므로 '
        '독립 자체 평가 점수는 계산하지 않습니다. 완료 후 제출 ZIP을 생성하고 별도 프로세스에서 '
        '오프라인 재현성을 확인합니다. A100은 최대 30분 후 해제됩니다. '
        '이 노트북은 Colab 재현용이며 실행 결과는 실제 실행 후에만 생성됩니다.')]
    cells.append(nb.v4.new_code_cell(textwrap.dedent('''
        from pathlib import Path
        import hashlib, json, os, shutil, subprocess, sys, time, threading, traceback, zipfile
        from google.colab import drive, runtime
        drive.mount('/content/drive')
        DRIVE = Path('/content/drive/MyDrive/DACON_236749')
        WORK = Path('/content/dacon_full_refit')
        WORK.mkdir(exist_ok=True)
        START = time.time()
        DEADLINE = START + 1680
        RUN = DRIVE / ('full_refit_' + time.strftime('%Y%m%d_%H%M%S',time.gmtime(START)))
        RUN.mkdir()
        timer = threading.Timer(1680,runtime.unassign)
        timer.daemon = True
        timer.start()
        print('RESULTS',RUN)
    ''').strip()))
    for name in SOURCES:
        cells.append(nb.v4.new_code_cell(f'%%writefile /content/dacon_full_refit/{name}\n'+(ROOT/name).read_text()))
    archives = json.loads((ROOT/'v2_validation/drive_inputs.json').read_text())
    cells.append(nb.v4.new_code_cell('ARCHIVES = '+repr(archives)))
    cells.append(nb.v4.new_code_cell(textwrap.dedent('''
        OUTPUT = WORK / 'run'
        try:
            with (RUN/'install.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
                    'scikit-learn==1.8.0','soundfile==0.12.1','safetensors'],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            for archive in ARCHIVES:
                packed_path = WORK/archive['name']
                with packed_path.open('wb') as target:
                    for part in archive['parts']:
                        with (DRIVE/part['name']).open('rb') as source:
                            assert hashlib.file_digest(source,'sha256').hexdigest()==part['sha256']
                            source.seek(0)
                            shutil.copyfileobj(source,target,8*1024**2)
                with packed_path.open('rb') as source:
                    assert hashlib.file_digest(source,'sha256').hexdigest()==archive['sha256']
                with zipfile.ZipFile(packed_path) as packed:
                    packed.extractall(WORK)
            assets = next(p.parent for p in WORK.rglob('model.safetensors') if p.parent.name=='ast')
            (RUN/'code').mkdir()
            for source in WORK.iterdir():
                if source.suffix in {'.py','.txt'}:
                    shutil.copy2(source,RUN/'code'/source.name)
            command = [sys.executable,'-u',str(WORK/'run_full_refit.py'),
                '--assets',str(assets),'--data',str(WORK/'v2_validation'),
                '--output',str(OUTPUT),'--device','cuda','--seconds',str(int(DEADLINE-time.time()-120))]
            with (RUN/'experiment.log').open('w') as log:
                process = subprocess.Popen(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'USE_TF':'0'})
                while process.poll() is None:
                    if time.time()>DEADLINE-90:
                        process.terminate()
                        process.wait(timeout=30)
                        raise TimeoutError('full-refit runtime budget reached')
                    time.sleep(15)
                    print((RUN/'experiment.log').read_text().splitlines()[-1:] or ['Preparing...'],flush=True)
                if process.returncode:
                    raise RuntimeError((RUN/'experiment.log').read_text()[-8000:])
            print(json.loads((OUTPUT/'finished.json').read_text()))
        except Exception:
            (RUN/'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            try:
                if OUTPUT.exists():
                    for source in OUTPUT.iterdir():
                        if source.is_file():
                            shutil.copy2(source,RUN/source.name)
                        elif source.name in {'warmup','finetune','final_checkpoint'}:
                            shutil.copytree(source,RUN/source.name,ignore=shutil.ignore_patterns('cache'))
                    for source in OUTPUT.glob('*.zip'):
                        with source.open('rb') as a,(RUN/source.name).open('rb') as b:
                            assert hashlib.file_digest(a,'sha256').hexdigest()==hashlib.file_digest(b,'sha256').hexdigest()
                (RUN/'release_requested.json').write_text(json.dumps({'elapsed_seconds':time.time()-START}))
            finally:
                timer.cancel()
                runtime.unassign()
    ''').strip()))
    notebook = nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'gpuType':'A100','provenance':[]}})
    for cell in cells:
        if cell.cell_type=='code' and not cell.source.startswith('%%'):
            compile(cell.source,'colab_cell','exec')
    nb.validate(notebook)
    nb.write(notebook,ROOT/'DACON_236749_full_refit.ipynb')


if __name__=='__main__':
    main()
