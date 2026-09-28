"""Build the new Colab notebook with embedded, reviewable experiment sources."""

import json
import textwrap
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = (
    "graph_features.py", "graph_fusion.py", "graph_scoring.py", "run_experiment.py",
    "package_submission.py", "ast_multitask.py", "v2_scoring.py", "train_v2.py",
    "v2_inference.py", "package_v2_submission.py", "run_v2.py",
    "v2_data.py", "prepare_v2_sources.py", "prepare_v2_validation.py",
)


def code(source: str):
    return nb.v4.new_code_cell(textwrap.dedent(source).strip())


def main() -> None:
    assets = json.loads((ROOT / "v2_validation" / "drive_inputs.json").read_text())
    cells = [nb.v4.new_markdown_cell((ROOT / "PIPELINE_GRAPH_V2.md").read_text())]
    cells.append(nb.v4.new_markdown_cell("## 1. Drive 연결\n데이터와 모델 업로드를 마친 뒤 실행합니다. 최대 할당 시간은 2시간이며, 실행 오류와 정상 종료 모두 로그를 저장한 뒤 GPU를 해제합니다."))
    cells.append(code('''
        import time, json, shutil, subprocess, sys, zipfile, hashlib, threading, traceback
        from pathlib import Path
        from google.colab import drive, runtime
        SESSION_START = time.time()
        HARD_DEADLINE = SESSION_START + 7080
        drive.mount('/content/drive')
        DRIVE = Path('/content/drive/MyDrive/DACON_236749')
        WORK = Path('/content/dacon236749_v2')
        WORK.mkdir(exist_ok=True)
        RUN = DRIVE / 'runs_v2' / time.strftime('%Y%m%d_%H%M%S', time.gmtime(SESSION_START))
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / 'budget.json').write_text(json.dumps({'start_unix':SESSION_START, 'hard_deadline':HARD_DEADLINE,
            'max_seconds_after_first_cell':7080, 'allocation_limit_seconds':7200}))
        def budget_stop():
            try:
                (RUN / 'budget_stop.json').write_text(json.dumps({'unix':time.time(), 'reason':'allocation_limit'}))
            finally:
                runtime.unassign()
        watchdog = threading.Timer(max(1,HARD_DEADLINE-time.time()), budget_stop)
        watchdog.daemon = True
        watchdog.start()
        print('RESULTS_DIRECTORY', RUN, flush=True)
    '''))
    cells.append(nb.v4.new_markdown_cell("## 2. 실제 실행 코드\n이 셀들의 내용을 실행 폴더와 Drive에 저장합니다. 데이터 분리·학습·선택·평가·패키징을 코드에서 확인할 수 있습니다."))
    for name in RUNTIME_FILES:
        source = (ROOT / name).read_text()
        compile(source, name, "exec")
        cells.append(nb.v4.new_code_cell(f"%%writefile /content/dacon236749_v2/{name}\n" + source))
    cells.append(nb.v4.new_code_cell("%%writefile /content/dacon236749_v2/AST_LICENSE.txt\n" + (ROOT / "AST_LICENSE.txt").read_text()))
    cells.append(nb.v4.new_code_cell("INPUT_ARCHIVES = " + repr(assets)))
    cells.append(nb.v4.new_markdown_cell("## 3. 설치·데이터 검증·A100 비교·저장\n학습에는 개발 manifest만 전달합니다. 선택과 가중치 해시를 확정한 다음 새 평가 집합을 엽니다. 각 epoch, 예측, 선택 근거, 제출 ZIP 재실행 결과는 Drive 실행 폴더에 남습니다."))
    cells.append(code('''
        import os
        try:
            (RUN / 'code').mkdir(exist_ok=True)
            for p in WORK.glob('*.py'):
                shutil.copy2(p, RUN / 'code' / p.name)
            shutil.copy2(WORK / 'AST_LICENSE.txt', RUN / 'code' / 'AST_LICENSE.txt')
            (RUN / 'input_archives.json').write_text(json.dumps(INPUT_ARCHIVES, indent=2))
            packages = ['demucs==4.0.1','panns-inference==0.1.1','transformers==4.57.6',
                        'librosa==0.10.2.post1','diffq==0.2.4','einops==0.8.1',
                        'scikit-learn==1.8.0','soundfile==0.12.1','safetensors']
            with (RUN / 'install.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pip','install','-q',*packages],
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            print('INSTALL_FINISHED', flush=True)
            def sha256(path):
                with path.open('rb') as source:
                    return hashlib.file_digest(source,'sha256').hexdigest()
            for archive in INPUT_ARCHIVES:
                destination = WORK / archive['name']
                with destination.open('wb') as output:
                    for part in archive['parts']:
                        source = DRIVE / part['name']
                        assert sha256(source) == part['sha256'], f"Drive part differs: {source}"
                        with source.open('rb') as incoming:
                            shutil.copyfileobj(incoming,output,8*1024**2)
                assert sha256(destination) == archive['sha256'], f"Archive differs: {destination}"
                with zipfile.ZipFile(destination) as packed:
                    packed.extractall(WORK)
                print('VERIFIED_ARCHIVE', destination.name, flush=True)
            shutil.copy2(DRIVE / 'open.zip', WORK / 'open.zip')
            with zipfile.ZipFile(WORK / 'open.zip') as packed:
                packed.extractall(WORK / 'official')
            original = next((WORK / 'official').rglob('baseline_submit.zip'))
            assert sha256(original) == 'cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70'
            BASELINE = WORK / 'baseline'
            with zipfile.ZipFile(original) as packed:
                packed.extractall(BASELINE)
            public = next((WORK / 'official').rglob('sample_submission.csv')).parent
            shutil.copytree(public,BASELINE / 'data',dirs_exist_ok=True)
            os.chdir(WORK)
            cmd = [sys.executable,'-u','run_v2.py','--assets',str(WORK / 'ast'),
                   '--data',str(WORK / 'v2_validation'),'--baseline',str(BASELINE),
                   '--original-zip',str(original),'--output',str(RUN),'--work',str(WORK),
                   '--deadline',str(HARD_DEADLINE)]
            with (RUN / 'experiment.log').open('w',buffering=1) as log:
                process = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if time.time() >= HARD_DEADLINE - 60:
                        process.terminate()
                        process.wait(timeout=30)
                        raise TimeoutError('Stopped at allocation limit')
                    time.sleep(15)
                    lines = (RUN / 'experiment.log').read_text().splitlines()
                    print(lines[-1][:1000] if lines else 'Loading models...', flush=True)
                if process.returncode:
                    print((RUN / 'experiment.log').read_text()[-16000:])
                    raise RuntimeError(f'Experiment exited with {process.returncode}')
            from IPython.display import display
            display(json.loads((RUN / 'comparison.json').read_text()))
            display(json.loads((RUN / 'packaged_verification.json').read_text()))
            print('V2_SAVED',RUN,'Elapsed seconds',round(time.time()-SESSION_START),flush=True)
        except Exception:  # noqa: BROAD_EXCEPT_OK
            (RUN / 'failure.txt').write_text(traceback.format_exc())
            print(traceback.format_exc(),flush=True)
            raise
        finally:
            (RUN / 'release_requested.json').write_text(json.dumps({'unix':time.time(),
                'elapsed_seconds':time.time()-SESSION_START}))
            watchdog.cancel()
            runtime.unassign()
    '''))
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "DACON_236749_v2_ast.ipynb", "provenance": [], "gpuType": "A100"}})
    nb.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code" and not cell.source.startswith("%%writefile"):
            compile(cell.source, "colab_cell", "exec")
    nb.write(notebook, ROOT / "DACON_236749_v2_ast.ipynb")


if __name__ == "__main__":
    main()
