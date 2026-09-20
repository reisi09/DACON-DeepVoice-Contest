"""Assemble an inspectable Colab notebook with the actual runtime source cells."""

import textwrap
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parent


def code(source: str):
    return nb.v4.new_code_cell(textwrap.dedent(source).strip())


def main() -> None:
    graph = (ROOT / 'PIPELINE_GRAPH.md').read_text()
    graph = graph.replace('(COMPETITION_CONTRACT.md)', '(https://drive.google.com/file/d/1-LAwzjSsw6a61x2axDcvfdY4spmPIzct/view)')
    graph = graph.replace('(DATA_SOURCES.md)', '(https://drive.google.com/file/d/1dG3zSqvw_zQZXcs8lrkdzc0JhmocHVFN/view)')
    cells = [nb.v4.new_markdown_cell(graph)]
    cells.append(nb.v4.new_markdown_cell('## 1. Drive 연결과 2시간 예산\nA100을 선택한 뒤 순서대로 실행합니다. 데이터·결과는 `MyDrive/DACON_236749`에 저장합니다. 종료 시 런타임을 해제합니다.'))
    cells.append(code('''
        import time, json, shutil, subprocess, sys, zipfile, hashlib, threading
        from pathlib import Path
        SESSION_START = globals().get('SESSION_START', time.time())
        from google.colab import drive, runtime
        drive.mount('/content/drive')
        DRIVE = Path('/content/drive/MyDrive/DACON_236749')
        WORK = Path('/content/dacon236749')
        WORK.mkdir(exist_ok=True)
        RUN = DRIVE / 'runs' / time.strftime('%Y%m%d_%H%M%S', time.gmtime(SESSION_START))
        RUN.mkdir(parents=True, exist_ok=True)
        HARD_DEADLINE = SESSION_START + 7200
        (RUN / 'budget.json').write_text(json.dumps({'start_unix': SESSION_START, 'hard_deadline': HARD_DEADLINE, 'max_gpu_seconds': 7200}))
        def budget_stop():
            try:
                (RUN / 'budget_stop.json').write_text(json.dumps({'time': time.time(), 'reason': 'two_hour_limit'}))
            finally:
                runtime.unassign()
        watchdog = threading.Timer(max(1, HARD_DEADLINE - time.time() - 30), budget_stop)
        watchdog.daemon = True
        watchdog.start()
        print('Drive:', DRIVE, '\\nResults:', RUN, '\\nBudget: <=120 minutes')
    '''))
    cells.append(nb.v4.new_markdown_cell('## 2. 실행 환경과 Drive 데이터 준비\n기존 Colab의 torch/torchaudio를 사용하고 필요한 추론 라이브러리를 설치합니다. 실제 설치 버전과 GPU는 결과에 기록됩니다.'))
    cells.append(code('''
        packages = ['demucs==4.0.1', 'panns-inference==0.1.1', 'transformers==4.57.6',
                    'librosa==0.10.2.post1', 'diffq==0.2.4', 'einops==0.8.1', 'scikit-learn==1.8.0', 'soundfile']
        with (RUN / 'install.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
        print((RUN / 'install.log').read_text()[-6000:])
        result.check_returncode()
        for name in ('open.zip', 'validation.zip'):
            if not (DRIVE / name).is_file():
                raise FileNotFoundError(f'Upload {name} to {DRIVE} before running')
            shutil.copy2(DRIVE / name, WORK / name)
        with zipfile.ZipFile(WORK / 'open.zip') as archive:
            archive.extractall(WORK / 'official')
        original = next((WORK / 'official').rglob('baseline_submit.zip'))
        with original.open('rb') as stream:
            actual_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual_sha == 'cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70', 'Official ZIP differs'
        BASELINE = WORK / 'baseline'
        with zipfile.ZipFile(original) as archive:
            archive.extractall(BASELINE)
        public_data = next(p for p in (WORK / 'official').rglob('sample_submission.csv')).parent
        shutil.copytree(public_data, BASELINE / 'data', dirs_exist_ok=True)
        with zipfile.ZipFile(WORK / 'validation.zip') as archive:
            archive.extractall(WORK)
        print('Prepared:', BASELINE, WORK / 'validation')
    '''))
    cells.append(nb.v4.new_markdown_cell('## 3. 재현 가능한 실행 코드\n아래 셀은 실제 특징 추출·5개 그래프·공식 지표·패키징 코드입니다.'))
    for name in ('graph_fusion.py', 'graph_scoring.py', 'graph_features.py', 'run_experiment.py', 'package_submission.py'):
        cells.append(nb.v4.new_code_cell(f'%%writefile /content/dacon236749/{name}\n' + (ROOT / name).read_text()))
    cells.append(nb.v4.new_markdown_cell('## 4. 실제 A100 비교 실행\n특징 추출은 파일마다 Drive에 저장됩니다. 마지막 20분은 패키징·재실행·결과 저장에 남깁니다. 아래 로그의 완료 여부를 확인합니다.'))
    cells.append(code('''
        import os
        os.chdir(WORK)
        experiment_deadline = min(SESSION_START + 6000, HARD_DEADLINE - 1200)
        cmd = [sys.executable, '-u', 'run_experiment.py', '--baseline', str(BASELINE),
               '--validation', str(WORK / 'validation'), '--output', str(RUN),
               '--deadline', str(experiment_deadline)]
        with (RUN / 'experiment.log').open('w', buffering=1) as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.time() >= HARD_DEADLINE - 900:
                    process.terminate()
                    process.wait(timeout=30)
                    raise TimeoutError('Stopped to preserve packaging and save time')
                time.sleep(15)
                tail = (RUN / 'experiment.log').read_text().splitlines()
                print(tail[-1] if tail else 'Loading models...', flush=True)
        if process.returncode:
            print((RUN / 'experiment.log').read_text()[-15000:])
            raise RuntimeError(f'Experiment exited with {process.returncode}; inspect log and resume cached files')
        print((RUN / 'experiment.log').read_text()[-12000:])
    '''))
    cells.append(nb.v4.new_markdown_cell('## 5. 결과 비교와 실제 ZIP 재실행\n총점이 `None`이면 성분 정답이나 클래스가 부족한 것입니다. 부분 지표 기반 선택값을 DACON 점수로 해석하지 않습니다.'))
    cells.append(code('''
        import csv, numpy as np
        from IPython.display import display, Markdown
        from package_submission import build_submission, verify_packaged
        comparison = json.loads((RUN / 'comparison.json').read_text())
        display(comparison)
        selected = comparison['selection']['graph']
        fusion = json.loads((RUN / 'fusion.json').read_text())
        submission_zip = RUN / f'{selected}_submission.zip'
        build_submission(original, submission_zip, {'graph': selected, 'fusion': fusion})
        fresh = WORK / 'packaged_check'
        with zipfile.ZipFile(submission_zip) as archive:
            archive.extractall(fresh)
        shutil.copytree(BASELINE / 'data', fresh / 'data', dirs_exist_ok=True)
        rows = verify_packaged(fresh, HARD_DEADLINE - 120)
        with (RUN / f'{selected}_dummy_submission.csv').open() as stream:
            reference = list(csv.DictReader(stream))
        assert [r['ID'] for r in rows] == [r['ID'] for r in reference]
        columns = [k for k in rows[0] if k != 'ID']
        difference = max(abs(float(a[k]) - float(b[k])) for a, b in zip(rows, reference) for k in columns)
        assert difference <= 1e-5, f'Packaged predictions differ: {difference}'
        report = {'graph': selected, 'fresh_offline_script_executed': True,
                  'max_probability_delta': difference, 'submission_zip': str(submission_zip),
                  'completed_unix': time.time()}
        (RUN / 'packaged_verification.json').write_text(json.dumps(report, indent=2))
        display(report)
    '''))
    cells.append(nb.v4.new_markdown_cell('## 6. 저장 확인 후 GPU 해제\nColab 실행 기록은 이 노트북에 저장되며, 원본 특징·예측·지표·로그·선택 ZIP은 위의 Drive 결과 폴더에 남습니다.'))
    cells.append(code('''
        (RUN / 'finished.json').write_text(json.dumps({'status': 'finished', 'elapsed_seconds': time.time() - SESSION_START,
            'comparison': str(RUN / 'comparison.json'), 'note': 'DACON scoring requires a separate server submission'}))
        print('FINISHED', RUN, 'GPU elapsed seconds:', round(time.time() - SESSION_START))
        runtime.unassign()
        watchdog.cancel()
    '''))
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': 'DACON_236749_five_graphs.ipynb', 'provenance': [], 'gpuType': 'A100'}})
    nb.validate(notebook)
    nb.write(notebook, ROOT / 'DACON_236749_five_graphs.ipynb')


if __name__ == '__main__':
    main()
