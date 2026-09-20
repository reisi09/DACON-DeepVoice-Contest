"""Build the bounded GPU diagnosis and dev-locked hybrid deployment notebook."""
import textwrap
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parent


def cell(source):
    return nb.v4.new_code_cell(textwrap.dedent(source).strip())


def main():
    cells = [nb.v4.new_markdown_cell("# V2 GPU 정밀도 진단과 고정 조합 재실행\n\n첫 실험의 학습·새 평가 결과는 보존합니다. 여기서는 같은 가중치로 ZIP 예측 차이를 재현하고, 원인이 확인될 때만 런타임 정밀도 설정을 수정합니다. 개발 자료에서 선택한 G0_AST_MUSIC 조합도 실제 ZIP으로 실행합니다. 추가 학습이나 새 평가를 이용한 모델 선택은 하지 않습니다. 이번 할당은 최대 30분입니다.")]
    cells.append(cell('''
        import time, json, shutil, subprocess, sys, zipfile, hashlib, threading, traceback, os
        from pathlib import Path
        from google.colab import drive, runtime
        SESSION_START = time.time()
        DEADLINE = SESSION_START + 1680
        drive.mount('/content/drive')
        DRIVE = Path('/content/drive/MyDrive/DACON_236749')
        PRIMARY = DRIVE / 'runs_v2/20260912_174330'
        RUN = PRIMARY / 'replay_verified'
        RUN.mkdir(exist_ok=True)
        WORK = Path('/content/dacon236749_replay')
        WORK.mkdir(exist_ok=True)
        (RUN / 'budget.json').write_text(json.dumps({'start_unix':SESSION_START,'deadline':DEADLINE,
            'first_allocation_elapsed_seconds':812.4415035247803,'second_allocation_limit_seconds':1800}))
        watchdog = threading.Timer(max(1,DEADLINE-time.time()),runtime.unassign)
        watchdog.daemon = True
        watchdog.start()
        for source in (PRIMARY / 'code').iterdir():
            if source.is_file():
                shutil.copy2(source,WORK / source.name)
        print('REPLAY_RESULTS',RUN,flush=True)
    '''))
    for name in ("package_hybrid_submission.py", "verify_v2_gpu_precision.py"):
        source = (ROOT / name).read_text()
        compile(source, name, "exec")
        cells.append(nb.v4.new_code_cell(f"%%writefile /content/dacon236749_replay/{name}\n" + source))
    cells.append(cell('''
        try:
            packages = ['demucs==4.0.1','panns-inference==0.1.1','transformers==4.57.6',
                        'librosa==0.10.2.post1','diffq==0.2.4','einops==0.8.1',
                        'scikit-learn==1.8.0','soundfile==0.12.1','safetensors']
            with (RUN / 'install.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pip','install','-q',*packages],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            checkpoint = WORK / 'checkpoint'
            shutil.copytree(PRIMARY / 'training/AST_LAST_BLOCK/best',checkpoint,dirs_exist_ok=True)
            inputs = WORK / 'inputs'
            shutil.copytree(PRIMARY / 'package_inputs',inputs,dirs_exist_ok=True)
            old_package = WORK / 'old_AST_LAST_BLOCK_submission.zip'
            shutil.copy2(PRIMARY / 'AST_LAST_BLOCK_submission.zip',old_package)
            os.chdir(WORK)
            sys.path.insert(0,str(WORK))
            from graph_features import atomic_json, digest
            from package_submission import verify_packaged
            from package_v2_submission import build_ast_submission
            from package_hybrid_submission import build_hybrid_submission, route_predictions
            from run_experiment import read_csv, write_csv
            from graph_scoring import SUBMISSION_COLUMNS
            lock = json.loads((PRIMARY / 'hybrid_candidate_lock.json').read_text())
            assert lock['selected'] == 'G0_AST_MUSIC'
            assert digest(PRIMARY / 'hybrid_selection.json') == lock['selection_sha256']
            for name, expected in lock['checkpoints']['AST_LAST_BLOCK'].items():
                assert digest(checkpoint / name) == expected, f'Checkpoint changed: {name}'
            diagnostic = WORK / 'precision'
            cmd = [sys.executable,'-u','verify_v2_gpu_precision.py','--checkpoint',str(checkpoint),
                   '--package',str(old_package),'--inputs',str(inputs),'--work',str(diagnostic),
                   '--output',str(RUN / 'precision_diagnostic.json'),'--deadline',str(DEADLINE-600)]
            with (RUN / 'precision_run.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,DEADLINE-time.time()-600))
            diagnosis = json.loads((RUN / 'precision_diagnostic.json').read_text())
            for source in diagnostic.iterdir():
                if source.is_file():
                    shutil.copy2(source,RUN / source.name)
            print('PRECISION_DIAGNOSIS',diagnosis['diagnosis'],diagnosis['variants'],flush=True)
            assert diagnosis['diagnosis'] in ('confirmed_cudnn_tf32','confirmed_process_precision_state'), 'Precision cause not established'
            source_path = WORK / 'v2_inference.py'
            before = source_path.read_text()
            anchor = '    checkpoint = root / "model" / "ast"'
            assert before.count(anchor) == 1
            fixed = before.replace(anchor,'    torch.backends.cuda.matmul.allow_tf32 = False\\n    torch.backends.cudnn.allow_tf32 = False\\n'+anchor,1)
            source_path.write_text(fixed)
            compile(fixed,'v2_inference.py','exec')
            atomic_json(RUN / 'precision_fix.json',{'reason':diagnosis['diagnosis'],
                'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
                'after_sha256':digest(source_path),'threshold':1e-5,'weights_changed':False})
            with zipfile.ZipFile(DRIVE / 'open.zip') as archive:
                member = next(n for n in archive.namelist() if n.endswith('baseline_submit.zip'))
                original = Path(archive.extract(member,WORK / 'original'))
            assert digest(original) == lock['original_zip_sha256']
            baseline = WORK / 'baseline'
            with zipfile.ZipFile(original) as archive:
                archive.extractall(baseline)
            from run_v2 import baseline_predictions
            input_rows = [{'ID':r['ID'],'path':'test/'+r['ID']+'.wav'} for r in read_csv(inputs / 'sample_submission.csv')]
            g0 = baseline_predictions(baseline,input_rows,inputs,RUN,DEADLINE-300)
            ast_reference = [{c:r[c] for c in SUBMISSION_COLUMNS} for r in read_csv(diagnostic / 'reference.csv')]
            expected = {'AST_LAST_BLOCK':ast_reference,'G0_AST_MUSIC':route_predictions(g0,ast_reference)}
            guard = WORK / 'network_guard'
            guard.mkdir(exist_ok=True)
            (guard / 'sitecustomize.py').write_text('import socket\\ndef deny(*args, **kwargs):\\n    raise RuntimeError("network forbidden in offline replay")\\nsocket.socket.connect = deny\\nsocket.create_connection = deny\\n')
            receipts = {}
            for name in expected:
                package = WORK / (name+'_submission.zip')
                if name == 'AST_LAST_BLOCK':
                    build_ast_submission(checkpoint,package)
                else:
                    build_hybrid_submission(original,checkpoint,package)
                unpacked = WORK / ('verified_'+name)
                with zipfile.ZipFile(package) as archive:
                    expanded = sum(x.file_size for x in archive.infolist())
                    archive.extractall(unpacked)
                shutil.copytree(inputs,unpacked / 'data',dirs_exist_ok=True)
                old_pythonpath = os.environ.get('PYTHONPATH')
                os.environ['PYTHONPATH'] = str(guard)
                started = time.perf_counter()
                try:
                    predicted = verify_packaged(unpacked,DEADLINE-120)
                finally:
                    if old_pythonpath is None:
                        os.environ.pop('PYTHONPATH',None)
                    else:
                        os.environ['PYTHONPATH'] = old_pythonpath
                seconds = time.perf_counter()-started
                assert [r['ID'] for r in predicted] == [r['ID'] for r in expected[name]]
                delta = max(abs(float(a[c])-float(b[c])) for a,b in zip(predicted,expected[name],strict=True) for c in SUBMISSION_COLUMNS[1:])
                assert delta <= 1e-5, f'Package parity: {name} {delta}'
                shutil.copy2(package,RUN / package.name)
                assert digest(package) == digest(RUN / package.name)
                shutil.copy2(unpacked / 'output/packaged_inference.log',RUN / (name+'_packaged.log'))
                write_csv(RUN / (name+'_reference.csv'),expected[name])
                write_csv(RUN / (name+'_packaged_predictions.csv'),predicted)
                receipts[name] = {'max_probability_delta':delta,'files':len(predicted),
                    'script_seconds_including_load':seconds,'zip_bytes':package.stat().st_size,
                    'expanded_bytes':expanded,'zip_sha256':digest(package),
                    'network_socket_guard':True,'stress_60s_included':True,'L4_runtime_limit':'unverified'}
                atomic_json(RUN / 'packaged_verification.json',receipts)
                print('PACKAGE_VERIFIED',name,receipts[name],flush=True)
                shutil.rmtree(unpacked)
            (RUN / 'code').mkdir(exist_ok=True)
            for source in WORK.glob('*.py'):
                shutil.copy2(source,RUN / 'code' / source.name)
            atomic_json(RUN / 'finished.json',{'status':'finished','selected':'G0_AST_MUSIC',
                'elapsed_seconds':time.time()-SESSION_START,'submission_status':'not_submitted'})
            print('REPLAY_SAVED',RUN,flush=True)
        except Exception:
            (RUN / 'failure.txt').write_text(traceback.format_exc())
            print(traceback.format_exc(),flush=True)
            raise
        finally:
            (RUN / 'release_requested.json').write_text(json.dumps({'unix':time.time(),
                'elapsed_seconds':time.time()-SESSION_START}))
            watchdog.cancel()
            runtime.unassign()
    '''))
    notebook = nb.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3"},
        "language_info":{"name":"python"},"accelerator":"GPU","colab":{"name":"DACON_236749_v2_replay.ipynb","gpuType":"A100","provenance":[]}})
    for item in cells:
        if item.cell_type == 'code' and not item.source.startswith('%%writefile'):
            compile(item.source,'replay_cell','exec')
    nb.validate(notebook)
    nb.write(notebook,ROOT / 'DACON_236749_v2_replay.ipynb')


if __name__ == '__main__':
    main()
