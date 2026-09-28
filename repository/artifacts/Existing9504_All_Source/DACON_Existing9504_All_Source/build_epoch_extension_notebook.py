"""Build the bounded Colab continuation experiment from reviewed local sources."""
import json
import textwrap
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parent
SOURCES = ("train_v2.py", "v2_inference.py", "run_epoch_extension.py", "package_hybrid_submission.py")


def code(text):
    return nb.v4.new_code_cell(textwrap.dedent(text).strip())


def main():
    archive = next(a for a in json.loads((ROOT / "v2_validation/drive_inputs.json").read_text())
                   if a["name"] == "v2_validation.zip")
    cells = [nb.v4.new_markdown_cell("# AST epoch 연장 실험\n\n기존 6 epoch의 가중치·AdamW·난수 상태를 이어받아 최대 30 epoch까지 학습합니다. 학습률과 학습 633개·개발 평가 122개는 그대로 유지합니다. 개발 proxy가 5 epoch 연속 개선되지 않으면 중단하고 최적 checkpoint를 복원합니다. 기존 평가 274개는 후속 비교용이며 새로운 독립 검증이 아닙니다. 이번 A100 할당은 최대 30분입니다.")]
    cells.append(code('''
        import time, json, shutil, subprocess, sys, zipfile, hashlib, threading, traceback, os
        from pathlib import Path
        from google.colab import drive, runtime
        SESSION_START = time.time()
        DEADLINE = SESSION_START + 1680
        drive.mount('/content/drive')
        DRIVE = Path('/content/drive/MyDrive/DACON_236749')
        PRIMARY = DRIVE / 'runs_v2/20260912_174330'
        RUN = DRIVE / 'runs_v2' / ('epoch_extension_' + time.strftime('%Y%m%d_%H%M%S',time.gmtime(SESSION_START)))
        RUN.mkdir()
        WORK = Path('/content/dacon_epoch_extension')
        WORK.mkdir(exist_ok=True)
        watchdog = threading.Timer(max(1,DEADLINE-time.time()),runtime.unassign)
        watchdog.daemon = True
        watchdog.start()
        (RUN / 'budget.json').write_text(json.dumps({'start_unix':SESSION_START,'deadline':DEADLINE,'allocation_limit_seconds':1800}))
        for source in (PRIMARY / 'code').iterdir():
            if source.is_file():
                shutil.copy2(source,WORK / source.name)
        print('RESULTS_DIRECTORY',RUN,flush=True)
    '''))
    for name in SOURCES:
        source = (ROOT / name).read_text()
        compile(source, name, "exec")
        cells.append(nb.v4.new_code_cell(f"%%writefile /content/dacon_epoch_extension/{name}\n" + source))
    cells.append(nb.v4.new_code_cell("DATA_ARCHIVE = " + repr(archive)))
    cells.append(code('''
        try:
            (RUN / 'code').mkdir()
            for source in WORK.iterdir():
                if source.is_file():
                    shutil.copy2(source,RUN / 'code' / source.name)
            with (RUN / 'install.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
                    'scikit-learn==1.8.0','soundfile==0.12.1','safetensors'],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            import torch, importlib.metadata
            assert torch.cuda.is_available()
            (RUN / 'environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,
                'gpu':torch.cuda.get_device_name(),'transformers':importlib.metadata.version('transformers')}))
            def sha(path):
                with path.open('rb') as f:
                    return hashlib.file_digest(f,'sha256').hexdigest()
            archive_path = WORK / DATA_ARCHIVE['name']
            with archive_path.open('wb') as out:
                for part in DATA_ARCHIVE['parts']:
                    source = DRIVE / part['name']
                    assert sha(source) == part['sha256']
                    with source.open('rb') as incoming:
                        shutil.copyfileobj(incoming,out,8*1024**2)
            assert sha(archive_path) == DATA_ARCHIVE['sha256']
            with zipfile.ZipFile(archive_path) as packed:
                packed.extractall(WORK)
            os.chdir(WORK)
            command = [sys.executable,'-u','run_epoch_extension.py','--primary',str(PRIMARY),
                '--data',str(WORK / 'v2_validation'),'--work',str(WORK),'--output',str(RUN),'--deadline',str(DEADLINE)]
            with (RUN / 'experiment.log').open('w') as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if time.time() > DEADLINE-60:
                        process.terminate()
                        process.wait(timeout=30)
                        raise TimeoutError('epoch experiment deadline reached')
                    time.sleep(15)
                    lines = (RUN / 'experiment.log').read_text().splitlines()
                    print(lines[-1][:700] if lines else 'Preparing checkpoint...',flush=True)
                if process.returncode:
                    raise RuntimeError((RUN / 'experiment.log').read_text()[-10000:])
            report = json.loads((RUN / 'training_report.json').read_text())
            import matplotlib.pyplot as plt
            history = report['reports']
            fig, axes = plt.subplots(1,2,figsize=(11,4))
            axes[0].plot([r['epoch'] for r in history],[r['mean_loss'] for r in history],marker='o')
            axes[0].set(title='Training loss',xlabel='Epoch',ylabel='Masked BCE')
            axes[1].plot([r['epoch'] for r in history],[r['proxy'] for r in history],marker='o')
            axes[1].set(title='Development proxy',xlabel='Epoch',ylabel='Proxy (higher is better)')
            for axis in axes:
                axis.axvline(6,color='gray',linestyle='--',label='Previous stop')
                axis.axvline(report['best_epoch'],color='green',linestyle=':',label='Selected epoch')
                axis.legend()
                axis.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(RUN / 'learning_curve.png',dpi=160)
            plt.show()
            print('RESULT',json.loads((RUN / 'finished.json').read_text()))
            print('DRIVE',RUN)
        except Exception:
            (RUN / 'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            (RUN / 'release_requested.json').write_text(json.dumps({'unix':time.time(),'elapsed_seconds':time.time()-SESSION_START}))
            watchdog.cancel()
            runtime.unassign()
    '''))
    notebook = nb.v4.new_notebook(cells=cells, metadata={"kernelspec":{"name":"python3","display_name":"Python 3"},
        "language_info":{"name":"python"},"accelerator":"GPU",
        "colab":{"name":"DACON_236749_epoch_extension.ipynb","gpuType":"A100","provenance":[]}})
    for cell in cells:
        if cell.cell_type == "code" and not cell.source.startswith("%%writefile"):
            compile(cell.source, "extension_cell", "exec")
    nb.validate(notebook)
    nb.write(notebook, ROOT / "DACON_236749_epoch_extension.ipynb")


if __name__ == "__main__":
    main()
