"""CPU-only v3 handoff review: actual 9504 IDs, synthetic evidence, no model run.

Protects locked evidence integrity during packaging and user-downloadable ZIP bytes.
Run with _workspace/v3_review_venv/bin/python from the repository root.
"""
import csv
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
CODE = HERE / 'code'
sys.path.insert(0, str(CODE))
from dacon9504a100 import handoff as h
from dacon9504a100.common import PipelineError, canonical_sha, sha256_file
from dacon9504a100.constants import DATASET_EXPECTATIONS, HANDOFF_NAME, RUNTIME_MANIFEST_SHA256
from dacon9504a100.dataset import read_manifest, validate_manifest
from dacon9504a100.orchestrator import build_config


def main():
    report = {'gpu_executed': False, 'scope': 'macOS CPU; synthetic prediction/checkpoint bytes; actual dataset IDs'}
    notebook = json.loads((HERE / 'DACON_Existing9504_A100.ipynb').read_text())
    archive_sha = sha256_file(HERE / 'author_code.zip')
    code_cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assert archive_sha in code_cells[0]
    for cell in code_cells:
        compile(cell, '<notebook-cell>', 'exec')
    report['author_sha256'] = archive_sha
    manifest = HERE.parent / 'manifest.csv'
    assert sha256_file(manifest) == RUNTIME_MANIFEST_SHA256
    rows = read_manifest(manifest)
    audit = validate_manifest(rows, DATASET_EXPECTATIONS)
    report['data'] = {k:audit[k] for k in ('rows','splits','cross_split_components')}
    with tempfile.TemporaryDirectory(prefix='dacon-v3-review-') as scratch:
        work = Path(scratch).resolve()  # macOS /var aliases /private/var; Jupyter requires a canonical root
        deploy, job = work/'objvol', work/'job'
        deploy.mkdir()
        cfg = build_config(deploy, job, CODE, HERE/'author_code.zip', HERE/'DACON_Existing9504_A100.ipynb')
        for name in ('inputs','cache','features','train','eval','parity','pilot','config','env'):
            (job/name).mkdir(parents=True)
        def write(name, value):
            path = job/name
            path.write_text(json.dumps(value))
            return sha256_file(path)
        shutil.copyfile(manifest, job/'inputs/manifest.csv')
        index_sha = write('cache/index.json', {'items':{r['ID']:{} for r in rows}})
        write('features/features_complete.json', {'rows':9504,'index_sha256':index_sha})
        for split in ('val','test'):
            with (job/'eval'/f'{split}_predictions.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['ID','FILE_FAKE_PROB'])
                writer.writerows((r['ID'],0.1) for r in rows if r['split']==split)
        initial_sha = write('cache/voice_classifier_initial.pt', {'synthetic':True})
        checkpoints = {f'{r}_{s}.pt':write(f'train/{r}_{s}.pt',{'synthetic':True})
                       for r in ('music','voice') for s in (41,42,43)}
        identity = {'synthetic':True}
        write('train/run_identity.json',identity)
        results_sha = write('train/results.json',{'synthetic':True})
        val_pred = sha256_file(job/'eval/val_predictions.csv')
        val_metric = write('eval/val_metrics.json',{'predictions_sha256':val_pred})
        lock = {'checkpoints':checkpoints,'run_identity_sha256':canonical_sha(identity),
                'manifest_sha256':RUNTIME_MANIFEST_SHA256,'feature_index_sha256':index_sha,
                'voice_classifier_initial_sha256':initial_sha,'results_sha256':results_sha,
                'validation_predictions_sha256':val_pred,'validation_metrics_sha256':val_metric}
        lock_sha = write('train/candidate_lock.json',{**lock,'lock_body_sha256':canonical_sha(lock)})
        test_pred = sha256_file(job/'eval/test_predictions.csv')
        test_metric = write('eval/test_metrics.json',{'predictions_sha256':test_pred,'lock_sha256':lock_sha})
        write('eval/test_receipt.json',{'lock_sha256':lock_sha,'predictions_sha256':test_pred,'metrics_sha256':test_metric})
        write('parity/parity_receipt.json',{'status':'bitwise','lock_sha256':lock_sha,'samples':[]})
        (job/'source.zip').write_bytes(b'fixture source archive')
        (job/'assets.zip').write_bytes(b'fixture assets archive')
        (job/'ffmpeg').write_bytes(b'fixture executable placeholder')
        write('inputs/staging_receipt.json',{'paths':{'source_zip':str(job/'source.zip'),'assets_zip':str(job/'assets.zip')}})
        write('env/receipt.json',{'static_ffmpeg':{'target':str(job/'ffmpeg')}})
        write('pilot/pilot_receipt.json',{'projection':{'projected_feature_hours':0,'rows':9504,'audio_hours':0},'mert_batch':1,'memory':{}})
        write('config/run_config.json',cfg)
        def gate(): return h.coverage_report(cfg,h.file_reader(h.collect(cfg,'complete')))
        assert not gate()['problems']
        rejected=[]
        for name,label in [('eval/test_predictions.csv','test predictions vs test_receipt'),
                           ('eval/val_predictions.csv','val predictions vs candidate_lock'),
                           ('eval/test_metrics.json','test metrics vs test_receipt'),
                           ('eval/val_metrics.json','val metrics vs candidate_lock'),
                           ('parity/parity_receipt.json','parity_receipt.lock_sha256')]:
            path=job/name; original=path.read_bytes()
            if name.endswith('.csv'): path.write_bytes(original.replace(b',0.1',b',0.9',1))
            elif name.startswith('parity/'): write(name,{'status':'bitwise','lock_sha256':'0'*64,'samples':[]})
            else: path.write_bytes(original+b' ')
            try:
                assert any(label in p for p in gate()['problems'])
                try: h.build(cfg,'complete',{'stages':{}},log=lambda _:None)
                except PipelineError as e: assert 'coverage check failed' in str(e)
                else: raise AssertionError('Corrupt evidence packaged as complete')
                rejected.append(label)
            finally: path.write_bytes(original)
        report['rejected_mutations']=rejected
        original_gate=h.coverage_report
        path=job/'eval/test_predictions.csv'; original=path.read_bytes(); calls=[]
        def race(config,read):
            result=original_gate(config,read); calls.append(1)
            if len(calls)==1: path.write_bytes(original.replace(b',0.1',b',0.9',1))
            return result
        h.coverage_report=race
        try:
            try: h.build(cfg,'complete',{'stages':{}},log=lambda _:None)
            except PipelineError as e: assert 'Packaged handoff bytes' in str(e)
            else: raise AssertionError('Post-gate modification accepted')
        finally: h.coverage_report=original_gate; path.write_bytes(original)
        assert len(calls)==2
        report['post_gate_file_change']='rejected on archive reread'
        sidecar=h.build(cfg,'complete',{'stages':{}},log=lambda _:None)
        with zipfile.ZipFile(sidecar['file']) as archive:
            assert archive.testzip() is None
            assert h.coverage_report(cfg,h.zip_reader(archive))==json.loads(archive.read('handoff_manifest.json'))['coverage']
        report['complete_zip']='CRC and evidence binding verified on actual packaged bytes'

        # Execute the actual FINAL cell; reuse the author's local root-scoped server.
        spec=importlib.util.spec_from_file_location('download_test',CODE/'tests/test_download_link.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        InteractiveShell.instance()
        server_root=work/'server'; notebook_dir=server_root/'uploaded'; notebook_dir.mkdir(parents=True)
        before=Path.cwd()
        try:
            os.chdir(notebook_dir)
            with capture_output() as captured:
                exec(compile(code_cells[-1],'<actual FINAL cell>','exec'),{'DEPLOYMENT_ROOT':deploy,'Path':Path})
        finally: os.chdir(before)
        html=[o.data.get('text/html','') for o in captured.outputs]
        assert any('href="'+HANDOFF_NAME+'"' in s or "href='"+HANDOFF_NAME+"'" in s for s in html),html
        assert os.path.samefile(notebook_dir/HANDOFF_NAME,sidecar['local_copy'])
        server=module.Server(server_root,work/'jupyter.log')
        try:
            status,body=server.get(module.files_url('uploaded',HANDOFF_NAME))
            assert status==200 and hashlib.sha256(body).hexdigest()==sidecar['sha256'], (status, (work/'jupyter.log').read_text()[-5000:].replace(server.token, '[redacted]'))
            old_status,_=server.get(module.files_url('uploaded',os.path.relpath(sidecar['file'],notebook_dir)))
            assert old_status!=200
        finally: server.stop()
        report['download']={'new_link_http':status,'old_link_http':old_status,'downloaded_sha256_matches':True,'hardlink':True,
                            'scope':'Actual FINAL cell in IPython; local Jupyter /files handler, not live VESSL UI'}
        record=h.expose_download(Path(sidecar['file']+'.json'),deploy)
        assert record['mode']=='already_present_verified'
        elsewhere=work/'separate'; elsewhere.mkdir(); (elsewhere/HANDOFF_NAME).write_bytes(b'stale')
        real_link=h.os.link
        def exdev(*args,**kwargs): raise OSError(errno.EXDEV,'cross-device fixture')
        h.os.link=exdev
        try: record=h.expose_download(Path(sidecar['file']+'.json'),elsewhere)
        finally: h.os.link=real_link
        assert record['mode']=='verified_byte_copy' and sha256_file(elsewhere/HANDOFF_NAME)==sidecar['sha256']
        report['download']['same_directory_and_cross_filesystem_copy']='verified'
    (HERE/'codex_cpu_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
