"""Protect offline notebook execution and separation-cache provenance on deployment."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types

import nbformat
import numpy as np


def test_packaged_worker_and_cache_provenance() -> None:
    notebook = nbformat.read('MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb', as_version=4)
    module = types.ModuleType('fullft8_check')
    sys.modules[module.__name__] = module
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source,'fullft8_notebook','exec',dont_inherit=True),module.__dict__)
    worker = '\n\n'.join(c.source for c in notebook.cells if 'worker' in c.metadata.get('tags',[]))
    isolated = types.ModuleType('fullft8_worker_check')
    sys.modules[isolated.__name__] = isolated
    exec(compile(worker,'worker.py','exec',dont_inherit=True),isolated.__dict__)
    assert isolated.SEPARATOR_CODE_SHA256 == hashlib.sha256(Path('fullft_separator.py').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)/'audio_cache'
        cache.mkdir()
        row = module.SPLITS['S0']['train'][0]
        path = cache/f'{row["ID"]}.npz'
        np.savez(path,train=np.arange(10,dtype='float32'))
        config = dict(output=temporary,separator_checkpoint_sha256='original',audio_recipe='original',seconds=5.,pool_windows=8)
        record = dict(identity=isolated.audio_cache_identity(row,'weights',config),
                      sha256=isolated.file_hash(path),source_sha256=row['audio_sha256'],separator_sha256='weights')
        path.with_suffix('.json').write_text(json.dumps(record))
        isolated.validate_audio_cache(row,config)
        for changed in [dict(config,separator_checkpoint_sha256='adapted'),dict(config,audio_recipe='changed')]:
            try:
                isolated.validate_audio_cache(row,changed)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Old separator cache was accepted by another preprocessing identity')
        np.savez(path,train=np.arange(10,dtype='float32')+1)
        try:
            isolated.validate_audio_cache(row,config)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Corrupt waveform content passed its recorded hash')


def test_mirror_preserves_finished_versions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        os.environ['FULLFT_REMOTE'] = str(root/'remote')
        os.environ['FULLFT_JOB_STARTED_AT'] = '1'
        spec = importlib.util.spec_from_file_location('fullft8_runtime_test','fullft8_runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        runtime.ROOT = root/'local'
        output = runtime.ROOT/'h100x1_fullft8_run'/'01_S0_MERT_frozen'
        output.mkdir(parents=True)
        (output/'best.pt').write_bytes(b'finished checkpoint version one')
        (output/'best.partial.pt').write_bytes(b'incomplete new checkpoint')
        seen = {}
        runtime.mirror(seen)
        remote = runtime.REMOTE/'h100x1_fullft8_run'/'01_S0_MERT_frozen'
        assert (remote/'best.pt').read_bytes()==(output/'best.pt').read_bytes()
        assert not (remote/'best.partial.pt').exists()
        (output/'best.partial.pt').replace(output/'best.pt')
        runtime.mirror(seen)
        assert (remote/'best.pt').read_bytes()==b'incomplete new checkpoint'


def test_eat_snapshot_loads_with_empty_module_cache() -> None:
    # A cold remote machine must resolve eat_model -> model_core before weight loading.
    with tempfile.TemporaryDirectory() as cache:
        subprocess.run([sys.executable, '-c', '''
import nbformat, types, sys
from pathlib import Path
notebook = nbformat.read('MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb', 4)
module = types.ModuleType('cold_worker')
sys.modules[module.__name__] = module
worker = '\\n\\n'.join(c.source for c in notebook.cells if 'worker' in c.metadata.get('tags', []))
exec(compile(worker, 'worker.py', 'exec', dont_inherit=True), module.__dict__)
model = module.MusicClassifier('eat', False, Path('_workspace/fullft7_upload/models'))
assert any(p.requires_grad for p in model.encoder.parameters())
print('Cold EAT snapshot and transitive imports loaded successfully')
'''], env=dict(os.environ, HF_MODULES_CACHE=cache), check=True)


if __name__ == '__main__':
    test_packaged_worker_and_cache_provenance()
    test_mirror_preserves_finished_versions()
    test_eat_snapshot_loads_with_empty_module_cache()
    print('Validated standalone worker, separator cache isolation, content integrity, and artifact mirroring.')
