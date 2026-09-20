"""Protect Drive reconstruction against corrupt parts and loss of an existing partial file."""
import hashlib
import json
from pathlib import Path
import runpy
import pytest


def test_restore_validates_bytes_and_preserves_existing_partial(tmp_path):
    restore = runpy.run_path(str(Path(__file__).parent/'submissions/AST_full1029_epoch17/restore_submission.py'))['restore']
    chunks = [b'first archive chunk', b'second archive chunk']
    sha = lambda content: hashlib.sha256(content).hexdigest()
    parts = []
    for i, content in enumerate(chunks):
        name = f'part{i}'
        (tmp_path/name).write_bytes(content)
        parts.append({'name':name,'size':len(content),'sha256':sha(content)})
    content = b''.join(chunks)
    manifest = {'output_filename':'final.zip','size':len(content),'sha256':sha(content),'parts':parts}
    (tmp_path/'split_manifest.json').write_text(json.dumps(manifest))
    assert restore(tmp_path).read_bytes() == content
    (tmp_path/'final.zip').unlink()
    (tmp_path/'part0').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt backup part'):
        restore(tmp_path)
    assert not (tmp_path/'final.zip').exists()
    (tmp_path/'part0').write_bytes(chunks[0])
    partial = tmp_path/'final.zip.partial'
    partial.write_bytes(b'keep previous recovery work')
    with pytest.raises(FileExistsError):
        restore(tmp_path)
    assert partial.read_bytes() == b'keep previous recovery work'
