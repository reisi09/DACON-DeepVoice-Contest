# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat"]
# ///
# Run: .venv/bin/python prepare_expanded_input.py
"""Package the executed protocol, metadata, and exact local source closure without duplicate audio."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import nbformat

from build_expanded_notebook import build
from build_redesign_notebook import source_payload
from eight_contract import read_manifest, sha256_file
from expanded_data import ROOT, OUTPUT, audit_groups


def main() -> None:
    work = ROOT / '_workspace/expanded_h100_20260918'
    metadata = json.loads((OUTPUT / 'data_preparation.json').read_text())
    manifest = work / 'dataset/manifest.csv'
    if metadata['manifest_sha256'] != sha256_file(manifest) or not metadata['all_shards_readback_verified']:
        raise ValueError('Completed audio preparation is required')
    rows = [dict(r) for r in read_manifest(manifest)]
    audit_groups(rows)
    exposure = json.loads((OUTPUT / 'sampling_preflight.json').read_text())
    if any(value['unseen_parents'] for cohort in exposure.values() for value in cohort.values()):
        raise ValueError('Every training parent must be exposed before GPU launch')
    notebook = ROOT / 'DACON_Expanded_H100_Experiments.ipynb'
    nbformat.write(build(), notebook)
    _, _, modules = source_payload(('expanded_experiment.py', 'localization_features.py', 'localization_export.py'))
    files = {'source/' + name: ROOT / name for name in modules}
    files.update({'dataset/' + path.name: path for path in (work / 'dataset').iterdir() if path.is_file()})
    for name in ('split_summary.json', 'data_preparation.json', 'audio_upload.json', 'sampling_preflight.json', 'PROTOCOL.md'):
        files[name] = OUTPUT / name
        shutil.copy2(OUTPUT / name, work / name)
    files[notebook.name] = notebook
    files['asset_hashes.json'] = ROOT / '_workspace/mert12_a100_20260918/upload/asset_hashes.json'
    files['ast_assets_receipt.json'] = ROOT / '_workspace/localization_a100_20260918/ast_assets_receipt.json'
    upload = work / 'upload'
    upload.mkdir(exist_ok=True)
    hashes = {name: sha256_file(path) for name, path in files.items()}
    with zipfile.ZipFile(upload / 'input.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
        for name, path in sorted(files.items()):
            packed.write(path, name)
        packed.writestr('INPUT_SHA256.json', json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(upload / 'input.zip') as packed:
        if packed.testzip() is not None:
            raise ValueError('Input package failed CRC verification')
    shutil.copy2(ROOT / 'expanded_h100_bootstrap.sh', upload / 'bootstrap.sh')
    receipt = {name: {'bytes': (upload / name).stat().st_size, 'sha256': sha256_file(upload / name)} for name in ('input.zip', 'bootstrap.sh')}
    (upload / 'upload_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (OUTPUT / 'input_build.json').write_text(json.dumps({'files': receipt, 'member_sha256': hashes,
                 'dataset': metadata, 'gpu_started': False, 'legacy_bundle_sha256': '00e495257b76e09bcbd91646f98ef9e8286bfc403349679e0018bddaa3dd1c4d'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
