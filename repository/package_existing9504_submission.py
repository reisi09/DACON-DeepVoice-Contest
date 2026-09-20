"""Package the verified 9,504-row H100 handoff without changing its inference graph."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

from package_redesign_submission import _write_member
from v6_data import sha256

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/existing9504_h100_20260920'
sys.path.insert(0, str(RUN / 'author_code.zip'))
from dacon9504a100.common import checked_members, extract_member, extract_exact
from dacon9504a100.handoff import coverage_report, zip_reader, verify_zip

CANDIDATE = 'existing9504_MERT330M_WindowMIL_DFfc5_701515'
RUNTIME = ('mert12_submission.py', 'eight_heads.py', 'v6_runtime.py', 'v6_core.py',
           'v6_data.py', 'graph_scoring.py', 'localization_models.py', 'ast_multitask.py',
           'full_bank_submission.py')
REQUIREMENTS = '--extra-index-url https://download.pytorch.org/whl/cu128\ntorchvision==0.22.1+cu128\n'


def package(handoff: Path, output: Path) -> None:
    """Bind assets, runtime and selected checkpoints to the downloaded handoff."""
    receipt = json.loads((handoff.parent / 'download_receipt.json').read_text())
    if sha256(handoff) != receipt['sha256']:
        raise ValueError('Handoff differs from its verified download')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='existing9504_package_', dir=output.parent) as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(handoff) as source:
            source_members = checked_members(source)
            config = json.loads(source.read('config/run_config.json'))
            coverage = coverage_report(config, zip_reader(source))
            if coverage['problems'] or coverage['splits'] != {'train': 6652, 'val': 1426, 'test': 1426}:
                raise ValueError(f'Invalid completed handoff: {coverage}')
            if hashlib.sha256(source.read('code/author_code.zip')).hexdigest() != sha256(RUN / 'author_code.zip'):
                raise ValueError('Handoff author code differs from reviewed H100 code')
            lock = json.loads(source.read('model/candidate_lock.json'))
            extract_member(source, source_members['assets/assets.zip'], work / 'assets.zip')
            extract_exact(work / 'assets.zip', work / 'assets',
                          {**config['asset_hashes'], 'asset_hashes.json': None})
            members = {f'model/assets/{name}': work / 'assets' / name for name in config['asset_hashes']}
            members['model/assets/asset_hashes.json'] = work / 'assets/asset_hashes.json'
            with zipfile.ZipFile(io.BytesIO(source.read('code/source.zip'))) as runtime:
                for name in RUNTIME:
                    data = runtime.read('source/' + name)
                    if hashlib.sha256(data).hexdigest() != config['source_hashes'][name]:
                        raise ValueError('Training runtime source changed: ' + name)
                    if name == 'full_bank_submission.py':
                        old = b"CANDIDATE = 'balanced7144_MERT330M_WindowMIL_DFfc5_8020'"
                        if data.count(old) != 1:
                            raise ValueError('Unexpected inference candidate declaration')
                        data = data.replace(old, f'CANDIDATE = {CANDIDATE!r}'.encode())
                    path = work / 'runtime' / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(data)
                    members['model/runtime/' + name] = path
            for name, expected in lock['checkpoints'].items():
                path = work / 'trained' / name
                extract_member(source, source_members['model/checkpoints/' + name], path, expected_sha256=expected)
                members['model/trained/' + name] = path
            for src, dst in [('model/candidate_lock.json', 'model/trained/candidate_lock.json'),
                             ('model/run_identity.json', 'model/trained/run_identity.json'),
                             ('metrics/val_metrics.json', 'model/trained/validation_metrics.json'),
                             ('environment/bin/ffmpeg', 'model/runtime/bin/ffmpeg')]:
                path = work / dst
                extract_member(source, source_members[src], path)
                members[dst] = path
        choice = {'candidate': CANDIDATE, 'train_rows': 6652, 'validation_rows': 1426, 'test_rows': 1426,
                  'seeds': [41, 42, 43], 'file_fusion': 'max', 'selection': 'validation_only',
                  'independent_test': False, 'historically_exposed_validation_rows': 997,
                  'historically_exposed_test_rows': 1001, 'voice_head': 'backbone.conformer.fc5',
                  'htdemucs': 'htdemucs', 'mert': 'MERT-v1-330M', 'mert_layer': 12,
                  'inference_gain': 'peak capped at 0.6; legacy gain-preserving QA is separate',
                  'handoff_sha256': receipt['sha256'], 'candidate_lock_sha256': sha256(members['model/trained/candidate_lock.json'])}
        entry = "import sys\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parent\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(ROOT/'model/runtime'))\nfrom full_bank_submission import run\nif __name__=='__main__':\n    run(ROOT)\n"
        temporary = output.with_suffix('.partial')
        written = {}
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, path in sorted(members.items()):
                written[name] = _write_member(archive, name, path)
            for name, data in {'script.py': entry.encode(), 'requirements.txt': REQUIREMENTS.encode()}.items():
                archive.writestr(zipfile.ZipInfo(name), data)
                written[name] = hashlib.sha256(data).hexdigest()
            manifest = (json.dumps({'choice': choice, 'sha256': written}, indent=2) + '\n').encode()
            archive.writestr(zipfile.ZipInfo('model/manifest.json'), manifest)
        written['model/manifest.json'] = hashlib.sha256(manifest).hexdigest()
        verify_zip(temporary, written)
        with zipfile.ZipFile(temporary) as archive:
            expanded = sum(info.file_size for info in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise ValueError('Competition archive size limit exceeded')
        temporary.replace(output)
    report = {'file': str(output.resolve()), 'sha256': sha256(output), 'bytes': output.stat().st_size,
              'expanded_bytes': expanded, 'choice': choice, 'member_sha256': written,
              'crc_verified': True, 'inference_parity_verified': False, 'official_l4_validated': False}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'member_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.handoff, args.output)
