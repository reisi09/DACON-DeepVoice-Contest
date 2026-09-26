# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run with the prepared offline Python: package_panns_submission.py --help
"""Stream a selected PANNs suffix into a reference-bound offline submission ZIP."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import import_module
import json
from pathlib import Path, PurePosixPath
from typing import Final, TypedDict
import zipfile

from panns_submission import Selection, SubmissionError, sha256

ENTRYPOINT: Final = b"import sys\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parent\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(ROOT/'model/runtime'))\nfrom panns_submission import run\nif __name__=='__main__':\n    run(ROOT)\n"


class PackageReport(TypedDict):
    file: str
    sha256: str
    bytes: int
    expanded_bytes: int
    member_sha256: dict[str, str]
    reference_sha256: str
    checkpoint_sha256: str | None
    crc_verified: bool
    inference_parity_verified: bool
    official_l4_validated: bool


def package(reference: Path, export: Path | None, output: Path) -> PackageReport:
    """Preserve reference bytes and fail before publication on changed assets or archive limits."""
    selection = Selection.read(export/'selection.json') if export is not None else None
    reference_hash = sha256(reference)
    if selection is not None and export is not None:
        if selection.reference_sha256 != reference_hash:
            raise SubmissionError('Selected training reference changed')
        if sha256(export/'best_model.pt') != selection.weights_sha256:
            raise SubmissionError('Selected PANNs weights changed')
    lock = json.loads(reference.read_text())
    base = Path(lock['package']).resolve()
    base_manifest = json.loads((base/'model/manifest.json').read_text())
    expected = lock['members']
    if base_manifest['sha256'] != {name: value for name, value in expected.items() if name != 'model/manifest.json'}:
        raise SubmissionError('Base manifest differs from the training reference')
    sources = {}
    for name in expected:
        member = PurePosixPath(name)
        path = (base/name).resolve()
        if (member.is_absolute() or '..' in member.parts or '\\' in name
                or not path.is_relative_to(base) or member.parts[0] not in ('model', 'script.py', 'requirements.txt')):
            raise SubmissionError('Unsafe base package member: '+name)
        sources[name] = path
    required = {'script.py', 'requirements.txt', 'model/trained/presence_head.json',
                'model/trained/file_ensemble.json', 'model/runtime/full_bank_submission.py'}
    if not required <= sources.keys():
        raise SubmissionError('Missing preserved runtime or decision readout')
    partial, receipt = output.with_suffix('.partial'), output.with_suffix('.json')
    if any(path.exists() for path in (output, partial, receipt)):
        raise FileExistsError('Use a fresh submission output path')
    additions = {}
    if export is not None:
        additions = {'model/trained/panns_selection.json': export/'selection.json',
                     'model/trained/panns_best_model.pt': export/'best_model.pt'}
        root = Path(__file__).resolve().parent
        additions['model/runtime/panns_submission.py'] = root/'panns_submission.py'
        additions['model/runtime/panns_adaptation.py'] = Path(import_module('panns_adaptation').__file__)
    if sources.keys() & additions.keys():
        raise SubmissionError('The base already contains a PANNs adaptation')
    expected = {**expected, **{name: sha256(path) for name, path in additions.items()}}
    sources.update(additions)
    total = sum(path.stat().st_size for name, path in sources.items() if name not in ('script.py', 'model/manifest.json'))
    if total >= 10_000_000_000:
        raise SubmissionError('Uncompressed submission would exceed the 10 GB ZIP limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    with zipfile.ZipFile(partial, 'x', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(sources.items()):
            if selection is not None and name in ('script.py', 'model/manifest.json'):
                if sha256(path) != expected[name]:
                    raise SubmissionError('Base entry point or manifest changed')
                continue
            digest = hashlib.sha256()
            with path.open('rb') as source, archive.open(zipfile.ZipInfo(name), 'w', force_zip64=True) as target:
                while chunk := source.read(8 << 20):
                    target.write(chunk)
                    digest.update(chunk)
            written[name] = digest.hexdigest()
            if written[name] != expected[name]:
                raise SubmissionError('Package input changed: '+name)
        if selection is not None:
            archive.writestr(zipfile.ZipInfo('script.py'), ENTRYPOINT)
            written['script.py'] = hashlib.sha256(ENTRYPOINT).hexdigest()
            choice = dict(base_manifest['choice'], candidate=f'panns_{selection.arm}_seed41_DataV4',
                          parent_candidate=base_manifest['choice']['candidate'], panns_adaptation=asdict(selection))
            manifest = (json.dumps(dict(choice=choice, sha256=written), indent=2, sort_keys=True)+'\n').encode()
            archive.writestr(zipfile.ZipInfo('model/manifest.json'), manifest)
            written['model/manifest.json'] = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(partial) as archive:
        if len(archive.namelist()) != len(written) or set(archive.namelist()) != set(written):
            raise SubmissionError('Duplicate, missing or unexpected archive member')
        expanded = sum(member.file_size for member in archive.infolist())
        if partial.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise SubmissionError('Submission archive size limit exceeded')
        for name, digest in written.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise SubmissionError('Archive member changed: '+name)
    partial.replace(output)
    report = PackageReport(file=str(output), sha256=sha256(output), bytes=output.stat().st_size,
        expanded_bytes=expanded, member_sha256=written, reference_sha256=reference_hash,
        checkpoint_sha256=selection.checkpoint_sha256 if selection is not None else None, crc_verified=True,
        inference_parity_verified=False, official_l4_validated=False)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--export', type=Path)
    choice.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    report = package(args.reference, args.export, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'member_sha256'}, indent=2))


if __name__ == '__main__':
    main()
