"""Package the receipt-bound full-data B ensemble and exactly its frozen assets."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

import torch

from redesign_assets import validate_assets
from redesign_checkpoint import sha256
from redesign_features_cache import receipt_payload
from redesign_submission import Choice, EPOCHS, SEEDS, SubmissionError, load_heads

RUNTIME: Final = (
    'redesign_submission.py', 'redesign_checkpoint.py', 'redesign_models.py',
    'redesign_features.py', 'redesign_features_cache.py', 'redesign_features_runtime.py',
    'redesign_assets.py', 'music_temporal.py', 'temporal_augmentation.py', 'v7_ablation.py',
    'v6_core.py', 'v6_data.py', 'v6_train.py', 'v6_runtime.py', 'graph_scoring.py',
)
ASSET_NAMES: Final = frozenset(
    {'baseline/script.py'}
    | {'baseline/model/df_arena_1b/' + name for name in (
        '.gitattributes', '.gitignore', '__init__.py', 'backbone.py', 'config.json',
        'configuration_antispoofing.py', 'conformer.py', 'facebook/wav2vec2-xls-r-1b/config.json',
        'feature_extraction_antispoofing.py', 'LICENSE.txt', 'modeling_antispoofing.py',
        'pipeline_antispoofing.py', 'preprocessor_config.json', 'pytorch_model.bin', 'README.md')}
    | {'baseline/model/htdemucs/955717e8-8726e21a.th', 'baseline/model/htdemucs/htdemucs.yaml'}
    | {'baseline/model/panns/' + name for name in (
        'class_labels_indices.csv', 'Cnn14_mAP=0.431.pth', 'component_labels.json')}
    | {'mert/' + name for name in ('config.json', 'configuration_MERT.py', 'modeling_MERT.py',
                                   'preprocessor_config.json', 'pytorch_model.bin', 'README.md')}
)
REQUIREMENTS: Final = '''--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
diffq==0.2.4
einops==0.8.1
'''
ENTRYPOINT: Final = '''import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'model/runtime'))
from redesign_submission import main
if __name__ == '__main__':
    main(ROOT)
'''


class CheckpointReceipt(TypedDict):
    seed: int
    epochs: int
    checkpoint_sha256: str


class FullRefitReceipt(TypedDict):
    training_mode: str
    arm: str
    seeds: list[int]
    epochs: list[int]
    checkpoints: list[CheckpointReceipt]
    train_ids: list[str]
    label_eligible_music: int
    independent_validation: bool
    file_fusion: str
    feature_receipt: dict[str, str]


class PackageReport(TypedDict):
    file: str
    sha256: str
    bytes: int
    expanded_bytes: int
    choice: Choice
    member_sha256: dict[str, str]
    crc_verified: bool
    gpu_end_to_end_verified: bool
    official_environment: str


def _choice(receipt: FullRefitReceipt, checkpoints: Sequence[Path]) -> Choice:
    """Reject incomplete, selected-on-validation, mismatched, or duplicate seed heads."""
    if (receipt['training_mode'] != 'full_data_refit' or receipt['arm'] != 'B'
            or receipt['seeds'] != list(SEEDS) or receipt['epochs'] != list(EPOCHS)
            or receipt['independent_validation'] is not False or receipt['file_fusion'] != 'noisy_or'
            or len(receipt['train_ids']) != 1696 or len(set(receipt['train_ids'])) != 1696
            or receipt['label_eligible_music'] != 1216 or len(receipt['checkpoints']) != 3):
        raise SubmissionError('Full-refit receipt does not match the locked B recipe')
    load_heads(checkpoints)
    for path, expected, seed, epochs in zip(checkpoints, receipt['checkpoints'], SEEDS, EPOCHS, strict=True):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['arm'] != 'B' or expected['seed'] != seed or expected['epochs'] != epochs
                or sha256(path) != expected['checkpoint_sha256']
                or payload['report']['train_ids'] != sorted(receipt['train_ids'])):
            raise SubmissionError(f'Checkpoint differs from full-refit receipt: seed{seed}')
    return Choice(candidate='B_raw_stem_full1696_3seed', seeds=list(SEEDS), epochs=list(EPOCHS),
                  checkpoints=[f'model/trained/B_seed{seed}.pt' for seed in SEEDS], file_fusion='noisy_or')


def _write_member(archive: zipfile.ZipFile, name: str, path: Path) -> str:
    """Hash the exact streamed bytes, using stable member timestamps for reproducibility."""
    digest = hashlib.sha256()
    with path.open('rb') as incoming, archive.open(name, 'w', force_zip64=True) as outgoing:
        while chunk := incoming.read(8 * 1024 * 1024):
            outgoing.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def package(root: Path, output: Path, *, assets: Path, checkpoints: Sequence[Path],
            fullrefit_receipt: Path) -> PackageReport:
    """Build and verify a deterministic ZIP without bundling audio, caches, or validation data."""
    hashes = validate_assets(assets)
    if set(hashes) != ASSET_NAMES:
        raise SubmissionError('Expected exactly the 27 frozen baseline and MERT asset paths')
    receipt = json.loads(fullrefit_receipt.read_text(encoding='utf-8'))
    if receipt['feature_receipt'] != receipt_payload(assets, hashes):
        raise SubmissionError('Full-refit feature receipt differs from packaged encoders or extractor')
    choice = _choice(receipt, checkpoints)
    members = {f'model/runtime/{name}': root / name for name in RUNTIME}
    members.update({f'model/assets/{name}': assets / name for name in sorted(hashes)})
    members['model/assets/asset_hashes.json'] = assets / 'asset_hashes.json'
    members.update(zip(choice['checkpoints'], checkpoints, strict=True))
    members['model/trained/fullrefit_receipt.json'] = fullrefit_receipt
    expected = {f'model/assets/{name}': digest for name, digest in hashes.items()}
    expected.update({name: item['checkpoint_sha256']
                     for name, item in zip(choice['checkpoints'], receipt['checkpoints'], strict=True)})
    # Bind source bytes before archiving, including the three unchanged extractor modules.
    expected.update({name: sha256(path) for name, path in members.items() if name not in expected})
    content = {'script.py': ENTRYPOINT.encode(), 'requirements.txt': REQUIREMENTS.encode()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    written: dict[str, str] = {}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            written[name] = _write_member(archive, name, path)
            if written[name] != expected[name]:
                raise SubmissionError(f'Input changed while packaging: {name}')
        for name, value in sorted(content.items()):
            archive.writestr(zipfile.ZipInfo(name), value)
            written[name] = hashlib.sha256(value).hexdigest()
        manifest = json.dumps(dict(schema_version=1, choice=choice, sha256=written),
                              indent=2, sort_keys=True).encode() + b'\n'
        archive.writestr(zipfile.ZipInfo('model/manifest.json'), manifest)
    # The external receipt hashes the manifest too; an embedded manifest cannot hash itself.
    written['model/manifest.json'] = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(temporary) as archive:
        expanded = sum(member.file_size for member in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise SubmissionError('Archive exceeds competition size limits')
        if len(archive.namelist()) != len(written) or set(archive.namelist()) != set(written):
            raise SubmissionError('Archive contains missing or duplicate members')
        for name, digest in written.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise SubmissionError(f'Archived member hash mismatch: {name}')
        if archive.testzip() is not None:
            raise SubmissionError('Submission archive CRC verification failed')
    temporary.replace(output)
    report = PackageReport(file=str(output), sha256=sha256(output), bytes=output.stat().st_size,
        expanded_bytes=expanded, choice=choice, member_sha256=written, crc_verified=True,
        gpu_end_to_end_verified=False, official_environment='Python 3.11, torch 2.7.1 CUDA 12.8')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report
