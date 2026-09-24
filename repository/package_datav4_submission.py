"""Package the data v4 duration-balanced heads + decision layer on top of the verified Existing9504 archive.

  .venv/bin/python package_datav4_submission.py --output NEW/submit.zip
Use --legacy-decision-layer for candidates evaluated with unmodified PANNs presence and max FILE fusion.
That mode removes inherited fitted decision heads and packages only the configured candidate checkpoints.
Frozen assets, FFmpeg, script.py, requirements.txt and the unchanged runtime modules are streamed byte-for-byte
from submissions/Existing9504_MIL/submit.zip (its recorded SHA-256 is checked first). Replaced: the six head
checkpoints and their lock/identity/metrics (train_duration_balanced), presence_head.json (phase B),
file_ensemble.json (fitted on val for these heads), full_bank_submission.py (decision layer) and the new
decision_heads.py. Every member is hashed into model/manifest.json, which the runtime re-checks at start.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

from v6_data import sha256
from full_bank_contract import saved_config

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'submissions/Existing9504_MIL'
RESULTS = ROOT / 'results/data_v4_20260923'
HEADS = RESULTS / 'data_v4_phase_c_20260923T1653Z/train_duration_balanced'
PRESENCE = RESULTS / 'data_v4_results_20260923T1639Z/train/presence_head.json'
OUTPUT = ROOT / 'submissions/DataV4_DurationBalanced/submit.zip'
CANDIDATE = 'datav4_durbal_MERT330M_WindowMIL_DFfc5_presence_ensemble'
OLD_CANDIDATE = b"CANDIDATE = 'balanced7144_MERT330M_WindowMIL_DFfc5_8020'"

REQUIRED_BASE = {'script.py', 'requirements.txt', 'model/runtime/bin/ffmpeg',
                 'model/assets/asset_hashes.json', 'model/assets/baseline/script.py',
                 'model/assets/baseline/model/df_arena_1b/pytorch_model.bin',
                 'model/assets/baseline/model/htdemucs/955717e8-8726e21a.th',
                 'model/assets/baseline/model/panns/Cnn14_mAP=0.431.pth',
                 'model/assets/mert/pytorch_model.bin'} | {
    'model/runtime/' + name for name in ('ast_multitask.py', 'eight_heads.py', 'graph_scoring.py',
        'localization_models.py', 'mert12_submission.py', 'v6_core.py', 'v6_data.py', 'v6_runtime.py')}


def replacements(heads: Path, presence: Path | None, candidate: str) -> tuple[dict[str, bytes], list[str]]:
    lock = json.loads((heads / 'candidate_lock.json').read_text())
    config = saved_config(heads)
    trained = [f'{r}_{seed}.pt' for r in ('music', 'voice') for seed in config['seeds']]
    if set(lock['checkpoints']) != set(trained):
        raise ValueError('Candidate checkpoint set differs from configured seeds')
    for name, digest in lock['checkpoints'].items():
        if sha256(heads / name) != digest:
            raise ValueError(f'checkpoint differs from its candidate lock: {name}')
    if lock.get('config', config) != config:
        raise ValueError('Candidate configuration differs from run identity')
    from full_bank_submission import load_heads
    load_heads(heads, 'cpu')  # Reject internally inconsistent or invalid weights before creating an archive.
    trained += ['candidate_lock.json', 'run_identity.json', 'validation_metrics.json']
    if presence is None:
        if any((heads / name).exists() for name in ('file_ensemble.json', 'presence_head.json')):
            raise ValueError('Legacy decision mode cannot discard candidate decision artifacts')
    else:
        trained.append('file_ensemble.json')
        ensemble = json.loads((heads / 'file_ensemble.json').read_text())
        if ensemble.get('fitted_on') not in ('val', 'calibration') or 'prev' in ensemble['weights']:
            raise ValueError('file_ensemble must be val-fitted and self-contained (no prev candidate)')
    runtime = (ROOT / 'full_bank_submission.py').read_bytes()
    if runtime.count(OLD_CANDIDATE) != 1:
        raise ValueError('unexpected candidate declaration in full_bank_submission.py')
    out = {'model/trained/' + n: (heads / n).read_bytes() for n in trained}
    if presence is not None:
        out['model/trained/presence_head.json'] = presence.read_bytes()
    out['model/runtime/full_bank_submission.py'] = runtime.replace(OLD_CANDIDATE, f'CANDIDATE = {candidate!r}'.encode())
    out['model/runtime/decision_heads.py'] = (ROOT / 'decision_heads.py').read_bytes()
    out['model/runtime/full_bank_contract.py'] = (ROOT / 'full_bank_contract.py').read_bytes()
    return out, trained


def package(candidate_lock: Path = HEADS / "candidate_lock.json", base_archive: Path = BASE / "submit.zip",
            output: Path = OUTPUT, presence: Path | None = PRESENCE, candidate: str = CANDIDATE) -> dict:
    heads = candidate_lock.parent
    if candidate_lock.name != 'candidate_lock.json':
        raise ValueError('Candidate lock must be named candidate_lock.json beside its checkpoints')
    if output.exists() or output.with_suffix('.partial').exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Use a fresh package output; existing artifacts are preserved')
    base_receipt = json.loads(base_archive.with_suffix('.json').read_text())
    if sha256(base_archive) != base_receipt['sha256']:
        raise ValueError('base archive changed since its verification')
    new, trained = replacements(heads, presence, candidate)
    producer = json.loads((heads / 'run_identity.json').read_text()).get('feature_contract')
    if not producer or not producer.get('assets') or not producer.get('implementation'):
        raise ValueError('Packaging requires verified encoder asset and implementation provenance')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial')
    written = {}
    with zipfile.ZipFile(base_archive) as src, zipfile.ZipFile(partial, 'x', zipfile.ZIP_STORED, allowZip64=True) as dst:
        names = src.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Unsafe or duplicate base archive member')
        if not REQUIRED_BASE <= set(names):
            raise ValueError('Base archive lacks required frozen runtime members')
        kept = [n for n in names if n != 'model/manifest.json' and n not in new
                and not (n.startswith('model/trained/') and n.rsplit('/', 1)[1] not in trained)]
        for name in kept:
            digest = hashlib.sha256()
            with src.open(name) as fin, dst.open(zipfile.ZipInfo(name), 'w', force_zip64=True) as fout:
                while chunk := fin.read(8 << 20):
                    fout.write(chunk); digest.update(chunk)
            if name in base_receipt['member_sha256'] and digest.hexdigest() != base_receipt['member_sha256'][name]:
                raise ValueError('base member differs from its receipt: ' + name)
            written[name] = digest.hexdigest()
        for name, data in sorted(new.items()):
            dst.writestr(zipfile.ZipInfo(name), data)
            written[name] = hashlib.sha256(data).hexdigest()
        expected_members = {'model/assets/' + name: digest for name, digest in producer['assets'].items()}
        expected_members.update({'model/runtime/' + name: digest for name, digest in producer['implementation'].items()
                                 if name != 'full_bank_features.py'})
        run_identity = json.loads((heads / 'run_identity.json').read_text())
        for key, name in (('model', 'localization_models.py'), ('head', 'eight_heads.py')):
            if key in run_identity:
                expected_members['model/runtime/' + name] = run_identity[key]
        if any(written.get(name) != digest for name, digest in expected_members.items()):
            raise ValueError('Packaged encoder assets or preprocessing differ from training')
        results = json.loads((heads / 'results.json').read_text())
        choice = {'candidate': candidate, 'base_archive_sha256': base_receipt['sha256'],
                  'heads': 'per-route checkpoint selection; see run identity', 'config': saved_config(heads),
                  'split': 'see candidate lock and run identity; independent validation not asserted',
                  'presence': ('legacy PANNs max' if presence is None else
                               'presence_head.json logistic heads over PANNs tag statistics + stem energy'),
                  'file_fusion': ('max' if presence is None else
                                  json.loads((heads / 'file_ensemble.json').read_text())['weights']),
                  'val_route_eer': {k: v['eer'] for k, v in results.items() if k.endswith('_ensemble')},
                  'per_file_independent': True, 'inference_gain': 'peak capped at 0.6'}
        manifest = (json.dumps({'choice': choice, 'sha256': written}, indent=2) + '\n').encode()
        dst.writestr(zipfile.ZipInfo('model/manifest.json'), manifest)
        written['model/manifest.json'] = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(partial) as z:
        if z.testzip() is not None:
            raise ValueError('CRC failure')
        names = set(z.namelist())
        tops = {n.split('/', 1)[0] for n in names}
        if tops != {'model', 'script.py', 'requirements.txt'}:
            raise ValueError(f'archive roots must be model/, script.py, requirements.txt: {tops}')
        for name, digest in written.items():
            with z.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('member changed on write: ' + name)
        expanded = sum(i.file_size for i in z.infolist())
    if partial.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
        raise ValueError('competition archive size limit exceeded')
    partial.replace(output)
    report = {'file': str(output), 'sha256': sha256(output), 'bytes': output.stat().st_size, 'expanded_bytes': expanded,
              'choice': choice, 'member_sha256': written, 'crc_verified': True,
              'replaced_members': sorted(new), 'official_l4_validated': False}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-lock', type=Path, default=HEADS / 'candidate_lock.json')
    parser.add_argument('--base-archive', type=Path, default=BASE / 'submit.zip')
    parser.add_argument('--presence-head', type=Path, default=PRESENCE)
    parser.add_argument('--legacy-decision-layer', action='store_true',
                        help='Preserve legacy PANNs presence and max FILE fusion; include no fitted decision heads')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', default=CANDIDATE)
    args = parser.parse_args()
    print(json.dumps(package(args.candidate_lock, args.base_archive, args.output,
                             None if args.legacy_decision_layer else args.presence_head, args.candidate), indent=2))
