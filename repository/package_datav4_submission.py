"""Package the data v4 duration-balanced heads + decision layer on top of the verified Existing9504 archive.

  .venv/bin/python package_datav4_submission.py
Frozen assets, FFmpeg, script.py, requirements.txt and the unchanged runtime modules are streamed byte-for-byte
from submissions/Existing9504_MIL/submit.zip (its recorded SHA-256 is checked first). Replaced: the six head
checkpoints and their lock/identity/metrics (train_duration_balanced), presence_head.json (phase B),
file_ensemble.json (fitted on val for these heads), full_bank_submission.py (decision layer) and the new
decision_heads.py. Every member is hashed into model/manifest.json, which the runtime re-checks at start.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from v6_data import sha256

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'submissions/Existing9504_MIL'
RESULTS = ROOT / 'results/data_v4_20260923'
HEADS = RESULTS / 'data_v4_phase_c_20260923T1653Z/train_duration_balanced'
PRESENCE = RESULTS / 'data_v4_results_20260923T1639Z/train/presence_head.json'
OUTPUT = ROOT / 'submissions/DataV4_DurationBalanced/submit.zip'
CANDIDATE = 'datav4_durbal_MERT330M_WindowMIL_DFfc5_presence_ensemble'
HEADS_NOTE = 'data v4 train split, --duration-balance, seeds 41/42/43, validation-only checkpoint selection'
SPLIT_NOTE = 'analysis-driven holdout (data_v4/split_rules.json): trainval 35,589 / test 6,450'
OLD_CANDIDATE = b"CANDIDATE = 'balanced7144_MERT330M_WindowMIL_DFfc5_8020'"
TRAINED = [f'{r}_{s}.pt' for r in ('music', 'voice') for s in (41, 42, 43)] + [
    'candidate_lock.json', 'run_identity.json', 'validation_metrics.json', 'file_ensemble.json']
# optional component fusion (data_v4/fusion_fit.py output dir) and its zero-shot member weights / vendored code
FUSION = None
DETECTORS: dict[str, Path] = {}
SONICS = ROOT / '_workspace/models/sonics-spectttra-alpha-120s'
VENDORED = {'sonics': ROOT / '_workspace/models/sonics-main/sonics', 'timm': ROOT / '.venv/lib/python3.11/site-packages/timm'}


def replacements() -> dict[str, bytes]:
    lock = json.loads((HEADS / 'candidate_lock.json').read_text())
    for name, digest in lock['checkpoints'].items():
        if sha256(HEADS / name) != digest:
            raise ValueError(f'checkpoint differs from its candidate lock: {name}')
    ensemble = json.loads((HEADS / 'file_ensemble.json').read_text())
    if ensemble.get('fitted_on') != 'val' or 'prev' in ensemble['weights']:
        raise ValueError('file_ensemble must be val-fitted and self-contained (no prev candidate)')
    runtime = (ROOT / 'full_bank_submission.py').read_bytes()
    if runtime.count(OLD_CANDIDATE) != 1:
        raise ValueError('unexpected candidate declaration in full_bank_submission.py')
    out = {'model/trained/' + n: (HEADS / n).read_bytes() for n in TRAINED}
    for n in [f'voice_ad_{s}.pt' for s in (41, 42, 43)] + ['music_layers.json']:   # optional new-backbone heads
        if (HEADS / n).exists():
            if n.endswith('.pt') and lock['checkpoints'].get(n) != sha256(HEADS / n):
                raise ValueError(f'checkpoint differs from its candidate lock: {n}')
            out['model/trained/' + n] = (HEADS / n).read_bytes()
    out['model/trained/presence_head.json'] = PRESENCE.read_bytes()
    out['model/runtime/full_bank_submission.py'] = runtime.replace(OLD_CANDIDATE, f'CANDIDATE = {CANDIDATE!r}'.encode())
    out['model/runtime/decision_heads.py'] = (ROOT / 'decision_heads.py').read_bytes()
    if FUSION:
        fusion = json.loads((FUSION / 'component_fusion.json').read_text())
        if fusion.get('fitted_on') != 'data v4 val' or json.loads((FUSION / 'file_ensemble.json').read_text()).get('fitted_on') != 'val':
            raise ValueError('fusion and its refitted file ensemble must be val-fitted')
        out['model/trained/component_fusion.json'] = (FUSION / 'component_fusion.json').read_bytes()
        out['model/trained/file_ensemble.json'] = (FUSION / 'file_ensemble.json').read_bytes()   # refitted on fused components
        for name in sorted({n.rsplit('_', 1)[0] for n in fusion['voice']['weights'] if n not in ('ours', 'df0')}):
            for f in ('antideepfake_config.json', 'model.safetensors'):
                out[f'model/assets/antideepfake/{name}/{f}'] = (DETECTORS[name] / f).read_bytes()
        if fusion['music']['weights'].get('sonics'):   # SONICS and its vendored code (with timm) only when fused
            for f in ('config.json', 'pytorch_model.bin'):
                out[f'model/assets/sonics/{f}'] = (SONICS / f).read_bytes()
            for package, src in VENDORED.items():
                for f in sorted(src.rglob('*')):
                    if f.is_file() and '__pycache__' not in f.parts:
                        out[f'model/runtime/{package}/{f.relative_to(src).as_posix()}'] = f.read_bytes()
        for f in ('component_fusion.py', 'antideepfake_runtime.py'):
            out[f'model/runtime/{f}'] = (ROOT / f).read_bytes()
    return out


def package() -> dict:
    base_receipt = json.loads((BASE / 'submit.json').read_text())
    if sha256(BASE / 'submit.zip') != base_receipt['sha256']:
        raise ValueError('base archive changed since its verification')
    new = replacements()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix('.partial')
    written = {}
    with zipfile.ZipFile(BASE / 'submit.zip') as src, zipfile.ZipFile(partial, 'w', zipfile.ZIP_STORED, allowZip64=True) as dst:
        kept = [n for n in src.namelist() if n != 'model/manifest.json' and n not in new
                and not (n.startswith('model/trained/') and n.rsplit('/', 1)[1] not in TRAINED)]
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
        results = json.loads((HEADS / 'results.json').read_text())
        choice = {'candidate': CANDIDATE, 'base_archive_sha256': base_receipt['sha256'],
                  'heads': HEADS_NOTE,
                  'split': SPLIT_NOTE,
                  'presence': 'presence_head.json logistic heads over PANNs tag statistics + stem energy',
                  'file_fusion': 'file_ensemble.json equal-weight CDF mean of max / noisy_or / max_nopresence (val-fitted)',
                  'val_route_eer': {k: v['eer'] for k, v in results.items() if k.endswith('_ensemble')},
                  'per_file_independent': True, 'inference_gain': 'peak capped at 0.6'}
        if FUSION:
            fusion = json.loads((FUSION / 'component_fusion.json').read_text())
            choice['component_fusion'] = {route: fusion[route]['weights'] for route in ('voice', 'music')}
            choice['file_fusion'] = 'file_ensemble.json refitted on val over the fused VOICE/MUSIC scores'
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
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('member changed on write: ' + name)
        expanded = sum(i.file_size for i in z.infolist())
    if partial.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
        raise ValueError('competition archive size limit exceeded')
    partial.replace(OUTPUT)
    report = {'file': str(OUTPUT), 'sha256': sha256(OUTPUT), 'bytes': OUTPUT.stat().st_size, 'expanded_bytes': expanded,
              'choice': choice, 'member_sha256': written, 'crc_verified': True,
              'replaced_members': sorted(new), 'official_l4_validated': False}
    OUTPUT.with_name('submit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='defaults reproduce the recorded DataV4 duration-balanced package')
    ap.add_argument('--heads', type=Path, default=HEADS)
    ap.add_argument('--output', type=Path, default=OUTPUT)
    ap.add_argument('--candidate', default=CANDIDATE)
    ap.add_argument('--heads-note', default=HEADS_NOTE)
    ap.add_argument('--split-note', default=SPLIT_NOTE)
    ap.add_argument('--fusion', type=Path, help='fusion_fit.py output dir (component_fusion.json + file_ensemble.json)')
    ap.add_argument('--antideepfake', nargs='*', default=[], help='name=dir with antideepfake_config.json + model.safetensors')
    a = ap.parse_args()
    HEADS, OUTPUT, CANDIDATE, HEADS_NOTE, SPLIT_NOTE = a.heads.resolve(), a.output.resolve(), a.candidate, a.heads_note, a.split_note
    FUSION = a.fusion.resolve() if a.fusion else None
    DETECTORS = {n: Path(d).resolve() for n, d in (s.split('=', 1) for s in a.antideepfake)}
    r = package()
    print(json.dumps({k: v for k, v in r.items() if k != 'member_sha256'}, indent=2))
