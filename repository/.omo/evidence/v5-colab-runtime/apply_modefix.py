"""Apply the verified training-mode repair before the held runner imports models."""
import hashlib
import json
import shutil
import time
from pathlib import Path

drive = Path('/content/drive/MyDrive/DACON_236749')
work = Path('/content/dacon_v5')
run = drive / 'runs_compare_v5/20260913_141033'
expected = {
    'v5_training': '61da419fe94e6f0d6099bb1abea0c22025991471e20e10c5a4cba2a689e80b2f',
    'v5_compare': '6d86c529dc6bd996d3668ae37fb4405f6b2313fea64a14eb15cfc139604cc473',
}
assert (work / 'HOLD_TRAINING').is_file(), 'Runner must still be held before imports'
assert time.time() < 1789315095, 'Original allocation deadline has elapsed'
for name, checksum in expected.items():
    source = drive / f'{name}_modefix.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum, name
    shutil.copy2(source, work / f'{name}.py')
    shutil.copy2(source, run / 'code' / f'{name}.py')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (run / 'code').glob('*.py')}
for name, value in (
    ('source_hashes.json', hashes),
    ('source_bundle_receipt.json', {'source_count_py': len(hashes), 'sources': hashes}),
    ('runtime_mode_patch.json', {'unix': time.time(), 'source_sha256': expected,
                               'reason': 'Restore train mode after evaluation and disable undeclared augmentation',
                               'before_model_imports': True, 'original_deadline': 1789315095}),
):
    (run / name).write_text(json.dumps(value, indent=2))
shutil.copy2(__file__, run / 'apply_modefix.py')
(work / 'HOLD_TRAINING').unlink()
print('V5_MODE_FIX_APPLIED_AND_GATE_RELEASED', flush=True)
