"""Replace requirements only and prove every other archive member is unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

WORK = Path('/content/v5_repair')
DRIVE = Path('/content/drive/MyDrive/DACON_236749/runs_final_v5/20260913_215338/submissions_B1_B2')
ORIGINAL_HASHES = {
    'B1': '77f949d126ab7f22f3ebbaa2839febcc30b5c2bc22761b143e4e1b28beef8e9d',
    'B2': '5e46eb2a91f9dbdb38ba79756c3c366fd002f35b4022066f7b9afc4cfef70721',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipts = []
    for arm, original_hash in ORIGINAL_HASHES.items():
        original = WORK / f'{arm}_original.zip'
        fixed = WORK / f'V5_{arm}_fixed.zip'
        if not original.exists():
            shutil.copyfile(DRIVE / f'V5_{arm}.zip', original)
        assert digest(original) == original_hash, f'{arm}: wrong original archive'
        if not fixed.exists():
            shutil.copyfile(original, fixed)
        with zipfile.ZipFile(fixed) as archive:
            needs_update = archive.read('requirements.txt') != (WORK / 'requirements.txt').read_bytes()
        if needs_update:
            subprocess.run(['zip', '-u', str(fixed), 'requirements.txt'], cwd=WORK, check=True)
        members = {}
        with zipfile.ZipFile(original) as before, zipfile.ZipFile(fixed) as after:
            assert len(after.namelist()) == len(set(after.namelist()))
            assert set(before.namelist()) == set(after.namelist())
            assert after.read('requirements.txt') == (WORK / 'requirements.txt').read_bytes()
            for name in before.namelist():
                if name == 'requirements.txt':
                    continue
                with before.open(name) as left, after.open(name) as right:
                    old = hashlib.file_digest(left, 'sha256').hexdigest()
                    new = hashlib.file_digest(right, 'sha256').hexdigest()
                assert old == new, f'{arm}: changed member {name}'
                members[name] = new
        receipt = dict(arm=arm, original_sha256=original_hash, fixed_sha256=digest(fixed),
                       bytes=fixed.stat().st_size, changed_members=['requirements.txt'],
                       unchanged_member_sha256=members)
        receipts.append(receipt)
        (WORK / 'comparison_receipt.json').write_text(json.dumps(receipts, indent=2))
        print('ONLY_REQUIREMENTS_CHANGED', arm, len(members), flush=True)
    for receipt in receipts:
        arm = receipt['arm']
        shutil.copyfile(WORK / f'V5_{arm}_fixed.zip', DRIVE / f'V5_{arm}.zip')
        print('COPIED_TO_DRIVE', arm, flush=True)
    shutil.copyfile(WORK / 'comparison_receipt.json', DRIVE / 'dependency_fix_comparison.json')
    print('PAIR_REPAIR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
