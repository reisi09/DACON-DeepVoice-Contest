"""Verify the persisted feature archive against extraction receipts before shutdown."""
import hashlib
import json
from pathlib import Path
import zipfile
root=Path('/root/C08_MERT330_A100')
obj=Path('/root-1/temporal/mert330-a100-20260916')
p=obj/'C08_MERT330_A100_features.zip'
receipt=json.loads(p.with_suffix('.json').read_text())
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
assert p.stat().st_size==receipt['bytes']
records=json.loads((root/'results/features_manifest.json').read_text())['records']
with zipfile.ZipFile(p) as z:
    assert set(z.namelist())=={'feature_cache/'+r['file'] for r in records.values()}
    for item in records.values():
        assert hashlib.sha256(z.read('feature_cache/'+item['file'])).hexdigest()==item['sha256']
result=dict(**receipt,members_verified=len(records),crc_and_member_hashes_valid=True)
(root/'feature_archive_verification.json').write_text(json.dumps(result,indent=2))
(obj/'feature_archive_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
