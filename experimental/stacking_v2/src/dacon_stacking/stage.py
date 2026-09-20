"""Hash-guarded application to an explicitly supplied LOCAL submission source copy."""
from pathlib import Path
import shutil
from .common import ContractError,read_json,write_json,sha256,check_hash


def stage_runtime(source_root,return_root):
    source_root=Path(source_root).resolve();return_root=Path(return_root).resolve()
    info=read_json(return_root/'configs/runtime_patch.json')
    target=source_root/'model/runtime/full_bank_submission.py'
    check_hash(target,info['original_sha256'])
    patch=return_root/'patches/full_bank_submission.py';check_hash(patch,info['patched_sha256'])
    manifest_path=source_root/'model/manifest.json';manifest=read_json(manifest_path);old_manifest=sha256(manifest_path)
    package=source_root/'model/runtime/dacon_stacking'
    if package.exists(): raise ContractError('Refuse to overwrite installed stacking package')
    shutil.copyfile(patch,target);package.mkdir()
    for p in sorted((return_root/'src/dacon_stacking').glob('*.py')):
        shutil.copyfile(p,package/p.name);manifest['sha256'][str((package/p.name).relative_to(source_root))]=sha256(package/p.name)
    manifest['sha256']['model/runtime/full_bank_submission.py']=sha256(target)
    manifest['stacking_patch']={'original_manifest_sha256':old_manifest,'baseline_candidate_unchanged':True,'activation':'requires DACON_STACKING_LOCK and DACON_STACKING_LOCK_SHA256','gpu_parity':'NOT_RUN'}
    write_json(manifest_path,manifest)
    missing=[n for n in manifest['sha256'] if not (source_root/n).exists()]
    receipt={'patched_file_sha256':sha256(target),'new_manifest_sha256':sha256(manifest_path),'missing_package_members':missing,'deployment_ready':False,'reason':'Stage only; asset availability, GPU parity, real meta-data and candidate lock require separate verification.'}
    write_json(source_root/'stacking_stage_receipt.json',receipt)
    return receipt
