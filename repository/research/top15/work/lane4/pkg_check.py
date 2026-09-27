"""Lane 4 check P: are the S3 archive members byte-identical to model/manifest.json (what run() re-verifies),
to submit.json, to the source heads dir (_workspace/final_backbone_20260927) + candidate_lock, and are the frozen
encoder assets identical to the ones the training caches were extracted with (cache_v4 index identity)?
Read-only; streams the 8.3 GB zip once."""
import hashlib, json, zipfile, sys
from pathlib import Path

R = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
Z = R / 'submissions/DataV4_S3_Backbones/submit.zip'
F = R / '_workspace/final_backbone_20260927'
out = {}
with zipfile.ZipFile(Z) as z:
    manifest = json.loads(z.read('model/manifest.json'))
    receipt = json.loads((Z.parent / 'submit.json').read_text())
    actual = {}
    for info in z.infolist():
        h = hashlib.sha256()
        with z.open(info) as f:
            while chunk := f.read(8 << 20):
                h.update(chunk)
        actual[info.filename] = h.hexdigest()
mf = manifest['sha256']
out['members'] = len(actual)
out['manifest_entries'] = len(mf)
out['manifest_mismatch'] = [n for n in mf if actual.get(n) != mf[n]]
out['unlisted_in_manifest'] = sorted(set(actual) - set(mf) - {'model/manifest.json'})
out['receipt_mismatch'] = [n for n, d in receipt['member_sha256'].items() if actual.get(n) != d]
lock = json.loads((F / 'candidate_lock.json').read_text())['checkpoints']
src = {}
for p in sorted(F.iterdir()):
    n = 'model/trained/' + p.name
    if n in actual:
        src[p.name] = hashlib.sha256(p.read_bytes()).hexdigest() == actual[n]
out['trained_equal_to_final_dir'] = src
out['lock_vs_zip'] = {n: actual['model/trained/' + n] == d for n, d in lock.items()}
# P1 DF-Arena voice heads copied from p1_refit_cpu
out['voice_equal_to_p1'] = {s: hashlib.sha256((R / f'_workspace/p1_refit_cpu/voice_{s}.pt').read_bytes()).hexdigest()
                            == actual[f'model/trained/voice_{s}.pt'] for s in (41, 42, 43)}
# presence head identical to the one used in train_backbone_heads.final (PRESENCE)
pres = R / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'
out['presence_equal'] = hashlib.sha256(pres.read_bytes()).hexdigest() == actual['model/trained/presence_head.json']
# encoder assets vs the cache_v4 extraction identity (DF-Arena, HTDemucs, PANNs, MERT, baseline script)
ident = json.loads((R / '_workspace/cache_v4/cache_tv/index.json').read_text())['identity']
out['assets_vs_cache_v4_identity'] = {n: actual.get('model/assets/' + n) == d for n, d in ident['assets'].items()}
out['assets_all_equal'] = all(out['assets_vs_cache_v4_identity'].values())
# runtime implementation files vs cache_v4 identity (mert12_submission, decision_heads)
for n in ('mert12_submission.py', 'decision_heads.py'):
    out[f'impl_{n}_equal_cache_identity'] = actual.get('model/runtime/' + n) == ident['implementation'][n]
# AntiDeepfake safetensors vs the local copy used for training init (proj_fc) and extraction
ad = R / '_workspace/models/xls-r-1b-anti-deepfake/model.safetensors'
if ad.exists():
    h = hashlib.sha256()
    with ad.open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    out['antideepfake_safetensors_equal_local'] = h.hexdigest() == actual['model/assets/antideepfake/ad/model.safetensors']
    out['antideepfake_sha256'] = h.hexdigest()
json.dump(out, sys.stdout, indent=1)
print()
