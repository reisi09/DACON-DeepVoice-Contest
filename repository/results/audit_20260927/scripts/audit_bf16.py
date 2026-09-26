"""Audit A1: packaged P1/S2a runtimes ran the heads under CUDA bf16 autocast; heads were trained/selected in fp32.
Question: how much does bf16 head inference change VOICE/MUSIC scores and val EER? Real cached features, CPU bf16 autocast
as the bf16 proxy (CUDA bf16 autocast also casts matmul inputs to bf16). Criterion: |dEER| < 0.002 and Spearman > 0.999."""
import csv, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.')
from full_bank_evaluate import load_selected_heads
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve
B = 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
H = Path(sys.argv[1])
rows = [r for r in csv.DictReader(open(B + 'bank_trainval.csv')) if r['split'] == 'val']
music, voice, fi, _ = load_selected_heads(H, 'cpu')
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); k = int(np.argmin(np.abs(fpr - (1 - tpr)))); return (fpr[k] + 1 - tpr[k]) / 2
out = {m: {'v': [], 'm': []} for m in ('fp32', 'bf16')}
with torch.no_grad():
    for r in rows:
        with np.load(f"_workspace/cache_v4/cache_tv/{r['ID']}.npz") as d:
            f = torch.from_numpy(d['mert_windows']); e = torch.from_numpy(d['voice_embeddings']); silent = bool(d['voice_silent'])
        for mode in ('fp32', 'bf16'):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=mode == 'bf16'):
                ms = [float((h.head((f - h.mean) / h.scale)[0].float().logsumexp(0) - np.log(len(f))).sigmoid()) for h in music]
                vs = [0. if silent else float(h(e).float().softmax(-1)[:, fi].max()) for h in voice]
            out[mode]['m'].append(np.mean(ms)); out[mode]['v'].append(np.mean(vs))
for comp, key in (('voice', 'v'), ('music', 'm')):
    C = comp.upper()
    idx = [i for i, r in enumerate(rows) if r[f'{C}_FAKE_mask'] == '1' and r[f'{C}_PRESENT'] == '1']
    y = [int(rows[i][f'{C}_FAKE']) for i in idx]
    a = np.array(out['fp32'][key])[idx]; b = np.array(out['bf16'][key])[idx]
    print(comp, 'n', len(idx), 'max|d|', round(float(np.abs(a - b).max()), 4), 'spearman', round(spearmanr(a, b)[0], 5),
          'EER fp32', round(eer(y, a), 5), 'bf16', round(eer(y, b), 5))
