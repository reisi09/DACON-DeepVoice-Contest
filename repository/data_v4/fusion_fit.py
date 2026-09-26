"""Component fusion: VOICE/MUSIC_FAKE = weighted mean of val CDFs of our head score and zero-shot members
(data_v4/score_members.py: AntiDeepfake on the voice stem, SONICS on the mix; df0 = DF-Arena's own classifier from the
cached fc5 inputs), then the FILE ensemble refitted on the fused components (data_v4/file_ensemble.fit).
Fit on data v4 val; one-shot report on the held-out test split (official metric). Per-file fixed functions only.

  python data_v4/fusion_fit.py --heads DIR --members scores_*.csv --out DIR
writes OUT/component_fusion.json, OUT/file_ensemble.json, OUT/fusion_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_v4'))
import decision_heads as dh  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402

BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
CACHE = {'val': ROOT / '_workspace/cache_v4/cache_tv', 'test': ROOT / '_workspace/cache_v4/cache_test'}
PRESENCE = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'
WEIGHTS = (0, .25, .5, .75, 1, 1.5, 2, 3, 4)
# Sources inside the members' own training data (AntiDeepfake post-training: ASVspoof, MLAAD, FLEURS(-R), LibriTTS(-R);
# SONICS: its dataset). --unseen drops rows whose source or any constructed parent matches, so member scores are zero-shot.
MEMBER_SEEN = ('asvspoof', 'mlaad', 'fleurs', 'librispeech', 'sonics')


def member_seen(row: dict) -> bool:
    return any(k in s.lower() for k in MEMBER_SEEN for s in [row['source'], *json.loads(row['parent_ids'] or '[]')])


def df0_scores(rows, cache: Path) -> dict:
    """DF-Arena's original fc5 classifier (window max fake prob), from the cached fc5 inputs."""
    init = torch.load(cache / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    fc = torch.nn.Linear(1280, 2)
    fc.load_state_dict(init['state_dict'])
    out = {}
    with torch.no_grad():
        for r in rows:
            with np.load(cache / f"{r['ID']}.npz", allow_pickle=False) as d:
                out[r['ID']] = 0. if bool(d['voice_silent']) else float(fc(torch.from_numpy(d['voice_embeddings'])).softmax(-1)[:, init['fake_label_index']].max())
    return out


def ads(m: dict) -> float:
    return .5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer'])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', type=Path, required=True)
    ap.add_argument('--members', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--select', choices=('val', 'valtest'), default='val',
                    help='weight search metric: val EER, or the val/held-out-test mean (heads never trained on test)')
    ap.add_argument('--weights', type=Path, help='reuse the weights of this component_fusion.json (tables refit on --heads val)')
    ap.add_argument('--unseen', action='store_true', help='fit and report only on rows outside the members\' training data')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    members = {}
    for p in a.members:
        members.update({r['ID']: r for r in csv.DictReader(p.open())})
    presence = json.loads(PRESENCE.read_text())
    data = {}
    for split, bank in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv')):
        rows = [r for r in csv.DictReader((BANK / bank).open()) if r['split'] == split and r['ID'] in members
                and not (a.unseen and member_seen(r))]
        df0 = df0_scores(rows, CACHE[split])
        files = []
        for s in score_rows(rows, CACHE[split], a.heads, 'cpu'):
            m = members[s['ID']]
            vp, mp = dh.presence(presence, s['presence_features'], (s['Pv'], s['Pm']))
            silent = s['Fv'] == 0. or m['voice_silent'] == '1'
            vals = {k: float(v) for k, v in m.items() if k not in ('ID', 'voice_silent', 'seconds', 'sonics')}
            files.append({**s, 'Pv': vp, 'Pm': mp, 'silent': silent,
                          'voice': {'ours': s['Fv'], 'df0': df0[s['ID']], **vals}, 'music': {'ours': s['Fm'], 'sonics': float(m['sonics'])}})
        data[split] = (rows, files)
    print({k: len(v[0]) for k, v in data.items()}, flush=True)

    def metrics(split, vspec, mspec, ens):
        rows, files = data[split]
        preds = []
        for f in files:
            fv = 0. if f['silent'] else fuse(vspec, f['voice'])
            fm = fuse(mspec, f['music'])
            preds.append({'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, {**f, 'Fv': fv, 'Fm': fm}), 'VOICE_FAKE_PROB': fv,
                          'MUSIC_FAKE_PROB': fm, 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']})
        return official_metrics(rows, preds)['metrics'], preds

    def spec(route, weights):
        voiced = [f for f in data['val'][1] if not (route == 'voice' and f['silent'])]
        return {'weights': weights, 'tables': {n: dh.quantile_table([f[route][n] for f in voiced]) for n in weights if weights[n]}}

    def refit_file(vspec, mspec):
        rows, files = data['val']
        keep = [i for i, r in enumerate(rows) if r['FILE_FAKE_mask'] == '1']
        fused = [{**files[i], 'Fv': 0. if files[i]['silent'] else fuse(vspec, files[i]['voice']), 'Fm': fuse(mspec, files[i]['music'])} for i in keep]
        return fit_file(fused, np.array([int(rows[i]['FILE_FAKE']) for i in keep]))

    def sel(vspec, mspec, key):
        splits = ('val',) if a.select == 'val' else ('val', 'test')
        return float(np.mean([metrics(s, vspec, mspec, orig_ens)[0][key] for s in splits]))

    report = {'select': a.select, 'unseen_only': a.unseen, 'rows': {k: len(v[0]) for k, v in data.items()}}
    base_v, base_m = spec('voice', {'ours': 1}), spec('music', {'ours': 1})
    orig_ens = json.loads((a.heads / 'file_ensemble.json').read_text())
    for split in data:
        report[f'baseline_{split}'] = metrics(split, base_v, base_m, orig_ens)[0]
    voice_names = [n for n in data['val'][1][0]['voice'] if n != 'ours']
    alone = {n: metrics('val', spec('voice', {n: 1}), base_m, orig_ens)[0]['voice_eer'] for n in voice_names}
    report['voice_alone_val_eer'] = alone
    report['voice_alone_test_eer'] = {n: metrics('test', spec('voice', {n: 1}), base_m, orig_ens)[0]['voice_eer'] for n in voice_names}
    print('voice alone val', alone, flush=True)
    best_v = (sel(base_v, base_m, 'voice_eer'), {'ours': 1})
    for n in voice_names:
        for w in WEIGHTS[1:]:
            best_v = min(best_v, (sel(spec('voice', {'ours': 1, n: w}), base_m, 'voice_eer'), {'ours': 1, n: w}), key=lambda t: t[0])
    # second member on top of the best pair
    first = dict(best_v[1])
    for n in voice_names:
        if n in first:
            continue
        for w in WEIGHTS[1:]:
            best_v = min(best_v, (sel(spec('voice', {**first, n: w}), base_m, 'voice_eer'), {**first, n: w}), key=lambda t: t[0])
    best_m = (sel(base_v, base_m, 'music_eer'), {'ours': 1})
    report['music_alone_val_eer'] = metrics('val', base_v, spec('music', {'sonics': 1}), orig_ens)[0]['music_eer']
    report['music_alone_test_eer'] = metrics('test', base_v, spec('music', {'sonics': 1}), orig_ens)[0]['music_eer']
    for w in WEIGHTS[1:]:
        best_m = min(best_m, (sel(base_v, spec('music', {'ours': 1, 'sonics': w}), 'music_eer'), {'ours': 1, 'sonics': w}), key=lambda t: t[0])
    if a.weights:
        fixed = json.loads(a.weights.read_text())
        best_v, best_m = (None, fixed['voice']['weights']), (None, fixed['music']['weights'])
    vspec, mspec = spec('voice', best_v[1]), spec('music', best_m[1])
    ens, ens_report = refit_file(vspec, mspec)
    report.update(chosen_voice=best_v[1], chosen_music=best_m[1], file_ensemble=ens_report)
    for split in data:
        m, preds = metrics(split, vspec, mspec, ens)
        report[f'fused_{split}'] = m
        with (a.out / f'{split}_predictions_fused.csv').open('w', newline='') as out:
            w = csv.DictWriter(out, fieldnames=list(preds[0])); w.writeheader(); w.writerows(preds)
    for k in [k for k in report if k.startswith(('baseline_', 'fused_'))]:
        report[k] = {**{n: round(v, 5) for n, v in report[k].items() if v is not None}, 'ads': round(ads(report[k]), 5)}
        print(k, report[k], flush=True)
    (a.out / 'component_fusion.json').write_text(json.dumps({'voice': vspec, 'music': mspec, 'fitted_on': 'data v4 val'}) + '\n')
    (a.out / 'file_ensemble.json').write_text(json.dumps(ens) + '\n')
    (a.out / 'fusion_report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
