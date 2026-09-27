"""Lane 3 read-only checks (CPU, pandas/numpy/scipy only). Writes JSON next to this file.
  python lane3_checks.py split|duration|subset|coverage
"""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
OUT = Path(__file__).parent
PRED = ROOT / '_keep_best_20260926/train_duration_balanced'


def bank():
    tv = pd.read_csv(BANK / 'bank_trainval.csv', dtype=str, keep_default_na=False)
    te = pd.read_csv(BANK / 'bank_test.csv', dtype=str, keep_default_na=False)
    d = pd.concat([tv, te], ignore_index=True)
    dur = {}
    for c in ('cache_tv', 'cache_test'):
        items = json.loads((ROOT / '_workspace/cache_v4' / c / 'index.json').read_text())['items']
        dur.update({k: v['samples'] / 16000 for k, v in items.items()})
    d['dur'] = d.ID.map(dur)
    d['has_bb'] = [(ROOT / '_workspace/backbone_cache' / f'{i}.npz').exists() for i in d.ID]
    return d


def split_check(d):
    r = {'rows': len(d), 'splits': d.split.value_counts().to_dict(), 'dur_missing': int(d.dur.isna().sum())}
    for col in ('ID', 'orig_id', 'audio_sha256', 'source_sha256', 'group_id', 'component_id'):
        sp = d.groupby(col).split.nunique()
        r[f'{col}_multi_split'] = int((sp > 1).sum())
        r[f'{col}_dup_rows_within_split'] = int(d.duplicated([col, 'split']).sum())
    # constructed parents: parent strings are orig_id-style keys
    parents = d.parent_ids.map(json.loads)
    orig_split = dict(zip(d.orig_id, d.split))
    orig_group = dict(zip(d.orig_id, d.group_id))
    rows_with_parents = parents.map(len) > 0
    r['rows_with_parents'] = int(rows_with_parents.sum())
    in_bank = cross = cross_group = 0
    par_splits = {}
    for ps, s, g in zip(parents[rows_with_parents], d.split[rows_with_parents], d.group_id[rows_with_parents]):
        for p in ps:
            par_splits.setdefault(p, set()).add(s)
            if p in orig_split:
                in_bank += 1
                cross += orig_split[p] != s
                cross_group += orig_group[p] != g
    r['parent_refs_resolving_to_bank_rows'] = in_bank
    r['parent_refs_in_other_split'] = cross
    r['parent_refs_in_other_group'] = cross_group
    r['parent_items_used_in_multiple_splits'] = sum(len(v) > 1 for v in par_splits.values())
    # coarse keys: underlying track/speaker behind a crop id (openmic track, zeroth speaker, haim folder item)
    def coarse(k):
        src, _, rest = k.partition(':')
        if src == 'openmic':
            return 'openmic:' + rest.split('_')[0]
        if src in ('zeroth', 'speechfake') and re.search(r'(\d+)_(\d+)_\d+', rest):
            m = re.search(r'(\d+)_(\d+)_\d+', rest)
            return f'{src}_spk:{m.group(1)}'
        return k
    keys = {}
    for oid, s in zip(d.orig_id, d.split):
        keys.setdefault(coarse(oid), set()).add(s)
    for p, ss in par_splits.items():
        keys.setdefault(coarse(p), set()).update(ss)
    multi = {k: sorted(v) for k, v in keys.items() if len(v) > 1}
    r['coarse_keys_multi_split'] = len(multi)
    r['coarse_keys_multi_split_by_prefix'] = pd.Series([k.split(':')[0] for k in multi]).value_counts().to_dict() if multi else {}
    r['coarse_examples'] = dict(list(multi.items())[:8])
    return r


def eer_threshold(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


BINS = [0, 4, 8, 15, 30, 60, 1e9]
LAB = ['<4', '4-8', '8-15', '15-30', '30-60', '>60']


def duration_check(d):
    res = {}
    for split, f in (('test', 'test_predictions_decision.csv'), ('val', 'val_predictions_decision.csv')):
        p = pd.read_csv(PRED / f)
        x = d[d.split == split].merge(p, on='ID', how='inner')
        res[f'{split}_rows_bank_vs_pred'] = [int((d.split == split).sum()), len(x)]
        fm = x[x.FILE_FAKE_mask == '1'].copy()
        fm['y'] = fm.FILE_FAKE.astype(int)
        thr, eer = eer_threshold(fm.y, fm.FILE_FAKE_PROB)
        out = {'file_eer': round(eer, 5), 'threshold': thr}
        fm['bin'] = pd.cut(fm.dur, BINS, labels=LAB, right=False)
        fm['fp'] = (fm.FILE_FAKE_PROB >= thr) & (fm.y == 0)
        fm['fn'] = (fm.FILE_FAKE_PROB < thr) & (fm.y == 1)
        real, fake = fm[fm.y == 0], fm[fm.y == 1]
        rho_all = spearmanr(real.FILE_FAKE_PROB, real.dur)
        out['pooled_real'] = {'n': len(real), 'spearman': round(rho_all.statistic, 3), 'p': float(rho_all.pvalue)}
        rho_f = spearmanr(fake.FILE_FAKE_PROB, fake.dur)
        out['pooled_fake'] = {'n': len(fake), 'spearman': round(rho_f.statistic, 3), 'p': float(rho_f.pvalue)}
        out['fpr_by_bin_all_real'] = {b: [int(g.fp.sum()), len(g), round(g.fp.mean(), 3) if len(g) else None] for b, g in real.groupby('bin', observed=False)}
        out['fnr_by_bin_all_fake'] = {b: [int(g.fn.sum()), len(g), round(g.fn.mean(), 3) if len(g) else None] for b, g in fake.groupby('bin', observed=False)}
        per = {}
        zs, ws = [], []
        for src, g in real.groupby('source'):
            e = {'n': len(g), 'groups': g.group_id.nunique(), 'dur_q10_50_90': [round(q, 1) for q in g.dur.quantile([.1, .5, .9])],
                 'fpr': round(g.fp.mean(), 3), 'fpr_by_bin': {b: [int(h.fp.sum()), len(h)] for b, h in g.groupby('bin', observed=False) if len(h)}}
            if len(g) >= 20 and g.dur.nunique() > 5:
                rr = spearmanr(g.FILE_FAKE_PROB, g.dur)
                e['spearman'], e['p'] = round(rr.statistic, 3), float(rr.pvalue)
                zs.append(np.arctanh(np.clip(rr.statistic, -.999, .999))); ws.append(len(g) - 3)
                # within source x codec (codec also varies inside sources)
                cc = []
                for c, h in g.groupby('codec_chain'):
                    if len(h) >= 20 and h.dur.nunique() > 5:
                        cc.append([c, len(h), round(spearmanr(h.FILE_FAKE_PROB, h.dur).statistic, 3)])
                e['by_codec'] = cc
            per[src] = e
        out['real_by_source'] = per
        out['within_source_fisher_mean_rho'] = round(float(np.tanh(np.average(zs, weights=ws))), 3) if zs else None
        # within-source permutation test of the pooled statistic: sum over sources of n*rho
        rng = np.random.default_rng(0)
        srcs = [g for _, g in real.groupby('source') if len(g) >= 20 and g.dur.nunique() > 5]
        stat = lambda perm: sum(len(g) * spearmanr(g.FILE_FAKE_PROB, g.dur.values if not perm else rng.permutation(g.dur.values)).statistic for g in srcs) / sum(len(g) for g in srcs)
        obs = stat(False)
        null = np.array([stat(True) for _ in range(300)])
        out['within_source_perm'] = {'obs_weighted_rho': round(obs, 3), 'null_mean': round(null.mean(), 4), 'null_sd': round(null.std(), 4),
                                     'p_two_sided': float((np.abs(null) >= abs(obs)).mean())}
        # voice component among real-voice files
        vm = x[(x.VOICE_FAKE_mask == '1') & (x.VOICE_PRESENT == '1')]
        vr = vm[vm.VOICE_FAKE == '0']
        out['voice_real_by_source'] = {src: {'n': len(g), 'spearman': round(spearmanr(g.VOICE_FAKE_PROB, g.dur).statistic, 3)}
                                       for src, g in vr.groupby('source') if len(g) >= 20 and g.dur.nunique() > 5}
        out['voice_real_pooled'] = round(spearmanr(vr.VOICE_FAKE_PROB, vr.dur).statistic, 3)
        # fakes per source too (is it 'longer -> higher score' for everything?)
        out['fake_by_source'] = {src: {'n': len(g), 'spearman': round(spearmanr(g.FILE_FAKE_PROB, g.dur).statistic, 3), 'fnr': round(g.fn.mean(), 3)}
                                 for src, g in fake.groupby('source') if len(g) >= 20 and g.dur.nunique() > 5}
        res[split] = out
    return res


def subset_check(d):
    r = {'bb_rows': int(d.has_bb.sum()), 'share': round(d.has_bb.mean(), 4)}
    r['by_split'] = d.groupby('split').has_bb.agg(['sum', 'count', 'mean']).round(3).to_dict('index')
    r['by_route'] = d.groupby(['split', 'route']).has_bb.mean().round(3).unstack().to_dict('index')
    r['by_file_label'] = d.groupby(['split', 'FILE_FAKE']).has_bb.mean().round(3).unstack().to_dict('index')
    s = d.groupby('source').has_bb.agg(['sum', 'count', 'mean'])
    r['sources_zero_coverage'] = s[s['sum'] == 0].index.tolist()
    r['sources_full_coverage'] = s[s['mean'] == 1].index.tolist()
    r['by_source_test'] = d[d.split == 'test'].groupby('source').has_bb.agg(['sum', 'count']).to_dict('index')
    r['by_source'] = s.round(3).to_dict('index')
    r['by_id_first_hex'] = d.groupby(d.ID.str[3]).has_bb.mean().round(3).to_dict()
    q = [.1, .25, .5, .75, .9]
    r['dur_quantiles'] = {f'{sp}_{h}': [round(v, 2) for v in g.dur.quantile(q)] for (sp, h), g in d.groupby(['split', 'has_bb'])}
    # shift of the subset vs full, by (split, source): total variation distance of source mix
    tvd = {}
    for sp, g in d.groupby('split'):
        a = g.source.value_counts(normalize=True)
        b = g[g.has_bb].source.value_counts(normalize=True)
        tvd[sp] = round(0.5 * a.subtract(b, fill_value=0).abs().sum(), 3)
    r['source_mix_tvd_subset_vs_full'] = tvd
    # DB-head official metrics on the full held-out test vs its backbone subset (same predictions)
    sys.path.insert(0, str(ROOT))
    from graph_scoring import official_metrics
    p = pd.read_csv(PRED / 'test_predictions_decision.csv')
    x = d[d.split == 'test'].merge(p, on='ID')
    ads = lambda m: .5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer'])
    for name, part in (('full_test', x), ('bb_subset', x[x.has_bb]), ('non_bb', x[~x.has_bb])):
        rows = part.drop(columns=[c for c in p.columns if c != 'ID']).to_dict('records')
        preds = part[p.columns].to_dict('records')
        m = official_metrics(rows, preds)['metrics']
        r[f'db_heads_{name}'] = {k: round(m[k], 4) for k in ('file_eer', 'voice_eer', 'music_eer')} | {'ads': round(ads(m), 5), 'n': len(part)}
    return r


def coverage(d):
    t = []
    for src, g in d.groupby('source'):
        t.append({'source': src, 'rows': len(g), 'groups': g.group_id.nunique(), 'rows_per_group': round(len(g) / g.group_id.nunique(), 1),
                  'splits': '/'.join(f"{s[:2]}{n}" for s, n in g.split.value_counts().items()),
                  'route': '/'.join(g.route.unique()), 'file_fake': '/'.join(f"{k or '?'}:{n}" for k, n in g.FILE_FAKE.value_counts().items()),
                  'voice_fake': '/'.join(f"{k or 'NA'}:{n}" for k, n in g.VOICE_FAKE.value_counts().items()),
                  'music_fake': '/'.join(f"{k or 'NA'}:{n}" for k, n in g.MUSIC_FAKE.value_counts().items()),
                  'dur_p10_p50_p90': [round(v, 1) for v in g.dur.quantile([.1, .5, .9])],
                  'lt4s': round((g.dur < 4).mean(), 2), 'tel': round(g.codec_chain.str.startswith('tel').mean(), 2),
                  'lang_top': g.language.replace('', 'NA').value_counts().head(2).to_dict()})
    r = {'table': t}
    r['duration_by_split'] = {s: {'lt4': round((g.dur < 4).mean(), 3), 'in4_60': round(((g.dur >= 4) & (g.dur <= 60)).mean(), 3), 'gt60': round((g.dur > 60).mean(), 3),
                                  'p50': round(g.dur.median(), 1), 'p90': round(g.dur.quantile(.9), 1), 'p99': round(g.dur.quantile(.99), 1), 'max': round(g.dur.max(), 1)}
                              for s, g in d.groupby('split')}
    r['duration_by_split_label'] = {f'{s}|{y}': {'lt4': round((g.dur < 4).mean(), 3), 'p50': round(g.dur.median(), 1), 'p90': round(g.dur.quantile(.9), 1)}
                                    for (s, y), g in d.groupby(['split', 'FILE_FAKE'])}
    r['codec_by_split'] = pd.crosstab(d.codec_chain, d.split).to_dict()
    r['codec_x_label_train'] = pd.crosstab(d[d.split == 'train'].codec_chain, d[d.split == 'train'].FILE_FAKE).to_dict()
    r['language_top_by_split'] = {s: g.language.replace('', 'NA').value_counts().head(6).to_dict() for s, g in d.groupby('split')}
    r['lang_x_label_voice'] = {s: pd.crosstab(g.language.replace('', 'NA').where(g.language.isin(['ko', 'en', '']), 'other'), g.FILE_FAKE).to_dict()
                               for s, g in d[d.route == 'voice'].groupby('split')}
    typ = np.select([(d.VOICE_PRESENT == '1') & (d.MUSIC_PRESENT == '1'), d.VOICE_PRESENT == '1', d.MUSIC_PRESENT == '1', (d.VOICE_PRESENT == '0') & (d.MUSIC_PRESENT == '0')],
                    ['mixed', 'voice', 'music', 'none'], 'unknown')
    r['type_x_label_by_split'] = pd.crosstab([d.split, typ], d.FILE_FAKE).to_dict('index')
    r['mask_missing'] = {c: int((d[c + '_mask'] != '1').sum()) for c in ('VOICE_PRESENT', 'MUSIC_PRESENT', 'VOICE_FAKE', 'MUSIC_FAKE', 'FILE_FAKE')}
    # label-risk probes
    enh = d[d.source == 'constructed_enhanced']
    r['constructed_enhanced'] = {'generator': enh.generator.value_counts().to_dict(), 'FILE_FAKE': enh.FILE_FAKE.value_counts().to_dict(),
                                 'VOICE_FAKE': enh.VOICE_FAKE.value_counts().to_dict(), 'split': enh.split.value_counts().to_dict()}
    songs = d[(d.VOICE_PRESENT == '1') & (d.MUSIC_PRESENT == '1')]
    r['mixed_rows_voice_label'] = pd.crosstab(songs.source, [songs.VOICE_FAKE.replace('', 'NA'), songs.MUSIC_FAKE.replace('', 'NA')]).to_dict('index')
    hm = d[d.source.isin(['haim', 'existing:HAIM'])]
    r['haim_generators'] = pd.crosstab(hm.generator.replace('', 'NA'), [hm.FILE_FAKE, hm.VOICE_FAKE.replace('', 'NA'), hm.MUSIC_FAKE.replace('', 'NA')]).to_dict('index')
    r['historical_development_exposure'] = pd.crosstab(d.historical_development_exposure.replace('', 'NA'), d.split).to_dict()
    return r


if __name__ == '__main__':
    what = sys.argv[1]
    d = bank()
    fn = {'split': split_check, 'duration': duration_check, 'subset': subset_check, 'coverage': coverage}[what]
    res = fn(d)
    sk = lambda o: {str(k): sk(v) for k, v in o.items()} if isinstance(o, dict) else [sk(v) for v in o] if isinstance(o, list) else o
    res = sk(res)
    (OUT / f'{what}.json').write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    print(json.dumps(res, default=str, ensure_ascii=False)[:6000])
