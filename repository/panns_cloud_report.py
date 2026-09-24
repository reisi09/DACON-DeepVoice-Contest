"""Paired parent-component development diagnostics, never a leaderboard claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from graph_scoring import official_metrics
from panns_adaptation_data import write_json, sha256


def curve(y, p, weights):
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order];w=weights[order]
    ends=np.r_[np.flatnonzero(np.diff(p))+1,len(p)]-1
    positives=(y*w).sum();negatives=((1-y)*w).sum()
    if not positives or not negatives:return np.nan,np.nan
    t=np.r_[0,np.cumsum(y*w)[ends]/positives]
    f=np.r_[0,np.cumsum((1-y)*w)[ends]/negatives]
    k=np.argmin(np.abs(f-(1-t)))
    return (f[k]+1-t[k])/2,float(np.sum(np.diff(f)*(t[1:]+t[:-1])/2))


def vector(labels,predictions,weights):
    return np.array([curve(labels[:,i],predictions[:,i],weights)[0 if i==0 else 1] for i in range(3)])


def report(config, repeats=2000):
    root=Path(config['output'])
    rows={r['ID']:r for r in map(json.loads,Path(config['rows']).read_text().splitlines())}
    base=json.loads((root/'evaluation_R0_41.json').read_text())
    refs={p['ID']:p for p in base['predictions']}
    targets=('FILE_FAKE','VOICE_PRESENT','MUSIC_PRESENT')
    files=[root/f'evaluation_{arm}_{seed}.json' for arm in config['arms'] for seed in config['seeds']]
    candidates=[json.loads(p.read_text()) for p in files]
    results={}
    for role in ('development','selection'):
        all_ids=[i for i in refs if rows[i]['role']==role]
        if not all_ids:continue
        ids=[i for i in all_ids if all(rows[i][k] in (0,1) for k in targets)]
        if not ids:
            results[role]=dict(reason='No complete common G labels',excluded=len(all_ids));continue
        labels=np.array([[rows[i][k] for k in targets] for i in ids],dtype=float)
        groups=sorted({rows[i]['component_id'] for i in ids});mapping={g:n for n,g in enumerate(groups)}
        group_index=np.array([mapping[rows[i]['component_id']] for i in ids])
        baseline=np.array([[refs[i][k+'_PROB'] for k in targets] for i in ids])
        def draws():
            rng=np.random.default_rng(20260925)
            for _ in range(repeats):
                yield np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[group_index]
        base_draws=np.array([vector(labels,baseline,w) for w in draws()])
        comparisons=[]
        paired_deltas={}
        for candidate in candidates:
            pred={p['ID']:p for p in candidate['predictions']}
            if set(pred)!=set(refs):raise ValueError('Candidate panel differs from frozen R0')
            for i in refs:
                if any(pred[i][k]!=refs[i][k] for k in ('VOICE_FAKE_PROB','MUSIC_FAKE_PROB')):
                    raise ValueError('Fake head probabilities changed')
            probabilities=np.array([[pred[i][k+'_PROB'] for k in targets] for i in ids])
            delta=np.array([vector(labels,probabilities,w) for w in draws()])-base_draws
            gain=delta@np.array([-.45,.05,.05])
            valid=np.isfinite(delta).all(1)
            def interval(x):
                return np.quantile(x[valid],[.025,.975]).tolist() if valid.any() else None
            point=vector(labels,probabilities,np.ones(len(ids)))-vector(labels,baseline,np.ones(len(ids)))
            paired_deltas[(candidate['arm'],candidate['seed'])]=(point,delta)
            presence={}
            for key in targets[1:]:
                eligible=[i for i in all_ids if rows[i][key] in (0,1)]
                y=np.array([rows[i][key] for i in eligible]);q=np.clip([pred[i][key+'_PROB'] for i in eligible],1e-12,1-1e-12)
                presence[key]=dict(known=len(eligible),total=len(all_ids),
                    brier=float(np.mean((q-y)**2)) if len(y) else None,
                    log_loss=float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q))) if len(y) else None)
            strata={}
            for field in ('source','presence_combination'):
                def category(i):
                    return str(rows[i]['source']) if field=='source' else str((rows[i]['VOICE_PRESENT'],rows[i]['MUSIC_PRESENT']))
                for value in sorted({category(i) for i in all_ids}):
                    subset=[i for i in all_ids if category(i)==value]
                    strata[field+'|'+value]=dict(rows=len(subset),candidate=official_metrics([rows[i] for i in subset],[pred[i] for i in subset]),
                        baseline=official_metrics([rows[i] for i in subset],[refs[i] for i in subset]))
            def finite(value):
                return float(value) if np.isfinite(value) else None
            comparisons.append(dict(arm=candidate['arm'],seed=candidate['seed'],G=finite(point@np.array([-.45,.05,.05])),
                G_interval95=interval(gain),voice_auc_delta=finite(point[1]),music_auc_delta=finite(point[2]),
                voice_auc_interval95=interval(delta[:,1]),music_auc_interval95=interval(delta[:,2]),
                invalid_bootstrap_fraction=float(1-valid.mean()),presence=presence,strata=strata))
        structural_contrasts=[]
        for seed in config['seeds']:
            if ('R3',seed) not in paired_deltas or ('R1',seed) not in paired_deltas:
                continue
            p3,d3=paired_deltas[('R3',seed)];p1,d1=paired_deltas[('R1',seed)]
            delta=d3-d1;valid=np.isfinite(delta).all(1)
            values=delta@np.array([-.45,.05,.05])
            point=(p3-p1)@np.array([-.45,.05,.05])
            structural_contrasts.append(dict(contrast='R3-R1',seed=seed,G=float(point) if np.isfinite(point) else None,
                G_interval95=np.quantile(values[valid],[.025,.975]).tolist() if valid.any() else None,
                invalid_bootstrap_fraction=float(1-valid.mean())))
        results[role]=dict(structural_contrasts=structural_contrasts,complete_G_rows=len(ids),excluded_from_G=len(all_ids)-len(ids),parent_components=len(groups),comparisons=comparisons)
    result=dict(roles=results,bootstrap_repeats=repeats,bootstrap_seed=20260925,
        common_panel_selected_by_labels_only=True,independent_evaluation=False,adoption_authorized=False,
        inputs={p.name:sha256(p) for p in [root/'evaluation_R0_41.json',*files]})
    write_json(root/'development_report.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();report(json.loads(a.run.read_text()))
