"""Paired parent-component development diagnostics, never a leaderboard claim."""
from __future__ import annotations
import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Final, TypedDict
import numpy as np
from numpy.typing import NDArray
from graph_scoring import LABEL_COLUMNS, _label, official_metrics, validate_predictions
from panns_adaptation_data import write_json, sha256, validate_config

GAIN_WEIGHTS: Final = np.array([-.45, .05, .05])
NEW_PARAMETERS: Final = {'R1': 0, 'R2': 132160, 'R3': 132160, 'R5': 2048}


class DeltaSummary(TypedDict):
    G: float | None
    G_interval95: list[float] | None
    voice_auc_delta: float | None
    music_auc_delta: float | None
    voice_auc_interval95: list[float] | None
    music_auc_interval95: list[float] | None
    invalid_bootstrap_fraction: float


class CandidateSummary(DeltaSummary):
    arm: str
    new_parameters: int


def delta_summary(pairs: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]]) -> DeltaSummary:
    point = np.mean([pair[0] for pair in pairs], axis=0)
    delta = np.mean([pair[1] for pair in pairs], axis=0)
    valid = np.isfinite(delta).all(axis=1)
    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None
    def interval(values: NDArray[np.float64]) -> list[float] | None:
        return np.quantile(values[valid], [.025, .975]).tolist() if valid.any() else None
    return DeltaSummary(G=finite(point @ GAIN_WEIGHTS), G_interval95=interval(delta @ GAIN_WEIGHTS),
        voice_auc_delta=finite(point[1]), music_auc_delta=finite(point[2]),
        voice_auc_interval95=interval(delta[:, 1]), music_auc_interval95=interval(delta[:, 2]),
        invalid_bootstrap_fraction=float(1 - valid.mean()))


def rank_candidates(candidates: Sequence[CandidateSummary]) -> list[str]:
    eligible = []
    for candidate in candidates:
        gain, voice, music = candidate['G'], candidate['voice_auc_delta'], candidate['music_auc_delta']
        if (gain is not None and voice is not None and music is not None
                and np.isfinite([gain, voice, music]).all() and gain > 0 and voice >= 0 and music >= 0):
            eligible.append((-gain, candidate['new_parameters'], candidate['arm']))
    return [arm for _, _, arm in sorted(eligible)]


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
    from panns_cloud_run import training_identity
    from panns_training_budget import load_budget
    from panns_training_state import validate_completion
    validate_config(config)
    if sorted(config['seeds']) != [41,42,43] or repeats < 1:
        message='Report requires the three locked seeds and positive bootstrap repeats'
        raise ValueError(message)
    root=Path(config['output'])
    rows={r['ID']:r for r in map(json.loads,Path(config['rows']).read_text().splitlines())}
    truth={i:{key:_label(row,key) for key in LABEL_COLUMNS} for i,row in rows.items()}
    base=json.loads((root/'evaluation_R0_41.json').read_text())
    targets=('FILE_FAKE','VOICE_PRESENT','MUSIC_PRESENT')
    files=[root/f'evaluation_{arm}_{seed}.json' for arm in config['arms'] for seed in config['seeds']]
    candidates=[json.loads(p.read_text()) for p in files]
    conditions=[('R0',41),*[(arm,seed) for arm in config['arms'] for seed in config['seeds']]]
    expected_ids=[i for i in config['panel_ids'] if rows[i]['role'] in ('development','selection')]
    fixed={p['ID']:p for p in json.loads(Path(config['fixed_predictions']).read_text())['entries']}
    training_conditions=[]
    for evaluation,(arm,seed) in zip([base,*candidates],conditions,strict=True):
        expected=dict(arm=arm,seed=seed,config_identity=config['identity'],rows_sha256=sha256(config['rows']),
            fixed_predictions_sha256=sha256(config['fixed_predictions']),run_identity=None,checkpoint_sha256=None,
            completion_sha256=None)
        if arm!='R0':
            completed=json.loads((root/f'train_{arm}_{seed}'/'complete.json').read_text())
            budget,_=load_budget(config,evaluation['device'],validate_runtime=False)
            expected.update(run_identity=training_identity(config,arm,seed,evaluation['device'],validate_runtime=False),
                checkpoint_sha256=sha256(root/f'train_{arm}_{seed}'/'checkpoint.pt'),
                completion_sha256=sha256(root/f'train_{arm}_{seed}'/'complete.json'))
            validate_completion(completed, expected['run_identity'], expected['checkpoint_sha256'], budget)
            training_conditions.append(dict(arm=arm, seed=seed, **{key:completed[key] for key in
                ('steps','epochs_completed','best_epoch','best_loss','stop_reason','checkpoint_sha256')}))
        if any(evaluation.get(k)!=value for k,value in expected.items()):
            message='Evaluation provenance differs from the locked run'
            raise ValueError(message)
        evaluation['predictions']=validate_predictions(evaluation['predictions'],expected_ids)
        for prediction in evaluation['predictions']:
            if any(prediction[k]!=fixed[prediction['ID']][k] for k in ('VOICE_FAKE_PROB','MUSIC_FAKE_PROB')):
                message='Fake head probabilities changed'
                raise ValueError(message)
    refs={p['ID']:p for p in base['predictions']}
    results={}
    for role in ('development','selection'):
        all_ids=[i for i in refs if rows[i]['role']==role]
        if not all_ids:continue
        ids=[i for i in all_ids if all(truth[i][k] is not None for k in targets)
             and all(truth[i][prefix+'_PRESENT']==0 or truth[i][prefix+'_FAKE'] is not None for prefix in ('VOICE','MUSIC'))]
        common_ids=set(ids)
        labels=np.array([[truth[i][k] for k in targets] for i in ids],dtype=float).reshape(-1,3)
        fake_labels=np.array([[truth[i][prefix+'_FAKE'] if truth[i][prefix+'_PRESENT'] else -1
                              for prefix in ('VOICE','MUSIC')] for i in ids]).reshape(-1,2)
        groups=sorted({rows[i]['component_id'] for i in ids});mapping={g:n for n,g in enumerate(groups)}
        group_index=np.array([mapping[rows[i]['component_id']] for i in ids],dtype=int)
        baseline=np.array([[refs[i][k+'_PROB'] for k in targets] for i in ids]).reshape(-1,3)
        baseline_metrics=official_metrics([rows[i] for i in ids],[refs[i] for i in ids]) if ids else None
        g_defined=baseline_metrics is not None and baseline_metrics['score'] is not None
        def draws():
            rng=np.random.default_rng(20260925)
            for _ in range(repeats):
                yield (np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[group_index]
                       if groups else np.zeros(0))
        base_draws=np.array([vector(labels,baseline,w) if all(((w[:,None]*(fake_labels==value)).sum(0)>0).all()
                              for value in (0,1)) else np.full(3,np.nan) for w in draws()])
        comparisons=[]
        paired_deltas={}
        for candidate in candidates:
            pred={p['ID']:p for p in candidate['predictions']}
            probabilities=np.array([[pred[i][k+'_PROB'] for k in targets] for i in ids]).reshape(-1,3)
            delta=np.array([vector(labels,probabilities,w) for w in draws()])-base_draws
            point=vector(labels,probabilities,np.ones(len(ids)))-vector(labels,baseline,np.ones(len(ids)))
            if not g_defined:point[0]=np.nan
            paired_deltas[(candidate['arm'],candidate['seed'])]=(point,delta)
            presence={}
            for key in targets[1:]:
                eligible=[i for i in all_ids if truth[i][key] is not None]
                y=np.array([truth[i][key] for i in eligible])
                metrics={}
                for name,values in (('candidate',pred),('baseline',refs)):
                    q=np.array([values[i][key+'_PROB'] for i in eligible]);clipped=np.clip(q,1e-12,1-1e-12)
                    auc=curve(y,q,np.ones(len(y)))[1]
                    metrics[name]=dict(auc=float(auc) if np.isfinite(auc) else None,
                        brier=float(np.mean((q-y)**2)) if len(y) else None,
                        log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))) if len(y) else None)
                presence[key]=dict(known=len(eligible),total=len(all_ids),coverage=len(eligible)/len(all_ids),
                    **metrics['candidate'],baseline=metrics['baseline'])
            strata={}
            for field in ('source','presence_combination'):
                def category(i):
                    return str(rows[i]['source']) if field=='source' else str((truth[i]['VOICE_PRESENT'],truth[i]['MUSIC_PRESENT']))
                for value in sorted({category(i) for i in all_ids}):
                    subset=[i for i in all_ids if category(i)==value]
                    strata[field+'|'+value]=dict(rows=len(subset),candidate=official_metrics([rows[i] for i in subset],[pred[i] for i in subset]),
                        baseline=official_metrics([rows[i] for i in subset],[refs[i] for i in subset]))
            comparisons.append(dict(arm=candidate['arm'],seed=candidate['seed'],
                **delta_summary([(point,delta)]),presence=presence,strata=strata))
        seed_means=[dict(arm=arm,seeds=config['seeds'],new_parameters=NEW_PARAMETERS[arm],
            **delta_summary([paired_deltas[(arm,seed)] for seed in config['seeds']])) for arm in config['arms']]
        structural_contrasts=[]
        structural_seed_means=[]
        for arm in ('R3','R5'):
            if arm not in config['arms'] or 'R1' not in config['arms']:continue
            pairs=[]
            for seed in config['seeds']:
                p_new,d_new=paired_deltas[(arm,seed)];p1,d1=paired_deltas[('R1',seed)]
                pairs.append((p_new-p1,d_new-d1))
                structural_contrasts.append(dict(contrast=arm+'-R1',seed=seed,**delta_summary([pairs[-1]])))
            structural_seed_means.append(dict(contrast=arm+'-R1',seeds=config['seeds'],**delta_summary(pairs)))
        results[role]=dict(structural_contrasts=structural_contrasts,structural_seed_means=structural_seed_means,
            complete_G_rows=len(ids),excluded_from_G=len(all_ids)-len(ids),
            excluded_from_G_by_source=dict(Counter(rows[i]['source'] for i in all_ids if i not in common_ids)),
            G_defined=g_defined,baseline_common_panel=baseline_metrics,parent_components=len(groups),
            comparisons=comparisons,seed_means=seed_means)
        if role=='selection':
            ranking=rank_candidates(seed_means)
            r6_ranking=rank_candidates([mean for mean in seed_means if mean['arm'] in ('R3','R5')])
            results[role]['candidate_selection']=dict(ranking=ranking,
                selected_arm=ranking[0] if ranking else None,deployment_seed=41,
                r6_module_ranking=r6_ranking,r6_combination=r6_ranking if len(r6_ranking)==2 else [])
    result=dict(roles=results,bootstrap_repeats=repeats,bootstrap_seed=20260925,
        training_conditions=training_conditions,early_stopping=config['early_stopping'],equal_step_comparison=False,
        selection_adaptive_reuse=True,
        selection_warning='Selection controls early stopping and arm ranking; these are adaptive development diagnostics.',
        selection_role='selection',selected_candidate=results.get('selection',{}).get('candidate_selection',{}).get('selected_arm'),
        deployment_seed=41,seed_mean_is_ensemble=False,
        uncertainty_scope='Paired component sampling conditional on seeds 41, 42, 43; development diagnostics only',
        common_panel_selected_by_labels_only=True,independent_evaluation=False,adoption_authorized=False,
        config_identity=config['identity'],rows_sha256=sha256(config['rows']),
        inputs={p.name:sha256(p) for p in [root/'evaluation_R0_41.json',*files]})
    write_json(root/'development_report.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();report(json.loads(a.run.read_text()))
