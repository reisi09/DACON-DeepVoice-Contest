"""Exact discrete ROC/EER, fold-internal model selection, paired group bootstrap."""
from __future__ import annotations
from collections import defaultdict
import warnings
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from .common import ContractError, NotRunError
from ._snapshot_scoring import LABEL_COLUMNS, PREDICTION_COLUMNS, _roc_metrics, official_metrics, validate_predictions
from .features import feature_frame
from .models import fit_estimator, predict_estimator
from .provenance import connected_groups


def known_label(row: dict, name: str) -> int | None:
    mask=row.get(name+'_mask',1)
    if str(mask) not in ('0','1','0.0','1.0','False','True'):
        raise ContractError(f'Invalid label mask: {name}')
    if not bool(float(mask)): return None
    value=row.get(name)
    if value is None or str(value).strip().lower() in ('','nan','none','null','unknown','-1'):
        return None
    if value in (0,1,'0','1',0.,1.,'0.0','1.0'): return int(float(value))
    raise ContractError(f'Invalid binary label: {name}={value!r}')


def _applicable(row,name):
    if name in ('VOICE_FAKE','MUSIC_FAKE') and known_label(row,name.replace('FAKE','PRESENT'))!=1:
        return False
    return known_label(row,name) is not None


def metrics_report(rows: list[dict], predictions: list[dict]) -> dict:
    truth=[{'ID':r['ID'],**{n:known_label(r,n) for n in LABEL_COLUMNS}} for r in rows]
    official=official_metrics(truth,predictions)
    probability={}
    for name in LABEL_COLUMNS:
        use=[i for i,r in enumerate(rows) if _applicable(r,name)]
        y=[known_label(rows[i],name) for i in use];p=[predictions[i][name+'_PROB'] for i in use]
        probability[name+'_PROB']={'n':len(use),'brier':float(brier_score_loss(y,p)) if use else None,'logloss':float(log_loss(y,p,labels=[0,1])) if use else None,
                                  'single_class':len(set(y))<2,'unweighted':True}
    return {'official':official,'probability':probability,'note':'EER is discrete ranking/error balance, not a calibration metric; this is local replay, not an official submission score.'}


def baseline_predictions(rows,records,rule='max'):
    if [r['ID'] for r in rows]!=[r['ID'] for r in records]: raise ContractError('Row IDs/order mismatch')
    output=[]
    for row,record in zip(rows,records,strict=True):
        p={'ID':row['ID'],**record['baseline']}
        if rule=='noisy_or':
            a=p['VOICE_PRESENT_PROB']*p['VOICE_FAKE_PROB'];b=p['MUSIC_PRESENT_PROB']*p['MUSIC_FAKE_PROB']
            p['FILE_FAKE_PROB']=1-(1-a)*(1-b)
        elif rule!='max': raise ContractError('Unknown baseline')
        output.append(p)
    return validate_predictions(output,[r['ID'] for r in rows])


def replace_file(baseline: list[dict], file_prob) -> list[dict]:
    if len(baseline)!=len(file_prob): raise ContractError('File prediction length mismatch')
    result=[{**row,'FILE_FAKE_PROB':float(p)} for row,p in zip(baseline,file_prob,strict=True)]
    return validate_predictions(result,[r['ID'] for r in baseline])


def paired_bootstrap(rows, baseline, candidate, repeats=1000, seed=41, min_groups=5):
    groups=np.asarray(connected_groups(rows));unique=np.unique(groups)
    if len(unique)<min_groups:
        return {'status':'NOT_EVALUATED','reason':f'fewer than {min_groups} connected original groups','groups':len(unique)}
    y=np.array([-1 if known_label(r,'FILE_FAKE') is None else known_label(r,'FILE_FAKE') for r in rows])
    a=np.array([r['FILE_FAKE_PROB'] for r in baseline]);b=np.array([r['FILE_FAKE_PROB'] for r in candidate])
    valid=y>=0
    if len(np.unique(y[valid]))!=2: return {'status':'NOT_EVALUATED','reason':'File labels lack two classes'}
    point=_roc_metrics(y[valid].tolist(),b[valid].tolist())[0]-_roc_metrics(y[valid].tolist(),a[valid].tolist())[0]
    index={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(seed);deltas=[];skipped=0
    for _ in range(repeats):
        selected=np.concatenate([index[g] for g in rng.choice(unique,len(unique),replace=True)])
        selected=selected[y[selected]>=0]
        if len(np.unique(y[selected]))!=2: skipped+=1;continue
        deltas.append(_roc_metrics(y[selected].tolist(),b[selected].tolist())[0]-_roc_metrics(y[selected].tolist(),a[selected].tolist())[0])
    ok=len(deltas)>=max(30,int(.5*repeats))
    return {'status':'OK' if ok else 'NOT_EVALUATED','delta_file_eer_candidate_minus_baseline':float(point),
            'paired_group_percentile_95ci':np.quantile(deltas,[.025,.975]).tolist() if ok else None,
            'valid_draws':len(deltas),'class_empty_draws':skipped,'requested_draws':repeats,'groups':len(unique),
            'retraining_in_bootstrap':False,'interpretation':'conditional paired interval on fixed predictions; not model-selection uncertainty'}


def condition_reports(rows, baseline, candidate):
    conditions=defaultdict(list)
    for i,r in enumerate(rows):
        for key in ('category','composition','codec_condition','duration_bin','snr_bin'):
            if r.get(key) is not None: conditions[f'{key}={r[key]}'].append(i)
        for key in ('source','generator_family'):
            value='+'.join(sorted({p[key] for p in r.get('parents',[]) if p.get(key)}))
            if value: conditions[f'{key}={value}'].append(i)
    return {name:{'files':len(idx),'baseline':metrics_report([rows[i] for i in idx],[baseline[i] for i in idx]),'candidate':metrics_report([rows[i] for i in idx],[candidate[i] for i in idx])} for name,idx in sorted(conditions.items())}


def _folds(y,groups,n,seed):
    if n<2 or len(set(groups))<n: raise ContractError('Insufficient independent groups for requested CV folds')
    splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
    return list(splitter.split(np.zeros(len(y)),y,groups))


def select_parameters(kind,grid,x,y,groups,folds,seed,weighting):
    """Every hyperparameter and XGB stopping decision excludes its validation fold."""
    if not grid or len(grid)>4: raise ContractError('Predeclared search allows 1..4 combinations per model')
    scores=[];audit=[]
    for candidate,params in enumerate(grid):
        losses=[];traces=[]
        for fold,(train,valid) in enumerate(folds):
            tr=train[y[train]>=0];ev=valid[y[valid]>=0]
            if not len(ev) or len(np.unique(y[tr]))!=2 or len(np.unique(y[ev]))!=2:
                raise ContractError('Empty/single-class inner fold; candidate NOT_EVALUATED')
            model,trace=fit_estimator(kind,params,x.iloc[tr],y[tr],groups[tr],seed+fold,weighting)
            p=predict_estimator(model,x.iloc[ev]);losses.append((float(log_loss(y[ev],p,labels=[0,1])),len(ev)))
            trace['external_evaluation_groups']=sorted(set(groups[ev]));trace['inner_fold']=fold
            if set(trace['fit_groups'])&set(trace['external_evaluation_groups']): raise ContractError('Inner group leak')
            traces.append(trace)
        score=sum(loss*n for loss,n in losses)/sum(n for _,n in losses)
        scores.append(score);audit.append({'candidate':candidate,'params':params,'inner_logloss':score,'fits':traces})
    chosen=int(np.argmin(scores))
    return dict(grid[chosen]),{'selected_index':chosen,'selection_metric':'inner held-out file logloss','candidates':audit}


def evaluate_dev(rows,records,config):
    """Outer OOF of fixed experts + nested meta tuning, same rows/folds for all 6 systems.
    Never call this on base-head train/selection data. workflow validates provenance.
    """
    if any(r['split']!='meta_dev' for r in rows): raise ContractError('dev command does not accept locked OOD')
    baseline=baseline_predictions(rows,records);noisy=baseline_predictions(rows,records,'noisy_or')
    groups=np.asarray(connected_groups(rows,config.get('group_mode','parent')))
    y=np.array([known_label(r,'FILE_FAKE') if known_label(r,'FILE_FAKE') is not None else -1 for r in rows])
    folds=_folds(y,groups,config['outer_folds'],config['seed'])
    report={'status':'OK','estimand':'Frozen experts on independently acquired meta-dev; outer folds only estimate meta-head performance.',
            'warning':'Choosing the best of the six systems on these outer scores introduces selection optimism. Only locked OOD estimates the final chosen system.',
            'group_mode':config.get('group_mode','parent'),'ids':[r['ID'] for r in rows],
            'baseline':metrics_report(rows,baseline),'noisy_or':metrics_report(rows,noisy),
            'folds':[{'fold':j,'train_ids':[rows[i]['ID'] for i in tr],'eval_ids':[rows[i]['ID'] for i in ev],'train_groups':sorted(set(groups[tr])),'eval_groups':sorted(set(groups[ev]))} for j,(tr,ev) in enumerate(folds)],'systems':{}}
    predictions={'max':baseline,'noisy_or':noisy}
    for features in config['feature_sets']:
        try: x=feature_frame(records,features)
        except ContractError as exc:
            for kind in config['grids']:report['systems'][features+'/'+kind]={'status':'NOT_RUN','reason':str(exc)}
            continue
        for kind,grid in config['grids'].items():
            key=features+'/'+kind;out=np.full(len(rows),np.nan);audits=[]
            try:
                for fold,(train,valid) in enumerate(folds):
                    knowntr=train[y[train]>=0];knownev=valid[y[valid]>=0]
                    if len(np.unique(y[knowntr]))!=2 or len(np.unique(y[knownev]))!=2:
                        raise ContractError('Empty/single-class outer fold; no partial-row comparison permitted')
                    inner=_folds(y[train],groups[train],config['inner_folds'],config['seed']+fold+100)
                    params,selection=select_parameters(kind,grid,x.iloc[train],y[train],groups[train],inner,config['seed']+fold,config['weighting'])
                    model,fit=fit_estimator(kind,params,x.iloc[knowntr],y[knowntr],groups[knowntr],config['seed']+fold,config['weighting'])
                    out[valid]=predict_estimator(model,x.iloc[valid])
                    fit['external_evaluation_groups']=sorted(set(groups[valid]))
                    if set(fit['fit_groups'])&set(fit['external_evaluation_groups']): raise ContractError('Outer group leak')
                    audits.append({'fold':fold,'selection':selection,'final_fold_fit':fit})
                candidate=replace_file(baseline,out);predictions[key]=candidate
                report['systems'][key]={'status':'OK','metrics':metrics_report(rows,candidate),'paired_bootstrap':paired_bootstrap(rows,baseline,candidate,config['bootstrap'],config['seed']), 'conditions':condition_reports(rows,baseline,candidate),'audit':audits}
            except (ContractError,NotRunError) as exc:
                report['systems'][key]={'status':'NOT_RUN' if isinstance(exc,NotRunError) else 'NOT_EVALUATED','reason':str(exc),'completed_fold_audits':audits}
    return report,predictions


def fit_final_dev(rows,records,kind,features,config):
    """Tune within dev, then fit known File labels. No OOD argument exists here."""
    if any(r['split']!='meta_dev' for r in rows): raise ContractError('Final fit cannot access OOD')
    x=feature_frame(records,features);groups=np.asarray(connected_groups(rows,config.get('group_mode','parent')))
    y=np.array([known_label(r,'FILE_FAKE') if known_label(r,'FILE_FAKE') is not None else -1 for r in rows])
    folds=_folds(y,groups,config['inner_folds'],config['seed'])
    params,selection=select_parameters(kind,config['grids'][kind],x,y,groups,folds,config['seed'],config['weighting'])
    keep=y>=0
    model,fit=fit_estimator(kind,params,x.loc[keep],y[keep],groups[keep],config['seed'],config['weighting'])
    return model,{'selection':selection,'fit':fit,'excluded_unknown_file_labels':int((~keep).sum())}
