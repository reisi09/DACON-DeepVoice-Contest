"""Small probability models; fold-local fitting; no pickle loading fallback."""
from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from .common import ContractError, NotRunError, check_hash, read_json, sha256, write_json, preprocessing_policy
from .features import strict_frame


def parent_weights(groups):
    groups=np.asarray(groups);_,indices,counts=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1/counts[indices];return weights/weights.mean()


def _binary(y):
    y=np.asarray(y)
    if y.ndim!=1 or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ContractError('Training requires two classes with known binary labels')
    return y.astype(int)


def _early_split(y, groups, seed):
    # This validation is carved solely from the CURRENT fitting partition.
    n=min(3,len(np.unique(groups)))
    if n<2: raise ContractError('Insufficient independent groups for internal early stopping')
    splitter=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
    for tr,ev in splitter.split(np.zeros(len(y)),y,groups):
        if len(np.unique(y[tr]))==2 and len(np.unique(y[ev]))==2:
            return tr,ev
    raise ContractError('Internal early-stop folds lack two classes; NOT_EVALUATED')


def fit_estimator(kind: str, params: dict, x: pd.DataFrame, y, groups, seed=41, weighting='parent_equal'):
    order=list(x.columns); x=strict_frame(x,order); y=_binary(y);groups=np.asarray(groups,dtype=str)
    if len(y)!=len(x) or len(groups)!=len(x): raise ContractError('Training rows misaligned')
    if weighting not in ('parent_equal','file_equal'): raise ContractError('Unknown weighting policy')
    w=parent_weights(groups) if weighting=='parent_equal' else np.ones(len(y))
    audit={'fit_groups':sorted(set(groups)),'train_rows':len(y),'weighting':weighting,'class_weight':None,'loss':'binary logloss/BCE; tree split criterion log_loss', 'preprocessing_fit':'current fitting partition only','calibration':'none','imputation':'none','feature_selection':'none','params':dict(params)}
    if kind=='lr':
        estimator=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=float(params['C']),solver='lbfgs',max_iter=2000,random_state=seed))])
        estimator.fit(x,y,model__sample_weight=w)
    elif kind=='et':
        kwargs=dict(n_estimators=128,max_depth=3,min_samples_leaf=12,max_features=1.,bootstrap=True,oob_score=False,criterion='log_loss',n_jobs=1,random_state=seed)
        kwargs.update(params)
        if kwargs['max_depth'] is None or kwargs['max_depth']>6 or kwargs['min_samples_leaf']<2 or not kwargs['bootstrap']:
            raise ContractError('ExtraTrees must be shallow, regularized, bagged')
        estimator=ExtraTreesClassifier(**kwargs).fit(x,y,sample_weight=w)
    elif kind=='xgb':
        try: from xgboost import XGBClassifier
        except ImportError as exc: raise NotRunError('XGBoost NOT_RUN: install requirements-xgb.txt') from exc
        kwargs=dict(n_estimators=200,max_depth=2,min_child_weight=10,reg_lambda=10.,reg_alpha=.1,learning_rate=.05,subsample=.8,colsample_bytree=1.,tree_method='hist',device='cpu',objective='binary:logistic',eval_metric='logloss',n_jobs=1,random_state=seed)
        kwargs.update(params)
        if kwargs['max_depth']>4 or kwargs['max_depth']<1: raise ContractError('XGBoost depth must be 1..4')
        tr,ev=_early_split(y,groups,seed)
        trainw=parent_weights(groups[tr]) if weighting=='parent_equal' else np.ones(len(tr))
        # Validation logloss is unweighted: natural file prevalence is reported.
        probe=XGBClassifier(**kwargs,early_stopping_rounds=15)
        probe.fit(x.iloc[tr],y[tr],sample_weight=trainw,eval_set=[(x.iloc[ev],y[ev])],verbose=False)
        rounds=int(probe.best_iteration)+1
        kwargs['n_estimators']=rounds
        estimator=XGBClassifier(**kwargs).fit(x,y,sample_weight=w,verbose=False)
        audit.update(early_stop_train_groups=sorted(set(groups[tr])),early_stop_eval_groups=sorted(set(groups[ev])),selected_rounds=rounds,refit='all current training rows with frozen number of boosting rounds')
    else: raise ContractError(f'Unknown model, no silent substitution: {kind}')
    return {'kind':kind,'order':order,'estimator':estimator},audit


def predict_estimator(model: dict, x: pd.DataFrame) -> np.ndarray:
    x=strict_frame(x,model['order'])
    p=np.asarray(model['estimator'].predict_proba(x)[:,1],np.float64)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ContractError('Invalid model probabilities')
    return p


def _versions(kind):
    result={k:version(k) for k in ('numpy','scipy','scikit-learn')}
    if kind=='xgb':
        try: result['xgboost']=version('xgboost')
        except PackageNotFoundError as exc: raise NotRunError('XGBoost NOT_RUN: install requirements-xgb.txt') from exc
    else:
        try: result['skops']=version('skops')
        except Exception as exc: raise NotRunError('skops unavailable: safe LR/ExtraTrees serialization NOT_RUN; no pickle fallback') from exc
    return result


def save_bundle(directory: str | Path, model: dict, provenance: dict) -> str:
    """Return manifest digest to be pinned OUTSIDE the mutable model directory."""
    if provenance.get('purpose')=='production_candidate':
        preprocessing_policy(provenance.get('preprocessing_policy'))
    versions=_versions(model['kind'])
    if list(model['estimator'].feature_names_in_) != model['order']:
        raise ContractError('Estimator/manifest feature order mismatch')
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()): raise ContractError('Refuse to overwrite an existing model bundle')
    directory.mkdir(parents=True,exist_ok=True)
    if model['kind']=='xgb':
        filename='model.json';model['estimator'].save_model(directory/filename)
    else:
        try: import skops.io as sio
        except ImportError as exc: raise NotRunError('skops unavailable: safe LR/ExtraTrees serialization NOT_RUN; no pickle fallback') from exc
        filename='model.skops';sio.dump(model['estimator'],directory/filename)
        untrusted=sio.get_untrusted_types(file=directory/filename)
        if untrusted:
            (directory/filename).unlink()
            raise ContractError(f'Unreviewed skops types forbidden: {untrusted}')
    manifest={'schema':'dacon.stacking.bundle.v1','kind':model['kind'],'feature_order':model['order'],'file':filename,'sha256':sha256(directory/filename),'versions':versions,'provenance':provenance,'serialization':'XGBoost JSON' if model['kind']=='xgb' else 'skops; no extra trusted types'}
    write_json(directory/'manifest.json',manifest)
    return sha256(directory/'manifest.json')


def load_bundle(directory: str | Path, expected_manifest_sha256: str) -> dict:
    directory=Path(directory);check_hash(directory/'manifest.json',expected_manifest_sha256);meta=read_json(directory/'manifest.json')
    if meta['schema']!='dacon.stacking.bundle.v1': raise ContractError('Unknown bundle schema')
    if meta['provenance'].get('purpose')=='production_candidate':
        preprocessing_policy(meta['provenance'].get('preprocessing_policy'))
    kind=meta['kind'];expected_name='model.json' if kind=='xgb' else 'model.skops'
    if kind not in ('lr','et','xgb') or meta['file']!=expected_name: raise ContractError('Unexpected model format')
    check_hash(directory/expected_name,meta['sha256'])
    if _versions(kind)!=meta['versions']: raise ContractError('Runtime library versions differ from tested training bundle')
    if kind=='xgb':
        try: from xgboost import XGBClassifier
        except ImportError as exc: raise NotRunError('XGBoost NOT_RUN: install requirements-xgb.txt') from exc
        estimator=XGBClassifier();estimator.load_model(directory/expected_name)
    else:
        try: import skops.io as sio
        except ImportError as exc: raise NotRunError('skops unavailable: safe LR/ExtraTrees serialization NOT_RUN; no pickle fallback') from exc
        if sio.get_untrusted_types(file=directory/expected_name): raise ContractError('Unreviewed skops object types')
        estimator=sio.load(directory/expected_name,trusted=[])
    if list(estimator.feature_names_in_) != meta['feature_order']:
        raise ContractError('Stored model/manifest feature order mismatch')
    return {'kind':kind,'order':meta['feature_order'],'estimator':estimator,'provenance':meta['provenance']}
