"""Guards prevent leakage, evaluation-statistic fitting, and accidental deployment."""
import copy
import numpy as np
import pandas as pd
import pytest


def row(i, parent, y=0, split='meta_dev'):
    return {'ID':str(i),'split':split,'category':'Voice-only','parents':[{'parent_id':parent,'audio_sha256':str(i).zfill(64),'speaker':'sp'+parent,'track':'','source':'new_source','generator_family':'new_fake' if y else 'real'}],
            'audio_sha256':str(i).zfill(64),'base_exposure':{'head_train':'no','head_selection':'no','historical_development':'no','upstream':'unknown','evidence':['signed-acquisition-v1'],'reviewed_parent_ids':[parent]},
            'FILE_FAKE':y,'FILE_FAKE_mask':1,'VOICE_FAKE':y,'VOICE_FAKE_mask':1,'VOICE_PRESENT':1,'VOICE_PRESENT_mask':1,'MUSIC_FAKE':None,'MUSIC_FAKE_mask':0,'MUSIC_PRESENT':0,'MUSIC_PRESENT_mask':1}


def test_parent_bridges_and_training_exposure_are_refused():
    from dacon_stacking.provenance import connected_groups, validate_manifest
    a,b,c=row(1,'A'),row(2,'B'),row(3,'C')
    c['category']='Mixed';c['parents']=[a['parents'][0],b['parents'][0]];c['composition']='RR';c['base_exposure']['reviewed_parent_ids']=['A','B'];c['MUSIC_PRESENT']=1;c['MUSIC_FAKE']=0;c['MUSIC_FAKE_mask']=1
    assert len(set(connected_groups([a,b,c])))==1
    c['split']='locked_ood'
    with pytest.raises(ValueError,match='crosses split'):
        validate_manifest([a,b,c],registry=set())
    a['base_exposure']['head_selection']='yes'
    with pytest.raises(ValueError,match='exposure'):
        validate_manifest([a],registry=set())
    a=row(1,'A')
    with pytest.raises(ValueError,match='registry'):
        validate_manifest([a],registry={'parent:A'})


def test_masked_unknown_is_not_real_and_total_stays_null():
    from dacon_stacking.evaluation import metrics_report, known_label
    assert known_label({'FILE_FAKE':0,'FILE_FAKE_mask':0},'FILE_FAKE') is None
    assert known_label({'FILE_FAKE':'','FILE_FAKE_mask':1},'FILE_FAKE') is None
    rows=[row(1,'A',0),row(2,'B',1),row(3,'C',0)]
    rows[2]['FILE_FAKE_mask']=0
    ps=[{'ID':r['ID'],'FILE_FAKE_PROB':p,'VOICE_FAKE_PROB':p,'MUSIC_FAKE_PROB':.2,'VOICE_PRESENT_PROB':.8,'MUSIC_PRESENT_PROB':.1} for r,p in zip(rows,[.1,.9,.8])]
    report=metrics_report(rows,ps)
    assert report['official']['counts']['file_eer']==2
    assert report['official']['score'] is None
    assert report['probability']['FILE_FAKE_PROB']['n']==2


def test_fold_preprocessing_and_estimator_use_training_only():
    from dacon_stacking.models import fit_estimator, predict_estimator
    x=pd.DataFrame({'Pv':np.arange(24,dtype=float)/24,'Pm':np.ones(24)*.2,'Fv':np.arange(24)%2,'Fm':np.ones(24)*.3})
    y=np.arange(24)%2;groups=np.array([str(i//2) for i in range(24)])
    model,audit=fit_estimator('lr',{'C':1.},x,y,groups,seed=41)
    np.testing.assert_allclose(model['estimator'].named_steps['scale'].mean_,x.mean().to_numpy())
    before=model['estimator'].named_steps['scale'].mean_.copy()
    extreme=x.copy();extreme['Pv']=10000
    predict_estimator(model,extreme)
    np.testing.assert_array_equal(model['estimator'].named_steps['scale'].mean_,before)
    assert not (set(audit['fit_groups']) & set(audit.get('external_evaluation_groups',[])))
    with pytest.raises(ValueError,match='two classes'):
        fit_estimator('lr',{'C':1.},x,np.zeros(24),groups,seed=41)


def test_xgb_json_reload_order_and_single_file_independence(tmp_path):
    from dacon_stacking.models import fit_estimator, predict_estimator, save_bundle, load_bundle
    x=pd.DataFrame(np.random.default_rng(2).uniform(size=(64,4)),columns=['Pv','Pm','Fv','Fm'])
    y=np.arange(64)%2;g=np.array([str(i//2) for i in range(64)])
    model,audit=fit_estimator('xgb',{'max_depth':2,'n_estimators':12,'min_child_weight':1},x,y,g,seed=41)
    p=predict_estimator(model,x)
    receipt=save_bundle(tmp_path/'bundle',model,{'purpose':'synthetic_wiring_only','audit':audit})
    loaded=load_bundle(tmp_path/'bundle',receipt)
    np.testing.assert_allclose(p,predict_estimator(loaded,x),atol=1e-12,rtol=0)
    np.testing.assert_allclose(p[[5,2]],predict_estimator(loaded,x.iloc[[5,2]]),atol=0,rtol=0)
    np.testing.assert_allclose(p[[5]],predict_estimator(loaded,x.iloc[[5]]),atol=0,rtol=0)
    with pytest.raises(ValueError,match='order'):
        predict_estimator(loaded,x.iloc[:,::-1])
    with pytest.raises(ValueError,match='SHA'):
        load_bundle(tmp_path/'bundle','0'*64)
    assert set(audit['early_stop_train_groups']).isdisjoint(audit['early_stop_eval_groups'])


def test_ood_requires_immutable_production_lock(tmp_path):
    from dacon_stacking.workflow import verify_ood_lock
    with pytest.raises(ValueError,match='lock'):
        verify_ood_lock(tmp_path/'missing.json','a'*64,'b'*64)


def test_manifest_requires_parent_scope_and_consistent_component_truth():
    from dacon_stacking.provenance import validate_manifest
    a=row(1,'A')
    a['base_exposure']['reviewed_parent_ids']=[]
    with pytest.raises(ValueError,match='parent'):
        validate_manifest([a],registry=set())
    a['base_exposure']['reviewed_parent_ids']=['A']
    a['FILE_FAKE']=1;a['VOICE_FAKE']=0
    with pytest.raises(ValueError,match='label'):
        validate_manifest([a],registry=set())


def test_paired_bootstrap_skips_one_class_draws_without_imputing():
    from dacon_stacking.evaluation import paired_bootstrap
    rows=[row(i,str(i),int(i==0)) for i in range(6)]
    base=[{'ID':r['ID'],'FILE_FAKE_PROB':.7 if i==0 else .2} for i,r in enumerate(rows)]
    result=paired_bootstrap(rows,base,base,repeats=200,seed=41)
    assert result['class_empty_draws']>0
    assert result['valid_draws']+result['class_empty_draws']==200
    assert result['paired_group_percentile_95ci']==[0.,0.]
