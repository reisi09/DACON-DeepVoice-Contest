"""Explicit synthetic wiring fixture. Not acquired audio, OOF evidence, or a model recommendation."""
from __future__ import annotations
import numpy as np
from .common import write_json
from .features import from_measurements
from .evaluation import evaluate_dev, fit_final_dev
from .models import save_bundle, load_bundle, predict_estimator
from .features import feature_frame


def synthetic_data(groups=36, seed=907):
    rng=np.random.default_rng(seed);rows=[];records=[]
    for group in range(groups):
        for augmentation in range(2):
            y=int(group%2);identifier=f'SYNTHETIC_g{group}_a{augmentation}'
            pv,pm=rng.uniform(.05,.99,2);fv,fm=rng.uniform(.02,.98,2)
            baseline={'FILE_FAKE_PROB':max(pv*fv,pm*fm),'VOICE_FAKE_PROB':fv,'MUSIC_FAKE_PROB':fm,'VOICE_PRESENT_PROB':pv,'MUSIC_PRESENT_PROB':pm}
            measurements={'voice_seed':np.clip(fv+rng.normal(0,.03,3),0,1),'music_seed':np.clip(fm+rng.normal(0,.03,3),0,1),'voice_window':rng.uniform(.01,.99,(3,2)),'music_window':rng.uniform(.01,.99,(3,3)),'df_original':rng.uniform(.01,.99)}
            parent={'parent_id':f'SYNTHETIC_parent{group}','audio_sha256':f'{group+1:064x}','speaker':f'SYNTHETIC_speaker{group}','track':f'SYNTHETIC_track{group}','source':'SYNTHETIC_source'+str(group%3),'generator_family':'SYNTHETIC_fake' if y else 'real'}
            rows.append({'ID':identifier,'split':'meta_dev','category':'Mixed','composition':'FF' if y else 'RR','parents':[parent],
                         'audio_sha256':f'{10000+group*2+augmentation:064x}','synthetic':True,'base_exposure':{'head_train':'no','head_selection':'no','historical_development':'no','upstream':'unknown','evidence':['SYNTHETIC WIRING ONLY']},
                         'FILE_FAKE':y,'FILE_FAKE_mask':1,'VOICE_FAKE':y,'VOICE_FAKE_mask':1,'MUSIC_FAKE':y,'MUSIC_FAKE_mask':1,'VOICE_PRESENT':1,'VOICE_PRESENT_mask':1,'MUSIC_PRESENT':1,'MUSIC_PRESENT_mask':1})
            records.append({'ID':identifier,'features':from_measurements(baseline,measurements),'baseline':baseline})
    return rows,records


def smoke_config():
    return {'seed':41,'outer_folds':3,'inner_folds':2,'bootstrap':40,'weighting':'parent_equal','group_mode':'parent','feature_sets':['F0','F1'],
            'grids':{'lr':[{'C':.1},{'C':1.}], 'et':[{'max_depth':3,'min_samples_leaf':4,'n_estimators':24}],
                     'xgb':[{'max_depth':2,'n_estimators':16,'min_child_weight':1}]},'purpose':'synthetic_wiring_only; reduced runtime grid'}


def run_smoke(output):
    from pathlib import Path
    from .common import NotRunError
    rows,records=synthetic_data();config=smoke_config();report,predictions=evaluate_dev(rows,records,config)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report['purpose']='SYNTHETIC WIRING ONLY. Metrics here are NOT evidence of audio detection performance.'
    write_json(output/'synthetic_cv.json',report)
    serial={}
    for kind in ('lr','et','xgb'):
        model,audit=fit_final_dev(rows,records,kind,'F1',config)
        try:
            receipt=save_bundle(output/('synthetic_'+kind),model,{'purpose':'synthetic_wiring_only','audit':audit})
            loaded=load_bundle(output/('synthetic_'+kind),receipt);x=feature_frame(records,'F1')
            error=float(np.max(np.abs(predict_estimator(model,x)-predict_estimator(loaded,x))))
            serial[kind]={'status':'PASS','max_abs_roundtrip_error':error,'manifest_sha256':receipt}
        except NotRunError as exc:serial[kind]={'status':'NOT_RUN','reason':str(exc)}
    write_json(output/'serialization_results.json',serial)
    return {'purpose':report['purpose'],'systems':{k:v['status'] for k,v in report['systems'].items()},'serialization':serial}
