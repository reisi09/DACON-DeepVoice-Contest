"""Independent calibration and a predeclared, conservative seed gate."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from six_h100_experiments import fuse, metrics


def calibrate(rows,predictions,records):
    calibration=[r for r in rows if r['split']=='calibration' and r['variant']=='base']
    coefficients={}
    for name,column,index,present in (('DF','VOICE_FAKE',1,'VOICE_PRESENT'),
        ('Anti','VOICE_FAKE',1,'VOICE_PRESENT'),('Music','MUSIC_FAKE',2,'MUSIC_PRESENT')):
        eligible=[r for r in calibration if r[column] in ('0','1') and r[present]=='1']
        predictor=predictions['Anti_stem' if name=='Anti' else 'M330']
        x=np.asarray([logit(np.clip(predictor[r['ID']][index],1e-6,1-1e-6)) for r in eligible])[:,None]
        y=np.asarray([int(r[column]) for r in eligible])
        if set(y)!={0,1}:
            raise ValueError('Calibration requires both classes')
        model=LogisticRegression(C=1.,random_state=41).fit(x,y)
        a,b=float(model.coef_[0,0]),float(model.intercept_[0])
        accepted=a>0
        coefficients[name]=dict(slope=a if accepted else 1.,intercept=b if accepted else 0.,
            positive_slope_fit_accepted=accepted,ids=[r['ID'] for r in eligible])
    def apply(name,p):
        fit=coefficients[name]
        return float(expit(fit['slope']*logit(np.clip(p,1e-6,1-1e-6))+fit['intercept']))
    for mode in ('max','noisy_or'):
        predictions['calibrated_'+mode]={}
        for row in rows:
            key=row['ID']; value=list(predictions['M330'][key])
            value[1]=.5*(apply('DF',value[1])+apply('Anti',predictions['Anti_stem'][key][1]))
            value[2]=apply('Music',value[2])
            predictions['calibrated_'+mode][key]=fuse(value,records[key],mode)
    return dict(fitted_split='calibration',evaluated_split='dev',coefficients=coefficients,
        test_statistics_used=False,parameters_frozen=True,
        limitation='One calibration voice speaker; monotonic Platt cannot improve each branch EER by itself')


def seed_gate(rows,predictions):
    checks={}; eligible=True; gain=0.
    for variant in ('base','4s','telephone','mp3_32k'):
        panel=[r for r in rows if r['split']=='dev' and r['variant']==variant and r['kind']=='music']
        before=metrics(panel,predictions['music_control'])['metrics']
        after=metrics(panel,predictions['music_new'])['metrics']
        checks[variant]={}
        for key,weight in (('file_eer',.45),('music_eer',.27)):
            old,new=before[key],after[key]
            passed=old is not None and new is not None and new<=old+1e-12
            checks[variant][key]=dict(before=old,after=new,no_regression=passed)
            eligible &= passed
            if variant=='base' and old is not None and new is not None:
                gain+=weight*(old-new)
    selected=bool(eligible and gain>1e-12)
    return dict(additional_seeds=selected,weighted_base_gain=gain,checks=checks,
        selection_split='dev',test_used=False,seed_epochs={'41':17,'42':17,'43':17},
        tie_policy='keep control; no further compute',promotion_allowed=False,
        reason='Diagnostic panel below the independent-source/label-review deployment gate')
