"""Integration tests: source fidelity, measured cache identity, and File-only opt-in.
Stub encoders below test wiring ONLY; they do not establish GPU/audio parity.
"""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from dacon_stacking.common import digest, sha256
from dacon_stacking.features import FixtureAdapter, read_extended, write_extended, SCHEMA
from dacon_stacking import _snapshot_math as math

ROOT=Path(os.environ.get('STACKING_INPUT_ROOT','/mnt/data/stacking_input'))
RETURN=Path(__file__).resolve().parents[1]


def test_numerical_symbols_remain_verbatim_current_not_legacy():
    copied={n.name:n for n in ast.parse((RETURN/'src/dacon_stacking/_snapshot_math.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    for item in json.loads((RETURN/'configs/snapshot_symbols.json').read_text()):
        source=ROOT/item['source']
        assert sha256(source)==item['sha256']
        original={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        assert ast.dump(copied[item['symbol']])==ast.dump(original[item['symbol']])
    assert sha256(RETURN/'src/dacon_stacking/_snapshot_scoring.py')==sha256(ROOT/'current/submission_source/model/runtime/graph_scoring.py')


def test_extended_cache_binds_provenance_and_rejects_tampering(tmp_path):
    record=FixtureAdapter(ROOT).replay()[0]
    provenance={'audio_sha256':'a'*64,'code_sha256':{'runtime':'b'*64},'checkpoint_sha256':{'head':'c'*64},'timebase':math.window_layout(70000).tolist(),'preprocessing':{'gain':'cap_0.6'},'base_exposure':{'upstream':'unknown'},'numeric_mode':'cpu_fp32','feature_schema':SCHEMA}
    p=tmp_path/'features.npz';receipt=write_extended(p,record,provenance)
    reread=read_extended(p,receipt,digest(provenance))
    assert reread['features']==record['features']
    with pytest.raises(ValueError,match='identity'):
        read_extended(p,receipt,'e'*64)
    with p.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        read_extended(p,receipt,digest(provenance))


def _class_from_file(path):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
    env={'torch':torch,'np':np,'Path':Path,'tempfile':__import__('tempfile'),'window_layout':math.window_layout,'windows':math.windows,'PREDICTION_COLUMNS':__import__('dacon_stacking._snapshot_scoring',fromlist=['PREDICTION_COLUMNS']).PREDICTION_COLUMNS}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['Engine']


def _fake_engine(cls):
    e=cls.__new__(cls);e.device=torch.device('cpu');e.batch=8;e.fake_index=0;e.vi=[0];e.mi=[1];e.demucs=None
    raw=np.sin(np.arange(100000,dtype=np.float32)/100)*.8
    class Panns:
        def inference(self,x):
            p=np.full((1,527),.01,np.float32);p[0,0]=.8;p[0,1]=.5
            return p,np.zeros((1,2048),np.float32)
    e.panns=Panns()
    class DF:
        def __call__(self,**kwargs):
            e.embedding=torch.ones((1,1280),dtype=torch.float32)*.1
            return {'logits':torch.tensor([[.2,-.2]])}
    e.df=DF()
    def presence(model,vi,mi,segment):
        p,_=model.inference(segment);return float(p[:,vi].max()),float(p[:,mi].max())
    e.base=SimpleNamespace(load_audio=lambda path:raw.copy(),separate_voice_and_music=lambda *args:(raw*.2,raw*.7),get_segment_starts=math.get_segment_starts,extract_segment=math.extract_segment,predict_presence=presence,calculate_rms=math.calculate_rms,SILENCE_RMS=1e-5)
    e.processor=lambda audio,**kw:SimpleNamespace(input_values=torch.zeros((len(audio),4)))
    e.mert=lambda values,**kw:SimpleNamespace(hidden_states=[None]*12+[torch.ones((len(values),2,1024))*.1])
    music,voice,_=math.load_heads(ROOT/'fixtures/heads','cpu');e.music_heads=music;e.voice_heads=voice
    return e


def test_opt_in_telemetry_leaves_baseline_and_four_outputs_unchanged():
    from dacon_stacking.gpu import record_from_engine
    old=_fake_engine(_class_from_file(ROOT/'current/submission_source/model/runtime/full_bank_submission.py'))
    patched=_fake_engine(_class_from_file(RETURN/'patches/full_bank_submission.py'))
    expected=old.predict(Path('synthetic.wav'))
    assert patched.predict(Path('synthetic.wav'))==expected
    record=record_from_engine(patched,Path('synthetic.wav'),'synthetic',preserve_gain=False,raw_comparison=False)
    assert record['baseline']==expected
    assert 'voice_aligned_mean' in record['features']
    assert len(record['features'])>32


def test_six_matched_cv_systems_keep_same_outer_rows_and_four_outputs():
    from dacon_stacking.demo import synthetic_data, smoke_config
    from dacon_stacking.evaluation import evaluate_dev
    rows,records=synthetic_data(36)
    report,predictions=evaluate_dev(rows,records,smoke_config())
    assert len(report['systems'])==6
    for name,result in report['systems'].items():
        assert result['status']=='OK',(name,result.get('reason'))
        assert len(predictions[name])==len(rows)
        for base,candidate in zip(predictions['max'],predictions[name],strict=True):
            assert all(base[n]==candidate[n] for n in ('VOICE_FAKE_PROB','MUSIC_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB'))
        for outer in result['audit']:
            evaluation=set(outer['final_fold_fit']['external_evaluation_groups'])
            for candidate in outer['selection']['candidates']:
                for fit in candidate['fits']:
                    assert not evaluation.intersection(fit['fit_groups'])
                    assert not evaluation.intersection(fit.get('early_stop_eval_groups',[]))
