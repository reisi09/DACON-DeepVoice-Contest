"""Regression: stage 2 decodes real prepared WAVs without imageio_ffmpeg.

Only the GPU encoder is replaced; this check measures decoder wiring, not model quality.
Run: .venv/bin/python check_voice_music_decoder.py
"""
import importlib.abc
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import librosa
import numpy as np

import voice_music_h100_experiments as runtime


def check_resume():
    from redesign_notebook_runtime import export_results
    from redesign_h100_run import sha256
    import redesign_features_runtime
    source=Path('_workspace/voice_music_h100_20260917/source')
    saved=Path('results/voice_music_h100_20260917/downloaded/unpacked/output')
    row=next(csv.DictReader((source/'probes.csv').open())); key=row['ID']
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary); old=root/'old'; output=old/'output'; (output/'features').mkdir(parents=True)
        (old/'separated').mkdir()
        audio=librosa.load(source/row['path'],sr=16000,mono=True,dtype=np.float32)[0]
        # The decoder fixture tests cache I/O only; separator quality is checked on H100.
        np.savez(old/'separated'/(key+'.npz'),vocal=audio,music=audio)
        shutil.copy2(saved/'features'/(key+'.npz'),output/'features'/(key+'.npz'))
        for name in ('config.json','experiment_1.json','df_raw_probabilities.json'):
            shutil.copy2(saved/name,output/name)
        prediction=json.loads((saved/'predictions_M330.json').read_text())[key]
        (output/'predictions_M330.json').write_text(json.dumps({key:prediction}))
        export_results((output,),old/'results.zip')
        run=runtime.Experiments.__new__(runtime.Experiments)
        run.root=root/'retry';run.out=run.root/'output';run.out.mkdir(parents=True)
        run.cfg=dict(resume_root=str(old),resume_archive_sha256=sha256(old/'results.zip'),previous=str(old),
            manifest_sha256=json.loads((output/'config.json').read_text())['manifest_sha256'])
        run.rows=[row];run.records={};run.predictions={};run.mark=lambda *args,**kwargs:None
        with patch.object(redesign_features_runtime,'_load_baseline',return_value=SimpleNamespace(load_audio=lambda p:audio)):
            run.resume_experiment_1()
        assert run.predictions['M330'][key]==prediction
        with np.load(output/'features'/(key+'.npz')) as before:
            for name in before.files:
                assert np.array_equal(run.records[key][name],before[name]), 'Resumption changed model features'
        assert (run.root/'separated'/(key+'.npz')).read_bytes()==(old/'separated'/(key+'.npz')).read_bytes()
        with (old/'results.zip').open('ab') as f:f.write(b'corruption')
        try:run.resume_experiment_1()
        except ValueError as error:assert 'SHA-256' in str(error)
        else:raise AssertionError('Corrupted cache archive was accepted')


class WithoutImageio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'imageio_ffmpeg':
            raise ModuleNotFoundError("No module named 'imageio_ffmpeg'", name=fullname)
        return None


def main():
    check_resume()
    source=Path('_workspace/voice_music_h100_20260917/source')
    audio_path=next((source/'audio').glob('*_base.wav'))
    expected=librosa.load(audio_path,sr=16000,mono=True,dtype=np.float32)[0]
    calls=[]
    class EncoderProbe:
        def __init__(self,*args): pass
        def predict(self,audio,*args):
            assert np.array_equal(audio,expected), 'Prepared raw audio changed during decoding'
            calls.append(len(audio))
            return [.25,.25,.25]
    sys.modules.pop('v2_data',None)
    sys.modules.pop('imageio_ffmpeg',None)
    finder=WithoutImageio();sys.meta_path.insert(0,finder)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            run=runtime.Experiments.__new__(runtime.Experiments)
            run.root=Path(temporary);run.source=source;run.deadline=float('inf')
            run.rows=[dict(ID=audio_path.stem,path=str(audio_path),kind='speech')]
            (run.root/'separated').mkdir()
            np.savez(run.root/'separated'/(audio_path.stem+'.npz'),vocal=expected)
            run.records={audio_path.stem:{}}
            run.predictions={'M330':{audio_path.stem:[.2,.3,.4,.5,.6]}}
            run.load_audio=lambda path:librosa.load(path,sr=16000,mono=True,dtype=np.float32)[0]
            run.mark=lambda *args,**kwargs:None
            run.save=lambda *args:None
            run.reports=lambda *args:{}
            with patch.object(runtime,'AntiDeepfake',EncoderProbe), patch.object(runtime,'verify_native',return_value={}):
                run.experiment_2()
            assert len(calls)==2
            for name in ('Anti_stem','DF_Anti_mean','Anti_top30'):
                assert run.predictions[name][audio_path.stem][2:]==[.4,.5,.6]
    finally:
        sys.meta_path.remove(finder)
    print(json.dumps(dict(stage_2_without_imageio_ffmpeg=True,raw_waveform_preserved=True,
        music_and_presence_unchanged=True,resumed_features_and_predictions_identical=True,
        corrupted_resume_archive_rejected=True,gpu_model_quality_checked_here=False)))


if __name__=='__main__':
    main()
