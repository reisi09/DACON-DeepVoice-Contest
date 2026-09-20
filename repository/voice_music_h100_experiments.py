"""Six bounded follow-up stages; old held-out voice data is diagnostic only."""
from pathlib import Path
import csv
import gc
import json
import shutil
import time

import numpy as np
import torch

from antideepfake_runtime import AntiDeepfake, verify_native
from redesign_features import load_records
from redesign_features_runtime import FrozenEncoders
from redesign_fit import fit_music
from redesign_h100_run import atomic_json, sha256
from redesign_submission import score_ensemble
from six_h100_experiments import fuse, metrics


class Experiments:
    def __init__(self, config_path: str):
        self.cfg=json.loads(Path(config_path).read_text())
        self.root=Path(self.cfg['root']); self.source=self.root/'source'; self.out=self.root/'output'
        self.out.mkdir(parents=True,exist_ok=True)
        self.deadline=self.cfg['pause_deadline_epoch']-600
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Authorized H100 required')
        if not 0<self.cfg['pause_deadline_epoch']-self.cfg['started_epoch']<=3600 or time.time()>=self.deadline:
            raise ValueError('Expired or invalid one-hour allocation')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        self.rows=list(csv.DictReader((self.source/'probes.csv').open()))
        if sha256(self.source/'probes.csv')!=self.cfg['manifest_sha256']:
            raise ValueError('Probe manifest changed')
        self.records={}; self.predictions={}; self.heads={}
        for row in self.rows:
            row['path']=str(self.source/row['path'])
            if sha256(Path(row['path']))!=row['audio_sha256']:
                raise ValueError('Input audio checksum changed')
        self.save('data_receipt',json.loads((self.source/'data_receipt.json').read_text()))
        self.mark('preflight',gpu=torch.cuda.get_device_name(0),rows=len(self.rows))

    def mark(self,stage,**extra):
        value=dict(stage=stage,time=time.time(),**extra)
        atomic_json(self.out/'progress.json',value)
        atomic_json(Path(self.cfg['shared'])/'progress.json',value)
        print(json.dumps(value),flush=True)

    def save(self,name,value):
        atomic_json(self.out/(name+'.json'),value)
        return value

    def subset(self,split='dev',variant='base'):
        return [r for r in self.rows if r['split']==split and r['variant']==variant]

    def reports(self,names,split='dev'):
        return {variant:{name:metrics(self.subset(split,variant),self.predictions[name]) for name in names}
                for variant in sorted({r['variant'] for r in self.rows}) if self.subset(split,variant)}

    def predict_music(self,name,heads):
        self.heads[name]=heads
        self.predictions[name]={r['ID']:score_ensemble(heads,self.records[r['ID']],'cuda') for r in self.rows}
        self.save('predictions_'+name,self.predictions[name])

    def resume_experiment_1(self):
        from redesign_features_runtime import _load_baseline
        from redesign_h100_watch import verify_archive
        previous=Path(self.cfg['resume_root'])
        restored=self.root/'resumed_original'
        receipt=verify_archive(previous/'results.zip',self.cfg['resume_archive_sha256'],restored)
        original=restored/'output'
        old_config=json.loads((original/'config.json').read_text())
        if old_config['manifest_sha256']!=self.cfg['manifest_sha256']:
            raise ValueError('Cannot reuse features for a different input manifest')
        expected={row['ID'] for row in self.rows}
        if {p.stem for p in (original/'features').glob('*.npz')}!=expected:
            raise ValueError('Resumed features do not cover exactly the requested audio')
        stem_hashes={}
        for row in self.rows:
            key=row['ID']; path=previous/'separated'/(key+'.npz')
            with np.load(path,allow_pickle=False) as stems:
                for name in ('vocal','music'):
                    value=stems[name]
                    if value.ndim!=1 or not np.isfinite(value).all() or abs(len(value)-round(float(row['duration_seconds'])*16000))>2:
                        raise ValueError('Invalid preserved separator output: '+key)
            stem_hashes[key]=sha256(path)
            with np.load(original/'features'/(key+'.npz'),allow_pickle=False) as data:
                self.records[key]={name:data[name].copy() for name in data.files}
        self.predictions['M330']=json.loads((original/'predictions_M330.json').read_text())
        if set(self.predictions['M330'])!=expected:
            raise ValueError('Resumed baseline predictions do not match the input IDs')
        shutil.copytree(original/'features',self.out/'features')
        for name in ('experiment_1.json','predictions_M330.json','df_raw_probabilities.json'):
            shutil.copy2(original/name,self.out/name)
        (self.root/'separated').symlink_to(previous/'separated',target_is_directory=True)
        self.load_audio=_load_baseline(Path(self.cfg['previous'])/'assets/baseline/script.py').load_audio
        self.save('resume_receipt',dict(previous_root=str(previous),reused_stages=[1],
            original_deadline_unchanged=old_config['pause_deadline_epoch'],archive_verification=receipt,
            stem_sha256=stem_hashes,stem_hashes_first_recorded_during_resume=True))
        self.mark('experiment_1_reused',rows=len(self.rows),features_recomputed=False)

    def experiment_1(self):
        from music_temporal import load_temporal_head, predict_temporal_music
        self.mark('experiment_1_matched_controls')
        runtime=FrozenEncoders(Path(self.cfg['previous'])/'assets','cuda',self.deadline)
        self.load_audio=runtime.base.load_audio
        stems=self.root/'separated'; stems.mkdir(exist_ok=True)
        original=runtime.base.separate_voice_and_music
        # Capture the separator's real output without running Demucs a second time.
        def capture(path,model,device):
            vocal,music=original(path,model,device)
            np.savez(stems/(Path(path).stem+'.npz'),vocal=vocal,music=music)
            return vocal,music
        runtime.base.separate_voice_and_music=capture
        raw_scores={}; started=time.time()
        cache=self.out/'features'; cache.mkdir(exist_ok=True)
        for index,row in enumerate(self.rows):
            record=runtime.extract(Path(row['path']))
            self.records[row['ID']]=record
            np.savez_compressed(cache/(row['ID']+'.npz'),**record)
            if row['kind']=='speech':
                with torch.inference_mode():
                    raw=runtime.base.load_audio(Path(row['path']))
                    _,_,values=runtime._voice_windows(raw,record['starts'])
                raw_scores[row['ID']]=float(values.max())
            if index%25==0:
                self.mark('experiment_1_features',done=index+1,total=len(self.rows),elapsed=time.time()-started)
        self.save('df_raw_probabilities',raw_scores)
        del runtime; gc.collect(); torch.cuda.empty_cache()
        heads=[load_temporal_head(p,'cuda') for p in sorted((self.source/'heads/M330').glob('*.pt'))]
        if len(heads)!=5:
            raise ValueError('The submitted M330 ensemble must have five heads')
        adapted={key:dict(music=r['stem'],music_audible=r['stem_audible'],starts=r['starts']) for key,r in self.records.items()}
        music=np.mean([predict_temporal_music(h,self.rows,adapted,'cuda') for h in heads],axis=0)
        self.predictions['M330']={}
        for row,pm in zip(self.rows,music,strict=True):
            record=self.records[row['ID']]
            vf,vp,mp=record['evidence'].tolist()
            self.predictions['M330'][row['ID']]=[max(vf*vp,float(pm)*mp),vf,float(pm),vp,mp]
        self.save('predictions_M330',self.predictions['M330'])
        paired=[]
        for row in self.rows:
            if not row.get('paired_voice_id'):
                continue
            pure=row['paired_voice_id']; mixed=row['ID']
            paired.append(dict(ID=mixed,pure=pure,voice_label=self.predictions['M330'][pure][1],
                pure_df_raw=raw_scores[pure],pure_df_stem=self.predictions['M330'][pure][1],
                mixed_df_raw=raw_scores[mixed],mixed_df_stem=self.predictions['M330'][mixed][1],
                known_file_label=row['FILE_FAKE'],known_voice_label=row['VOICE_FAKE']))
        self.save('experiment_1',dict(extraction_seconds=time.time()-started,paired_controls=paired,
            reports=self.reports(['M330']),interpretation='Matched 4-second voice samples and fixed voice gain; music-source voice labels remain unknown'))

    def experiment_2(self):
        self.mark('experiment_2_antideepfake')
        directory=self.source/'antideepfake'
        self.save('native_parity_gpu',verify_native(directory,'cuda'))
        model=AntiDeepfake(directory,'cuda')
        diagnostics={}; started=time.time()
        for index,row in enumerate(self.rows):
            with np.load(self.root/'separated'/(row['ID']+'.npz'),allow_pickle=False) as separated:
                vocal=separated['vocal'].astype(np.float32)
            result={'stem':model.predict(vocal,self.deadline)}
            if row['kind']=='speech':
                result['raw']=model.predict(self.load_audio(Path(row['path'])),self.deadline)
            diagnostics[row['ID']]=result
            if index%25==0:
                self.mark('experiment_2_inference',done=index+1,total=len(self.rows))
        for name in ('Anti_stem','DF_Anti_mean','Anti_top30'):
            values={}
            for row in self.rows:
                key=row['ID']; value=list(self.predictions['M330'][key]); anti=diagnostics[key]['stem']
                value[1]=anti[1] if name=='Anti_top30' else anti[0] if name=='Anti_stem' else .5*(value[1]+anti[0])
                values[key]=fuse(value,self.records[key],'max')
            self.predictions[name]=values
            self.save('predictions_'+name,values)
        self.save('antideepfake_windows',diagnostics)
        self.save('experiment_2',dict(inference_seconds=time.time()-started,
            reports=self.reports(['M330','Anti_stem','DF_Anti_mean','Anti_top30']),
            fixed='Music M330, presence PANNs, File max; Anti fake class 0; native preprocessing'))
        del model; gc.collect(); torch.cuda.empty_cache()

    def old_training(self):
        if hasattr(self,'old_rows'):
            return
        self.old_rows=list(csv.DictReader((Path(self.cfg['old_run'])/'source/full_training.csv').open()))
        if len(self.old_rows)!=1696:
            raise ValueError('Original training manifest changed')
        self.all_records=load_records(self.old_rows,Path(self.cfg['previous'])/'cache') | self.records

    def experiment_3(self):
        from redesign_voice import fit_voice, predict_voice, voice_gate
        self.mark('experiment_3_speech_residual_pilot')
        self.old_training()
        mapping=json.loads((self.source/'voice_groups.json').read_text())
        speech=[]
        for row in self.old_rows:
            if row['ID'] not in mapping:
                continue
            speech.append(row | dict(group_id=mapping[row['ID']],kind='speech',
                event_intervals=json.dumps([dict(component='voice',start=0,end=float(row['duration_seconds']),label=int(row['VOICE_FAKE']))])))
        speech += [r for r in self.rows if r['split']=='train' and r['kind']=='speech' and r['MUSIC_PRESENT']=='0']
        held={g for r in self.rows if r['split']!='train' for g in r['group_id'].split('|')}
        if held & {r['group_id'] for r in speech}:
            raise ValueError('Speech speaker overlaps a held-out split')
        gate=voice_gate(speech)
        result=fit_voice(speech,self.all_records,self.out/'models/voice_pilot',seed=41,epochs=30,
            patience=8,device='cuda',deadline=self.deadline,require_gate=False)
        if result.checkpoint is None:
            raise RuntimeError(result.reason)
        voice=predict_voice(result.checkpoint,self.rows,self.records,device='cuda')
        self.predictions['speech_residual']={}
        for row in self.rows:
            key=row['ID']; value=list(self.predictions['M330'][key]); value[1]=voice[key]['voice']
            self.predictions['speech_residual'][key]=fuse(value,self.records[key],'max')
        self.save('experiment_3',dict(status='speech_only_pilot',deployment_gate=gate,
            training_rows=len(speech),training_speakers=len({r['group_id'] for r in speech}),
            checkpoint=str(result.checkpoint),reports=self.reports(['M330','speech_residual']),
            promotion_allowed=False,reason='Unreviewed speech labels and missing real/fake singing prevent deployment'))
        self.save('predictions_speech_residual',self.predictions['speech_residual'])

    def fit(self,name,rows,seed):
        result=fit_music(rows,self.all_records,[],self.out/'models'/name,arm='B',seed=seed,
            epochs=17,patience=1,batch_size=64,device='cuda',deadline=self.deadline,fixed_epochs=True)
        if result.report['status']!='complete' or result.report['optimizer_updates']<=0:
            raise RuntimeError('Head optimization did not complete')
        self.predict_music(name,[result.model])
        self.save('training_'+name,result.report)
        return result.model

    def experiment_4(self):
        self.mark('experiment_4_new_music_sources')
        self.old_training()
        control=[r for r in self.old_rows if r['source']!='nsynth']
        added=[r for r in self.rows if r['split']=='train' and r['kind']=='music' and r['variant']=='base']
        self.music_train=control+added
        self.fit('music_control',control,41)
        self.fit('music_new',self.music_train,41)
        self.save('experiment_4',dict(control_rows=len(control),added_rows=len(added),
            epochs=17,seed=41,nsynth_removed_from_both=True,
            reports=self.reports(['M330','music_control','music_new'])))

    def experiment_5(self):
        from voice_music_h100_selection import calibrate
        self.mark('experiment_5_fixed_calibration')
        reports=calibrate(self.rows,self.predictions,self.records)
        self.save('calibration_locked',reports)
        self.save('experiment_5',self.reports(['DF_Anti_mean','calibrated_max','calibrated_noisy_or']))

    def experiment_6(self):
        from voice_music_h100_selection import seed_gate
        self.mark('experiment_6_robustness_and_seeds')
        gate=seed_gate(self.rows,self.predictions)
        self.save('selection_locked',gate)
        if gate['additional_seeds']:
            heads=list(self.heads['music_new'])
            for seed in (42,43):
                heads.append(self.fit('music_new_seed'+str(seed),self.music_train,seed))
            self.predict_music('music_new_ensemble',heads)
        self.save('experiment_6',dict(dev=self.reports(list(self.predictions)),
            test=self.reports(list(self.predictions),'test'),selection=gate))
        self.save('summary',dict(status='complete',executed_stages=[2,3,4,5,6] if 'resume_root' in self.cfg else [1,2,3,4,5,6],
            reused_stages=[1] if 'resume_root' in self.cfg else [],
            additional_seeds=gate['additional_seeds'],official_score=None,
            leaderboard_improvement_verified=False,promotion_allowed=False))
        self.mark('complete',additional_seeds=gate['additional_seeds'])
