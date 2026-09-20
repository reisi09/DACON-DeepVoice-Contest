"""Optional local GPU extraction using the CURRENT patched Engine.
No network, checkpoint download, VESSL API, or GPU launch is performed on import.
F1/F2 capture existing calls; F2R adds raw DF+MERT calls explicitly.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np
import torch
from .common import ContractError, check_hash, contained, digest, read_json, sha256, write_json, preprocessing_policy, manifest_preprocessing
from .features import SCHEMA, from_measurements, time_layout, write_extended
from .models import load_bundle
from .workflow import infer_record


class RecordingPanns:
    """Transparent local proxy: record predictions from existing inference calls."""
    def __init__(self,model):
        self.model=model;self.clip_predictions=[]
    def inference(self,*args,**kwargs):
        result=self.model.inference(*args,**kwargs)
        self.clip_predictions.append(np.asarray(result[0]).copy())
        return result
    def __getattr__(self,name):
        return getattr(self.model,name)


@torch.inference_mode()
def raw_branch_comparison(engine, measurements):
    """Extra same-encoder raw calls. This is a reliability hypothesis, not a label."""
    import librosa
    from ._snapshot_math import window_layout, windows
    raw=np.asarray(measurements['_normalized_audio'],np.float32)
    starts=window_layout(len(raw)); features=[]
    for offset in range(0,len(starts),engine.batch):
        audio=[librosa.resample(w,orig_sr=16000,target_sr=24000,res_type='soxr_hq') for w in windows(raw,starts[offset:offset+engine.batch])]
        values=engine.processor(audio,sampling_rate=24000,return_tensors='pt',padding=True).input_values.to(engine.device)
        features.append(engine.mert(values,output_hidden_states=True).hidden_states[12].mean(1).float())
    x=torch.cat(features);embeddings=[]
    for start in engine.base.get_segment_starts(len(raw)):
        engine.df(input_values=torch.from_numpy(engine.base.extract_segment(raw,start)).to(engine.device))
        embeddings.append(engine.embedding)
    z=torch.cat(embeddings);music=[];voice=[]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=engine.device.type=='cuda'):
        for head in engine.music_heads:
            logits,_=head.head((x-head.mean)/head.scale)
            music.append(float((logits.float().logsumexp(0)-np.log(len(x))).sigmoid()))
        for head in engine.voice_heads:
            voice.append(float(head(z).float().softmax(-1)[:,engine.fake_index].max()))
    measurements['raw_music_fake']=float(np.mean(music))
    measurements['raw_voice_fake']=0. if engine.base.calculate_rms(raw)<engine.base.SILENCE_RMS else float(np.mean(voice))
    denom=float(np.square(raw,dtype=np.float64).mean())+1e-12
    for branch in ('voice','music'):
        measurements[branch+'_raw_energy_ratio']=float(np.square(measurements['_'+branch+'_audio'],dtype=np.float64).mean()/denom)


def record_from_engine(engine,path: Path,identifier: str,preserve_gain=False,raw_comparison=False):
    policy=manifest_preprocessing({'preserve_gain':preserve_gain})
    baseline,measurements=engine.predict(path,preserve_gain=policy['preserve_gain'],collect_features=True)
    if raw_comparison: raw_branch_comparison(engine,measurements)
    preprocessing={**policy,'source_peak':measurements['source_peak'],'normalization_gain':measurements['normalization_gain']}
    preprocessing_policy(preprocessing)
    return {'ID':identifier,'baseline':baseline,'features':from_measurements(baseline,measurements),
            'preprocessing':preprocessing,
            'measurements':{k:v for k,v in measurements.items() if not k.startswith('_')}}


def runtime_identity(root: str | Path):
    root=Path(root)
    from ._base_contract import BASE_CONTRACT
    for name,expected in BASE_CONTRACT.items(): check_hash(contained(root,name),expected)
    code={str(p.relative_to(root)):sha256(p) for p in sorted((root/'model/runtime').glob('*.py'))}
    base=root/'model/assets/baseline/script.py'
    if not base.exists(): raise ContractError('Current baseline script is missing')
    code[str(base.relative_to(root))]=sha256(base)
    for p in sorted(Path(__file__).parent.glob('*.py')): code['stacking/'+p.name]=sha256(p)
    checkpoint={str(p.relative_to(root)):sha256(p) for p in sorted((root/'model/trained').glob('*.pt'))}
    if len(checkpoint)!=6: raise ContractError('Exactly six current head checkpoints required')
    assets=read_json(root/'model/assets/asset_hashes.json')
    # Record complete supplied asset hash manifest as well as verified package files.
    checkpoint['asset_hashes_manifest']=sha256(root/'model/assets/asset_hashes.json')
    return {'code_sha256':code,'checkpoint_sha256':checkpoint,'numeric_mode':'cuda_bfloat16_heads_fp32_encoder_features_no_tf32','feature_schema':SCHEMA}


def load_local_engine(root: str | Path,device='cuda',batch=8):
    root=Path(root).resolve()
    if device!='cuda' or not torch.cuda.is_available(): raise ContractError('GPU extraction needs a CUDA runtime; use fixture-replay for CPU cached data')
    manifest=read_json(root/'model/manifest.json')
    missing=[name for name in manifest['sha256'] if not (root/name).exists()]
    if missing: raise ContractError('Missing local frozen assets/checkpoints: '+', '.join(missing[:12]))
    for name,expected in manifest['sha256'].items(): check_hash(contained(root,name),expected)
    runtime_identity(root)  # Reject any other base snapshot before model loading.
    runtime=root/'model/runtime'
    for name in ('mert12_submission','localization_models','eight_heads'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent!=runtime:
            raise ContractError('Another runtime snapshot already imported; use a fresh process')
    sys.path.insert(0,str(runtime))
    decoder=runtime/'bin/ffmpeg'
    if not decoder.exists(): raise ContractError('Bundled offline ffmpeg missing')
    decoder.chmod(decoder.stat().st_mode|0o111);os.environ['PATH']=str(decoder.parent)+os.pathsep+os.environ.get('PATH','')
    spec=importlib.util.spec_from_file_location('_dacon_current_engine',runtime/'full_bank_submission.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.CANDIDATE!='existing9504_MERT330M_WindowMIL_DFfc5_701515': raise ContractError('Wrong current base candidate')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    return module.Engine(root/'model/assets',root/'model/trained',device,batch=batch)


def extract_audio(rows,root,output,raw_comparison=False):
    """Local file paths only; parent split/exposure audit is done BEFORE this call."""
    # Validate ALL manifest policies before touching the native loader or output.
    for row in rows: manifest_preprocessing(row)
    output=Path(output)
    if output.exists(): raise ContractError('Refuse to overwrite an extraction cache')
    engine=load_local_engine(root);identity=runtime_identity(root);output.mkdir(parents=True)
    index={'schema':'dacon.stacking.cache_index.v1','items':{}}
    for row in rows:
        path=Path(row['audio_path']);check_hash(path,row['audio_sha256'])
        preserve=manifest_preprocessing(row)['preserve_gain']
        record=record_from_engine(engine,path,row['ID'],preserve,raw_comparison)
        n=record['measurements']['samples']
        provenance={**identity,'audio_sha256':row['audio_sha256'],
            'timebase':{b:time_layout(n,b) for b in ('music','voice','pann')},
            'preprocessing':record['preprocessing'],
            'base_exposure':row['base_exposure'],'extra_encoder_calls':'raw DF + raw MERT' if raw_comparison else 'none',
            'feature_measurement_mode':'native_current_GPU'}
        file=digest({'ID':row['ID'],'audio':row['audio_sha256']})[:32]+'.npz'
        side_hash=write_extended(output/file,record,provenance)
        # Store measured windows and PANN clips separately; metadata stays in sidecar.
        measured={k:np.asarray(v) for k,v in record['measurements'].items()}
        with (output/(file+'.telemetry.npz')).open('wb') as stream: np.savez_compressed(stream,**measured)
        index['items'][row['ID']]={'path':file,'sidecar_sha256':side_hash,'identity':digest(provenance),'telemetry_path':file+'.telemetry.npz','telemetry_sha256':sha256(output/(file+'.telemetry.npz'))}
        write_json(output/'index.json',index)
    return index


def load_deployment(lock_path,expected_lock_sha256,root):
    lock_path=Path(lock_path);check_hash(lock_path,expected_lock_sha256);lock=read_json(lock_path)
    if lock.get('purpose')!='production_candidate' or lock.get('schema')!='dacon.stacking.lock.v1': raise ContractError('Synthetic or unlocked bundle cannot be deployed')
    policy=preprocessing_policy(lock.get('preprocessing_policy'))
    identity=runtime_identity(root)
    if identity!=lock['feature_extraction_identity']: raise ContractError('Deployment encoder/head/code/numeric identity differs from trained cache')
    model=load_bundle(lock_path.parent/'bundle',lock['bundle_manifest_sha256'])
    if model['provenance'].get('purpose')!='production_candidate' or preprocessing_policy(model['provenance'].get('preprocessing_policy'))!=policy:
        raise ContractError('Deployment bundle preprocessing/purpose differs from candidate lock')
    return model,lock


def predict_engine_file(engine,path,identifier,model,contract):
    if contract.get('purpose')!='production_candidate': raise ContractError('Only a production candidate can deploy')
    policy=preprocessing_policy(contract.get('preprocessing_policy'))
    if model.get('provenance',{}).get('purpose')!='production_candidate' or preprocessing_policy(model['provenance'].get('preprocessing_policy'))!=policy:
        raise ContractError('Deployment model preprocessing/purpose differs from candidate lock')
    if contract['feature_set'] not in ('F0','F1','F2','F2R','F1S'): raise ContractError('Unknown deployed feature set')
    record=record_from_engine(engine,Path(path),identifier,preserve_gain=False,raw_comparison=contract['feature_set']=='F2R')
    return infer_record(record,model)
