"""Inspect supplied snapshot evidence without treating it as independent development data."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .common import read_json,sha256,check_hash,contained,write_json
from .features import FixtureAdapter
from .evaluation import metrics_report
from ._snapshot_scoring import PREDICTION_COLUMNS


def audit_input(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(root/'INPUT_MANIFEST.json');verified=[]
    for entry in manifest['files']:
        p=contained(root,entry['path']);check_hash(p,entry['sha256'])
        if p.stat().st_size!=entry['bytes']:raise ValueError('Input byte count mismatch: '+entry['path'])
        verified.append(entry['path'])
    adapter=FixtureAdapter(root);records=adapter.replay();perkey={};cache=[]
    for record,entry in zip(records,adapter.index,strict=True):
        errors={k:abs(float(record['diagnostic'][k])-float(v)) for k,v in entry['reference_predictions'].items()}
        for k,v in errors.items():perkey[k]=max(perkey.get(k,0.),v)
        with np.load(root/entry['cache'],allow_pickle=False) as a:
            cache.append({'ID':entry['ID'],'sha256':entry['sha256'],'identity':str(a['identity']),'samples':int(a['samples']),'duration_seconds':int(a['samples'])/16000,'music_windows':len(a['mert_windows']),'voice_windows':len(a['voice_embeddings']),'music_starts':a['starts'].tolist(),'music_valid_samples':a['valid_samples'].tolist(),'voice_silent':bool(a['voice_silent']),'arrays':{k:{'shape':list(a[k].shape),'dtype':str(a[k].dtype)} for k in a.files},'max_reference_abs_error':max(errors.values()),'reference_comparison':'current CPU versus provided historical GPU diagnostic; tolerance .002, not GPU parity'})
    heads={}
    for p in sorted((root/'fixtures/heads').glob('*.pt')):
        c=torch.load(p,weights_only=True,map_location='cpu')
        heads[p.name]={'sha256':sha256(p),'route':c['route'],'seed':c['seed'],'epoch':c.get('epoch'),'fake_label_index':c['fake_label_index'],'train_ids_count':len(c['train_ids']),'state_shapes':{k:list(v.shape) for k,v in c['model'].items()},'normalization': 'checkpoint mean/scale fitted from training only' if c['route']=='music' else 'no added scaler; original fc5-initialized linear head'}
    datasets={}
    missing_tokens={'','none','null','nan','unknown','[]'}
    for filename in ('selected_list.csv','runtime_manifest.csv','catalog_list.csv'):
        frame=pd.read_csv(root/'data'/filename,dtype=str,keep_default_na=False)
        fields={}
        for name in ('parent_id','parent_ids','speaker_id','track_id','track','generator','generator_family','source','source_corpus','component_id','group_id','audio_sha256','source_sha256','native_pcm_sha256','pcm16k_sha256'):
            if name in frame:
                known=frame[name].map(lambda x:x.strip().lower() not in missing_tokens)
                fields[name]={'nonempty_rows':int(known.sum()),'unique_nonempty_strings':int(frame.loc[known,name].nunique()),'missing_rows':int((~known).sum())}
        datasets[filename]={'rows':len(frame),'metadata_field_coverage':fields,
             'split_counts':frame['split'].value_counts().to_dict() if 'split' in frame else None,
             'historical_exposure_values':frame['historical_development_exposure'].value_counts().to_dict() if 'historical_development_exposure' in frame else None,
             'source_counts':frame['source'].value_counts().to_dict() if 'source' in frame else None,
             'interpretation':'Identifiers/string counts are not proof of independent originals; parent_ids JSON strings are NOT counted as individual parents.'}
    diagnostic={}
    for split in ('val','test'):
        frame=pd.read_csv(root/'diagnostic'/f'{split}_predictions.csv',dtype=str,keep_default_na=False)
        rows=frame.to_dict('records');predictions=[{'ID':r['ID'],**{n:float(r[n]) for n in PREDICTION_COLUMNS}} for r in rows]
        maxerror=max(abs(p['FILE_FAKE_PROB']-max(p['VOICE_PRESENT_PROB']*p['VOICE_FAKE_PROB'],p['MUSIC_PRESENT_PROB']*p['MUSIC_FAKE_PROB'])) for p in predictions)
        diagnostic[split]={'rows':len(rows),'columns':list(frame.columns),'baseline_max_formula_max_abs_error':maxerror,'metrics':metrics_report(rows,predictions),'independent_oof':False,'per_window_scores_available':False,'use':'diagnostic replay ONLY'}
    report={'input_zip_sha256':None,
            'input_manifest_sha256':sha256(root/'INPUT_MANIFEST.json'),'manifest_members_verified':len(verified),'base_commit':manifest['base_commit'],
            'heads':heads,'caches':cache,'max_reference_abs_errors_by_probability':perkey,'datasets':datasets,'diagnostic':diagnostic,
            'new_meta_dev_audio_acquired':0,'new_locked_ood_audio_acquired':0,'gpu_parity':'NOT_RUN','generalization_improvement':'NOT_EVALUATED'}
    # ZIP is optional after extraction; never derive a hash from a path or filename.
    input_zip=root.parent/'DACON_Chat_Stacking_Input.zip'
    report['input_zip_sha256']=sha256(input_zip) if input_zip.exists() else None
    write_json(output/'input_audit.json',report);write_json(output/'fixture_replay.json',records)
    return report
