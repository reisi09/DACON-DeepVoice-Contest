"""Compare a 637M AF-Whisper audio encoder with the sealed MERT330 MLP controls."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from afwhisper_features import extract
from music_temporal import TemporalRecipe, fit_temporal_head, load_temporal_head, predict_temporal_music
from temporal_backbone import REGULARIZED_C08
from temporal_experiment import evaluate_locked_test, music_metrics, training_masks
from temporal_resolution import load_pair, select_resolution
from v6_data import assert_disjoint, sha256, write_json


def main() -> None:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if 'A100' not in torch.cuda.get_device_name():raise ValueError('Expected A100')
    for name,digest in json.loads(Path('input_manifest.json').read_text())['sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError(f'Input hash: {name}')
    split=json.loads(Path('locked_split.json').read_text())
    for a,b in [('train','tune'),('train','test'),('tune','test')]:assert_disjoint(split[a],split[b])
    rows=[r for part in ('train','tune','test') for r in split[part]]
    records=[]
    for file,folder,dim in [('reference330_features_manifest.json','feature_cache',1024),
                            ('reference95_features_manifest.json','reference95_cache',768)]:
        manifest=json.loads(Path(file).read_text());selected={}
        for row in rows:
            item=manifest['records'][row['ID']];p=Path(folder)/item['file']
            if item['identity']!=row['audio_sha256']+':'+manifest['cache_id'] or sha256(p)!=item['sha256']:
                raise ValueError(f'Reference cache identity: {p}')
            selected[row['ID']],_=load_pair(p,item['identity'],dim)
        records.append(selected)
    mert,coarse=records
    output=Path('results');output.mkdir(exist_ok=True)
    deadline=float(os.environ['V7_RESUMED_EPOCH'])+7200-180
    plan=json.loads(Path('full_data_plan.json').read_text())
    protocol=dict(models=['m-a-p/MERT-v1-330M','nvidia/audio-flamingo-3/sound_tower'],
        afwhisper_parameters=636968960,input_dims=[1024,1280],seeds=[41,42,43],
        factors='backbone x label_smoothing(0,0.05); original compact MLP, fixed LR',
        afwhisper_metadata=json.loads(Path('afwhisper_metadata.json').read_text()),
        split_sha256=sha256(Path('locked_split.json')),nsynth_mass=0.,
        deadline=deadline,selection='three-seed tune music EER mean, std, BCE, name; freeze before historical test',
        controls='Reuse six sealed MERT330 MLP checkpoints and verify predictions; six new AF-Whisper fits',
        limitation='Different pretraining, architecture, input dimension and required frontend; not a parameter-count-only causal study. Historical test reused. No official score or submission update.')
    write_json(output/'search_protocol.json',protocol)
    af=extract(rows,mert,output,deadline-600)
    trials=[];summaries=[]
    for backbone,features,dim in [('M330',mert,1024),('AF637',af,1280)]:
        masked=training_masks(split['train'],features,plan['quality'])
        for smoothing in (0.,.05):
            setting=f'{backbone}_mlp_ls{int(smoothing*100)}';metrics=[]
            recipe=replace(REGULARIZED_C08,input_dim=dim,label_smoothing=smoothing)
            for seed in (41,42,43):
                name=f'{setting}_seed{seed}';folder=output/name;chosen=replace(recipe,seed=seed)
                write_json(output/'progress.json',dict(stage='train_or_check_control',candidate=name,
                    completed=len(trials),total=12,utc=time.time()))
                started=time.time()
                if backbone=='M330':
                    report=json.loads((folder/'training.json').read_text())
                    if TemporalRecipe(**report['recipe'])!=chosen:raise ValueError('MERT control recipe mismatch')
                    if sha256(folder/'temporal_music.pt')!=report['checkpoint_sha256']:
                        raise ValueError('MERT control checkpoint hash mismatch')
                    model=load_temporal_head(folder/'temporal_music.pt','cuda')
                else:
                    model,report=fit_temporal_head(split['train'],masked,(split['tune'],features),
                        folder,deadline,recipe=chosen,device='cuda')
                p=predict_temporal_music(model,split['tune'],features,'cuda',chosen.batch_size)
                if backbone=='M330':
                    with np.load(folder/'tune_predictions.npz') as z:
                        np.testing.assert_array_equal(z['ids'],[r['ID'] for r in split['tune']])
                        np.testing.assert_allclose(p,z['probability'],atol=1e-6,rtol=0)
                else:
                    np.savez_compressed(folder/'tune_predictions.npz',ids=np.array([r['ID'] for r in split['tune']]),probability=p)
                metric=music_metrics(split['tune'],p);metrics.append(metric)
                trials.append(dict(name=name,backbone=backbone,seed=seed,recipe=asdict(chosen),
                    parameters=report['parameters'],selected_epoch=report['selected_epoch'],
                    epochs_completed=report['epochs_completed'],reused_control=backbone=='M330',
                    seconds=time.time()-started,**metric))
                write_json(output/'tuning.json',trials);print(trials[-1],flush=True)
                del model
            summaries.append(dict(name=setting,recipe=asdict(recipe),candidates=[f'{setting}_seed{s}' for s in (41,42,43)],
                music_eer_mean=float(np.mean([m['music_eer'] for m in metrics])),
                music_eer_std=float(np.std([m['music_eer'] for m in metrics])),
                balanced_bce_mean=float(np.mean([m['balanced_bce'] for m in metrics]))))
    write_json(output/'selection.json',dict(winner=select_resolution(summaries),summaries=summaries,
        test_used=False,frozen_at=time.time(),evaluation_candidates=[t['name'] for t in trials],
        comparison_groups={s['name']:s['candidates'] for s in summaries}))
    evaluate_locked_test(split['test'],coarse,json.loads(Path('baseline_predictions.json').read_text()),output,
                         wide_records=mert,afwhisper_records=af)


if __name__=='__main__':main()
