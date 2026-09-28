"""Compare frozen MERT-330M against the existing MERT-95M C08 control."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from music_temporal import fit_temporal_head, predict_temporal_music
from temporal_experiment import evaluate_locked_test, music_metrics, training_masks
from temporal_resolution import extract_features, load_pair, select_resolution
from temporal_tuning import compact_candidates
from v6_data import sha256, write_json

REGULARIZED_C08 = replace(compact_candidates()['C08'], dropout=.1, weight_decay=.01,
                          softmax_output=True, mlp_output=True)
COSINE_C08 = replace(REGULARIZED_C08, mlp_output=False, cosine_output=True,
                     label_smoothing=.05, lr_plateau=True)


def verify_inputs(root: Path) -> None:
    """Reject missing or changed fixed controls before expensive extraction."""
    manifest = json.loads((root/'input_manifest.json').read_text())['sha256']
    required = {'reference95_features_manifest.json', 'results/linear_control.npz'}
    required.update(f'results/M95_seed{seed}/{name}' for seed in (41,42,43)
                    for name in ('temporal_music.pt','training.json','tune_predictions.npz'))
    if not required.issubset(manifest):
        raise ValueError('Fixed comparison controls are not sealed in the input manifest')
    for name, digest in manifest.items():
        path = root/name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Input integrity mismatch: {name}')


def main() -> None:
    """Extract once, fit matched regularized heads, then evaluate frozen selection."""
    verify_inputs(Path.cwd())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    assert os.environ.get('EXPERIMENT_GPU','H100') in torch.cuda.get_device_name()
    split = json.loads(Path('locked_split.json').read_text())
    rows = [r for part in ('train','tune','test') for r in split[part]]
    output = Path('results'); output.mkdir(exist_ok=True)
    write_json(output/'split.json',split)
    original = json.loads(Path('reference95_features_manifest.json').read_text())
    coarse = {}
    for row in rows:
        item = original['records'][row['ID']]
        path = Path('reference95_cache')/item['file']
        if (item['audio_sha256'] != row['audio_sha256'] or sha256(path) != item['sha256']
                or item['identity'] != row['audio_sha256']+':'+original['cache_id']):
            raise ValueError(f'MERT-95M control provenance mismatch: {row["ID"]}')
        coarse[row['ID']],_ = load_pair(path,item['identity'])
    deadline = float(os.environ['V7_RESUMED_EPOCH'])+7200-120
    wide,_ = extract_features(rows,Path('assets'),Path('audio'),Path('feature_cache'),output,deadline)
    for row in rows:
        np.testing.assert_array_equal(wide[row['ID']]['starts'],coarse[row['ID']]['starts'])
        np.testing.assert_array_equal(wide[row['ID']]['music_audible'],coarse[row['ID']]['music_audible'])
    plan = json.loads(Path('full_data_plan.json').read_text())
    recipe = replace(COSINE_C08,input_dim=1024)
    protocol = dict(model='m-a-p/MERT-v1-330M',revision=json.loads(Path('mert330_metadata.json').read_text())['revision'],
        gpu=torch.cuda.get_device_name(),
        recipe=asdict(recipe),seeds=[41,42,43],control='MERT95 cached features with newly trained matched regularized heads',
        comparison='Both backbones use 64x3 GELU convolutions, cosine classifier (scale 10), label smoothing 0.05, ReduceLROnPlateau, dropout 0.1 and AdamW weight decay 0.01',
        reference95_manifest_sha256=sha256(Path('reference95_features_manifest.json')),
        selection='Tune EER mean, std, BCE rounded to 12 decimals, then name; no test-based selection')
    write_json(output/'search_protocol.json',protocol)
    trials = []
    for setting,records,input_dim in (('M95_cosine',coarse,768),('M330_cosine',wide,1024)):
        masked = training_masks(split['train'],records,plan['quality'])
        for seed in (41,42,43):
            name = f'{setting}_seed{seed}'
            selected = replace(COSINE_C08,input_dim=input_dim,seed=seed)
            write_json(output/'progress.json',dict(stage='training',candidate=name,completed=len(trials),total=6,utc=time.time()))
            started = time.time()
            model,report = fit_temporal_head(split['train'],masked,(split['tune'],records),output/name,
                                             deadline,recipe=selected,device='cuda')
            p = predict_temporal_music(model,split['tune'],records,'cuda',selected.batch_size)
            trials.append(dict(name=name,setting=setting,seed=seed,recipe=asdict(selected),parameters=report['parameters'],
                               selected_epoch=report['selected_epoch'],seconds=time.time()-started,**music_metrics(split['tune'],p)))
            np.savez_compressed(output/name/'tune_predictions.npz',ids=np.array([r['ID'] for r in split['tune']]),probability=p)
            write_json(output/'tuning.json',trials)
            print(trials[-1],flush=True)
            del model
    summaries = []
    for setting in ('M95_cosine','M330_cosine'):
        metrics=[]
        for seed in (41,42,43):
            folder=output/f'{setting}_seed{seed}'
            with np.load(folder/'tune_predictions.npz') as z:
                assert z['ids'].tolist()==[r['ID'] for r in split['tune']]
                metrics.append(music_metrics(split['tune'],z['probability']))
        setting_recipe=json.loads((output/f'{setting}_seed41/training.json').read_text())['recipe']
        summaries.append(dict(name=setting,recipe=setting_recipe,candidates=[f'{setting}_seed{s}' for s in (41,42,43)],
            music_eer_mean=float(np.mean([m['music_eer'] for m in metrics])),
            music_eer_std=float(np.std([m['music_eer'] for m in metrics])),
            balanced_bce_mean=float(np.mean([m['balanced_bce'] for m in metrics]))))
    write_json(output/'selection.json',dict(winner=select_resolution(summaries),summaries=summaries,
        test_used=False,frozen_at=time.time(),comparison_groups={s['name']:s['candidates'] for s in summaries},
        evaluation_candidates=[n for s in summaries for n in s['candidates']]))
    evaluate_locked_test(split['test'],coarse,json.loads(Path('baseline_predictions.json').read_text()),output,
                         wide_records=wide)


if __name__=='__main__':
    main()
