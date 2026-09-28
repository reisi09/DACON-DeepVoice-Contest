"""Extract two temporal resolutions together, then run the locked 2x2 comparison."""

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

from music_temporal import TemporalRecipe, fit_temporal_head, predict_temporal_music
from temporal_experiment import music_metrics, training_masks, evaluate_locked_test
from temporal_tuning import compact_candidates
from v6_data import sha256, write_json
from v6_runtime import baseline_module
from v7_ablation import Recipe, fit_head


def pool_frames(hidden, starts, kernels, strides):
    """Disjoint bins; coordinates are convolutional receptive-field centres at 16 kHz."""
    receptive, jump = 1, 1
    for kernel, stride in zip(kernels, strides, strict=True):
        receptive += (kernel - 1) * jump
        jump *= stride
    indices = np.array_split(np.arange(hidden.shape[1]), 8)
    if any(len(i) == 0 for i in indices):
        raise ValueError('MERT window has fewer than eight frames')
    coarse = hidden.mean(1).float().cpu().numpy()
    fine = torch.stack([hidden[:, i[0]:i[-1]+1].mean(1) for i in indices], 1).float().cpu().numpy()
    counts = np.array([len(i) for i in indices])
    offsets = np.array([((receptive-1)/2 + i.mean()*jump)*16000/24000 for i in indices])
    coordinates = np.asarray(starts)[:, None] + offsets[None, :]
    np.testing.assert_allclose((fine*counts[None,:,None]).sum(1)/counts.sum(), coarse, atol=2e-6, rtol=2e-5)
    return coarse, fine, coordinates, counts


def load_pair(path, identity):
    with np.load(path, allow_pickle=False) as z:
        if z['identity'].item() != identity:
            raise ValueError(f'Stale feature cache: {path}')
        r = {k: z[k] for k in z.files if k != 'identity'}
    n = len(r['starts'])
    expected = dict(music_1=(n,768), music_8=(n,8,768), bin_times=(n,8),
                    music_audible=(n,), starts=(n,), frame_counts=(8,))
    if not n or any(r[k].shape != shape or not np.isfinite(r[k]).all() for k,shape in expected.items()):
        raise ValueError(f'Corrupt feature cache: {path}')
    if (np.any(r['frame_counts'] < 1) or np.any(np.diff(r['bin_times'], axis=1) <= 0)
            or not np.isin(r['music_audible'], [0,1]).all()):
        raise ValueError(f'Invalid feature timing or mask: {path}')
    np.testing.assert_allclose(np.average(r['music_8'], axis=1, weights=r['frame_counts']),
                               r['music_1'], atol=2e-6, rtol=2e-5)
    common = dict(starts=r['starts'], music_audible=r['music_audible'])
    return dict(common, music=r['music_1']), dict(common, music=r['music_8'], bin_times=r['bin_times'])


def extract_features(rows, assets, audio, cache, output, deadline):
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel

    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((assets/'manifest.json').read_text())
    for name, digest in manifest.items():
        if sha256(assets/name) != digest:
            raise ValueError(f'Asset hash mismatch: {name}')
    provenance = dict(schema='mert95-disjoint-eight-v1', assets=manifest,
        extractor_sha256=sha256(Path(__file__)), torch=torch.__version__, device=torch.cuda.get_device_name(),
        versions={k:importlib.metadata.version(k) for k in ('transformers','demucs','librosa','numpy','torchaudio')},
        sample_rate=16000, mert_sample_rate=24000, window_samples=64600,
        frame_time='convolution receptive-field centre; contextual MERT embeddings',
        tail='baseline last full window, short audio tiled', bins=[1,8], dtype='float32')
    cache_id = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    write_json(output/'extraction_protocol.json', dict(cache_id=cache_id, **provenance))
    base = baseline_module(assets)
    demucs = base.load_htdemucs_model().eval().requires_grad_(False)
    processor = AutoFeatureExtractor.from_pretrained(str(assets/'mert'),local_files_only=True)
    mert, info = AutoModel.from_pretrained(str(assets/'mert'),trust_remote_code=True,
                                          local_files_only=True,output_loading_info=True)
    if any(info.values()):
        raise ValueError(f'MERT weights not loaded exactly: {info}')
    mert.cuda().eval().requires_grad_(False)
    coarse, fine, receipts = {}, {}, {}
    began = time.time()
    for index, row in enumerate(rows):
        if time.time() > deadline:
            raise TimeoutError('Feature extraction deadline exceeded')
        source = audio/(row['ID']+'.wav')
        if sha256(source) != row['audio_sha256']:
            raise ValueError(f'Audio hash mismatch: {row["ID"]}')
        identity = row['audio_sha256']+':'+cache_id
        target = cache/(row['ID']+'.npz')
        file_started = time.time()
        if not target.exists():
            waveform = base.load_audio(source)
            starts = np.array(base.get_segment_starts(len(waveform)))
            with torch.inference_mode():
                _, backing = base.separate_voice_and_music(source, demucs, torch.device('cuda'))
                backing = np.pad(backing[:len(waveform)],(0,max(0,len(waveform)-len(backing))))
                low, high, times, audible = [], [], [], []
                for offset in range(0,len(starts),16):
                    pieces = [base.extract_segment(backing,int(s)) for s in starts[offset:offset+16]]
                    audible.extend(float(base.calculate_rms(p)>=base.SILENCE_RMS) for p in pieces)
                    inputs = processor([librosa.resample(p,orig_sr=16000,target_sr=24000,res_type='soxr_hq')
                                        for p in pieces],sampling_rate=24000,return_tensors='pt')
                    hidden = mert(inputs.input_values.cuda()).last_hidden_state
                    one, eight, coordinates, counts = pool_frames(hidden, starts[offset:offset+16],
                                                                  mert.config.conv_kernel,mert.config.conv_stride)
                    low.extend(one); high.extend(eight); times.extend(coordinates)
                values = dict(music_1=np.array(low), music_8=np.array(high), bin_times=np.array(times),
                              frame_counts=counts, starts=starts, music_audible=np.array(audible))
            temporary = target.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,identity=np.array(identity),**values)
            temporary.replace(target)
        coarse[row['ID']],fine[row['ID']] = load_pair(target,identity)
        receipts[row['ID']] = dict(audio_sha256=row['audio_sha256'],identity=identity,
            file=target.name,sha256=sha256(target),
            music_1_sha256=hashlib.sha256(coarse[row['ID']]['music'].tobytes()).hexdigest(),
            music_8_sha256=hashlib.sha256(fine[row['ID']]['music'].tobytes()).hexdigest())
        elapsed = time.time()-began
        progress = dict(stage='extract',completed=index+1,total=len(rows),ID=row['ID'],
                        file_seconds=time.time()-file_started,elapsed_seconds=elapsed,
                        projected_seconds=elapsed/(index+1)*len(rows),utc=time.time())
        write_json(output/'progress.json',progress)
        if index < 3 or (index+1)%25 == 0:
            print(progress,flush=True)
    write_json(output/'features_manifest.json',dict(cache_id=cache_id,records=receipts,seconds=time.time()-began))
    del mert,demucs
    torch.cuda.empty_cache()
    return coarse,fine


def run_factorial(train,tune,coarse,fine,quality,output,deadline):
    """Fixed C08 settings; no historical test rows enter training or selection."""
    base = compact_candidates()['C08']
    candidates = {f'R{bins}_{mode}':replace(base,temporal_bins=bins,softmax_output=mode=='softmax')
                  for bins in (1,8) for mode in ('sigmoid','softmax')}
    write_json(output/'search_protocol.json',dict(candidates={k:asdict(v) for k,v in candidates.items()},
        seeds=[41,42,43],comparison='temporal resolution x output-loss parameterization',
        selection='three-seed tune music EER mean, std, BCE, name; historical test not used',
        note='8 bins also shorten the convolution receptive field in seconds; no new HP search'))
    trials, summaries = [],[]
    for setting,recipe in candidates.items():
        records = coarse if recipe.temporal_bins==1 else fine
        masked = training_masks(train,records,quality)
        for seed in (41,42,43):
            name = f'{setting}_seed{seed}'
            selected_recipe = replace(recipe,seed=seed)
            write_json(output/'progress.json',dict(stage='training',candidate=name,completed=len(trials),total=12,utc=time.time()))
            started=time.time()
            model,report=fit_temporal_head(train,masked,(tune,records),output/name,deadline,
                                           recipe=selected_recipe,device='cuda')
            p=predict_temporal_music(model,tune,records,'cuda',recipe.batch_size)
            trial=dict(name=name,setting=setting,seed=seed,recipe=asdict(selected_recipe),
                       **music_metrics(tune,p),parameters=report['parameters'],
                       selected_epoch=report['selected_epoch'],seconds=time.time()-started)
            trials.append(trial)
            np.savez_compressed(output/name/'tune_predictions.npz',ids=np.array([r['ID'] for r in tune]),probability=p)
            write_json(output/'tuning.json',trials)
            print(trial,flush=True)
            del model
        members=trials[-3:]
        summaries.append(dict(name=setting,recipe=asdict(recipe),candidates=[r['name'] for r in members],
            music_eer_mean=float(np.mean([r['music_eer'] for r in members])),
            music_eer_std=float(np.std([r['music_eer'] for r in members])),
            balanced_bce_mean=float(np.mean([r['balanced_bce'] for r in members]))))
    winner=min(summaries,key=lambda r:(r['music_eer_mean'],r['music_eer_std'],r['balanced_bce_mean'],r['name']))
    write_json(output/'selection.json',dict(winner=winner,summaries=summaries,test_used=False,frozen_at=time.time(),
        evaluation_candidates=[r['name'] for r in trials],comparison_groups={s['name']:s['candidates'] for s in summaries}))
    linear,_,stats=fit_head(train,training_masks(train,coarse,quality),Recipe('linear_control',.05,True,epochs=300),
                            deadline,validation=(tune,coarse))
    linear.save(output/'linear_control.npz')
    write_json(output/'linear_control.json',stats)
    return trials


def main():
    import os
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    split=json.loads(Path('locked_split.json').read_text())
    rows=[r for part in ('train','tune','test') for r in split[part]]
    assert len({r['ID'] for r in rows})==len(rows)
    output=Path('results');output.mkdir(exist_ok=True)
    write_json(output/'split.json',split)
    deadline=float(os.environ['V7_RESUMED_EPOCH'])+float(os.environ['EXPERIMENT_COMPUTE_HOURS'])*3600-120
    coarse,fine=extract_features(rows,Path('assets'),Path('audio'),Path('feature_cache'),output,deadline)
    plan=json.loads(Path('full_data_plan.json').read_text())
    run_factorial(split['train'],split['tune'],coarse,fine,plan['quality'],output,deadline)
    evaluate_locked_test(split['test'],coarse,json.loads(Path('baseline_predictions.json').read_text()),output,
                         fine_records=fine)


if __name__=='__main__':
    main()
