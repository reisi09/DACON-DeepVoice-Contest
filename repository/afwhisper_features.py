"""Frozen AF-Whisper features, retaining the existing audio windows and silence mask."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Qwen2AudioEncoderConfig, WhisperFeatureExtractor
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

from music_temporal import Records, Rows
from v6_data import sha256, write_json
from v6_runtime import baseline_module


def pooled_features(model: Qwen2AudioEncoder, mel: torch.Tensor, valid_mel: int) -> torch.Tensor:
    """Trim masked keys after convolutions; preserve the full padded convolution context."""
    valid = (valid_mel + 1) // 2
    if valid < 2 or valid > model.config.max_source_positions:
        raise ValueError('Invalid AF-Whisper feature length')
    h = F.gelu(model.conv1(mel))
    h = F.gelu(model.conv2(h)).transpose(1,2)[:, :valid]
    h = h + model.embed_positions.weight[:valid]
    for layer in model.layers:
        h = layer(h, attention_mask=None, layer_head_mask=None, output_attentions=False)[0]
    h = model.avg_pooler(h.transpose(1,2)).transpose(1,2)
    return model.layer_norm(h).mean(1)


def load_encoder(folder: Path, device: str = 'cuda') -> Qwen2AudioEncoder:
    config = Qwen2AudioEncoderConfig(**json.loads((folder/'config.json').read_text()))
    config._attn_implementation = 'sdpa'
    model, info = Qwen2AudioEncoder.from_pretrained(folder, config=config, torch_dtype=torch.float32,
        local_files_only=True, output_loading_info=True)
    if any(info.values()):
        raise ValueError(f'AF-Whisper checkpoint mismatch: {info}')
    if sum(p.numel() for p in model.parameters()) != 636968960:
        raise ValueError('Unexpected AF-Whisper audio parameter count')
    return model.to(device).eval().requires_grad_(False)


def extract(rows: Rows, reference: Records, output: Path, deadline: float) -> Records:
    assets = Path('assets'); cache = Path('afwhisper_cache');cache.mkdir(exist_ok=True)
    meta = json.loads(Path('afwhisper_metadata.json').read_text())
    folder = assets/'afwhisper'
    for file,key in [('config.json','config_sha256'),('model.safetensors','weight_sha256')]:
        if sha256(folder/file) != meta[key]:raise ValueError(f'AF-Whisper asset hash: {file}')
    baseline_hashes = {str(p.relative_to(assets)):sha256(p) for p in (assets/'baseline').rglob('*') if p.is_file()}
    protocol = dict(model=meta, source_sha256=sha256(Path(__file__)), baseline_assets=baseline_hashes,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), dtype='float32',
        versions={k:importlib.metadata.version(k) for k in ('transformers','librosa','numpy','demucs')},
        sample_rate=16000, window_samples=64600, feature='128 mel, 30s padded; post-convolution masked-key trimming; valid-frame mean after native avgpool and layernorm',
        extractor='WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, return_attention_mask=True)',
        batch_size=8)
    identity_hash = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    write_json(output/'extraction_protocol.json',dict(cache_id=identity_hash, **protocol))
    base = baseline_module(assets)
    demucs = base.load_htdemucs_model().eval().requires_grad_(False)
    processor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, return_attention_mask=True)
    model = load_encoder(folder)
    records = {};receipts = {};began = time.time();parity_done = False
    for index,row in enumerate(rows):
        if time.time() >= deadline:raise TimeoutError('AF-Whisper extraction deadline')
        audio = Path('audio')/(row['ID']+'.wav')
        if sha256(audio) != row['audio_sha256']:raise ValueError(f'Audio hash: {row["ID"]}')
        target = cache/(row['ID']+'.npz');identity = row['audio_sha256']+':'+identity_hash
        started = time.time()
        if not target.exists():
            waveform = base.load_audio(audio)
            starts = np.array(base.get_segment_starts(len(waveform)))
            with torch.inference_mode():
                _,backing = base.separate_voice_and_music(audio,demucs,torch.device('cuda'))
                backing = np.pad(backing[:len(waveform)],(0,max(0,len(waveform)-len(backing))))
                features=[];audible=[]
                for offset in range(0,len(starts),8):
                    if time.time() >= deadline:raise TimeoutError('AF-Whisper batch deadline')
                    pieces=[base.extract_segment(backing,int(s)) for s in starts[offset:offset+8]]
                    audible.extend(float(base.calculate_rms(p)>=base.SILENCE_RMS) for p in pieces)
                    inputs=processor(pieces,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
                    lengths=inputs.attention_mask.sum(-1)
                    if not torch.all(lengths == lengths[0]):raise ValueError('Windows must have equal valid lengths')
                    mel=inputs.input_features.cuda();valid_mel=int(lengths[0]);valid=(valid_mel+1)//2
                    p=pooled_features(model,mel,valid_mel)
                    if not parity_done:
                        total=model.config.max_source_positions
                        mask=torch.zeros(1,1,total,total,device='cuda');mask[:,:,:,valid:]=-torch.inf
                        full=model(mel[:1],attention_mask=mask).last_hidden_state[:,:valid//2].mean(1)
                        np.testing.assert_allclose(full.cpu(),p[:1].cpu(),atol=1e-4,rtol=1e-5)
                        alone=pooled_features(model,mel[:1],valid_mel)
                        np.testing.assert_allclose(alone.cpu(),p[:1].cpu(),atol=1e-4,rtol=1e-5)
                        write_json(output/'encoder_parity.json',dict(full_masked_vs_trimmed_max_abs=float((full-p[:1]).abs().max()),
                            tolerance=1e-4, audio_id=row['ID'], valid_mel_frames=valid_mel, valid_encoder_frames=valid,
                            audio_parameters=sum(x.numel() for x in model.parameters()),
                            trainable_audio_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                            batch_vs_single_max_abs=float((alone-p[:1]).abs().max())))
                        parity_done=True
                    features.extend(p.cpu().numpy())
            with target.with_suffix('.partial').open('wb') as f:
                np.savez_compressed(f,identity=np.array(identity),music=np.array(features),music_audible=np.array(audible),starts=starts)
            target.with_suffix('.partial').replace(target)
        with np.load(target,allow_pickle=False) as z:
            if z['identity'].item()!=identity:raise ValueError(f'Stale AF-Whisper cache: {target}')
            record={k:z[k] for k in ('music','music_audible','starts')}
        if record['music'].shape != (len(record['starts']),1280) or not np.isfinite(record['music']).all():
            raise ValueError(f'Invalid AF-Whisper features: {target}')
        np.testing.assert_array_equal(record['starts'],reference[row['ID']]['starts'])
        np.testing.assert_array_equal(record['music_audible'],reference[row['ID']]['music_audible'])
        records[row['ID']]=record
        receipts[row['ID']]=dict(file=target.name,sha256=sha256(target),identity=identity,audio_sha256=row['audio_sha256'])
        elapsed=time.time()-began
        progress=dict(stage='extract',completed=index+1,total=len(rows),elapsed_seconds=elapsed,
            projected_seconds=elapsed/(index+1)*len(rows),file_seconds=time.time()-started,utc=time.time())
        write_json(output/'progress.json',progress)
        if index<3 or (index+1)%25==0:print(progress,flush=True)
    write_json(output/'afwhisper_features_manifest.json',dict(cache_id=identity_hash,records=receipts,seconds=time.time()-began))
    del model,demucs
    torch.cuda.empty_cache()
    return records
