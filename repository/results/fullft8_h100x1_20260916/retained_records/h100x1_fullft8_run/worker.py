import base64
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import zlib
from dataclasses import dataclass
from itertools import groupby
from typing import Literal, TypedDict

class AudioRow(TypedDict):
    ID: str
    path: str
    group_id: str
    source: str
    panel: str
    MUSIC_FAKE: str
    MUSIC_PRESENT: str
    FILE_FAKE: str
    VOICE_FAKE: str
    VOICE_PRESENT: str
    audio_sha256: str
    pcm_sha256: str

class Partition(TypedDict):
    train: list[AudioRow]
    tune: list[AudioRow]
    test: list[AudioRow]

@dataclass(frozen=True, slots=True)
class Trial:
    name: str
    split: str
    encoder: Literal['mert', 'eat']
    frozen: bool = False

TRIALS = [Trial('01_S0_MERT_frozen', 'S0', 'mert', True),
          Trial('02_S0_MERT_fullft', 'S0', 'mert'),
          Trial('03_S0_EAT_fullft', 'S0', 'eat'),
          Trial('04_S1_MERT_fullft', 'S1', 'mert'),
          Trial('05_S1_EAT_fullft', 'S1', 'eat'),
          Trial('06_S2_MERT_fullft', 'S2', 'mert'),
          Trial('07_S2_EAT_fullft', 'S2', 'eat')]


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def family(row: AudioRow) -> str:
    parts = row['source'].lower().split('/')
    return parts[1] if parts[0] == 'echoes' else parts[0]


def groups(rows: list[AudioRow]) -> set[str]:
    return {group for row in rows for group in row['group_id'].split('|') if group}


def require_disjoint(split: Partition) -> None:
    for name, rows in split.items():
        if len({r['ID'] for r in rows}) != len(rows):
            raise RuntimeError(f'Duplicate ID in {name}')
        if len({r['pcm_sha256'] for r in rows}) != len(rows):
            raise RuntimeError(f'Duplicate PCM within {name}')
        if any(not r['group_id'] or not r['pcm_sha256'] for r in rows):
            raise RuntimeError(f'Missing lineage/hash in {name}')
    for left, right in [('train', 'tune'), ('train', 'test'), ('tune', 'test')]:
        if groups(split[left]) & groups(split[right]):
            raise RuntimeError(f'Parent/artist leakage: {left}/{right}')
        for key in ['ID', 'audio_sha256', 'pcm_sha256']:
            if {r[key] for r in split[left]} & {r[key] for r in split[right]}:
                raise RuntimeError(f'{key} leakage: {left}/{right}')
    for name in ['train', 'tune', 'test']:
        if {r['MUSIC_FAKE'] for r in split[name] if r['MUSIC_PRESENT'] == '1'} < {'0', '1'}:
            raise RuntimeError(f'Both music classes required in {name}')


def make_splits(locked: Partition) -> dict[str, Partition]:
    # NSynth is excluded in every trial, before any model comparison.
    base = {k: [r for r in rows if family(r) != 'nsynth'] for k, rows in locked.items()}
    require_disjoint(base)
    result = {'S0': base}
    pool = base['train']
    # Transitive connected components, including compound parent IDs.
    parent = {g: g for g in groups(pool)}
    def root(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    for row in pool:
        tokens = row['group_id'].split('|')
        for token in tokens[1:]:
            parent[root(token)] = root(tokens[0])
    component = {r['ID']: root(r['group_id'].split('|')[0]) for r in pool}
    ordered = sorted(set(component.values()), key=lambda g: stable_hash(['split41', g]))
    count = max(1, round(len(ordered) * .2))
    test_groups, tune_groups = set(ordered[:count]), set(ordered[count:2*count])
    for name, held in [('S1', 'musicgen'), ('S2', 'audioldm')]:
        split = {'train': [], 'tune': [], 'test': []}
        for row in pool:
            group = component[row['ID']]
            partition = 'test' if group in test_groups else ('tune' if group in tune_groups else 'train')
            if partition != 'test' and family(row) == held:
                continue
            split[partition].append(row)
        require_disjoint(split)
        if any(family(r) == held for p in ['train', 'tune'] for r in split[p]):
            raise RuntimeError('Held generator leaked into optimization/selection')
        if not any(family(r) == held for r in split['test']):
            raise RuntimeError(f'No OOD examples for {held}')
        result[name] = split
    return result


def credit_estimate(hours: float, rate: float = 2.98) -> float:
    return round(hours * rate, 2)


def eer_auc(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """Reuse graph_scoring.py's full tied-score ROC / nearest-point EER rule."""
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if min(positives, negatives) == 0:
        raise RuntimeError('EER needs both classes')
    points, tp, fp = [(0., 0.)], 0, 0
    for _, tied in groupby(sorted(zip(scores, labels), reverse=True), key=lambda x: x[0]):
        for _, target in tied:
            tp += target
            fp += 1 - target
        points.append((fp / negatives, tp / positives))
    fpr, tpr = min(points, key=lambda p: abs(p[0] - (1 - p[1])))
    auc = sum((b[0]-a[0])*(b[1]+a[1])/2 for a, b in zip(points, points[1:]))
    return (fpr + (1 - tpr)) / 2, auc

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel


def check_deadline(config) -> None:
    if time.time() >= config['deadline']:
        raise TimeoutError('Session credit/time limit: training stopped; stop the VESSL resource to end billing.')


def audio_path(row: AudioRow, config) -> Path:
    relative = row['path'].split('/v4_validation/')[-1]
    path = Path(config['project_root']) / 'v4_validation' / relative
    if not path.is_file():
        raise FileNotFoundError(f'Mount original v4_validation/audio: {path}')
    return path


def verify_source(row: AudioRow, config) -> tuple[np.ndarray, int]:
    path = audio_path(row, config)
    if file_hash(path) != row['audio_sha256']:
        raise RuntimeError(f'Original file hash mismatch: {row["ID"]}')
    wav, rate = sf.read(path, dtype='float32')
    if rate != 16000 or wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
        raise RuntimeError(f'Invalid source audio: {row["ID"]}')
    if hashlib.sha256(wav.astype('<f4').tobytes()).hexdigest() != row['pcm_sha256']:
        raise RuntimeError(f'PCM mismatch: {row["ID"]}')
    return wav, rate


def audio_cache_identity(row: AudioRow, separator_hash: str, config) -> str:
    return stable_hash([row['audio_sha256'], separator_hash, config['separator_checkpoint_sha256'], config['audio_recipe'],
                        config['seconds'], config['pool_windows']])


def validate_audio_cache(row: AudioRow, config) -> None:
    path = Path(config['output'])/'audio_cache'/f'{row["ID"]}.npz'
    record = json.loads(path.with_suffix('.json').read_text())
    expected = audio_cache_identity(row, record['separator_sha256'], config)
    if (record['identity'] != expected or file_hash(path) != record['sha256']
            or record['source_sha256'] != row['audio_sha256']):
        raise RuntimeError(f'Waveform cache content or preprocessing recipe changed: {row["ID"]}')


def prepare_audio(rows: list[AudioRow], config) -> None:
    """Frozen HTDemucs once; preserve baseline normalization/separation settings."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    check_deadline(config)
    # Trusted, hash-checked upstream checkpoint needs legacy pickle loading in PyTorch 2.6+.
    original_load = torch.load
    def trusted_demucs_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = trusted_demucs_load
    try:
        model = get_model('htdemucs', repo=Path(config['model_root'])/'htdemucs')
    finally:
        torch.load = original_load
    if config.get('adapted_separator_path'):
        checkpoint = Path(config['adapted_separator_path'])
        if file_hash(checkpoint) != config['separator_checkpoint_sha256']:
            raise RuntimeError('Adapted separator checkpoint changed')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model = model.eval().requires_grad_(False).to('cuda')
    model_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        model_hash.update(name.encode())
        model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    separator_hash = model_hash.hexdigest()
    cache = Path(config['output']) / 'audio_cache'
    cache.mkdir(parents=True, exist_ok=True)
    for row in rows:
        check_deadline(config)
        target, record = cache / f'{row["ID"]}.npz', cache / f'{row["ID"]}.json'
        identity = audio_cache_identity(row, separator_hash, config)
        if target.exists() and record.exists():
            previous = json.loads(record.read_text())
            if previous['identity'] == identity and previous['sha256'] == file_hash(target):
                continue
            raise RuntimeError(f'Stale cache: use a new output directory: {target}')
        raw, rate = verify_source(row, config)
        wav = torch.from_numpy(raw).to('cuda')[None].repeat(model.audio_channels, 1)
        wav = torchaudio.functional.resample(wav, rate, model.samplerate)
        mean, std = wav.mean(0).mean(), wav.mean(0).std()
        if std.item() < 1e-8:
            music = np.zeros(len(raw), dtype=np.float32)
        else:
            with torch.inference_mode():
                sources = apply_model(model, ((wav-mean)/std)[None], device='cuda', shifts=0,
                                      split=True, overlap=.25, progress=False)[0] * std + mean
                music = sources[[i for i,s in enumerate(model.sources) if s != 'vocals']].sum(0).mean(0)
                music = torchaudio.functional.resample(music, model.samplerate, 16000).cpu().numpy()
        width = int(config['seconds'] * 16000)
        starts = list(range(0, max(1, len(music)-width+1), width))
        starts = sorted(set(starts + [max(0, len(music)-width)]))
        valid = []
        for start in starts:
            clip = music[start:start+width]
            frame_count = len(clip)//1600
            active = np.sqrt(np.mean(clip[:frame_count*1600].reshape(-1,1600).astype('float64')**2,1))
            valid.append(bool(frame_count and (active >= 10**(-50/20)).mean() >= .2))
        # Same eight fixed candidates for every encoder; only training uses the activity gate.
        pools, offsets, masks = [], [], []
        for candidates in [np.flatnonzero(valid).tolist() or [0], list(range(len(starts)))]:
            indices = np.linspace(0, len(candidates)-1, config['pool_windows']).round().astype(int)
            selected = [candidates[i] for i in indices]
            clips = []
            for i in selected:
                clip = music[starts[i]:starts[i]+width]
                clip = np.tile(clip, math.ceil(width / max(1,len(clip))))[:width]
                clips.append(clip)
            pools.append(np.stack(clips).astype('float32'))
            offsets.append([starts[i] for i in selected])
            masks.append([valid[i] for i in selected])
        temporary = target.with_suffix('.partial.npz')
        np.savez_compressed(temporary, train=pools[0], evaluation=pools[1],
                            offsets=np.array(offsets), valid=np.array(masks))
        temporary.replace(target)
        save_json(record, {'identity': identity, 'sha256': file_hash(target),
                          'source_sha256': row['audio_sha256'], 'separator_sha256': separator_hash,
                          'usable_train': bool(any(valid)), 'source_samples': len(raw),
                          'unique_train_windows': len(set(offsets[0])), 'quality_gate': '20% active 100ms >= -50dBFS'})
        print('PREPARED', row['ID'], flush=True)


class AudioBags(Dataset):
    """Mutable epoch counter controls reproducible window subsets, shared across encoders."""
    def __init__(self, rows: list[AudioRow], config, training: bool = False, feature_dir: Path | None = None):
        self.rows, self.config, self.training, self.feature_dir = rows, config, training, feature_dir
        self.epoch = 0
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int):
        row = self.rows[index]
        directory = self.feature_dir or Path(self.config['output'])/'audio_cache'
        with np.load(directory / f'{row["ID"]}.npz') as archive:
            values = archive['train' if self.training else 'evaluation'].copy()
        if self.training:
            seed = int(stable_hash([self.config['seed'], self.epoch, row['ID']])[:16], 16)
            chosen = np.sort(np.random.default_rng(seed).choice(len(values), self.config['train_windows'], replace=False))
            values = values[chosen]
        label = int(row['MUSIC_FAKE']) if row['MUSIC_FAKE'] in ('0','1') else 0
        return torch.from_numpy(values), label


def training_weights(rows: list[AudioRow]) -> torch.Tensor:
    counts = collections.Counter((r['MUSIC_FAKE'], family(r)) for r in rows)
    per_class = collections.Counter(label for label, _ in counts)
    return torch.tensor([.5 / per_class[r['MUSIC_FAKE']] / counts[(r['MUSIC_FAKE'],family(r))]
                         for r in rows], dtype=torch.double)

class ResidualTemporalBlock(nn.Module):
    """Non-causal temporal convolutions; mask every layer to isolate padding."""

    def __init__(self, dilation: int, dropout: float = .2, *, channels: int = 128,
                 layers: int = 2) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(layers)])
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        for norm, conv in zip(self.norms, self.convs, strict=True):
            hidden = norm(hidden.transpose(1, 2)).transpose(1, 2).masked_fill(mask, 0)
            hidden = self.dropout(F.gelu(conv(hidden))).masked_fill(mask, 0)
        return (residual + hidden).masked_fill(mask, 0)

class CompactHead(nn.Module):
    """Exact compact C08 MLP path from music_temporal.TemporalMusic; no head search."""
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(1024)
        self.projection = nn.Conv1d(1024,64,1)
        self.blocks = nn.ModuleList([ResidualTemporalBlock(1,.1,channels=64,layers=3)])
        self.output = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,2))
    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        features = features.masked_fill(padding_mask[...,None],0)
        mask = padding_mask[:,None,:]
        hidden = F.gelu(self.projection(self.input_norm(features).transpose(1,2))).masked_fill(mask,0)
        for block in self.blocks:
            hidden = block(hidden,mask)
        empty = padding_mask.all(1)
        pooled = hidden.masked_fill(mask,-torch.inf).amax(2).masked_fill(empty[:,None],0)
        logits = self.output(pooled)
        return torch.where(empty[:,None],logits.new_tensor([0.,-100.]),logits)

MODELS = {
    'mert': ('m-a-p/MERT-v1-330M', '5240c2708a5acaee1007f43fb9735c7dcd0b78c9'),
    'eat': ('worstchan/EAT-large_epoch20_pretrain', '1109aaae544915a2b82182c643c1894b1b4d9f27'),
}


class MusicClassifier(nn.Module):
    def __init__(self, encoder: Literal['mert','eat'], frozen: bool, model_root: Path):
        super().__init__()
        self.kind, self.frozen = encoder, frozen
        self.head = CompactHead()  # Identical head initialization before different encoder constructors.
        # Local HF snapshots need transitive Python imports copied into the module cache.
        from transformers.dynamic_module_utils import get_cached_module_file
        for source in sorted((model_root/encoder).glob('*.py')):
            get_cached_module_file(str(model_root/encoder), source.name, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_root/encoder,
            trust_remote_code=True, local_files_only=True)
        if frozen:
            self.encoder.requires_grad_(False)
        if encoder == 'mert':
            self.encoder.config.apply_spec_augment = False
            self.encoder.config.layerdrop = 0.0
            # Only the unused masking token is frozen; convolution + all Transformer layers train.
            if hasattr(self.encoder, 'masked_spec_embed'):
                self.encoder.masked_spec_embed.requires_grad_(False)
        # EAT's fixed sinusoidal positions remain fixed, as in upstream pretraining.

    def encode_windows(self, waves: torch.Tensor) -> torch.Tensor:
        match self.kind:
            case 'mert':
                with torch.autocast(waves.device.type, enabled=False):
                    wav = torchaudio.functional.resample(waves.float(), 16000, 24000)
                    wav = (wav-wav.mean(-1,keepdim=True)) / torch.sqrt(wav.var(-1,keepdim=True,unbiased=False)+1e-7)
                features = self.encoder(input_values=wav).last_hidden_state.mean(1)
            case 'eat':
                mels = []
                for wav in waves:
                    with torch.autocast(waves.device.type, enabled=False):
                        wav = wav.float()
                        mel = torchaudio.compliance.kaldi.fbank((wav-wav.mean())[None],
                            htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning',
                            num_mel_bins=128, dither=0., frame_shift=10)
                    # 5 seconds -> pad time to 512 frames, matching official normalization order.
                    mel = F.pad(mel, (0,0,0,max(0,512-len(mel))))[:512]
                    mels.append((mel + 4.268) / (4.569 * 2))
                tokens = self.encoder(torch.stack(mels)[:,None])
                features = tokens[:,1:].mean(1)
            case unknown:
                from typing import assert_never
                assert_never(unknown)
        return features

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        if bags.shape[-1] == 1024:
            if not self.frozen:
                raise RuntimeError('Cached embeddings cannot train a full encoder')
            features = bags
        else:
            count, windows, length = bags.shape
            if self.frozen:
                self.encoder.eval()
            with torch.set_grad_enabled(self.training and not self.frozen):
                features = self.encode_windows(bags.reshape(-1,length))
            features = features.reshape(count,windows,1024)
        mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
        return self.head(features, mask)


def cache_frozen_features(model: MusicClassifier, rows: list[AudioRow], config) -> Path:
    directory = Path(config['output'])/'mert_frozen_features'
    directory.mkdir(exist_ok=True)
    model.eval()
    for row in rows:
        check_deadline(config)
        path = directory/f'{row["ID"]}.npz'
        wave_path = Path(config['output'])/'audio_cache'/f'{row["ID"]}.npz'
        identity = stable_hash([file_hash(wave_path), MODELS['mert'], config['model_manifest_sha256'], config['worker_sha256']])
        record = path.with_suffix('.json')
        if path.exists() and record.exists():
            metadata = json.loads(record.read_text())
            if metadata['identity'] == identity and file_hash(path) == metadata['sha256']:
                continue
            raise RuntimeError(f'Frozen feature cache mismatch: {path}')
        outputs = {}
        with np.load(wave_path) as data, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for key in ['train','evaluation']:
                pieces = []
                for start in range(0,len(data[key]),config['train_windows']):
                    waves = torch.from_numpy(data[key][start:start+config['train_windows']]).cuda()
                    pieces.append(model.encode_windows(waves).float().cpu().numpy())
                outputs[key] = np.concatenate(pieces)
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **outputs)
        temporary.replace(path)
        save_json(record, {'identity': identity, 'sha256': file_hash(path)})
    return directory


def optimization_groups(model: MusicClassifier, config) -> list[dict]:
    groups_out = []
    for module, lr in [(model.encoder, config['encoder_lr']), (model.head, config['head_lr'])]:
        for decay in [False, True]:
            parameters = [p for name,p in module.named_parameters() if p.requires_grad and
                          (p.ndim >= 2 and not name.endswith('.bias')) == decay]
            if parameters:
                groups_out.append({'params': parameters, 'lr': lr,
                                   'weight_decay': config['weight_decay'] if decay else 0.})
    return groups_out


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step+1)/max(1,warmup)
    progress = min(1., (step-warmup)/max(1,total-warmup-1))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


def gradient_evidence(model: MusicClassifier) -> dict[str, float]:
    evidence = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f'Nonfinite gradient: {name}')
            evidence[name] = float(p.grad.detach().abs().sum())
    if not any(v > 0 for k,v in evidence.items() if k.startswith('head.')):
        raise RuntimeError('Classification head is disconnected')
    if model.frozen:
        if any(k.startswith('encoder.') for k in evidence):
            raise RuntimeError('Frozen encoder received gradients')
    else:
        names = [name for name,p in model.encoder.named_parameters() if p.requires_grad]
        if model.kind == 'mert':
            prefixes = ['feature_extractor.', 'encoder.layers.0.',
                        f'encoder.layers.{model.encoder.config.num_hidden_layers-1}.']
        else:
            prefixes = ['model.local_encoder.', 'model.blocks.0.',
                        f'model.blocks.{model.encoder.config.depth-1}.']
        for prefix in prefixes:
            expected = [name for name in names if name.startswith(prefix)]
            if not expected or not any(evidence.get('encoder.'+name,0)>0 for name in expected):
                raise RuntimeError(f'Full fine-tuning disconnected: {prefix}')
    return evidence

def music_report(rows: list[AudioRow], probabilities: list[float]) -> dict:
    indices = [i for i,r in enumerate(rows) if r['MUSIC_PRESENT']=='1' and r['MUSIC_FAKE'] in ('0','1')]
    labels = [int(rows[i]['MUSIC_FAKE']) for i in indices]
    scores = [float(probabilities[i]) for i in indices]
    result = {'count': len(labels), 'real': labels.count(0), 'fake': labels.count(1), 'eer': None, 'auc': None}
    if set(labels) == {0,1}:
        result['eer'], result['auc'] = eer_auc(labels,scores)
        losses = [-(y*math.log(max(p,1e-7))+(1-y)*math.log(max(1-p,1e-7))) for y,p in zip(labels,scores)]
        result['balanced_bce'] = sum(sum(v for v,y in zip(losses,labels) if y==c)/labels.count(c) for c in [0,1])/2
    return result


def batch_loader(data: AudioBags, config, sampler=None) -> DataLoader:
    workers = config['loader_workers']
    return DataLoader(data, batch_size=config['batch_size'], sampler=sampler,
        num_workers=workers, pin_memory=True, persistent_workers=False,
        multiprocessing_context='spawn' if workers else None,
        generator=torch.Generator().manual_seed(config['seed'] + data.epoch))


def batch_loss_weight(index: int, batch_count: int, total_count: int, config) -> float:
    effective = config['batch_size'] * config['accumulation']
    group_start = (index * config['batch_size']) // effective * effective
    return batch_count / min(effective, total_count - group_start)


def predict(model: MusicClassifier, data: AudioBags, config) -> list[float]:
    loader = batch_loader(data, config)
    output = []
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for bags, _ in loader:
            check_deadline(config)
            output.extend(model(bags.cuda(non_blocking=True)).float().softmax(-1)[:,1].cpu().tolist())
    return output


def atomic_checkpoint(path: Path, value) -> None:
    temporary = path.with_suffix('.partial.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def train_trial(trial: Trial, splits: dict[str, Partition], config) -> None:
    check_deadline(config)
    folder = Path(config['output'])/trial.name
    folder.mkdir(exist_ok=True)
    split = splits[trial.split]
    learning_keys = ['seed','seconds','train_windows','pool_windows','epochs','patience',
                     'encoder_lr','head_lr','weight_decay','warmup_fraction','accumulation','batch_size','loader_workers','audio_cache_fingerprint']
    identity = stable_hash([trial.__dict__ if hasattr(trial,'__dict__') else
                            [trial.name,trial.split,trial.encoder,trial.frozen], split,
                            {key:config[key] for key in learning_keys}, MODELS, config['model_manifest_sha256'], config['worker_sha256']])
    meta = folder/'identity.json'
    if meta.exists() and json.loads(meta.read_text())['identity'] != identity:
        raise RuntimeError(f'Configuration changed: use another output directory: {folder}')
    save_json(meta, {'identity':identity, 'historical_test':trial.split=='S0',
                     'held_generator': {'S1':'musicgen','S2':'audioldm'}.get(trial.split),
                     'selection':'tune Music EER, balanced BCE, then earliest epoch'})
    if (folder/'complete.json').exists():
        completed = json.loads((folder/'complete.json').read_text())
        if file_hash(folder/'best.pt') != completed['best_checkpoint_sha256']:
            raise RuntimeError('Completed checkpoint changed')
        print('SKIP_COMPLETE',trial.name,flush=True)
        return
    save_json(folder/'execution_config.json', config)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    usable = []
    excluded = []
    for row in split['train']:
        record = json.loads((Path(config['output'])/'audio_cache'/f'{row["ID"]}.json').read_text())
        (usable if record['usable_train'] else excluded).append(row)
    if {r['MUSIC_FAKE'] for r in usable} != {'0','1'}:
        raise RuntimeError('Quality gate removed a training class')
    save_json(folder/'training_rows.json', {'included':[r['ID'] for r in usable],
                                         'excluded_low_activity':[r['ID'] for r in excluded]})
    model = MusicClassifier(trial.encoder,trial.frozen,Path(config['model_root'])).cuda()
    feature_dir = None
    if trial.frozen:
        all_rows = {r['ID']:r for p in split.values() for r in p}
        feature_dir = cache_frozen_features(model,list(all_rows.values()),config)
    train = AudioBags(usable,config,True,feature_dir)
    tune = AudioBags(split['tune'],config,False,feature_dir)
    optimizer = torch.optim.AdamW(optimization_groups(model,config))
    steps_per_epoch = math.ceil(len(train)/(config['batch_size']*config['accumulation']))
    total_steps = steps_per_epoch*config['epochs']
    warmup = max(1,math.ceil(total_steps*config['warmup_fraction']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_factor(s,total_steps,warmup))
    best_key, stale, start_epoch, history = (math.inf,math.inf), 0, 0, []
    latest = folder/'last.pt'
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity'] != identity:
            raise RuntimeError('Resume checkpoint identity mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_key, stale, history = state['epoch']+1, tuple(state['best_key']), state['stale'], state['history']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        del state
    for epoch in range(start_epoch,config['epochs']):
        if stale >= config['patience']:
            break
        check_deadline(config)
        begun = time.time(); train.epoch = epoch
        sampler = WeightedRandomSampler(training_weights(usable),len(train),replacement=True,
                                        generator=torch.Generator().manual_seed(config['seed']+epoch))
        loader = batch_loader(train, config, sampler)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for index,(bags,labels) in enumerate(loader):
            check_deadline(config)
            weight = batch_loss_weight(index, len(bags), len(train), config)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = model(bags.cuda(non_blocking=True))
                loss = F.cross_entropy(logits.float(),labels.cuda(non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            (loss*weight).backward()
            if index == 0 and epoch == start_epoch:
                save_json(folder/'gradient_connectivity.json',gradient_evidence(model))
            loss_sum += float(loss.detach()) * len(bags)
            if (index+1)%config['accumulation']==0 or index+1==len(loader):
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        probabilities = predict(model,tune,config)
        metrics = music_report(split['tune'],probabilities)
        key = (metrics['eer'],metrics['balanced_bce'])
        if key < best_key:
            best_key,stale = key,0
            atomic_checkpoint(folder/'best.pt', {'model':model.state_dict(),'epoch':epoch,'identity':identity})
            save_json(folder/'selection.json', {'epoch':epoch+1, 'metrics':metrics,'identity':identity,'test_used':False})
            save_json(folder/'tune_predictions.json',[{'ID':r['ID'],'probability':p} for r,p in zip(split['tune'],probabilities)])
        else:
            stale += 1
        entry = {'epoch':epoch+1,'train_loss':loss_sum/len(train),'tune':metrics,
                 'seconds':time.time()-begun,'learning_rates':scheduler.get_last_lr()}
        history.append(entry); save_json(folder/'history.json',history)
        atomic_checkpoint(latest, {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'epoch':epoch,'identity':identity,'best_key':best_key,
            'stale':stale,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
        print(trial.name,json.dumps(entry),flush=True)
    save_json(folder/'trained.json', {'identity':identity,'epochs':len(history),
                                    'status':'early_stopped' if stale>=config['patience'] else 'max_epochs'})
    # The checkpoint is sealed before any evaluation. No test-based hyperparameter changes.
    state = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); del state
    test = AudioBags(split['test'],config,False,feature_dir)
    probabilities = predict(model,test,config)
    report = {'overall':music_report(split['test'],probabilities), 'subsets':{}, 'official_score':None}
    held = {'S1':'musicgen','S2':'audioldm'}.get(trial.split)
    names = sorted({r['panel'] for r in split['test']})
    subsets = {name:[i for i,r in enumerate(split['test']) if r['panel']==name] for name in names}
    if held:
        subsets['unseen_generator_plus_real'] = [i for i,r in enumerate(split['test']) if r['MUSIC_FAKE']=='0' or family(r)==held]
        subsets['seen_generators_plus_real'] = [i for i,r in enumerate(split['test']) if family(r)!=held]
    else:
        for panel in ['music_seen','music_ood']:
            subsets[panel+'_plus_real'] = [i for i,r in enumerate(split['test']) if r['panel'] in (panel,'music_real')]
    for name, indices in subsets.items():
        report['subsets'][name] = music_report([split['test'][i] for i in indices],[probabilities[i] for i in indices])
    save_json(folder/'test_predictions.json',[{'ID':r['ID'],'probability':p} for r,p in zip(split['test'],probabilities)])
    if trial.split == 'S0':
        baseline = {r['ID']:r for r in config['baseline_predictions']}
        labels, file_scores = [], []
        for row,p in zip(split['test'],probabilities):
            if row['ID'] not in baseline:
                raise RuntimeError('Missing fixed voice/presence prediction')
            fixed = baseline[row['ID']]
            if row['FILE_FAKE'] in ('0','1'):
                labels.append(int(row['FILE_FAKE']))
                file_scores.append(max(fixed['VOICE_PRESENT_PROB']*fixed['VOICE_FAKE_PROB'],fixed['MUSIC_PRESENT_PROB']*p))
        report['known_file_eer'],report['known_file_auc'] = eer_auc(labels,file_scores)
        report['voice_presence'] = 'Fixed historical predictions; voice branch unchanged. No official total.'
    save_json(folder/'evaluation.json',report)
    save_json(folder/'complete.json',{'status':'complete','identity':identity,'finished_at':time.time(),
                                    'best_checkpoint_sha256':file_hash(folder/'best.pt')})
    del model,optimizer
    torch.cuda.empty_cache()

import types, sys
separator_module = types.ModuleType('fullft_separator')
sys.modules['fullft_separator'] = separator_module
SEPARATOR_CODE_SHA256 = '678d43b46680da6f04049dc6e65216621b51febad17a3d0c8e26b90bb7f98078'
exec(compile('"""Fixed-teacher HTDemucs remix adaptation, inspired by (not full) RemixIT.\n\nPseudo-validation measures agreement with the fixed teacher, not separation truth.\nTeacher errors persist; no voice, classification, or official-score claim follows.\n"""\n# noqa: SIZE_OK - standalone notebook-embedded adaptation module; no external helper dependency.\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import NotRequired, TypedDict\nimport hashlib\nimport json\nimport random\nimport time\n\nimport numpy as np\nimport soundfile as sf\nimport torch\nfrom torch import nn\n\n\nclass SeparatorConfig(TypedDict):\n    project_root: str\n    output: str\n    model_root: str\n    deadline: float\n    seed: int\n    separator_steps: NotRequired[int]\n    separator_lr: NotRequired[float]\n    separator_seconds: NotRequired[float]\n    separator_validation_every: NotRequired[int]\n    separator_code_sha256: NotRequired[str]\n\n\nclass SeparatorError(RuntimeError):\n    """Invalid source, experiment state, or unsafe separator computation."""\n\n\nclass SourceRow(TypedDict):\n    ID: str\n    path: str\n    group_id: str\n    MUSIC_PRESENT: str\n    MUSIC_FAKE: str\n    audio_sha256: str\n    pcm_sha256: str\n\n\n@dataclass(frozen=True, slots=True)\nclass StemClip:\n    identifier: str\n    parents: frozenset[str]\n    path: Path\n\n\ndef digest(path: Path) -> str:\n    with path.open(\'rb\') as stream:\n        return hashlib.file_digest(stream, \'sha256\').hexdigest()\n\n\ndef atomic(path: Path, payload) -> None:\n    temporary = path.with_suffix(path.suffix + \'.partial\')\n    if path.suffix == \'.json\':\n        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))\n    else:\n        torch.save(payload, temporary)\n    temporary.replace(path)\n\n\ndef deadline(config: SeparatorConfig) -> None:\n    if time.time() >= config[\'deadline\']:\n        raise TimeoutError(\'Separator adaptation deadline reached; no incomplete checkpoint is selected.\')\n\n\ndef remix(clips: list[StemClip], rng: random.Random) -> tuple[torch.Tensor, list[StemClip]]:\n    """Four sources from four disjoint original parent sets, with bounded mixture peak."""\n    chosen, parents = [], set()\n    for clip in rng.sample(clips, len(clips)):\n        if not parents.intersection(clip.parents):\n            chosen.append(clip)\n            parents.update(clip.parents)\n        if len(chosen) == 4:\n            break\n    if len(chosen) != 4:\n        raise SeparatorError(\'Four parent-disjoint accepted crops required per remix\')\n    targets = torch.stack([torch.load(clip.path, weights_only=True)[index] for index, clip in enumerate(chosen)])\n    targets = targets * min(1., .95 / max(float(targets.sum(0).abs().max()), 1e-8))\n    return targets[None], chosen\n\n\ndef forward_stems(model: nn.Module, mixture: torch.Tensor) -> torch.Tensor:\n    """Differentiable BagOfModels equivalent; apply_model disables gradients."""\n    mean, std = mixture.mean(), mixture.mean(1).std().clamp_min(1e-5)\n    normalized = (mixture - mean) / std\n    members = getattr(model, \'models\', [model])\n    weights = getattr(model, \'weights\', [[1.] * 4])\n    denominator = torch.tensor(weights, device=mixture.device).sum(0)[None, :, None, None]\n    predictions = [member(normalized) * torch.tensor(weight, device=mixture.device)[None, :, None, None]\n                   for member, weight in zip(members, weights)]\n    return sum(predictions) / denominator * std + mean\n\n\ndef fit_remixes(model: nn.Module, train: list[StemClip], validation: list[StemClip], config: SeparatorConfig) -> Path:\n    """Optimize all parameters; select solely on fixed held-parent pseudo-remixes."""\n    if set().union(*(clip.parents for clip in train)).intersection(set().union(*(clip.parents for clip in validation))):\n        raise SeparatorError(\'Pseudo-validation shares original parents with optimization\')\n    folder = Path(config[\'output\']) / \'separator_adaptation\'\n    folder.mkdir(parents=True, exist_ok=True)\n    torch.manual_seed(config[\'seed\'])\n    rng = random.Random(config[\'seed\'])\n    held = [remix(validation, random.Random(config[\'seed\'] + 1000 + i)) for i in range(4)]\n    lineage = [{\'ids\': [clip.identifier for clip in chosen], \'parents\': [\'|\'.join(sorted(clip.parents)) for clip in chosen]} for _, chosen in held]\n    device = next(model.parameters()).device\n    model.requires_grad_(True)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get(\'separator_lr\', 1e-5), weight_decay=0.)\n    original = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}\n    coverage = sum(parameter.numel() for group in optimizer.param_groups for parameter in group[\'params\'])\n    evidence = {\'optimizer_parameter_count\': coverage, \'model_parameter_count\': sum(p.numel() for p in model.parameters())}\n    if {id(p) for p in model.parameters()} != {id(p) for group in optimizer.param_groups for p in group[\'params\']}:\n        raise SeparatorError(\'Optimizer omits separator parameters\')\n    history, best = [], float(\'inf\')\n    steps = config.get(\'separator_steps\', 400)\n    interval = config.get(\'separator_validation_every\', 25)\n    if steps < 1 or interval < 1:\n        raise SeparatorError(\'Positive separator steps and validation interval required\')\n    for step in range(steps + 1):\n        deadline(config)\n        if step:\n            model.train()\n            targets, _ = remix(train, rng)\n            targets = targets.to(device)\n            optimizer.zero_grad(set_to_none=True)\n            prediction = forward_stems(model, targets.sum(1))\n            loss = (prediction - targets).abs().mean()\n            if not torch.isfinite(loss):\n                raise SeparatorError(\'Nonfinite separator loss\')\n            loss.backward()\n            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)\n            if step == 1:\n                evidence[\'nonzero_gradient_parameters\'] = sum(p.numel() for p in model.parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0))\n                evidence[\'disconnected_parameters\'] = [name for name, p in model.named_parameters() if p.grad is None]\n                branches = (\'encoder\', \'decoder\', \'tencoder\', \'tdecoder\', \'crosstransformer\')\n                evidence[\'branch_gradient_l1\'] = {f\'{index}.{branch}\': sum(float(p.grad.abs().sum()) for name, p in member.named_parameters() if name.startswith(branch + \'.\') and p.grad is not None)\n                    for index, member in enumerate(getattr(model, \'models\', [model])) for branch in branches if hasattr(member, branch)}\n                if evidence[\'disconnected_parameters\'] or not evidence[\'nonzero_gradient_parameters\'] or any(value <= 0 for value in evidence[\'branch_gradient_l1\'].values()):\n                    raise SeparatorError(f\'Separator gradient gate failed: {evidence}\')\n            optimizer.step()\n        if step % interval and step != steps:\n            continue\n        model.eval()\n        with torch.no_grad():\n            losses = [(forward_stems(model, target.to(device).sum(1)) - target.to(device)).abs().mean().item() for target, _ in held]\n        measured = sum(losses) / len(losses)\n        if not np.isfinite(measured):\n            raise SeparatorError(\'Nonfinite pseudo-validation loss\')\n        history.append({\'step\': step, \'pseudo_validation_l1\': measured})\n        atomic(folder / \'history.json\', history)\n        if step and measured < best:\n            best = measured\n            evidence[\'parameter_update_l2\'] = sum(float((p.detach().cpu() - original[name]).square().sum()) for name, p in model.named_parameters()) ** .5\n            if not evidence[\'parameter_update_l2\']:\n                raise SeparatorError(\'Optimizer produced no separator parameter update\')\n            atomic(folder / \'best.pt\', {\'model\': model.state_dict(), \'step\': step, \'pseudo_validation_l1\': measured,\n                                       \'history\': history, \'gradient_evidence\': evidence, \'validation_lineage\': lineage})\n        print(\'SEPARATOR_ADAPTATION\', json.dumps(history[-1]), flush=True)\n    atomic(folder / \'gradient_evidence.json\', evidence)\n    return folder / \'best.pt\'\n\n\ndef adapt_separator(rows: list[SourceRow], config: SeparatorConfig) -> Path:\n    """Caller supplies only S0 TRAIN after train/tune/test lineage and hash audit."""\n    from demucs.apply import apply_model\n    from demucs.pretrained import get_model\n    import torchaudio\n    folder = Path(config[\'output\']) / \'separator_adaptation\'\n    folder.mkdir(parents=True, exist_ok=True)\n    eligible = [row for row in rows if row[\'MUSIC_PRESENT\'] == \'1\' and row[\'MUSIC_FAKE\'] in (\'0\', \'1\')]\n    seconds = config.get(\'separator_seconds\', 5.)\n    if not 1 <= seconds <= 7.8:\n        raise SeparatorError(\'Separator crop must be between 1 and 7.8 seconds\')\n    repository = Path(config[\'model_root\']) / \'htdemucs\'\n    identity_payload = {\'rows\': eligible, \'settings\': {key: value for key, value in config.items() if key.startswith(\'separator_\') or key == \'seed\'},\n                        \'models\': {str(p.relative_to(repository)): digest(p) for p in sorted(repository.rglob(\'*\')) if p.is_file()}, \'module\': config[\'separator_code_sha256\'] if \'separator_code_sha256\' in config else digest(Path(__file__))}\n    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()\n    meta, complete = folder / \'identity.json\', folder / \'complete.json\'\n    if meta.exists() and json.loads(meta.read_text())[\'identity\'] != identity:\n        raise SeparatorError(\'Separator identity changed; use a new output directory\')\n    if complete.exists():\n        record = json.loads(complete.read_text())\n        if record[\'identity\'] != identity or digest(folder / \'best.pt\') != record[\'sha256\']:\n            raise SeparatorError(\'Completed separator checkpoint integrity mismatch\')\n        return folder / \'best.pt\'\n    if (folder / \'training_started.json\').exists():\n        raise SeparatorError(\'Interrupted separator optimization cannot resume exactly; use a new output directory. Teacher cache alone is resumable.\')\n    atomic(meta, {\'identity\': identity, **identity_payload, \'method\': \'fixed-teacher remix adaptation inspired by RemixIT; teacher errors retained\', \'selection\': \'held S0 train parent groups only\'})\n    deadline(config)\n    original_load = torch.load\n    def trusted_load(*args, **kwargs):\n        kwargs.setdefault(\'weights_only\', False)\n        return original_load(*args, **kwargs)\n    torch.load = trusted_load\n    try:\n        teacher = get_model(\'htdemucs\', repo=repository).eval().requires_grad_(False).cuda()\n    finally:\n        torch.load = original_load\n    if list(teacher.sources) != [\'drums\', \'bass\', \'other\', \'vocals\'] or teacher.samplerate != 44100 or teacher.audio_channels != 2:\n        raise SeparatorError(\'Expected native four-stem 44.1 kHz stereo HTDemucs\')\n    cache = folder / \'teacher_cache\'\n    cache.mkdir(exist_ok=True)\n    accepted, rejected, quality = [], [], []\n    for row in eligible:\n        deadline(config)\n        path = Path(config[\'project_root\']) / \'v4_validation\' / row[\'path\'].split(\'/v4_validation/\')[-1]\n        if digest(path) != row[\'audio_sha256\']:\n            raise SeparatorError(f\'Source file hash changed: {row["ID"]}\')\n        wave, rate = sf.read(path, dtype=\'float32\')\n        if rate != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or hashlib.sha256(wave.astype(\'<f4\').tobytes()).hexdigest() != row[\'pcm_sha256\']:\n            raise SeparatorError(f\'Invalid PCM/hash: {row["ID"]}\')\n        key = hashlib.sha256(row[\'ID\'].encode()).hexdigest()\n        target, record = cache / f\'{key}.pt\', cache / f\'{key}.json\'\n        parents = frozenset(row[\'group_id\'].split(\'|\'))\n        if not parents or \'\' in parents:\n            raise SeparatorError(\'Missing original parent lineage\')\n        if record.exists():\n            prior = json.loads(record.read_text())\n            if prior[\'identity\'] != identity or prior[\'sha256\'] != digest(target):\n                raise SeparatorError(\'Teacher crop cache changed\')\n            accepted.append(StemClip(row[\'ID\'], parents, target))\n            quality.append(prior)\n            continue\n        width = int(seconds * rate)\n        start = max(0, (len(wave) - width) // 2)\n        crop = torch.from_numpy(wave[start:start + width].copy()).cuda()\n        crop = torch.nn.functional.pad(crop, (0, max(0, width - len(crop))))\n        crop = torchaudio.functional.resample(crop[None].repeat(2, 1), rate, 44100)\n        mean, std = crop.mean(), crop.mean(0).std()\n        if std < 1e-5 or crop.abs().max() > 1.001 or (crop.abs() >= .999).float().mean() > .01:\n            rejected.append({\'ID\': row[\'ID\'], \'reason\': \'silent_or_clipped_source\'})\n            continue\n        with torch.inference_mode():\n            stems = apply_model(teacher, ((crop - mean) / std)[None], shifts=0, split=True, overlap=.25, device=\'cuda\')[0] * std + mean\n        rms = stems.square().mean((1, 2)).sqrt()\n        backing_rms = stems[:3].sum(0).square().mean().sqrt()\n        if not torch.isfinite(stems).all() or stems.abs().max() > 2 or rms.max() > 20 * std or backing_rms < 1e-5:\n            rejected.append({\'ID\': row[\'ID\'], \'reason\': \'nonfinite_silent_backing_or_huge_gain\'})\n            continue\n        atomic(target, stems.cpu().clone())\n        entry = {\'ID\': row[\'ID\'], \'identity\': identity, \'sha256\': digest(target), \'rms\': rms.cpu().tolist(), \'absent_stems\': [name for name, energy in zip(teacher.sources, rms) if energy < 1e-5], \'start_16k\': start}\n        atomic(record, entry)\n        quality.append(entry)\n        accepted.append(StemClip(row[\'ID\'], parents, target))\n    del teacher\n    torch.cuda.empty_cache()\n    # Connected components prevent compound parent IDs leaking into pseudo-validation.\n    components = []\n    for clip in accepted:\n        connected = [group for group in components if group.intersection(clip.parents)]\n        merged = set(clip.parents).union(*connected)\n        components = [group for group in components if group not in connected] + [merged]\n    components.sort(key=lambda group: hashlib.sha256((str(config[\'seed\']) + \'|\'.join(sorted(group))).encode()).hexdigest())\n    if len(components) < 8:\n        raise SeparatorError(\'Need at least eight independent accepted parent groups for four-source train/validation remixes\')\n    held = set().union(*components[:max(4, round(len(components) * .1))])\n    validation = [clip for clip in accepted if clip.parents.intersection(held)]\n    train = [clip for clip in accepted if not clip.parents.intersection(held)]\n    atomic(folder / \'provenance.json\', {\'train_ids\': [c.identifier for c in train], \'validation_ids\': [c.identifier for c in validation], \'rejected\': rejected, \'quality\': quality})\n    torch.load = trusted_load\n    try:\n        student = get_model(\'htdemucs\', repo=repository).float().cuda()\n    finally:\n        torch.load = original_load\n    atomic(folder / \'training_started.json\', {\'identity\': identity, \'time\': time.time()})\n    result = fit_remixes(student, train, validation, config)\n    atomic(complete, {\'identity\': identity, \'sha256\': digest(result), \'selection_uses_test\': False})\n    del student\n    torch.cuda.empty_cache()\n    return result\n', 'fullft_separator.py', 'exec', dont_inherit=True), separator_module.__dict__)


def worker_main() -> None:
    config = json.loads(Path(os.environ['FULLFT_CONFIG']).read_text())
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('An H100 worker is required; no CPU/MPS training fallback')
    splits = config['splits']
    stage = os.environ['FULLFT_STAGE']
    try:
        match stage:
            case 'prepare':
                rows = {r['ID']:r for split in splits.values() for rows in split.values() for r in rows}
                prepare_audio([rows[k] for k in sorted(rows)],config)
            case 'separator':
                separator_module.adapt_separator(splits['S0']['train'], config)
            case 'train':
                trial = Trial(**config['extra_trial']) if config.get('extra_trial') else next(t for t in TRIALS if t.name==os.environ['FULLFT_TRIAL'])
                save_json(Path(config['output'])/trial.name/'status.json',{'status':'running','started_at':time.time()})
                train_trial(trial,splits,config)
            case unknown:
                raise RuntimeError(f'Unknown worker stage: {unknown}')
    except TimeoutError as error:
        target = Path(config['output'])/os.environ.get('FULLFT_TRIAL','preparation')/'status.json'
        save_json(target,{'status':'budget_stopped','reason':str(error)})
        raise

if __name__ == '__main__' and os.environ.get('FULLFT_WORKER') == '1':
    worker_main()