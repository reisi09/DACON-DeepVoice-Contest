"""Fixed-teacher HTDemucs remix adaptation, inspired by (not full) RemixIT.

Pseudo-validation measures agreement with the fixed teacher, not separation truth.
Teacher errors persist; no voice, classification, or official-score claim follows.
"""
# noqa: SIZE_OK - standalone notebook-embedded adaptation module; no external helper dependency.
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict
import hashlib
import json
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn


class SeparatorConfig(TypedDict):
    project_root: str
    output: str
    model_root: str
    deadline: float
    seed: int
    separator_steps: NotRequired[int]
    separator_lr: NotRequired[float]
    separator_seconds: NotRequired[float]
    separator_validation_every: NotRequired[int]
    separator_code_sha256: NotRequired[str]


class SeparatorError(RuntimeError):
    """Invalid source, experiment state, or unsafe separator computation."""


class SourceRow(TypedDict):
    ID: str
    path: str
    group_id: str
    MUSIC_PRESENT: str
    MUSIC_FAKE: str
    audio_sha256: str
    pcm_sha256: str


@dataclass(frozen=True, slots=True)
class StemClip:
    identifier: str
    parents: frozenset[str]
    path: Path


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    if path.suffix == '.json':
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    else:
        torch.save(payload, temporary)
    temporary.replace(path)


def deadline(config: SeparatorConfig) -> None:
    if time.time() >= config['deadline']:
        raise TimeoutError('Separator adaptation deadline reached; no incomplete checkpoint is selected.')


def remix(clips: list[StemClip], rng: random.Random) -> tuple[torch.Tensor, list[StemClip]]:
    """Four sources from four disjoint original parent sets, with bounded mixture peak."""
    chosen, parents = [], set()
    for clip in rng.sample(clips, len(clips)):
        if not parents.intersection(clip.parents):
            chosen.append(clip)
            parents.update(clip.parents)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise SeparatorError('Four parent-disjoint accepted crops required per remix')
    targets = torch.stack([torch.load(clip.path, weights_only=True)[index] for index, clip in enumerate(chosen)])
    targets = targets * min(1., .95 / max(float(targets.sum(0).abs().max()), 1e-8))
    return targets[None], chosen


def forward_stems(model: nn.Module, mixture: torch.Tensor) -> torch.Tensor:
    """Differentiable BagOfModels equivalent; apply_model disables gradients."""
    mean, std = mixture.mean(), mixture.mean(1).std().clamp_min(1e-5)
    normalized = (mixture - mean) / std
    members = getattr(model, 'models', [model])
    weights = getattr(model, 'weights', [[1.] * 4])
    denominator = torch.tensor(weights, device=mixture.device).sum(0)[None, :, None, None]
    predictions = [member(normalized) * torch.tensor(weight, device=mixture.device)[None, :, None, None]
                   for member, weight in zip(members, weights)]
    return sum(predictions) / denominator * std + mean


def fit_remixes(model: nn.Module, train: list[StemClip], validation: list[StemClip], config: SeparatorConfig) -> Path:
    """Optimize all parameters; select solely on fixed held-parent pseudo-remixes."""
    if set().union(*(clip.parents for clip in train)).intersection(set().union(*(clip.parents for clip in validation))):
        raise SeparatorError('Pseudo-validation shares original parents with optimization')
    folder = Path(config['output']) / 'separator_adaptation'
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    rng = random.Random(config['seed'])
    held = [remix(validation, random.Random(config['seed'] + 1000 + i)) for i in range(4)]
    lineage = [{'ids': [clip.identifier for clip in chosen], 'parents': ['|'.join(sorted(clip.parents)) for clip in chosen]} for _, chosen in held]
    device = next(model.parameters()).device
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('separator_lr', 1e-5), weight_decay=0.)
    original = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    coverage = sum(parameter.numel() for group in optimizer.param_groups for parameter in group['params'])
    evidence = {'optimizer_parameter_count': coverage, 'model_parameter_count': sum(p.numel() for p in model.parameters())}
    if {id(p) for p in model.parameters()} != {id(p) for group in optimizer.param_groups for p in group['params']}:
        raise SeparatorError('Optimizer omits separator parameters')
    history, best = [], float('inf')
    steps = config.get('separator_steps', 400)
    interval = config.get('separator_validation_every', 25)
    if steps < 1 or interval < 1:
        raise SeparatorError('Positive separator steps and validation interval required')
    for step in range(steps + 1):
        deadline(config)
        if step:
            model.train()
            targets, _ = remix(train, rng)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_stems(model, targets.sum(1))
            loss = (prediction - targets).abs().mean()
            if not torch.isfinite(loss):
                raise SeparatorError('Nonfinite separator loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1:
                evidence['nonzero_gradient_parameters'] = sum(p.numel() for p in model.parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0))
                evidence['disconnected_parameters'] = [name for name, p in model.named_parameters() if p.grad is None]
                branches = ('encoder', 'decoder', 'tencoder', 'tdecoder', 'crosstransformer')
                evidence['branch_gradient_l1'] = {f'{index}.{branch}': sum(float(p.grad.abs().sum()) for name, p in member.named_parameters() if name.startswith(branch + '.') and p.grad is not None)
                    for index, member in enumerate(getattr(model, 'models', [model])) for branch in branches if hasattr(member, branch)}
                if evidence['disconnected_parameters'] or not evidence['nonzero_gradient_parameters'] or any(value <= 0 for value in evidence['branch_gradient_l1'].values()):
                    raise SeparatorError(f'Separator gradient gate failed: {evidence}')
            optimizer.step()
        if step % interval and step != steps:
            continue
        model.eval()
        with torch.no_grad():
            losses = [(forward_stems(model, target.to(device).sum(1)) - target.to(device)).abs().mean().item() for target, _ in held]
        measured = sum(losses) / len(losses)
        if not np.isfinite(measured):
            raise SeparatorError('Nonfinite pseudo-validation loss')
        history.append({'step': step, 'pseudo_validation_l1': measured})
        atomic(folder / 'history.json', history)
        if step and measured < best:
            best = measured
            evidence['parameter_update_l2'] = sum(float((p.detach().cpu() - original[name]).square().sum()) for name, p in model.named_parameters()) ** .5
            if not evidence['parameter_update_l2']:
                raise SeparatorError('Optimizer produced no separator parameter update')
            atomic(folder / 'best.pt', {'model': model.state_dict(), 'step': step, 'pseudo_validation_l1': measured,
                                       'history': history, 'gradient_evidence': evidence, 'validation_lineage': lineage})
        print('SEPARATOR_ADAPTATION', json.dumps(history[-1]), flush=True)
    atomic(folder / 'gradient_evidence.json', evidence)
    return folder / 'best.pt'


def adapt_separator(rows: list[SourceRow], config: SeparatorConfig) -> Path:
    """Caller supplies only S0 TRAIN after train/tune/test lineage and hash audit."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import torchaudio
    folder = Path(config['output']) / 'separator_adaptation'
    folder.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in rows if row['MUSIC_PRESENT'] == '1' and row['MUSIC_FAKE'] in ('0', '1')]
    seconds = config.get('separator_seconds', 5.)
    if not 1 <= seconds <= 7.8:
        raise SeparatorError('Separator crop must be between 1 and 7.8 seconds')
    repository = Path(config['model_root']) / 'htdemucs'
    identity_payload = {'rows': eligible, 'settings': {key: value for key, value in config.items() if key.startswith('separator_') or key == 'seed'},
                        'models': {str(p.relative_to(repository)): digest(p) for p in sorted(repository.rglob('*')) if p.is_file()}, 'module': config['separator_code_sha256'] if 'separator_code_sha256' in config else digest(Path(__file__))}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    meta, complete = folder / 'identity.json', folder / 'complete.json'
    if meta.exists() and json.loads(meta.read_text())['identity'] != identity:
        raise SeparatorError('Separator identity changed; use a new output directory')
    if complete.exists():
        record = json.loads(complete.read_text())
        if record['identity'] != identity or digest(folder / 'best.pt') != record['sha256']:
            raise SeparatorError('Completed separator checkpoint integrity mismatch')
        return folder / 'best.pt'
    if (folder / 'training_started.json').exists():
        raise SeparatorError('Interrupted separator optimization cannot resume exactly; use a new output directory. Teacher cache alone is resumable.')
    atomic(meta, {'identity': identity, **identity_payload, 'method': 'fixed-teacher remix adaptation inspired by RemixIT; teacher errors retained', 'selection': 'held S0 train parent groups only'})
    deadline(config)
    original_load = torch.load
    def trusted_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = trusted_load
    try:
        teacher = get_model('htdemucs', repo=repository).eval().requires_grad_(False).cuda()
    finally:
        torch.load = original_load
    if list(teacher.sources) != ['drums', 'bass', 'other', 'vocals'] or teacher.samplerate != 44100 or teacher.audio_channels != 2:
        raise SeparatorError('Expected native four-stem 44.1 kHz stereo HTDemucs')
    cache = folder / 'teacher_cache'
    cache.mkdir(exist_ok=True)
    accepted, rejected, quality = [], [], []
    for row in eligible:
        deadline(config)
        path = Path(config['project_root']) / 'v4_validation' / row['path'].split('/v4_validation/')[-1]
        if digest(path) != row['audio_sha256']:
            raise SeparatorError(f'Source file hash changed: {row["ID"]}')
        wave, rate = sf.read(path, dtype='float32')
        if rate != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest() != row['pcm_sha256']:
            raise SeparatorError(f'Invalid PCM/hash: {row["ID"]}')
        key = hashlib.sha256(row['ID'].encode()).hexdigest()
        target, record = cache / f'{key}.pt', cache / f'{key}.json'
        parents = frozenset(row['group_id'].split('|'))
        if not parents or '' in parents:
            raise SeparatorError('Missing original parent lineage')
        if record.exists():
            prior = json.loads(record.read_text())
            if prior['identity'] != identity or prior['sha256'] != digest(target):
                raise SeparatorError('Teacher crop cache changed')
            accepted.append(StemClip(row['ID'], parents, target))
            quality.append(prior)
            continue
        width = int(seconds * rate)
        start = max(0, (len(wave) - width) // 2)
        crop = torch.from_numpy(wave[start:start + width].copy()).cuda()
        crop = torch.nn.functional.pad(crop, (0, max(0, width - len(crop))))
        crop = torchaudio.functional.resample(crop[None].repeat(2, 1), rate, 44100)
        mean, std = crop.mean(), crop.mean(0).std()
        if std < 1e-5 or crop.abs().max() > 1.001 or (crop.abs() >= .999).float().mean() > .01:
            rejected.append({'ID': row['ID'], 'reason': 'silent_or_clipped_source'})
            continue
        with torch.inference_mode():
            stems = apply_model(teacher, ((crop - mean) / std)[None], shifts=0, split=True, overlap=.25, device='cuda')[0] * std + mean
        rms = stems.square().mean((1, 2)).sqrt()
        backing_rms = stems[:3].sum(0).square().mean().sqrt()
        if not torch.isfinite(stems).all() or stems.abs().max() > 2 or rms.max() > 20 * std or backing_rms < 1e-5:
            rejected.append({'ID': row['ID'], 'reason': 'nonfinite_silent_backing_or_huge_gain'})
            continue
        atomic(target, stems.cpu().clone())
        entry = {'ID': row['ID'], 'identity': identity, 'sha256': digest(target), 'rms': rms.cpu().tolist(), 'absent_stems': [name for name, energy in zip(teacher.sources, rms) if energy < 1e-5], 'start_16k': start}
        atomic(record, entry)
        quality.append(entry)
        accepted.append(StemClip(row['ID'], parents, target))
    del teacher
    torch.cuda.empty_cache()
    # Connected components prevent compound parent IDs leaking into pseudo-validation.
    components = []
    for clip in accepted:
        connected = [group for group in components if group.intersection(clip.parents)]
        merged = set(clip.parents).union(*connected)
        components = [group for group in components if group not in connected] + [merged]
    components.sort(key=lambda group: hashlib.sha256((str(config['seed']) + '|'.join(sorted(group))).encode()).hexdigest())
    if len(components) < 8:
        raise SeparatorError('Need at least eight independent accepted parent groups for four-source train/validation remixes')
    held = set().union(*components[:max(4, round(len(components) * .1))])
    validation = [clip for clip in accepted if clip.parents.intersection(held)]
    train = [clip for clip in accepted if not clip.parents.intersection(held)]
    atomic(folder / 'provenance.json', {'train_ids': [c.identifier for c in train], 'validation_ids': [c.identifier for c in validation], 'rejected': rejected, 'quality': quality})
    torch.load = trusted_load
    try:
        student = get_model('htdemucs', repo=repository).float().cuda()
    finally:
        torch.load = original_load
    atomic(folder / 'training_started.json', {'identity': identity, 'time': time.time()})
    result = fit_remixes(student, train, validation, config)
    atomic(complete, {'identity': identity, 'sha256': digest(result), 'selection_uses_test': False})
    del student
    torch.cuda.empty_cache()
    return result
