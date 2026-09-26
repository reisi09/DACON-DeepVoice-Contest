"""Offline PANNs suffix adaptation over the preserved DataV4 submission engine."""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import soundfile as sf
import torch

import full_bank_submission as frozen
from mert12_submission import SubmissionError, sha256
from panns_adaptation import AdaptedCnn14
from panns_submission_selection import Selection


def apply_adaptation(base: torch.nn.Module, state: Mapping[str, torch.Tensor], arm: str) -> AdaptedCnn14:
    """Load exactly the learned suffix tensors, leaving the frozen prefix and buffers intact."""
    if arm not in ('R1', 'R3'):
        raise SubmissionError('Only R1 and R3 are admitted for the submission')
    model = AdaptedCnn14(base, channel_se=arm == 'R3', train_blocks=(6,))
    expected = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(state) != set(expected):
        raise SubmissionError('Selected PANNs trainable parameter set mismatch')
    for name, param in expected.items():
        value = state[name]
        if value.shape != param.shape or value.dtype != param.dtype or not torch.isfinite(value).all():
            raise SubmissionError('Invalid selected PANNs tensor: '+name)
    with torch.no_grad():
        for name, param in expected.items():
            param.copy_(state[name])
    return model.to(next(base.parameters()).device).eval().requires_grad_(False)


class PannsTagger:
    """Use the same float32 prefix and four-segment suffix batches as the cache consumer."""
    def __init__(self, model: AdaptedCnn14):
        self.model = model

    @torch.inference_mode()
    def inference(self, audio: np.ndarray) -> tuple[np.ndarray, None]:
        device = next(self.model.parameters()).device
        waveform = torch.from_numpy(audio)
        tags = torch.cat([self.model(part.to(device)).cpu() for part in waveform.split(4)])
        return tags.numpy(), None


def load_normalized_wav(path: Path, channels: int, sample_rate: int) -> torch.Tensor:
    """Match the FLOAT WAV decoder used to produce the training stems."""
    from demucs.audio import convert_audio
    with sf.SoundFile(path) as source:
        if (source.format, source.subtype, source.channels, source.samplerate) != ('WAV', 'FLOAT', 1, 16000):
            raise SubmissionError('Expected normalized mono 16 kHz FLOAT WAV')
        samples = source.read(dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise SubmissionError('Invalid normalized audio')
    return convert_audio(torch.from_numpy(samples.T.copy()), 16000, sample_rate, channels)


def pin_normalized_wav_loader(base: ModuleType) -> None:
    base.load_track = load_normalized_wav


class Engine(frozen.Engine):
    def __init__(self, assets: Path, trained: Path, device: str | torch.device, batch: int = 8):
        selection = Selection.read(trained/'panns_selection.json')
        weights = trained/'panns_best_model.pt'
        if sha256(weights) != selection.weights_sha256:
            raise SubmissionError('Selected PANNs weights changed')
        state = torch.load(weights, map_location='cpu', weights_only=True)
        original_base, original_decoder = frozen.baseline_module, frozen.pin_torchaudio_loader
        original_load = torch.load

        def safe_base(root: Path) -> ModuleType:
            base = original_base(root)

            def adapted_panns(target: torch.device):
                labels = Path.home()/'panns_data/class_labels_indices.csv'
                if not labels.exists():
                    base.prepare_panns_labels()
                if sha256(labels) != sha256(base.PANNS_DIR/'class_labels_indices.csv'):
                    raise SubmissionError('Offline PANNs label table differs from the package')
                from panns_inference.models import Cnn14
                cnn = Cnn14(32000, 1024, 320, 64, 50, 14000, 527)
                record = original_load(base.PANNS_DIR/'Cnn14_mAP=0.431.pth', map_location='cpu', weights_only=True)
                cnn.load_state_dict(record['model'], strict=True)
                return PannsTagger(apply_adaptation(cnn, state, selection.arm).to(target)), [], []

            base.load_panns_model = adapted_panns
            return base

        def restricted_load(*args, **kwargs):
            kwargs['weights_only'] = True
            return original_load(*args, **kwargs)

        from demucs.htdemucs import HTDemucs
        allowed = [HTDemucs, np.dtype, Fraction,
                   (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), type(np.dtype('float64'))]
        frozen.baseline_module, frozen.pin_torchaudio_loader = safe_base, pin_normalized_wav_loader
        torch.load = restricted_load
        try:
            with torch.serialization.safe_globals(allowed):
                super().__init__(assets, trained, device, batch)
        finally:
            torch.load = original_load
            frozen.baseline_module, frozen.pin_torchaudio_loader = original_base, original_decoder
        if self.presence_heads is None or self.ensemble is None:
            raise SubmissionError('Adaptation requires the preserved presence and FILE readouts')
        self.selection = selection


def run(root: Path) -> None:
    """Retain the verified CSV entry point and fake heads, substituting only PANNs."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    selection = Selection.read(root/'model/trained/panns_selection.json')
    original, candidate = frozen.Engine, frozen.CANDIDATE
    frozen.Engine = Engine
    frozen.CANDIDATE = f'panns_{selection.arm}_seed41_DataV4'
    try:
        frozen.run(root)
    finally:
        frozen.Engine = original
        frozen.CANDIDATE = candidate
