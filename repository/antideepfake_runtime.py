"""Pinned AntiDeepfake weights on the existing Transformers runtime.

Conversion must pass native Fairseq logits on real audio before GPU admission.
"""
from pathlib import Path
import json
import re
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model

REVISION = '60c96543210117bdb256636e7961e16463f99291'


class AntiDeepfake(torch.nn.Module):
    def __init__(self, directory: Path, device: str = 'cpu'):
        super().__init__()
        if torch.device(device).type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.set_float32_matmul_precision('highest')
        config = Wav2Vec2Config.from_json_file(str(directory/'antideepfake_config.json'))
        config.apply_spec_augment = False
        config._attn_implementation = 'eager'
        with torch.device('meta'):
            self.encoder = Wav2Vec2Model(config)
            self.classifier = torch.nn.Linear(config.hidden_size, 2)   # 1280 (1B) / 1920 (2B)
        native = {k: v.float() for k, v in load_file(str(directory/'model.safetensors')).items()}   # fp16 storage -> fp32 compute
        converted, unused = {}, []
        for name, value in native.items():
            if name.startswith('proj_fc.'):
                converted[name.replace('proj_fc.', 'classifier.')] = value
                continue
            key = name.removeprefix('m_ssl.model.')
            if key.startswith(('quantizer.', 'project_q.', 'final_proj.')):
                unused.append(name)
                continue
            match = re.fullmatch(r'feature_extractor.conv_layers.(\d+).(0|2.1).(weight|bias)', key)
            if match:
                layer, block, parameter = match.groups()
                key = f'feature_extractor.conv_layers.{layer}.{"conv" if block == "0" else "layer_norm"}.{parameter}'
            elif key.startswith('post_extract_proj.'):
                key = key.replace('post_extract_proj.', 'feature_projection.projection.')
            elif key.startswith('layer_norm.'):
                key = 'feature_projection.' + key
            elif key == 'mask_emb':
                key = 'masked_spec_embed'
            elif key.startswith('encoder.pos_conv.0.'):
                key = key.replace('encoder.pos_conv.0.', 'encoder.pos_conv_embed.conv.')
                key = key.replace('weight_g', 'parametrizations.weight.original0').replace('weight_v', 'parametrizations.weight.original1')
            elif key.startswith('encoder.layers.'):
                key = key.replace('.self_attn_layer_norm.', '.layer_norm.').replace('.self_attn.', '.attention.')
                key = key.replace('.fc1.', '.feed_forward.intermediate_dense.').replace('.fc2.', '.feed_forward.output_dense.')
            mapped = 'encoder.' + key
            if mapped in converted:
                raise ValueError(f'Duplicate mapped tensor: {mapped}')
            converted[mapped] = value
        self.load_state_dict(converted, strict=True, assign=True)
        self.to(device).eval().requires_grad_(False)
        self.unused_training_only = unused

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        waveform = torch.nn.functional.layer_norm(audio, audio.shape[-1:])
        hidden = self.encoder(waveform).last_hidden_state
        return self.classifier(hidden.mean(dim=1))

    def predict(self, audio: np.ndarray, deadline: float, window: int = 64600, batch: int = 16) -> list[float]:
        from math import ceil
        if not np.isfinite(audio).all() or len(audio) < 400:
            raise ValueError('Invalid decoded waveform')
        starts = list(range(0, max(1, len(audio)-window+1), window//2))
        if len(audio) > window and starts[-1] != len(audio)-window:
            starts.append(len(audio)-window)
        probabilities = []
        device = next(self.parameters()).device
        with torch.inference_mode():
            for offset in range(0, len(starts), batch):
                if time.time() >= deadline:
                    raise TimeoutError('AntiDeepfake inference deadline reached')
                # Windows share one length (only a sub-window file has a single shorter clip): no batch padding,
                # so native Fairseq pooling still includes every feature frame.
                clips = torch.from_numpy(np.stack([audio[s:s+window] for s in starts[offset:offset+batch]])).to(device)
                probabilities += self(clips).softmax(-1)[:, 0].tolist()
        top = sorted(probabilities, reverse=True)[:max(1, ceil(.3*len(probabilities)))]
        return [max(probabilities), float(np.mean(top)), *probabilities]


def verify_native(directory: Path, device: str = 'cpu') -> dict:
    """Protect class polarity, tensor mapping, normalization and pooling parity."""
    reference = json.loads((directory/'native_reference.json').read_text())
    model = AntiDeepfake(directory, device)
    errors = []
    with np.load(directory/'parity_audio.npz', allow_pickle=False) as audio:
        for name, expected in reference['logits'].items():
            with torch.inference_mode():
                actual = model(torch.from_numpy(audio[name]).to(device)[None]).cpu().numpy()[0]
            error = float(np.abs(actual-np.asarray(expected)).max())
            if not np.allclose(actual, expected, rtol=1e-4, atol=3e-4):
                raise ValueError(f'Native Fairseq parity failed on {name}: {actual}, {expected}')
            errors.append(error)
    return dict(max_logit_error=max(errors), real_audio_cases=len(errors), fake_class=0,
                native_revision=REVISION, unused_training_only=model.unused_training_only)


if __name__ == '__main__':
    import sys
    print(json.dumps(verify_native(Path(sys.argv[1]))))
