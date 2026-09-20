"""CPU-only parity reference in the isolated Python 3.10/Fairseq environment."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model


def main(directory: Path) -> None:
    torch.set_num_threads(4)
    cfg = Wav2Vec2Config(quantize_targets=True, extractor_mode='layer_norm', layer_norm_first=True,
        final_dim=1024, latent_temp=(2.0,.1,.999995), encoder_layerdrop=0., dropout_input=0.,
        dropout_features=0., dropout=0., attention_dropout=0., conv_bias=True,
        encoder_layers=48, encoder_embed_dim=1280, encoder_ffn_embed_dim=5120,
        encoder_attention_heads=16, feature_grad_mult=1.)
    model = Wav2Vec2Model(cfg).eval().requires_grad_(False)
    state = load_file(str(directory/'model.safetensors'))
    model.load_state_dict({k.removeprefix('m_ssl.model.'): v for k,v in state.items()
                           if k.startswith('m_ssl.model.')}, strict=True)
    logits = {}
    started = time.time()
    with np.load(directory/'parity_audio.npz', allow_pickle=False) as audio, torch.inference_mode():
        for name in audio.files:
            waveform = torch.from_numpy(audio[name])[None]
            waveform = torch.nn.functional.layer_norm(waveform, waveform.shape[-1:])
            features = model(waveform, mask=False, features_only=True)['x'].mean(1)
            values = torch.nn.functional.linear(features, state['proj_fc.weight'], state['proj_fc.bias'])[0]
            logits[name] = values.tolist()
            print(name, logits[name], flush=True)
    (directory/'native_reference.json').write_text(json.dumps(dict(logits=logits,
        elapsed_seconds=time.time()-started, torch=torch.__version__, fake_class=0), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
