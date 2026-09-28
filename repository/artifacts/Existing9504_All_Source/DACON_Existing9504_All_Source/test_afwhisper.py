"""Protect widened head optimization and AF-Whisper valid-frame pooling."""
from pathlib import Path

import torch
from music_temporal import TemporalMusic, load_temporal_head


def test_wider_audio_features_reach_loss_and_survive_reload(tmp_path: Path):
    torch.manual_seed(41)
    model = TemporalMusic(0., simplified=True, compact=True, softmax_output=True,
                          mlp_output=True, input_dim=1280).eval()
    x = torch.randn(4, 6, 1280)
    mask = torch.zeros(4, 6, dtype=torch.bool);mask[:, -2:] = True
    logits = model(x, mask)
    poisoned = x.clone();poisoned[mask] = float('nan')
    torch.testing.assert_close(logits, model(poisoned, mask), atol=0, rtol=0)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0,1,0,1]), label_smoothing=.05).backward()
    assert model.projection.weight.grad[:,1024:].abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.output.parameters())
    path = tmp_path/'temporal_music.pt'
    torch.save(dict(architecture=model.architecture, dropout=0., state_dict=model.state_dict()),path)
    torch.testing.assert_close(logits, load_temporal_head(path)(x,mask),atol=0,rtol=0)


def test_trimmed_encoder_matches_upstream_masked_features():
    # Removing masked keys must preserve valid embeddings, including convolution boundaries.
    import importlib.util
    import sys
    from transformers import Qwen2AudioEncoderConfig
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
    from afwhisper_features import pooled_features

    path=Path(__file__).parent/'results/afwhisper_a100_20260916/source/modeling_whisper.py'
    spec=importlib.util.spec_from_file_location('official_afwhisper',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=Qwen2AudioEncoderConfig(d_model=32,encoder_layers=2,encoder_attention_heads=4,
        encoder_ffn_dim=64,num_mel_bins=8,max_source_positions=20,pad_token_id=0,init_std=.02)
    config._attn_implementation='eager'
    torch.manual_seed(41)
    upstream=module.AFWhisperEncoder(config).eval()
    native=Qwen2AudioEncoder(config).eval();native.load_state_dict(upstream.state_dict(),strict=True)
    mel=torch.randn(2,8,40);mask=torch.zeros(2,1,20,20);mask[:,:,:,8:]=-torch.inf
    with torch.inference_mode():
        expected=upstream(mel,attention_mask=mask).last_hidden_state[:,:4].mean(1)
        native_full=native(mel,attention_mask=mask).last_hidden_state[:,:4].mean(1)
        trimmed=pooled_features(native,mel,16)
    torch.testing.assert_close(native_full,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(trimmed,expected,atol=1e-6,rtol=1e-6)
