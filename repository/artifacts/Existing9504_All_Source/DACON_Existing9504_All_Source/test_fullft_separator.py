"""Decision checks for separator lineage and real differentiable checkpoint selection."""
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from demucs.apply import BagOfModels


class TinySeparator(nn.Sequential):
    sources = ["drums", "bass", "other", "vocals"]
    audio_channels = 2
    samplerate = 44100

    def __init__(self):
        super().__init__(nn.Conv1d(2, 8, 1), nn.Unflatten(1, (4, 2)))


class ZeroGradientSeparator(TinySeparator):
    def forward(self, value):
        return super().forward(value) * 0


def test_remix_lineage_and_optimization() -> None:
    # Given independent parent clips and a small real differentiable separator.
    import fullft_separator as adaptation
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clips = []
        for index in range(8):
            wave = torch.sin(torch.linspace(0, 20, 128))[None].repeat(2, 1)
            target = torch.stack([wave * (.03 * (stem + 1) + index * .001) for stem in range(4)])
            path = root / f'{index}.pt'
            torch.save(target, path)
            clips.append(adaptation.StemClip(str(index), frozenset({str(index)}), path))
        torch.manual_seed(41)
        model = BagOfModels([TinySeparator()])
        config = {'output': str(root), 'deadline': time.time() + 60, 'seed': 41,
                  'separator_steps': 15, 'separator_lr': .03, 'separator_validation_every': 1}
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        # When true backpropagation optimizes remixed pseudo-targets.
        path = adaptation.fit_remixes(model, clips[:4], clips[4:], config)
        state = torch.load(path, weights_only=False)
        # Then held-parent selection follows measured loss and weights actually change.
        assert state['pseudo_validation_l1'] == min(entry['pseudo_validation_l1'] for entry in json.loads((path.parent / 'history.json').read_text()) if entry['step'])
        assert state['pseudo_validation_l1'] < .8 * state['history'][0]['pseudo_validation_l1']
        assert any(not torch.equal(initial[key], value) for key, value in state['model'].items())
        evidence = state['gradient_evidence']
        assert evidence['optimizer_parameter_count'] == evidence['model_parameter_count']
        assert evidence['nonzero_gradient_parameters'] > 0 and evidence['parameter_update_l2'] > 0
        assert all(len(set(entry['parents'])) == 4 for entry in state['validation_lineage'])
        assert {parent for entry in state['validation_lineage'] for parent in entry['parents']} == {'4', '5', '6', '7'}
        restored = BagOfModels([TinySeparator()])
        restored.load_state_dict(state['model'], strict=True)
        model.load_state_dict(state['model'], strict=True)
        sample, _ = adaptation.remix(clips[4:], __import__('random').Random(41))
        assert torch.equal(adaptation.forward_stems(model, sample.sum(1)), adaptation.forward_stems(restored, sample.sum(1)))
        try:
            adaptation.fit_remixes(BagOfModels([ZeroGradientSeparator()]), clips[:4], clips[4:], {**config, 'output': str(root / 'broken')})
        except adaptation.SeparatorError as error:
            assert 'gradient gate' in str(error)
        else:
            raise AssertionError('A disconnected separator was accepted')
        # Duplicate parent sources cannot silently produce a supposedly independent remix.
        duplicate = [adaptation.StemClip(clip.identifier, frozenset({'same'}), clip.path) for clip in clips[:4]]
        try:
            adaptation.remix(duplicate, __import__('random').Random(41))
        except adaptation.SeparatorError as error:
            assert 'parent-disjoint' in str(error)
        else:
            raise AssertionError('A remix with reused parents was accepted')


if __name__ == '__main__':
    test_remix_lineage_and_optimization()
    print('Validated parent-disjoint remix, gradient/update, and held-parent selection.')
