# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "nbformat"]
# ///
# Run: .venv/bin/python test_fullft8_batching.py
"""Protect optimizer weighting, sample order and epoch crops after batching."""
import importlib
from pathlib import Path
import sys
import tempfile

import nbformat
import numpy as np
import torch


def check() -> None:
    notebook = nbformat.read('MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb', 4)
    source = '\n\n'.join(c.source for c in notebook.cells if 'worker' in c.metadata.get('tags', []))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root/'batch_worker.py').write_text(source)
        sys.path.insert(0, str(root))
        worker = importlib.import_module('batch_worker')
        torch.manual_seed(41)
        inputs, labels = torch.randn(13, 3, dtype=torch.float64), torch.arange(13) % 2
        initial = torch.nn.Linear(3, 2).double().state_dict()
        final = []
        for batch, accumulation in [(1, 8), (4, 2)]:
            model = torch.nn.Linear(3, 2).double()
            model.load_state_dict(initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            config = dict(batch_size=batch, accumulation=accumulation)
            steps = 0
            for index, start in enumerate(range(0, len(inputs), batch)):
                x, y = inputs[start:start+batch], labels[start:start+batch]
                loss = torch.nn.functional.cross_entropy(model(x), y)
                (loss * worker.batch_loss_weight(index, len(x), len(inputs), config)).backward()
                if (index+1) % accumulation == 0 or start+batch >= len(inputs):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
            assert steps == 2
            final.append(model.state_dict())
        for key in final[0]:
            torch.testing.assert_close(final[0][key], final[1][key], rtol=1e-12, atol=1e-12)

        cache = root/'audio_cache'
        cache.mkdir()
        rows = [dict(ID=f'row-{i}', MUSIC_FAKE=str(i % 2)) for i in range(13)]
        for i, row in enumerate(rows):
            values = np.arange(8*12, dtype='float32').reshape(8, 12) + i*1000
            np.savez(cache/f'{row["ID"]}.npz', train=values, evaluation=values)
        config = dict(output=str(root), seed=41, train_windows=4, batch_size=4, loader_workers=4)
        data = worker.AudioBags(rows, config, True)
        previous = None
        for epoch in [0, 1]:
            data.epoch = epoch
            batches = list(worker.batch_loader(data, config))
            actual = torch.cat([x for x, _ in batches])
            expected = torch.stack([data[i][0] for i in range(len(data))])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(torch.cat([y for _, y in batches]), torch.arange(13) % 2)
            if previous is not None:
                assert not torch.equal(previous, actual), 'Workers reused stale epoch crops'
            previous = actual
        print('PASS: equivalent updates including partial group; spawn workers preserve order and epoch crops.')


if __name__ == '__main__':
    check()
