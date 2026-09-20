"""Final data union must reject provenance/label drift and include the local holdout."""
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from run_v5_final import load_final_pool


def test_final_pool():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = []
        fields = ['ID', 'path', 'group_id', 'panel', 'FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT']
        for index, split in enumerate(('train', 'dev', 'sealed')):
            audio = root / f'{split}.wav'
            sf.write(audio, np.linspace(-.1, .1, 16000, dtype=np.float32) * (index + 1), 16000)
            row = dict(zip(fields, [split, audio.name, split, 'speech', '0', '0', '', '1', '0']))
            with (root / f'{split}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            sources.append({**row, 'license': 'CC-BY-4.0', 'full_path': audio.name})
        provenance = root / 'provenance.json'
        provenance.write_text(json.dumps({'sources': sources}))
        rows, audit = load_final_pool(root)
        assert {row.id for row in rows} == {'train', 'dev', 'sealed'}
        assert all(np.isnan(row.labels[2]) for row in rows)
        assert audit['independent_holdout'] is False
        sources[1]['VOICE_FAKE'] = '1'
        provenance.write_text(json.dumps({'sources': sources}))
        try:
            load_final_pool(root)
        except ValueError as error:
            assert 'label' in str(error).lower()
        else:
            raise AssertionError('Changed provenance label accepted')


if __name__ == '__main__':
    test_final_pool()
    print('PASS: all three local splits included; unknown labels preserved; label drift rejected')
