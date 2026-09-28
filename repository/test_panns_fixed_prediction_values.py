import hashlib
import json
from pathlib import Path

import pytest

import panns_fixed_predictions as fixed_module
from panns_adaptation_data import identity, sha256, write_json
from graph_scoring import PREDICTION_COLUMNS


def injected_engine_config(tmp_path, monkeypatch):
    package = tmp_path/'package'
    runtime = package/'model/runtime'
    runtime.mkdir(parents=True)
    module = runtime/'full_bank_submission.py'
    module.write_text('import json\nfrom pathlib import Path\n'
                      'from graph_scoring import validate_predictions\n'
                      'def baseline_module(assets):\n'
                      ' raise AssertionError("Substituted Engine must not load real audio models")\n'
                      'class Engine:\n'
                      ' def __init__(self,*args,**kwargs): self.root=Path(__file__).parents[3]\n'
                      ' def predict(self,path): return json.loads((self.root/"engine_output.json").read_text())\n')
    monkeypatch.syspath_prepend(str(runtime))
    reference = tmp_path/'reference.json'
    write_json(reference, dict(package=str(package), members={'model/runtime/full_bank_submission.py': sha256(module)}))
    audio = b'Fault-injection wrapper payload; substituted Engine does not decode it.'
    archive = tmp_path/'audio.bin'
    archive.write_bytes(audio)
    row = dict(ID='expected-row', role='development', path='audio.wav',
               verified_audio=dict(sha256=hashlib.sha256(audio).hexdigest(), tar_offset=0, bytes=len(audio)))
    rows = tmp_path/'rows.jsonl'
    rows.write_text(json.dumps(row)+'\n')
    config = dict(reference=str(reference), rows=str(rows), archive=str(archive), panel_ids=[row['ID']],
                  output=str(tmp_path/'output'), fixed_predictions=str(tmp_path/'output/fixed.json'),
                  inputs={str(reference): sha256(reference), str(rows): sha256(rows)})
    config['identity'] = identity(config)
    write_json(tmp_path/'engine_output.json', dict.fromkeys(PREDICTION_COLUMNS, .5))
    return config, row


@pytest.mark.parametrize('invalid', [float('nan'), -.01, 1.01])
def test_new_engine_invalid_probability_is_rejected_before_publication(tmp_path, monkeypatch, invalid):
    config, _ = injected_engine_config(tmp_path, monkeypatch)
    prediction = dict.fromkeys(PREDICTION_COLUMNS, .5)
    prediction['MUSIC_FAKE_PROB'] = invalid
    write_json(tmp_path/'engine_output.json', prediction)
    with pytest.raises(ValueError, match='finite'):
        fixed_module.fixed_predictions(config, 'cpu')
    assert not list((Path(config['output'])/'fixed_predictions').glob('*.json'))
    assert not Path(config['fixed_predictions']).exists()


@pytest.mark.parametrize('fault', ['nan', 'range', 'row_id'])
def test_self_consistent_invalid_partial_entry_is_rejected_before_resume(tmp_path, monkeypatch, fault):
    config, row = injected_engine_config(tmp_path, monkeypatch)
    fixed_module.fixed_predictions(config, 'cpu')
    path = Path(config['output'])/'fixed_predictions'/(hashlib.sha256(row['ID'].encode()).hexdigest()+'.json')
    record = json.loads(path.read_text())
    assert record['entry']['audio_sha256'] == row['verified_audio']['sha256']
    if fault == 'row_id':
        record['entry']['ID'] = 'wrong-row'
    else:
        record['entry']['MUSIC_FAKE_PROB'] = float('nan') if fault == 'nan' else 1.01
    record['entry_sha256'] = identity(record['entry'])
    write_json(path, record)
    original = path.read_bytes()
    Path(config['fixed_predictions']).unlink()
    with pytest.raises(ValueError, match='finite|IDs'):
        fixed_module.fixed_predictions(config, 'cpu')
    assert path.read_bytes() == original
    assert not Path(config['fixed_predictions']).exists()


def test_aggregate_consumer_rejects_self_consistent_invalid_values(tmp_path, monkeypatch):
    config, _ = injected_engine_config(tmp_path, monkeypatch)
    fixed_module.fixed_predictions(config, 'cpu')
    path = Path(config['fixed_predictions'])
    saved = json.loads(path.read_text())
    saved['entries'][0]['VOICE_FAKE_PROB'] = float('nan')
    saved['entries_sha256'] = identity(saved['entries'])
    write_json(path, saved)
    with pytest.raises(ValueError, match='finite'):
        fixed_module.load_fixed_predictions(config, 'cpu')
