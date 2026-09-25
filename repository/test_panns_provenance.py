import copy
import json
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pytest
import torch

import panns_adaptation_cache as cache_module
import panns_fixed_predictions as fixed_module
from panns_adaptation_data import cache_provenance, identity, sha256, validate_config, write_json


def cache_artifact(tmp_path):
    config = dict(identity='locked-config', output=str(tmp_path), cache_device='cpu')
    row = dict(ID='audio:a', role='train', component_id='parent:a', VOICE_PRESENT=1,
               MUSIC_PRESENT=0, verified_audio=dict(sha256='audio-hash'))
    starts = [0, 64600]
    key = identity(dict(config=config['identity'], provenance=cache_provenance(),
        audio=row['verified_audio']['sha256'], starts=starts,
        preprocessing='preserved baseline gain .6, soxr_hq', dtype='float32'))
    path = tmp_path/'cached.npz'
    np.savez(path, identity=key, block5=np.ones((2, 1), np.float32),
             tags=np.full((2, 1), .4, np.float32), starts=starts)
    entry = dict(ID=row['ID'], role=row['role'], component_id=row['component_id'],
                 targets=[1, 0], identity=key, path=str(path), sha256=sha256(path),
                 stems=dict(mix_rms_db=-23.), tag_error=0., stats_error=0., presence_error=0.)
    saved = dict(config_identity=config['identity'], provenance=cache_provenance(),
                 entry_sha256=identity(entry), entry=entry)
    return config, row, saved


def test_partial_cache_refuses_old_runtime_before_adoption(tmp_path):
    # Given an intact activation whose sidecar was produced by a different runtime.
    config, row, saved = cache_artifact(tmp_path)
    saved['provenance']['versions']['torch'] = 'stale-runtime'
    # When resuming, Then no intact payload may be relabelled with current provenance.
    with pytest.raises(ValueError, match='provenance'):
        cache_module.validate_partial_entry(config, row, saved)


def test_cache_key_must_bind_current_audio_and_segment_starts(tmp_path):
    # Given an intact payload/sidecar with a key for another audio identity.
    config, row, saved = cache_artifact(tmp_path)
    row['verified_audio']['sha256'] = 'different-audio'
    # When checking its reuse, Then its valid payload hash cannot authorize it.
    with pytest.raises(ValueError, match='cache key'):
        cache_module.validate_partial_entry(config, row, saved)


def prediction_config(tmp_path):
    reference = tmp_path/'reference.json'; reference.write_text('{}')
    rows = tmp_path/'rows.jsonl'; rows.write_text('')
    return dict(identity='fixed-config', reference=str(reference), rows=str(rows),
                fixed_predictions=str(tmp_path/'fixed.json'))


def test_fixed_prediction_consumer_refuses_stale_dependency_contract(tmp_path):
    # Given saved predictions whose producer versions differ from this consumer.
    config = prediction_config(tmp_path)
    fixed_module.configure_numerics('cpu')
    contract = fixed_module.fixed_prediction_contract(config, 'cpu')
    stale = copy.deepcopy(contract)
    stale['runtime']['versions']['torch'] = 'stale-runtime'
    write_json(config['fixed_predictions'], dict(**stale, entries=[], entries_sha256=identity([])))
    # When loading predictions, Then reject them even with unchanged rows/reference.
    with pytest.raises(ValueError, match='provenance'):
        fixed_module.load_fixed_predictions(config, 'cpu')


def test_fixed_predictions_apply_reference_numerics(tmp_path):
    # Given an ambient process with TF32 and benchmark enabled.
    config = prediction_config(tmp_path)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # When configuring the actual producer, Then its contract records disabled flags.
    fixed_module.configure_numerics('cpu')
    contract = fixed_module.fixed_prediction_contract(config, 'cpu')
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cudnn.benchmark
    assert contract['numerics']['matmul_tf32'] is False


def test_fixed_prediction_consumer_rejects_payload_mutation(tmp_path):
    # Given a prediction payload changed after its contract was sealed.
    config = prediction_config(tmp_path)
    fixed_module.configure_numerics('cpu')
    contract = fixed_module.fixed_prediction_contract(config, 'cpu')
    entries = [dict(ID='a', VOICE_FAKE_PROB=.2)]
    write_json(config['fixed_predictions'], dict(**contract, entries=entries, entries_sha256=identity(entries)))
    entries[0]['VOICE_FAKE_PROB'] = .9
    payload = json.loads(Path(config['fixed_predictions']).read_text())
    payload['entries'] = entries
    write_json(config['fixed_predictions'], payload)
    # When loading it, Then reject modified numerical values despite a valid runtime.
    with pytest.raises(ValueError, match='payload'):
        fixed_module.load_fixed_predictions(config, 'cpu')


def test_changed_cdf_is_rejected_before_presence_evaluation(tmp_path):
    # Given a locked ensemble whose CDF bytes change after the reference is pinned.
    name = 'model/trained/file_ensemble.json'
    ensemble = tmp_path/name
    write_json(ensemble, dict(cdf=[0., .3, 1.]))
    reference = tmp_path/'reference.json'
    write_json(reference, dict(package=str(tmp_path), members={name: sha256(ensemble)}))
    config = dict(reference=str(reference), inputs={str(reference): sha256(reference)})
    config['identity'] = identity(config)
    write_json(ensemble, dict(cdf=[0., .8, 1.]))
    # When validating actual evaluation inputs, Then unchanged config bytes cannot hide CDF drift.
    with pytest.raises(ValueError, match='Reference asset hash mismatch'):
        validate_config(config)


@pytest.mark.parametrize('field', ['activations', 'tags', 'stems', 'errors'])
def test_nonfinite_cache_values_cannot_pass_parity(field: Literal['activations', 'tags', 'stems', 'errors']):
    arrays = [np.array([.4]), np.array([.2])]
    stems, errors = dict(mix_rms_db=-23.), [0., 0., 0.]
    match field:
        case 'activations': arrays[0][0] = np.nan
        case 'tags': arrays[1][0] = np.nan
        case 'stems': stems['mix_rms_db'] = np.nan
        case 'errors': errors[0] = np.nan
        case unreachable: assert_never(unreachable)
    with pytest.raises(ValueError, match='Nonfinite'):
        cache_module.validate_cache_numerics(arrays, stems, errors)
