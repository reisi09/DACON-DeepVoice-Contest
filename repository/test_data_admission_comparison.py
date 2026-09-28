"""Scored fixtures protect fit isolation; these are not real model quality measurements."""
import copy

import numpy as np
import pytest

from data_v4.compare_data_admission import fixed_f0_comparison
from data_v4.file_ensemble import fit, known_file_panel
from full_bank_evaluate import five_columns


def test_e00_calibration_freezes_f0_for_shifted_candidate_and_ignores_development_labels():
    # A shifted E04 scale must saturate the baseline CDF rather than silently be refitted.
    def row(identifier, role, label):
        return dict(ID=identifier, role=role, component_id=identifier, FILE_FAKE=str(label),
                    FILE_FAKE_mask='1', VOICE_FAKE=str(label), VOICE_PRESENT='1',
                    MUSIC_FAKE='', MUSIC_FAKE_mask='0', MUSIC_PRESENT='0')

    def scores(rows, values):
        return [dict(ID=r['ID'], Fv=value, Fm=value / 2, Pv=.9, Pm=.1, presence_features=None)
                for r, value in zip(rows, values, strict=True)]

    calibration = [row(f'cal{i}', 'calibration', int(i >= 2)) for i in range(4)]
    development = [row(f'dev{i}', 'development', int(i >= 2)) for i in range(4)]
    cal_scores = scores(calibration, [.1, .2, .3, .4])
    baseline = scores(development, [.15, .25, .35, .45])
    candidate = scores(development, [.65, .75, .85, .95])
    f0, predictions, report = fixed_f0_comparison(calibration, cal_scores, development, baseline, candidate)
    assert predictions['E04'] == [five_columns(s, ensemble=f0) for s in candidate]
    np.testing.assert_allclose([r['FILE_FAKE_PROB'] for r in predictions['E04']], 1., atol=1e-12)
    fit_files, labels = known_file_panel(development, candidate)
    incorrectly_refitted, _ = fit(fit_files, labels)
    assert min(five_columns(s, ensemble=incorrectly_refitted)['FILE_FAKE_PROB'] for s in candidate) < .1
    changed_labels = [dict(r, FILE_FAKE=str(1 - int(r['FILE_FAKE']))) for r in development]
    f0_again, predictions_again, _ = fixed_f0_comparison(calibration, cal_scores, changed_labels, baseline, candidate)
    assert f0_again == f0 and predictions_again == predictions, 'Development labels cannot affect fitting or outputs'
    assert report['metrics']['E00']['metrics']['file_eer'] == 0
    assert report['metrics']['E04']['metrics']['file_eer'] == .5
    # Unknown calibration truth must not shape CDF tables or weight selection.
    masked = dict(row('unknown', 'calibration', 1), FILE_FAKE_mask='0')
    masked_score = scores([masked], [.99])[0]
    f0_masked, _, _ = fixed_f0_comparison(calibration + [masked], cal_scores + [masked_score], development, baseline, candidate)
    assert f0_masked == f0
    contaminated = copy.deepcopy(calibration)
    contaminated[0]['component_id'] = development[0]['component_id']
    with pytest.raises(ValueError, match='overlap'):
        fixed_f0_comparison(contaminated, cal_scores, development, baseline, candidate)
    changed_presence = copy.deepcopy(candidate)
    changed_presence[0]['Pv'] = .8
    with pytest.raises(ValueError, match='presence changed'):
        fixed_f0_comparison(calibration, cal_scores, development, baseline, changed_presence)
