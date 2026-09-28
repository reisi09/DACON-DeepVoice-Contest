"""Protect the measured report from false completion and incorrect panel weighting."""

import json
import shutil
from pathlib import Path

import pytest

from v5_analyze import analyze


def test_actual_budget_stop_and_duplicate_sensitivity_are_preserved(tmp_path: Path) -> None:
    source = Path('.omo/evidence/v5-colab-runtime/results')
    shutil.copy(source / 'comparison.json', tmp_path)
    for arm in ('B0', 'B1'):
        shutil.copy(source / f'{arm}_predictions.csv', tmp_path)
    shutil.copytree(source / 'jobs', tmp_path / 'jobs')
    result = analyze(tmp_path, Path('v4_validation'))
    receipt = json.loads((source / 'comparison.json').read_text())
    for arm in ('B0', 'B1'):
        assert result['arms'][arm]['original']['metrics'] == pytest.approx(
            receipt[arm]['metrics']['metrics'], abs=1e-12)
    assert result['arms']['B1']['status'] == 'evaluated_budget_censored'
    assert result['arms']['B1']['component_training_status']['voice'] == 'budget_stop'
    assert result['row_count'] == 230 and result['unique_audio_count'] == 210
    assert result['arms']['B0']['unique_audio']['metrics']['file_eer'] == pytest.approx(.33798076923076925)
    assert result['arms']['B0']['identical_audio_max_probability_delta'] == 0

    predictions = tmp_path / 'B1_predictions.csv'
    lines = predictions.read_text().splitlines()
    predictions.write_text('\n'.join(lines[:-1]) + '\n')
    with pytest.raises(ValueError, match='incomplete or duplicate prediction IDs'):
        analyze(tmp_path, Path('v4_validation'))
