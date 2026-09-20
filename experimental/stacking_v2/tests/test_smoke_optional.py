"""Defect: optional XGBoost final-fit failure discards the serialization report.
Import interception models an unavailable dependency, not a fake estimator;
LR/ExtraTrees still fit real sklearn estimators on synthetic wiring features.
"""
import builtins
import json
from pathlib import Path
import pytest


def block_imports(monkeypatch, *roots):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in roots:
            raise ModuleNotFoundError(f'v2 TEST ONLY: simulated missing {name}', name=name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def test_smoke_missing_xgboost_keeps_other_candidates_and_reports(tmp_path, monkeypatch):
    from dacon_stacking import demo
    from dacon_stacking.common import NotRunError
    block_imports(monkeypatch, 'xgboost')
    result = demo.run_smoke(tmp_path)
    cv = json.loads((tmp_path/'synthetic_cv.json').read_text())
    serial = json.loads((tmp_path/'serialization_results.json').read_text())
    assert result['serialization'] == serial
    assert set(serial) == {'lr', 'et', 'xgb'}
    assert serial['xgb']['status'] == 'NOT_RUN'
    assert serial['xgb']['stage'] == 'fit'
    assert 'XGBoost' in serial['xgb']['reason']
    for features in ('F0', 'F1'):
        assert cv['systems'][features+'/xgb']['status'] == 'NOT_RUN'
        for kind in ('lr', 'et'):
            assert cv['systems'][features+'/'+kind]['status'] == 'OK'
    for kind in ('lr', 'et'):
        assert serial[kind]['status'] in ('PASS', 'NOT_RUN')
        if serial[kind]['status'] == 'NOT_RUN':
            assert 'skops' in serial[kind]['reason']
            assert serial[kind]['stage'] in ('save', 'load')
        else:
            assert serial[kind]['max_abs_roundtrip_error'] <= 1e-12
    assert not list(tmp_path.rglob('*.pkl'))
    assert not list(tmp_path.rglob('*.pickle'))


@pytest.mark.parametrize('stage', ['fit', 'save', 'load'])
def test_smoke_records_notrun_at_each_candidate_stage(tmp_path, monkeypatch, stage):
    from dacon_stacking import demo
    from dacon_stacking.common import NotRunError
    target = {'fit': 'fit_final_dev', 'save': 'save_bundle', 'load': 'load_bundle'}[stage]
    original = getattr(demo, target)
    def maybe_unavailable(*args, **kwargs):
        kind = args[2] if stage == 'fit' else (args[1]['kind'] if stage == 'save' else Path(args[0]).name.removeprefix('synthetic_'))
        if kind == 'xgb':
            raise NotRunError(f'TEST ONLY: dependency unavailable during {stage}')
        return original(*args, **kwargs)
    monkeypatch.setattr(demo, target, maybe_unavailable)
    result = demo.run_smoke(tmp_path)
    assert result['serialization']['xgb'] == {'status': 'NOT_RUN', 'stage': stage,
                                            'reason': f'TEST ONLY: dependency unavailable during {stage}'}
    assert set(json.loads((tmp_path/'serialization_results.json').read_text())) == {'lr', 'et', 'xgb'}


def test_smoke_missing_skops_reports_notrun_without_pickle(tmp_path, monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from dacon_stacking import demo, models
    original_version = models.version
    def missing_skops(name):
        if name == 'skops': raise PackageNotFoundError('skops')
        return original_version(name)
    monkeypatch.setattr(models, 'version', missing_skops)
    block_imports(monkeypatch, 'skops')
    result = demo.run_smoke(tmp_path)
    for kind in ('lr', 'et'):
        entry = result['serialization'][kind]
        assert entry['status'] == 'NOT_RUN' and entry['stage'] == 'save'
        assert 'skops' in entry['reason'] and 'pickle fallback' in entry['reason']
        assert result['systems']['F1/'+kind] == 'OK'
    assert (tmp_path/'serialization_results.json').is_file()
    assert not list(tmp_path.rglob('*.pkl')) and not list(tmp_path.rglob('*.pickle'))
