"""Re-create the two review findings from the user's text (not their missing attachment).

Run with PYTHONPATH pointing to either original or v2 src. All data are synthetic;
no GPU, native audio, service connection, or model substitution is used.
"""
from __future__ import annotations
import argparse
import builtins
import json
from pathlib import Path
import tempfile

from test_preprocessing_contract import cache_fixture, mutate_cache
from dacon_stacking.common import ContractError, NotRunError, read_json, sha256, write_json
from dacon_stacking import workflow, demo


def reproduce(output: Path, expect='fixed'):
    output.mkdir(parents=True, exist_ok=True)
    observations = {}
    with tempfile.TemporaryDirectory(prefix='stacking_v2_policy_') as work:
        for case in ('same_policy_distinct_measurements', 'mixed_flags', 'manifest_false_cache_true', 'all_true'):
            flags = (False, True) if case == 'mixed_flags' else ((True, True) if case == 'all_true' else (False, False))
            rows, _, index, _ = cache_fixture(Path(work)/case, flags=flags)
            if case == 'manifest_false_cache_true':
                mutate_cache(index, rows[1]['ID'], lambda p: p.update(preserve_gain=True))
            try:
                loaded = workflow.load_records(index, rows)
                observations[case] = {'outcome': 'ACCEPTED', 'rows': len(loaded),
                    'measured_peak_gain': [{k: r['provenance']['preprocessing'][k] for k in ('source_peak','normalization_gain')} for r in loaded]}
            except ContractError as exc:
                observations[case] = {'outcome': 'REJECTED', 'reason': str(exc)}
    original_import = builtins.__import__
    def without_xgboost(name, *args, **kwargs):
        if name.split('.')[0] == 'xgboost':
            raise ModuleNotFoundError('TEST ONLY: simulate XGBoost not installed', name=name)
        return original_import(name, *args, **kwargs)
    builtins.__import__ = without_xgboost
    try:
        result = demo.run_smoke(output/'missing_xgboost_smoke')
        observations['missing_xgboost_smoke'] = {'outcome': 'RETURNED', 'summary': result}
    except NotRunError as exc:
        observations['missing_xgboost_smoke'] = {'outcome': 'ABORTED', 'reason': str(exc)}
    finally:
        builtins.__import__ = original_import
    serial_path = output/'missing_xgboost_smoke/serialization_results.json'
    observations['missing_xgboost_smoke']['serialization_report_exists'] = serial_path.exists()
    observations['missing_xgboost_smoke']['cv_report_exists'] = (output/'missing_xgboost_smoke/synthetic_cv.json').exists()
    if expect == 'fixed':
        checks = [observations['same_policy_distinct_measurements']['outcome'] == 'ACCEPTED',
                  all(observations[c]['outcome'] == 'REJECTED' for c in ('mixed_flags','manifest_false_cache_true','all_true')),
                  observations['missing_xgboost_smoke']['outcome'] == 'RETURNED', serial_path.exists()]
        if serial_path.exists():
            serial = read_json(serial_path)
            checks += [set(serial) == {'lr','et','xgb'}, serial['xgb']['status'] == 'NOT_RUN', serial['xgb'].get('stage') == 'fit',
                       all(serial[k]['status'] in ('PASS','NOT_RUN') for k in ('lr','et'))]
    else:
        checks = [all(observations[c]['outcome'] == 'ACCEPTED' for c in ('same_policy_distinct_measurements','mixed_flags','manifest_false_cache_true','all_true')),
                  observations['missing_xgboost_smoke']['outcome'] == 'ABORTED', not serial_path.exists()]
    report = {'purpose': 'Synthetic regression reproduction only; no real audio or GPU parity',
              'source_of_reproducer': 'Written from user review text; reproduce_preprocessing_gap.py was not available',
              'expected': expect, 'status': 'PASS' if all(checks) else 'FAILED',
              'workflow_sha256': sha256(workflow.__file__), 'demo_sha256': sha256(demo.__file__),
              'observations': observations}
    write_json(output/'review_reproduction.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(checks) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expect', choices=('fixed','gaps'), default='fixed')
    args = parser.parse_args()
    raise SystemExit(reproduce(args.out, args.expect))
